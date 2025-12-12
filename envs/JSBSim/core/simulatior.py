import os
import logging
import numpy as np
from collections import deque
from abc import ABC, abstractmethod
from typing import Literal, Union, List, Dict
from dataclasses import dataclass
from enum import Enum, auto

# 引入 JSBSim 和项目内部工具
import jsbsim
from .catalog import Property, Catalog
from ..utils.utils import get_root_dir, LLA2NEU, NEU2LLA

# 定义队伍颜色类型
TeamColors = Literal["Red", "Blue", "Green", "Violet", "Orange"]


# ==========================================
# 1. 导弹配置系统 (Missile Configuration)
#    通过配置类将导弹参数与逻辑代码分离，方便扩展新型号
# ==========================================

@dataclass
class MissileConfig:
    name: str
    mass_0: float  # 初始质量 (kg)
    mass_loss_rate: float  # 燃油消耗率 (kg/s)
    thrust_duration: float  # 发动机工作时间 (s)
    isp: float  # 比冲 (s), 衡量发动机效率
    diameter: float  # 弹体直径 (m), 用于计算阻力截面积
    length: float  # 弹体长度 (m)
    drag_coeff: float  # 零升阻力系数 (简化气动模型)
    max_g: float  # 最大可用过载 (G), 决定导弹机动性
    seeker_fov: float  # 导引头视场半角 (度)。例如 20度表示圆锥角 40度。
    # 超过这个角度导弹就会丢失目标(脱锁)。
    nav_gain: float  # 比例导引系数 (Navigation Constant), 通常为 3.0-5.0
    explosion_radius: float  # 杀伤半径/近炸引信距离 (m)
    max_life_time: float  # 最大生存时间 (s), 超过时间自毁


# 导弹参数数据库
MISSILE_DB = {
    # === AIM-9L (响尾蛇) ===
    # 特点：红外制导，射程短，但视场较宽(离轴发射能力)，适合近距格斗
    "AIM-9L": MissileConfig(
        name="AIM-9L",
        mass_0=84.0,
        mass_loss_rate=6.0,
        thrust_duration=5.0,
        isp=180.0,
        diameter=0.127,
        length=2.87,
        drag_coeff=0.4,
        max_g=30.0,
        seeker_fov=40.0,  # 较宽的视场，不容易脱锁
        nav_gain=3.0,
        explosion_radius=200.0,
        max_life_time=60.0
    ),

    # === AIM-120B (AMRAAM) ===
    # 特点：中程主动雷达制导，速度快，射程远，但末端雷达视场较窄。
    # 模拟策略：发射即主动(Mad Dog)，如果敌机进行大幅度机动(Notching/Beaming)导致角度变化过大，容易脱锁。
    "AIM-120B": MissileConfig(
        name="AIM-120B",
        mass_0=152.0,
        mass_loss_rate=8.0,
        thrust_duration=10.0,
        isp=245.0,  # 更高的比冲，速度更快
        diameter=0.18,
        length=3.66,
        drag_coeff=0.35,  # 气动外形更好
        max_g=28.0,
        seeker_fov=20.0,  # 较窄的视场，对敌机机动敏感 (核心博弈点)
        nav_gain=4.0,  # 导引律更激进
        explosion_radius=400.0,
        max_life_time=120.0
    )
}


# ==========================================
# 2. 基础仿真类 (BaseSimulator)
#    飞机和导弹的父类，处理通用的位置、速度和属性
# ==========================================

class BaseSimulator(ABC):
    def __init__(self, uid: str, color: TeamColors, dt: float):
        """
        构造函数
        :param uid: 唯一标识符
        :param color: 队伍颜色 (用于敌我识别)
        :param dt: 仿真步长
        """
        self.__uid = uid
        self.__color = color
        self.__dt = dt
        self.model = ""
        self._geodetic = np.zeros(3)  # 经纬高 (Lon, Lat, Alt)
        self._position = np.zeros(3)  # 笛卡尔坐标 (North, East, Up)
        self._posture = np.zeros(3)  # 姿态 (Roll, Pitch, Yaw)
        self._velocity = np.zeros(3)  # 速度 (V_north, V_east, V_up)
        logging.debug(f"{self.__class__.__name__}:{self.__uid} is created!")

    @property
    def uid(self) -> str: return self.__uid

    @property
    def color(self) -> str: return self.__color

    @property
    def dt(self) -> float: return self.__dt

    def get_geodetic(self): return self._geodetic

    def get_position(self): return self._position

    def get_rpy(self): return self._posture

    def get_velocity(self): return self._velocity

    def reload(self):
        """重置状态"""
        self._geodetic = np.zeros(3)
        self._position = np.zeros(3)
        self._posture = np.zeros(3)
        self._velocity = np.zeros(3)

    @abstractmethod
    def run(self, **kwargs):
        pass

    def log(self):
        """生成日志字符串"""
        lon, lat, alt = self.get_geodetic()
        roll, pitch, yaw = self.get_rpy() * 180 / np.pi
        log_msg = f"{self.uid},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
        log_msg += f"Name={self.model.upper()},"
        log_msg += f"Color={self.color}"
        return log_msg

    @abstractmethod
    def close(self):
        pass

    def __del__(self):
        logging.debug(f"{self.__class__.__name__}:{self.uid} is deleted!")


# ==========================================
# 3. 雷达系统 (Radar Systems)
#    包含全向雷达(SA)和火控雷达(FCR)
# ==========================================

class OmniRadar:
    """
    全向态势感知雷达 (Situational Awareness Radar / Omni Radar)
    用途：模拟预警机数据链或RWR。
    特点：360度无死角，只要在距离内就能发现。
    功能：告诉AI“敌人在哪里”，用于战术决策。
    """

    def __init__(self, detection_range: float = 100000.0):
        self.detection_range = detection_range

    def scan(self, owner: 'BaseSimulator', all_aircrafts: List['BaseSimulator']) -> List['BaseSimulator']:
        detected = []
        owner_pos = owner.get_position()
        for aircraft in all_aircrafts:
            # 排除自己、队友、死人
            if aircraft.uid == owner.uid: continue
            if aircraft.color == owner.color: continue
            if not getattr(aircraft, 'is_alive', True): continue

            # 简单的距离判断
            dist = np.linalg.norm(aircraft.get_position() - owner_pos)
            if dist <= self.detection_range:
                detected.append(aircraft)
        return detected


class FireControlRadar:
    """
    火控雷达 (Fire Control Radar - FCR)
    用途：模拟飞机机头的雷达。
    特点：受视场角(FOV)限制，只能锁定机头前方的目标。具有多目标跟踪能力(TWS)。
    功能：只有被此雷达锁定的目标，才能作为导弹的攻击对象。
    """

    def __init__(self, max_range: float = 85000.0, scan_angle: float = 60.0, max_targets: int = 4):
        """
        :param max_range: 最大锁定距离 (m)
        :param scan_angle: 扫描半角 (度), +/- 60度是典型值
        :param max_targets: 最大同时锁定数量 (模拟雷达通道数)
        """
        self.max_range = max_range
        self.scan_angle = scan_angle
        self.max_targets = max_targets

    def _get_forward_vector(self, rpy):
        """根据姿态(Roll/Pitch/Yaw)计算机头指向向量"""
        # 注意：这里假设 rpy 单位是弧度，且遵循 standard JSBSim/Flight dynamics definition
        _, theta, psi = rpy
        # 在 NEU (North East Up) 坐标系下的指向计算
        vx = np.cos(theta) * np.cos(psi)
        vy = np.cos(theta) * np.sin(psi)
        vz = np.sin(theta)
        return np.array([vx, vy, vz])

    def scan(self, owner: 'BaseSimulator', all_aircrafts: List['BaseSimulator']) -> List['BaseSimulator']:
        """
        扫描并返回已锁定的目标列表
        """
        candidates = []
        owner_pos = owner.get_position()
        # 获取本机当前的机头朝向向量
        owner_dir = self._get_forward_vector(owner.get_rpy())

        for aircraft in all_aircrafts:
            if aircraft.uid == owner.uid: continue
            if aircraft.color == owner.color: continue
            if not getattr(aircraft, 'is_alive', True): continue

            # 1. 计算相对位置向量
            dist_vec = aircraft.get_position() - owner_pos
            dist = np.linalg.norm(dist_vec)

            # 2. 距离检查
            if dist > self.max_range: continue

            # 3. 角度检查 (Field of View Check)
            dist_unit = dist_vec / (dist + 1e-6)
            # 计算 机头指向 与 目标方向 的夹角余弦值
            dot_prod = np.clip(np.dot(owner_dir, dist_unit), -1.0, 1.0)
            angle = np.degrees(np.arccos(dot_prod))

            # 只有在雷达扫描锥内的目标才能被锁定
            if angle <= self.scan_angle:
                candidates.append((dist, aircraft))

        # 4. 多目标逻辑：按距离排序，锁定最近的 N 个目标
        candidates.sort(key=lambda x: x[0])
        locked_targets = [c[1] for c in candidates[:self.max_targets]]

        return locked_targets


# ==========================================
# 4. 飞机仿真类 (AircraftSimulator)
#    封装 JSBSim 并集成雷达系统
# ==========================================

class AircraftSimulator(BaseSimulator):
    """
    封装 JSBSim 实例，管理飞机状态、雷达和武器
    """
    ALIVE = 0
    CRASH = 1  # 坠毁 (地形碰撞或过载解体)
    SHOTDOWN = 2  # 被击落

    def __init__(self, uid: str = "A0100", color: TeamColors = "Red", model: str = 'f16',
                 init_state: dict = {}, origin: tuple = (120.0, 60.0, 0.0),
                 sim_freq: int = 60, **kwargs):
        super().__init__(uid, color, 1 / sim_freq)
        self.model = model
        self.init_state = init_state
        self.lon0, self.lat0, self.alt0 = origin
        self.bloods = 100
        self.__status = AircraftSimulator.ALIVE

        # --- 初始化双雷达系统 ---
        # 1. 全向雷达：100km 范围，用于感知态势
        self.omni_radar = OmniRadar(detection_range=100000.0)
        # 2. 火控雷达：85km 范围，+/-60度视场，用于武器锁定
        self.fcr = FireControlRadar(max_range=85000.0, scan_angle=60.0, max_targets=4)

        # === [修改位置 1] 新增告警标志属性 ===
        self.is_missile_locked = False      # 【致命】被导弹死死咬住 (RWR 急促报警)
        self.is_radar_locked = False        # 【危险】被敌方火控雷达锁定 (RWR 持续报警)
        self.is_radar_detected = False      # 【警告】被敌方全向/预警雷达发现 (态势感知)

        # 武器挂载配置
        for key, value in kwargs.items():
            if key == 'num_missiles':
                self.num_missiles = value
                self.num_left_missiles = self.num_missiles

        # 链接关系
        self.partners = []  # 队友列表
        self.enemies = []  # 敌人列表
        self.launch_missiles = []  # 我发射的导弹
        self.under_missiles = []  # 正在攻击我的导弹

        # 加载 JSBSim 模型
        self.reload()

    # --- 状态属性 ---
    @property
    def is_alive(self):
        return self.__status == AircraftSimulator.ALIVE

    @property
    def is_crash(self):
        return self.__status == AircraftSimulator.CRASH

    @property
    def is_shotdown(self):
        return self.__status == AircraftSimulator.SHOTDOWN

    def crash(self):
        self.__status = AircraftSimulator.CRASH

    def shotdown(self):
        self.__status = AircraftSimulator.SHOTDOWN

    def reload(self, new_state: Union[dict, None] = None, new_origin: Union[tuple, None] = None):
        """重载 JSBSim 模拟器，重置飞机状态"""
        super().reload()
        self.bloods = 100
        self.__status = AircraftSimulator.ALIVE
        self.launch_missiles.clear()
        self.under_missiles.clear()
        self.num_left_missiles = getattr(self, 'num_missiles', 4)

        # 初始化 JSBSim
        self.jsbsim_exec = jsbsim.FGFDMExec(os.path.join(get_root_dir(), 'data'))
        self.jsbsim_exec.set_debug_level(0)
        self.jsbsim_exec.load_model(self.model)
        Catalog.add_jsbsim_props(self.jsbsim_exec.query_property_catalog(""))
        self.jsbsim_exec.set_dt(self.dt)
        self.clear_defalut_condition()

        # 设置初始条件
        if new_state is not None: self.init_state = new_state
        if new_origin is not None: self.lon0, self.lat0, self.alt0 = new_origin
        for key, value in self.init_state.items():
            self.set_property_value(Catalog[key], value)

        success = self.jsbsim_exec.run_ic()
        if not success:
            raise RuntimeError("JSBSim failed to init simulation conditions.")

        # 引擎初始化
        propulsion = self.jsbsim_exec.get_propulsion()
        for j in range(propulsion.get_num_engines()):
            propulsion.get_engine(j).init_running()
        propulsion.get_steady_state()

        self._update_properties()

    def clear_defalut_condition(self):
        """设置默认的初始飞行条件"""
        default_condition = {
            Catalog.ic_long_gc_deg: 120.0, Catalog.ic_lat_geod_deg: 60.0, Catalog.ic_h_sl_ft: 20000,
            Catalog.ic_psi_true_deg: 0.0, Catalog.ic_u_fps: 800.0, Catalog.ic_v_fps: 0.0, Catalog.ic_w_fps: 0.0,
            Catalog.ic_p_rad_sec: 0.0, Catalog.ic_q_rad_sec: 0.0, Catalog.ic_r_rad_sec: 0.0,
            Catalog.ic_roc_fpm: 0.0, Catalog.ic_terrain_elevation_ft: 0,
        }
        for prop, value in default_condition.items():
            self.set_property_value(prop, value)

    def run(self):
        """
        运行一步仿真
        """
        if self.is_alive:
            if self.bloods <= 0: self.shotdown()

            # JSBSim 步进
            result = self.jsbsim_exec.run()
            # 注意: 这里根据 JSBSim版本不同，返回 False 可能意味着结束也可能意味着错误
            # 这里为了稳健通常只更新属性
            if not result: pass

            self._update_properties()

            self.update_warnings()
            return True
        else:
            return True

    def update_warnings(self):
        """
        每一帧调用，更新被探测和被锁定的状态标志
        """
        # 1. 重置状态
        self.is_missile_locked = False
        self.is_radar_locked = False
        self.is_radar_detected = False

        # 2. 检查导弹威胁 (Missile Warning)
        for missile in self.under_missiles:
            if missile.is_alive and missile.is_locking:
                self.is_missile_locked = True
                break  # 只要有一个导弹锁定，就是最高危状态

        # 3. 检查雷达威胁 (Radar Warning)
        # 遍历所有敌人，看谁“看”到了我
        for enemy in self.enemies:
            if not enemy.is_alive:
                continue

            # A. 检查是否被全向雷达(Omni)发现
            # 我们利用 radar.scan 传入 [self] 来快速检测自己是否在对方视野内
            if not self.is_radar_detected:
                # 如果对方扫到了我
                if len(enemy.omni_radar.scan(enemy, [self])) > 0:
                    self.is_radar_detected = True

            # B. 检查是否被火控雷达(FCR)锁定
            if not self.is_radar_locked:
                # 如果对方火控雷达锁定了 (即我在他的扫描列表里)
                if len(enemy.fcr.scan(enemy, [self])) > 0:
                    self.is_radar_locked = True
                    self.is_radar_detected = True  # 被火控锁定肯定也被发现了

            # 优化：如果三个状态全满了，就不用继续循环了
            if self.is_missile_locked and self.is_radar_locked and self.is_radar_detected:
                break

    def close(self):
        """关闭仿真释放资源"""
        if self.jsbsim_exec: self.jsbsim_exec = None
        self.partners = [];
        self.enemies = []

    def _update_properties(self):
        """从 JSBSim 读取最新状态同步到 Python 对象"""
        self._geodetic[:] = self.get_property_values(
            [Catalog.position_long_gc_deg, Catalog.position_lat_geod_deg, Catalog.position_h_sl_m])
        self._position[:] = LLA2NEU(*self._geodetic, self.lon0, self.lat0, self.alt0)
        self._posture[:] = self.get_property_values(
            [Catalog.attitude_roll_rad, Catalog.attitude_pitch_rad, Catalog.attitude_heading_true_rad])
        self._velocity[:] = self.get_property_values(
            [Catalog.velocities_v_north_mps, Catalog.velocities_v_east_mps, Catalog.velocities_v_down_mps])
        # JSBSim 的 Z 轴向下 (Down)，NEU 坐标系 Z 轴向上 (Up)，需要翻转
        self._velocity[2] = -self._velocity[2]

    # --- 属性读写辅助方法 ---
    def get_property_values(self, props):
        return [self.get_property_value(p) for p in props]

    def set_property_values(self, props, values):
        for p, v in zip(props, values): self.set_property_value(p, v)

    def get_property_value(self, prop):
        if isinstance(prop, Property): return self.jsbsim_exec.get_property_value(prop.name_jsbsim)
        raise ValueError(f"Unknown prop: {prop}")

    def set_property_value(self, prop, value):
        if isinstance(prop, Property):
            value = max(prop.min, min(prop.max, value))
            self.jsbsim_exec.set_property_value(prop.name_jsbsim, value)
        else:
            raise ValueError(f"Unknown prop: {prop}")

    def check_missile_warning(self):
        """
        RWR (雷达告警) 逻辑
        返回一个字典，告诉飞行员当前的威胁状态
        """
        warning_status = {
            'LOCKED': [],  # 致命威胁：导弹正在跟踪 (滴滴滴急促音)
            'SEARCHING': []  # 潜在威胁：导弹已发射但暂时丢失目标 (断续音/静默)
        }

        for missile in self.under_missiles:
            if not missile.is_alive:
                continue

            # 调用刚才在 MissileSimulator 里加的属性
            if missile.is_locking:
                warning_status['LOCKED'].append(missile)
            else:
                warning_status['SEARCHING'].append(missile)

        return warning_status

    # --- 对外接口：获取雷达探测结果 ---

    def get_detected_targets(self, all_aircrafts: List['AircraftSimulator']) -> List['AircraftSimulator']:
        """获取全向感知列表 (上帝视角/数据链)"""
        return self.omni_radar.scan(self, all_aircrafts)

    def get_locked_targets(self, all_aircrafts: List['AircraftSimulator']) -> List['AircraftSimulator']:
        """获取火控锁定列表 (在机头前方且距离内) - 用于发射导弹"""
        return self.fcr.scan(self, all_aircrafts)


# ==========================================
# 5. 导弹仿真类 (MissileSimulator)
#    实现有限状态机、视场角脱锁逻辑和比例导引
# ==========================================

class MissileState(Enum):
    INACTIVE = auto()
    TRACKING = auto()  # 锁定状态：目标在视场内，执行比例导引
    INS_GUIDANCE = auto()  # 丢失目标，正在依靠惯性预测飞行 (Memory Track)
    SEARCHING = auto()  # 彻底丢失，开启主动搜索模式 (Mad Dog)
    HIT = auto()  # 命中目标
    MISS = auto()  # 脱靶 (燃料耗尽、丢失目标过久等)


class MissileSimulator(BaseSimulator):
    # 兼容旧代码的状态常量
    INACTIVE = -1
    LAUNCHED = 0
    HIT = 1
    MISS = 2

    @classmethod
    def create(cls, parent: AircraftSimulator, target: AircraftSimulator, uid: str, missile_model: str = "AIM-120B"):
        """
        工厂方法：创建并发射导弹
        :param missile_model: 导弹型号 ("AIM-9L" 或 "AIM-120B")
        """
        assert parent.dt == target.dt, "integration timestep must be same!"
        missile = MissileSimulator(uid, parent.color, missile_model, parent.dt)
        missile.launch(parent)
        missile.target(target)
        return missile

    def __init__(self, uid="A0101", color="Red", model="AIM-120B", dt=1 / 60):
        super().__init__(uid, color, dt)

        # 1. 加载参数
        if model not in MISSILE_DB:
            logging.warning(f"Missile model {model} not found, using AIM-9L.")
            self.config = MISSILE_DB["AIM-9L"]
        else:
            self.config = MISSILE_DB[model]

        self.model = model
        self._state = MissileState.INACTIVE
        self.parent_aircraft = None
        self.target_aircraft = None
        self.render_explosion = False

        # 物理常数
        self._g = 9.81

        # 初始化变量
        self._m = self.config.mass_0
        self._dtheta, self._dphi = 0, 0
        self._distance_pre = np.inf
        self._distance_increment = deque(maxlen=int(5 / self.dt))
        self._t = 0.0

        self.last_known_pos = np.zeros(3)  # 目标最后已知位置
        self.last_known_vel = np.zeros(3)  # 目标最后已知速度
        self.time_since_lost = 0.0  # 丢失锁定的时间累积
        self.memory_limit = 5.0  # 记忆维持时间 (例如5秒)，超过则认为彻底跟丢

    # --- 状态判断属性 ---
    @property
    def is_alive(self):
        return self._state in [MissileState.SEARCHING, MissileState.TRACKING]

    @property
    def is_success(self):
        return self._state == MissileState.HIT

    @property
    def is_done(self):
        return self._state in [MissileState.HIT, MissileState.MISS]

    @property
    def Isp(self):
        # 只有在发动机燃烧时间内才有比冲
        return self.config.isp if self._t < self.config.thrust_duration else 0

    @property
    def S(self):
        """横截面积"""
        return np.pi * (self.config.diameter / 2) ** 2

    @property
    def rho(self):
        """根据高度估算空气密度"""
        return 1.225 * np.exp(-self._geodetic[-1] / 9300)

    @property
    def is_locking(self):
        """返回 True 表示导弹雷达正锁定目标，返回 False 表示导弹丢失目标/正在搜索"""
        return self._state == MissileState.TRACKING

    def launch(self, parent: AircraftSimulator):
        """发射逻辑：继承载机初始状态"""
        self.parent_aircraft = parent
        self.parent_aircraft.launch_missiles.append(self)

        self._geodetic[:] = parent.get_geodetic()
        self._position[:] = parent.get_position()
        self._velocity[:] = parent.get_velocity()
        self._posture[:] = parent.get_rpy()
        self._posture[0] = 0  # 导弹 Roll 初始归零
        self.lon0, self.lat0, self.alt0 = parent.lon0, parent.lat0, parent.alt0

        self._t = 0
        self._m = self.config.mass_0

        # Mad Dog Launch: 默认进入跟踪模式，下一帧 run() 会立刻检查 FOV
        self._state = MissileState.TRACKING

        self._distance_pre = np.inf
        self._distance_increment.clear()

    def target(self, target: AircraftSimulator):
        """指定攻击目标"""
        self.target_aircraft = target
        self.target_aircraft.under_missiles.append(self)

    def run(self):
        """导弹主循环：每帧调用"""
        if not self.is_alive:
            return

        self._t += self.dt

        # 1. 生存性检查 (时间、速度、目标存活)
        speed = np.linalg.norm(self.get_velocity())
        if (self._t > self.config.max_life_time) or \
                (speed < 50.0) or \
                (not self.target_aircraft.is_alive):
            self._state = MissileState.MISS
            return

        # 2. 计算相对几何关系
        pos_m = self.get_position()
        pos_t = self.target_aircraft.get_position()
        dist_vector = pos_t - pos_m
        distance = np.linalg.norm(dist_vector)

        # 3. 命中/脱靶判定
        self._distance_increment.append(distance > self._distance_pre)
        self._distance_pre = distance

        # 判定命中
        if distance < self.config.explosion_radius:
            self._state = MissileState.HIT
            self.target_aircraft.shotdown()
            return

        # 判定脱靶 (距离连续增加)
        if np.sum(self._distance_increment) >= self._distance_increment.maxlen:
            self._state = MissileState.MISS
            return

        # ========================================================
        # 4. 导引头逻辑 (Seeker Logic) - 核心：脱锁与复锁
        # ========================================================

        vel_m = self.get_velocity()
        # 单位向量化
        los_unit = dist_vector / (distance + 1e-6)
        vel_unit = vel_m / (speed + 1e-6)

        # 计算 Boresight Angle (导弹机头与目标连线的夹角)
        cos_angle = np.clip(np.dot(vel_unit, los_unit), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))

        # 检查目标是否在导引头视场(FOV)内
        in_fov = angle_deg < self.config.seeker_fov

        if in_fov:
            # 如果之前是搜索状态，现在找到了 -> 复锁
            if self._state == MissileState.SEARCHING:
                pass  # 可以加日志: print("Relocked!")
            self._state = MissileState.TRACKING
        else:
            # 目标超出视场 -> 脱锁
            if self._state == MissileState.TRACKING:
                pass  # 可以加日志: print("Lost Lock!")
            self._state = MissileState.SEARCHING

        # ========================================================
        # 5. 制导律计算
        # ========================================================

        action = np.array([0.0, 0.0])  # 默认无过载 (惯性/弹道飞行)

        if self._state == MissileState.TRACKING:
            # 仅在锁定时计算比例导引 (PN)
            action = self._guidance_pn(pos_m, vel_m, pos_t, self.target_aircraft.get_velocity())

        # 6. 执行物理更新
        self._state_trans(action)

    def _guidance_pn(self, p_m, v_m, p_t, v_t):
        """
        比例导引律 (Proportional Navigation)
        计算所需的法向过载 (ny, nz)
        """
        # 相对位置与相对速度
        R_vec = p_t - p_m
        R = np.linalg.norm(R_vec)
        V_R = v_t - v_m

        # 计算视线角速度 (LOS Rate)
        # Omega = (R x V_R) / R^2
        omega_vec = np.cross(R_vec, V_R) / (R ** 2 + 1e-6)

        speed_m = np.linalg.norm(v_m)

        # 导航计算
        # 这里保留你原有的投影算法风格，将 3D 加速度分解为 ny, nz
        dx_m, dy_m, dz_m = v_m
        x_m, y_m, z_m = p_m
        x_t, y_t, z_t = p_t
        dx_t, dy_t, dz_t = v_t

        Rxy = np.linalg.norm([x_m - x_t, y_m - y_t])
        Rxyz = R
        if Rxy < 1.0: return np.array([0.0, 0.0])

        # 计算方位角变化率 (dbeta) 和 俯仰角变化率 (deps)
        dbeta = ((dy_t - dy_m) * (x_t - x_m) - (dx_t - dx_m) * (y_t - y_m)) / Rxy ** 2
        deps = ((dz_t - dz_m) * Rxy ** 2 - (z_t - z_m) * (
                    (x_t - x_m) * (dx_t - dx_m) + (y_t - y_m) * (dy_t - dy_m))) / (Rxyz ** 2 * Rxy)

        theta_m = np.arcsin(np.clip(dz_m / (speed_m + 1e-6), -1, 1))

        # N: 导航比
        K = self.config.nav_gain

        # 计算指令过载
        ny = K * speed_m / self._g * np.cos(theta_m) * dbeta
        nz = K * speed_m / self._g * deps + np.cos(theta_m)  # 包含重力补偿

        # 限制过载幅度 (G-Limit)
        max_nyz = self.config.max_g
        return np.clip([ny, nz], -max_nyz, max_nyz)

    def _state_trans(self, action):
        """
        物理状态积分 (动力学与运动学)
        """
        dt = self.dt

        # 1. 更新位置
        self._position[:] += dt * self.get_velocity()
        self._geodetic[:] = NEU2LLA(*self.get_position(), self.lon0, self.lat0, self.alt0)

        # 2. 准备速度与姿态计算
        v_vec = self.get_velocity()
        v = np.linalg.norm(v_vec)
        if v < 1e-3: return

        theta, phi = self.get_rpy()[1:]

        # 3. 计算推力与阻力
        T = 0.0
        if self._t < self.config.thrust_duration:
            T = self._g * self.Isp * self.config.mass_loss_rate
            self._m -= dt * self.config.mass_loss_rate

        D = 0.5 * self.config.drag_coeff * self.S * self.rho * v ** 2

        # 切向过载 (加速度/减速度)
        nx = (T - D) / (self._m * self._g)

        # 法向过载 (来自制导律)
        ny, nz = action

        # 4. 运动学微分方程
        # dv/dt = g * (nx - sin(theta))
        dv = self._g * (nx - np.sin(theta))

        # 防止分母为0 (Gimbal lock protection)
        cos_theta = np.cos(theta)
        if abs(cos_theta) < 0.1: cos_theta = 0.1 * np.sign(cos_theta)

        # dphi/dt (Yaw rate)
        self._dphi = self._g / v * (ny / cos_theta)
        # dtheta/dt (Pitch rate)
        self._dtheta = self._g / v * (nz - np.cos(theta))

        # 5. 积分更新
        v += dt * dv
        phi += dt * self._dphi
        theta += dt * self._dtheta

        # 限制俯仰角范围
        theta = np.clip(theta, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)

        # 6. 重建速度向量和姿态
        self._velocity[:] = np.array([
            v * np.cos(theta) * np.cos(phi),
            v * np.cos(theta) * np.sin(phi),
            v * np.sin(theta)
        ])
        self._posture[:] = np.array([0, theta, phi])

    def log(self):
        """记录日志"""
        if self.is_alive:
            return super().log()
        elif self.is_done and (not self.render_explosion):
            self.render_explosion = True
            log_msg = f"-{self.uid}\n"
            lon, lat, alt = self.get_geodetic()
            roll, pitch, yaw = self.get_rpy() * 180 / np.pi
            log_msg += f"{self.uid}F,T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
            # 记录爆炸类型和半径
            log_msg += f"Type=Misc+Explosion,Color={self.color},Radius={self.config.explosion_radius}"
            return log_msg
        return None

    def close(self):
        self.target_aircraft = None