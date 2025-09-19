import math
from typing import List, Union

import numpy as np

from envs.JSBSim.utils.utils import LLA2NEU, get_AO_TA_R


class Aircraft:
    """
    描述一架飞机的完整状态信息。

    本类通过嵌套子类来组织飞机的各类参数，并提供了 update 和 get 方法
    来方便地与字典格式的观测数据进行交互。
    """

    def __init__(self, key: str, uid: str, origin: tuple = (0.0, 0.0, 0.0), lat_limit: tuple = (-0.9032379, 0.9032379),
                 lon_limit: tuple = (-0.8971999, 0.8971999)):
        """
        初始化飞机对象。

        :param key: 阵营标识，例如 "red" 或 "blue"。
        :param uid: 飞机的唯一标识符，例如 "red_0"。
        :param origin: 坐标系原点的大地坐标 (经度, 纬度, 高度)。
        """
        # --- 基础信息 ---
        self.key = key  # 阵营标识 ("red" 或 "blue")
        self.uid = uid  # 飞机的唯一标识符 ("red_0", "blue_1" 等)

        # --- 坐标系原点 ---
        self.lon0, self.lat0, self.alt0 = origin  # (经度, 纬度, 高度)
        # ---判断是否超出边界
        self.lat_limit = lat_limit
        self.lon_limit = lon_limit
        self.out_side_time = 0

        # --- 状态数据模块 ---
        # 初始化飞机各系统状态的子类实例
        self.position_and_attitude = self.PositionAndAttitude()  # 位置坐标及姿态
        self.inertia = self.Inertia()  # 质量与转动惯量
        self.velocities = self.Velocities()  # 速度相关参数
        self.accelerations = self.Accelerations()  # 加速度相关参数
        self.control_state = self.ControlState()  # 飞控状态
        self.engines = self.Engines()  # 引擎状态
        self.controls_command = self.ControlsCommand()  # 飞控指令
        self.simulation = self.Simulation()  # 模拟器信息
        self.initial_conditions = self.InitialConditions()  # 初始条件
        self.battle_info = self.BattleInfo()  # 战场与控制模型信息

        # --- 派生参数 (使用国际标准单位) ---
        # 这些参数由原始数据换算而来，方便计算和使用
        self._geodetic = np.zeros(3)  # 大地坐标 (经度, 纬度, 高度)，单位: [度, 度, 米]
        self._position = np.zeros(3)  # 本地坐标 (北, 东, 上)，单位: [米, 米, 米]
        self._posture = np.zeros(3)  # 姿态角 (滚转, 俯仰, 偏航)，单位: [弧度, 弧度, 弧度]
        self._velocity = np.zeros(3)  # 本地速度 (北向, 东向, 上向)，单位: [米/秒]

        # --- 关系与交互 ---
        self.partners: List['Aircraft'] = []  # 友机列表, 通讯模式0-0下全部友军
        self.enemies: List['Aircraft'] = []  # 敌机列表， 通讯模式0-0下全部敌军

        self.detected_enemies: List['Aircraft'] = []  # 全队共享的雷达探测敌军列表，每个友军的TargetIntoView求并集， 比如red_0，一个飞机看不到，  red_1能看到所有的。 red_0的detected_enemies也是所有的
        self.launched_missiles: List[Missile] = []  # 本机已发射的导弹列表
        self.under_missiles: List[Missile] = []  # 正在攻击本机的敌方导弹列表

        # --- 工具映射 ---
        # 创建一个从 obs 键到内部属性的映射，用于高效更新和获取数据
        self._attribute_map = self._create_attribute_map()

    # --- 便捷 Get 方法 (国际标准单位) ---
    def get_geodetic(self):
        """获取大地坐标系下的位置 (经度, 纬度, 高度)。

        :return: tuple (longitude, latitude, altitude)，单位为 (°/°/m)。
        """
        return self._geodetic

    def get_position(self):
        """获取本地坐标系下的位置 (北, 东, 上)。

        :return: tuple (north, east, up)，单位为 (m/m/m)。
        """
        return self._position

    def get_rpy(self):
        """获取飞机的姿态角 (滚转角, 俯仰角, 偏航角)。

        :return: tuple (roll, pitch, yaw)，单位为 (rad/rad/rad)。
        """
        return self._posture

    def get_velocity(self):
        """获取本地坐标系下的速度 (北向, 东向, 上向)。

        :return: tuple (v_north, v_east, v_up)，单位为 (m/s)。
        """
        return self._velocity

    def detected_missiles(self):
        detected_missiles = []
        for missile in self.under_missiles:
            agent_feature = np.hstack([self.get_position(), self.get_velocity()])
            missile_feature = np.hstack([missile.get_position(), missile.get_velocity()])
            _, _, R = get_AO_TA_R(agent_feature, missile_feature)
            if R < 25000:
                detected_missiles.append(missile)
        return detected_missiles

    def check_missile_warning(self):
        for missile in self.detected_missiles():
            if missile.is_alive:
                return missile
        return None



    # --- 核心数据交互方法 ---
    def update(self, obs: dict):
        """
        根据传入的字典 obs 更新飞机的所有参数。

        :param obs: 一个包含飞机状态数据的字典，键与原始参数表对应。
        """
        for key, value in obs.items():
            if key in self._attribute_map:
                obj, attr_name = self._attribute_map[key]
                setattr(obj, attr_name, value)

        # 在所有基础参数更新后，调用辅助函数来更新派生参数
        self._update_derived_parameters()

    def get(self, key: str):
        """
        根据键获取对应的原始参数值。

        :param key: 状态参数的键 (例如 'velocities/mach')。
        :return: 对应的参数值，如果键不存在则返回 None。
        """
        if key in self._attribute_map:
            obj, attr_name = self._attribute_map[key]
            return getattr(obj, attr_name)
        return None

    def gets(self, keys: List[str]) -> list:
        """
        根据键列表批量获取对应的原始参数值。

        :param keys: 一个包含多个状态参数键的列表。
        :return: 一个包含所有查询结果的值的列表，顺序与输入键一致。
                 如果某个键不存在，则对应位置为 None。
        """
        return [self.get(key) for key in keys]

    def get_property_value(self, key: str):
        """
        根据键获取对应的原始参数值。

        :param key: 状态参数的键 (例如 'velocities/mach')。
        :return: 对应的参数值，如果键不存在则返回 None。
        """
        if key in self._attribute_map:
            obj, attr_name = self._attribute_map[key]
            return getattr(obj, attr_name)
        return None

    def get_property_values(self, keys: List[str]) -> list:
        """
        根据键列表批量获取对应的原始参数值。

        :param keys: 一个包含多个状态参数键的列表。
        :return: 一个包含所有查询结果的值的列表，顺序与输入键一致。
                 如果某个键不存在，则对应位置为 None。
        """
        return [self.get(key) for key in keys]

    # --- 内部辅助方法 ---
    def _update_derived_parameters(self):
        """
        使用基础参数计算并更新所有派生参数 (如 _geodetic, _velocity 等)。
        此方法在 self.update() 的末尾被调用，以确保数据同步。
        """
        FT_TO_M = 0.3048  # 单位换算常量: 英尺到米

        # 1. 更新大地坐标 (Geodetic)
        lon_deg = self.position_and_attitude.long_gc_deg
        lat_deg = self.position_and_attitude.lat_geod_deg
        alt_m = self.position_and_attitude.h_sl_ft * FT_TO_M
        self._geodetic = np.array([lon_deg, lat_deg, alt_m])

        # 2. 更新本地坐标 (Position)
        self._position[:] = LLA2NEU(*self._geodetic, self.lon0, self.lat0, self.alt0)

        # 3. 更新姿态角 (RPY - Roll, Pitch, Yaw)
        roll_rad = self.position_and_attitude.roll_rad
        pitch_rad = self.position_and_attitude.pitch_rad
        yaw_rad = math.radians(self.position_and_attitude.psi_deg)  # 原始数据为度，需转为弧度
        self._posture = np.array([roll_rad, pitch_rad, yaw_rad])

        # 4. 更新速度 (Velocity)
        v_north_ms = self.velocities.v_north_fps * FT_TO_M
        v_east_ms = self.velocities.v_east_fps * FT_TO_M
        # v-down-fps 向下为正，我们需要 v_up 向上为正，故取反
        v_up_ms = -self.velocities.v_down_fps * FT_TO_M
        self._velocity = np.array([v_north_ms, v_east_ms, v_up_ms])

        # 5. 更新是否超出边界
        if not (self.lat_limit[0] < self.position_and_attitude.lat_geod_deg < self.lat_limit[1] and
                self.lon_limit[0] < self.position_and_attitude.long_gc_deg < self.lon_limit[1]):
            self.out_side_time += 1
        else:
            self.out_side_time = 0

    def _create_attribute_map(self) -> dict:
        """内部方法，用于生成观测字典键和类内部属性之间的映射关系。"""
        # 键中的连字符'-' 在类属性中用下划线'_'代替
        mapping = {
            # Position and Attitude
            'position/h-sl-ft': (self.position_and_attitude, 'h_sl_ft'),
            'attitude/pitch-rad': (self.position_and_attitude, 'pitch_rad'),
            'attitude/roll-rad': (self.position_and_attitude, 'roll_rad'),
            'attitude/psi-deg': (self.position_and_attitude, 'psi_deg'),
            'aero/beta-deg': (self.position_and_attitude, 'beta_deg'),
            'aero/alpha-deg': (self.position_and_attitude, 'alpha_deg'),
            'position/lat-geod-deg': (self.position_and_attitude, 'lat_geod_deg'),
            'position/long-gc-deg': (self.position_and_attitude, 'long_gc_deg'),
            # Inertia
            'inertia/mass-slugs': (self.inertia, 'mass_slugs'),
            'inertia/ixx-slugs_ft2': (self.inertia, 'ixx_slugs_ft2'),
            'inertia/iyy-slugs_ft2': (self.inertia, 'iyy_slugs_ft2'),
            'inertia/izz-slugs_ft2': (self.inertia, 'izz_slugs_ft2'),
            'inertia/ixy-slugs_ft2': (self.inertia, 'ixy_slugs_ft2'),
            'inertia/ixz-slugs_ft2': (self.inertia, 'ixz_slugs_ft2'),
            'inertia/iyz-slugs_ft2': (self.inertia, 'iyz_slugs_ft2'),
            # Velocities
            'velocities/u-fps': (self.velocities, 'u_fps'),
            'velocities/v-fps': (self.velocities, 'v_fps'),
            'velocities/w-fps': (self.velocities, 'w_fps'),
            'velocities/v-north-fps': (self.velocities, 'v_north_fps'),
            'velocities/v-east-fps': (self.velocities, 'v_east_fps'),
            'velocities/v-down-fps': (self.velocities, 'v_down_fps'),
            'velocities/p-rad_sec': (self.velocities, 'p_rad_sec'),
            'velocities/q-rad_sec': (self.velocities, 'q_rad_sec'),
            'velocities/r-rad_sec': (self.velocities, 'r_rad_sec'),
            'velocities/ve-fps': (self.velocities, 've_fps'),
            'velocities/h-dot-fps': (self.velocities, 'h_dot_fps'),
            'velocities/mach': (self.velocities, 'mach'),
            # Accelerations
            'accelerations/a-pilot-x-ft_sec2': (self.accelerations, 'a_pilot_x_ft_sec2'),
            'accelerations/a-pilot-y-ft_sec2': (self.accelerations, 'a_pilot_y_ft_sec2'),
            'accelerations/a-pilot-z-ft_sec2': (self.accelerations, 'a_pilot_z_ft_sec2'),
            'accelerations/n-pilot-x-norm': (self.accelerations, 'n_pilot_x_norm'),
            'accelerations/n-pilot-y-norm': (self.accelerations, 'n_pilot_y_norm'),
            'accelerations/n-pilot-z-norm': (self.accelerations, 'n_pilot_z_norm'),
            # Control State
            'forces/load-factor': (self.control_state, 'load_factor'),
            'fcs/left-aileron-pos-norm': (self.control_state, 'left_aileron_pos_norm'),
            'fcs/elevator-pos-norm': (self.control_state, 'elevator_pos_norm'),
            'fcs/rudder-pos-norm': (self.control_state, 'rudder_pos_norm'),
            'fcs/throttle-pos-norm': (self.control_state, 'throttle_pos_norm'),
            'gear/gear-pos-norm': (self.control_state, 'gear_pos_norm'),
            # Engines
            'propulsion/engine/set-running': (self.engines, 'set_running'),
            'propulsion/set-running': (self.engines, 'set_running'),
            'propulsion/engine/thrust-lbs': (self.engines, 'thrust_lbs'),
            'propulsion/tank/contents-lbs': (self.engines, 'contents_lbs'),
            'propulsion/tank/pct-full': (self.engines, 'pct_full'),
            # Controls Command
            'fcs/aileron-cmd-norm': (self.controls_command, 'aileron_cmd_norm'),
            'fcs/elevator-cmd-norm': (self.controls_command, 'elevator_cmd_norm'),
            'fcs/rudder-cmd-norm': (self.controls_command, 'rudder_cmd_norm'),
            'fcs/throttle-cmd-norm': (self.controls_command, 'throttle_cmd_norm'),
            'fcs/mixture-cmd-norm': (self.controls_command, 'mixture_cmd_norm'),
            'fcs/throttle-cmd-norm[1]': (self.controls_command, 'throttle_cmd_norm_1'),
            'fcs/mixture-cmd-norm[1]': (self.controls_command, 'mixture_cmd_norm_1'),
            'gear/gear-cmd-norm': (self.controls_command, 'gear_cmd_norm'),
            # Simulation
            'simulation/dt': (self.simulation, 'dt'),
            'simulation/sim-time-sec': (self.simulation, 'sim_time_sec'),
            # Initial Conditions
            'ic/h-sl-ft': (self.initial_conditions, 'h_sl_ft'),
            'ic/terrain-elevation-ft': (self.initial_conditions, 'terrain_elevation_ft'),
            'ic/long-gc-deg': (self.initial_conditions, 'long_gc_deg'),
            'ic/lat-geod-deg': (self.initial_conditions, 'lat_geod_deg'),
            'ic/u-fps': (self.initial_conditions, 'u_fps'),
            'ic/v-fps': (self.initial_conditions, 'v_fps'),
            'ic/w-fps': (self.initial_conditions, 'w_fps'),
            'ic/p-rad_sec': (self.initial_conditions, 'p_rad_sec'),
            'ic/q-rad_sec': (self.initial_conditions, 'q_rad_sec'),
            'ic/r-rad_sec': (self.initial_conditions, 'r_rad_sec'),
            'ic/roc-fpm': (self.initial_conditions, 'roc_fpm'),
            'ic/psi-true-deg': (self.initial_conditions, 'psi_true_deg'),
            'ic/phi-deg': (self.initial_conditions, 'phi_deg'),
            'ic/theta-deg': (self.initial_conditions, 'theta_deg'),
            # Battle Info
            'LifeCurrent': (self.battle_info, 'LifeCurrent'),
            'BulletCurrentNum': (self.battle_info, 'BulletCurrentNum'),
            'IfOverHeat': (self.battle_info, 'IfOverHeat'),
            'TargetIntoView': (self.battle_info, 'TargetIntoView'),
            'AllyIntoView': (self.battle_info, 'AllyIntoView'),
            'TargetEnterAttackRange': (self.battle_info, 'TargetEnterAttackRange'),
            'AimMode': (self.battle_info, 'AimMode'),
            'ACMaimMode': (self.battle_info, 'ACMaimMode'),
            'SRAAMCurrentNum': (self.battle_info, 'SRAAMCurrentNum'),
            'SRAAM1_CanReload': (self.battle_info, 'SRAAM1_CanReload'),
            'SRAAM2_CanReload': (self.battle_info, 'SRAAM2_CanReload'),
            'SRAAMTargetLocked': (self.battle_info, 'SRAAMTargetLocked'),
            'AMRAAMCurrentNum': (self.battle_info, 'AMRAAMCurrentNum'),
            'AMRAAM1_CanReload': (self.battle_info, 'AMRAAM1_CanReload'),
            'AMRAAM2_CanReload': (self.battle_info, 'AMRAAM2_CanReload'),
            'AMRAAM3_CanReload': (self.battle_info, 'AMRAAM3_CanReload'),
            'AMRAAM4_CanReload': (self.battle_info, 'AMRAAM4_CanReload'),
            'AMRAAMlockedTarget': (self.battle_info, 'AMRAAMlockedTarget'),
            'MissileAlert': (self.battle_info, 'MissileAlert'),
            'WarningNumber': (self.battle_info, 'WarningNumber'),
            'IsOutOfValidBattleArea': (self.battle_info, 'IsOutOfValidBattleArea'),
            'OutOfValidBattleAreaCurrentDuration': (self.battle_info, 'OutOfValidBattleAreaCurrentDuration'),
            'IfPresenceHitting': (self.battle_info, 'IfPresenceHitting'),
            'EnvelopeMax': (self.battle_info, 'EnvelopeMax'),
            'EnvelopeMin': (self.battle_info, 'EnvelopeMin'),
            'DeathEvent': (self.battle_info, 'DeathEvent'),
        }
        return mapping

    @property
    def is_alive(self) -> bool:
        """判断飞机是否存活"""
        # DeathEvent 为 99 表示飞机正常
        return self.battle_info.DeathEvent == 99

    @property
    def is_crash(self) -> bool:
        """判断飞机是否因撞击或出界而坠毁"""
        # DeathEvent 为 0 表示坠毁
        return self.battle_info.DeathEvent == 0

    @property
    def is_shotdown(self) -> bool:
        """判断飞机是否被击落 (被导弹或子弹)"""
        # DeathEvent 不为 99 (存活) 且不为 0 (坠毁)，则为被击落
        return self.battle_info.DeathEvent not in [99, 0]

    def get_health_percent(self) -> float:
        """获取当前生命值百分比"""
        # 初始生命值为 200
        max_health = 200.0
        return max(0.0, self.battle_info.LifeCurrent / max_health)

    def get_speed_mps(self) -> float:
        """获取飞机总速度大小 (米/秒)"""
        # _velocity 是 (v_north, v_east, v_up) 的numpy数组
        return np.linalg.norm(self._velocity)

    def get_altitude_m(self) -> float:
        """获取飞机当前高度 (米)"""
        # _position 是 (north, east, up) 的numpy数组, 第三个元素是高度
        return float(self._position[2])

    @property
    def is_climbing(self) -> bool:
        """判断飞机是否正在爬升 (俯仰角大于1度)"""
        # _posture[1] 是俯仰角 (pitch) in radians
        return self._posture[1] > math.radians(1.0)

    @property
    def is_diving(self) -> bool:
        """判断飞机是否正在俯冲 (俯仰角小于-1度)"""
        return self._posture[1] < -math.radians(1.0)

    @property
    def is_stall_warning(self) -> bool:
        """
        判断飞机是否处于失速风险中 (简易版：基于速度)
        注意: 这是一个简化判断，真实失速与攻角等多种因素相关。
        80 m/s 约等于 155 节，是一个常见的战斗机低速阈值。
        """
        return self.get_speed_mps() < 80.0

    @property
    def can_fire_missile(self) -> bool:
        """判断飞机是否可以发射导弹 (存活且有剩余导弹)"""
        has_missile = (self.battle_info.SRAAMCurrentNum > 0 or
                       self.battle_info.AMRAAMCurrentNum > 0)
        return self.is_alive and has_missile

    @property
    def is_under_attack(self) -> bool:
        """判断飞机当前是否正被至少一枚存活的导弹攻击"""
        # 检查 self.under_missiles 列表中是否有任何仍在正常飞行的导弹
        # 导弹的 Status 为 0 表示 "正常飞行"
        return any(missile.Status == 0 for missile in self.under_missiles)

    # --- 嵌套子类定义 ---
    class PositionAndAttitude:
        """位置坐标及姿态"""

        def __init__(self):
            self.h_sl_ft = 0.0  # 海拔高度 [英尺 ft]
            self.pitch_rad = 0.0  # 俯仰角 [弧度 rad]
            self.roll_rad = 0.0  # 翻滚角 [弧度 rad]
            self.psi_deg = 0.0  # 航向角 [度 deg]
            self.beta_deg = 0.0  # 侧滑角 [度 deg]
            self.alpha_deg = 0.0  # 攻角 [度 deg]
            self.lat_geod_deg = 0.0  # 纬度 [度 deg]
            self.long_gc_deg = 0.0  # 经度 [度 deg]

    class Inertia:
        """飞机当前质量与转动惯量"""

        def __init__(self):
            self.mass_slugs = 0.0  # 飞机当前质量 [斯勒格 slug]。1 slug = 14.5939 kg
            self.ixx_slugs_ft2 = 0.0  # x轴转动惯量 [slugs/ft²]
            self.iyy_slugs_ft2 = 0.0  # y轴转动惯量 [slugs/ft²]
            self.izz_slugs_ft2 = 0.0  # z轴转动惯量 [slugs/ft²]
            self.ixy_slugs_ft2 = 0.0  # xy轴转动惯量 [slugs/ft²]
            self.ixz_slugs_ft2 = 0.0  # xz轴转动惯量 [slugs/ft²]
            self.iyz_slugs_ft2 = 0.0  # yz轴转动惯量 [slugs/ft²]

    class Velocities:
        """速度相关参数"""

        def __init__(self):
            self.u_fps = 0.0  # 机体坐标系 x 轴速度 [英尺/秒 ft/s]
            self.v_fps = 0.0  # 机体坐标系 y 轴速度 [英尺/秒 ft/s]
            self.w_fps = 0.0  # 机体坐标系 z 轴速度 [英尺/秒 ft/s]
            self.v_north_fps = 0.0  # 北方向速度 [英尺/秒 ft/s]
            self.v_east_fps = 0.0  # 东方向速度 [英尺/秒 ft/s]
            self.v_down_fps = 0.0  # 向下方向速度 [英尺/秒 ft/s]
            self.p_rad_sec = 0.0  # 翻滚速率 [弧度/秒 rad/s]
            self.q_rad_sec = 0.0  # 俯仰速率 [弧度/秒 rad/s]
            self.r_rad_sec = 0.0  # 偏航速率 [弧度/秒 rad/s]
            self.ve_fps = 0.0  # 真实速度 [英尺/秒 ft/s]
            self.h_dot_fps = 0.0  # 高度变化速率 [英尺/秒 ft/s]
            self.mach = 0.0  # 马赫数 [M]

    class Accelerations:
        """加速度相关参数"""

        def __init__(self):
            self.a_pilot_x_ft_sec2 = 0.0  # 飞机坐标系 x 轴加速度 [英尺/秒² ft/s²]
            self.a_pilot_y_ft_sec2 = 0.0  # 飞机坐标系 y 轴加速度 [英尺/秒² ft/s²]
            self.a_pilot_z_ft_sec2 = 0.0  # 飞机坐标系 z 轴加速度 [英尺/秒² ft/s²]
            self.n_pilot_x_norm = 0.0  # 飞机坐标系 x 轴加速度
            self.n_pilot_y_norm = 0.0  # 飞机坐标系 y 轴加速度
            self.n_pilot_z_norm = 0.0  # 飞机坐标系 z 轴加速度

    class ControlState:
        """控制状态"""

        def __init__(self):
            self.load_factor = 0.0  # 负载系数
            self.left_aileron_pos_norm = 0.0  # 左副翼位置，范围 (-1, 1)
            self.elevator_pos_norm = 0.0  # 升降舵位置，范围 (-1, 1)
            self.rudder_pos_norm = 0.0  # 方向舵位置，范围 (-1, 1)
            self.throttle_pos_norm = 0.0  # 油门位置，范围 (0, 1)
            self.gear_pos_norm = 0.0  # 起落架位置，范围 (0, 1)

    class Engines:
        """引擎状态"""

        def __init__(self):
            self.set_running = False  # 发动机是否运转
            self.thrust_lbs = 0.0  # 发动机推力 [磅 lb]
            self.contents_lbs = 0.0  # 油箱中剩余油量 [磅 lb]
            self.pct_full = 0.0  # 油箱加注液位百分比，范围 (0 到 100)

    class ControlsCommand:
        """控制命令"""

        def __init__(self):
            self.aileron_cmd_norm = 0.0  # 副翼指令，范围 (-1, 1)
            self.elevator_cmd_norm = 0.0  # 升降舵指令，范围 (-1, 1)
            self.rudder_cmd_norm = 0.0  # 方向舵指令，范围 (-1, 1)
            self.throttle_cmd_norm = 0.0  # 油门指令，范围 (0, 1)
            self.mixture_cmd_norm = 0.0  # 发动机混合设置，范围 (0, 1)
            self.throttle_cmd_norm_1 = 0.0  # 油门1指令位置，范围 (0, 1)
            self.mixture_cmd_norm_1 = 0.0  # 油料混合调整阀1设置，范围 (0, 1)
            self.gear_cmd_norm = 0.0  # 所有起落架指令位置，范围 (0, 1)

    class Simulation:
        """模拟相关参数"""

        def __init__(self):
            self.dt = 0.0  # JSBSim 仿真时间步长 [秒 s]
            self.sim_time_sec = 0.0  # 模拟时间 [秒 s]

    class InitialConditions:
        """初始条件"""

        def __init__(self):
            self.h_sl_ft = 0.0  # 初始高度 [英尺 ft]
            self.terrain_elevation_ft = 0.0  # 初始地形高度 [英尺 ft]
            self.long_gc_deg = 0.0  # 初始经度 [度 deg]
            self.lat_geod_deg = 0.0  # 初始纬度 [度 deg]
            self.u_fps = 0.0  # 初始机体坐标系 x 轴速度 [英尺/秒 ft/s]
            self.v_fps = 0.0  # 初始机体坐标系 y 轴速度 [英尺/秒 ft/s]
            self.w_fps = 0.0  # 初始机体坐标系 z 轴速度 [英尺/秒 ft/s]
            self.p_rad_sec = 0.0  # 初始翻滚速率 [弧度/秒 rad/s]
            self.q_rad_sec = 0.0  # 初始俯仰速率 [弧度/秒 rad/s]
            self.r_rad_sec = 0.0  # 初始偏航速率 [弧度/秒 rad/s]
            self.roc_fpm = 0.0  # 初始爬升速率 [英尺/分钟 ft/min]
            self.psi_true_deg = 0.0  # 初始航向 [度 deg]
            self.phi_deg = 0.0  # 初始滚转 [度 deg]
            self.theta_deg = 0.0  # 初始俯仰 [度 deg]

    class BattleInfo:
        """战场与控制模型信息"""

        def __init__(self):
            self.LifeCurrent = 200.0  # 当前生命值 (初始值 200)
            self.BulletCurrentNum = 0  # 剩余子弹数
            self.IfOverHeat = 0  # 机枪是否过热 (0: 否, 1: 是)
            # 进入视野的敌机编号, 多位0/1表示, Eg: "00110" 从个位开始代表进入视野的敌机 blue_1、blue_2
            self.TargetIntoView = "00000"
            # 进入视野的盟友编号, 多位0/1表示, Eg: "00110" 从个位开始代表进入视野的友机 red_1、red_2
            self.AllyIntoView = "00000"
            # 进入攻击范围的目标编号, 格式同上
            self.TargetEnterAttackRange = "00000"
            # 切换导弹模式, 近程 AIM-9M 对应空战格斗模式(ACM), 中程 AIM-120B 对应复合雷达模式(CRM)
            self.AimMode = ""
            # ACM 扫描模式 (0: ACM-HUD 扫描, 1: ACM-垂扫)
            self.ACMaimMode = 0
            # 近程红外弹(SRAAM)剩余数量 (初始2枚)
            self.SRAAMCurrentNum = 2
            self.SRAAM1_CanReload = 1  # 近程红外弹1发射口是否已装配好 (0: 否, 1: 是)
            self.SRAAM2_CanReload = 1  # 近程红外弹2发射口是否已装配好 (0: 否, 1: 是)
            # 单目标状态下锁定敌方编号 (单个数字表示编号, 9 表示未锁定目标)
            self.SRAAMTargetLocked = 9
            # 中程雷达弹(AMRAAM)剩余数量 (初始4枚)
            self.AMRAAMCurrentNum = 4
            self.AMRAAM1_CanReload = 1  # 中程雷达弹1发射口是否已装配好 (0: 否, 1: 是)
            self.AMRAAM2_CanReload = 1  # 中程雷达弹2发射口是否已装配好 (0: 否, 1: 是)
            self.AMRAAM3_CanReload = 1  # 中程雷达弹3发射口是否已装配好 (0: 否, 1: 是)
            self.AMRAAM4_CanReload = 1  # 中程雷达弹4发射口是否已装配好 (0: 否, 1: 是)
            # 多目标状态下锁定敌方编号, 4位表示从个位开始锁定的敌机编号
            self.AMRAAMlockedTarget = "9999"
            self.MissileAlert = 0  # 是否被雷达弹锁定 (0: 否, 1: 是)
            # 雷达告警类型 (0: 无, 1: 正在被敌机雷达扫描, 2: 被敌机CRM锁定, 3: 被敌机ACM锁定)
            self.WarningNumber = 0
            self.IsOutOfValidBattleArea = 0  # 战机是否在战区外 (0: 否, 1: 是)
            self.OutOfValidBattleAreaCurrentDuration = 0.0  # 战机在战区外停留时长 [秒 s]
            self.IfPresenceHitting = 0  # 是否存在该战机发射的导弹 (0: 否, 1: 是)
            self.EnvelopeMax = 0.0  # 导弹包线远边界
            self.EnvelopeMin = 0.0  # 导弹包线近边界
            # 战机死亡事件 (99: 正常, 0: 坠毁, 10/11/12..: 被对应编号战机发射的导弹击毁, 20/21/22..: 被对应编号战机发射的子弹击毁)
            self.DeathEvent = 99


class Missile:
    """
    描述一枚导弹的状态信息。

    本类结构较为扁平，提供了 update 和 get 方法来方便地与字典格式的
    观测数据进行交互。
    """

    def __init__(self, missile_type: str, number: int):
        """
        初始化导弹对象。

        :param missile_type: 导弹类型, 例如 "SRAAM" 或 "AMRAAM"。
        :param number: 导弹的编号。
        """
        self.missile_type = missile_type  # 导弹类型, "SRAAM" 或 "AMRAAM"
        self.number = number  # 导弹的编号

        # --- 关系与交互 ---
        self.parent: Union[Aircraft, None] = None  # 发射此导弹的飞机
        self.target: Union[Aircraft, None] = None  # 此导弹攻击的目标飞机

        # --- 导弹状态属性 ---
        # 发射机 XX 两位。第一位 0:红方, 1:蓝方; 第二位: 飞机编号
        self.Owner = ""
        self.Longitude = 0.0  # 经度 [度 deg]
        self.Lattitude = 0.0  # 纬度 [度 deg]
        self.Altitude = 0.0  # 高度 [英尺 ft]
        self.Pitch = 0.0  # 俯仰角 [度 deg]
        self.Yaw = 0.0  # 偏航角 [度 deg]
        self.Roll = 0.0  # 翻滚角 [度 deg]
        self.Speed = 0.0  # 速度 [英尺/秒 ft/s]
        self.LaunchedTime = 0.0  # 发射后运行时长 [秒 s]
        # 飞行状态 (0:正常飞行, 1:命中, 2:撞地, 3:时间耗尽, 4:目标已被摧毁, 5:发射机被击毁且无目标)
        self.Status = 0
        # 导引头状态 (0:未启用, 2:启动导引头, 3:目标捕获, 4:锁定目标, 5:盲区)
        self.Mseek = 0
        # 制导阶段 (0:惯性制导, 3:中段制导, 6:末端制导)
        self.Mguid = 0
        self.DistanceTarget = 0.0  # 距离目标的距离 [英尺 ft]
        # 目标机 XX 两位。十位 0:红方, 1:蓝方; 个位: 飞机编号; 9:无目标
        self.Target = ""

        # --- 新增：派生参数 (使用国际标准单位, 与 Aircraft 保持一致) ---
        self._geodetic = np.zeros(3)  # 大地坐标 (经度, 纬度, 高度)，单位: [度, 度, 米]
        self._position = np.zeros(3)  # 本地坐标 (北, 东, 上)，单位: [米, 米, 米]
        self._posture = np.zeros(3)   # 姿态角 (滚转, 俯仰, 偏航)，单位: [弧度, 弧度, 弧度]
        self._velocity = np.zeros(3)  # 本地速度 (北向, 东向, 上向)，单位: [米/秒]

        # --- 新增：便捷 Get 方法 (国际标准单位) ---

    def get_geodetic(self):
        """获取大地坐标系下的位置 (经度, 纬度, 高度)。"""
        return self._geodetic

    def get_position(self):
        """获取本地坐标系下的位置 (北, 东, 上)。"""
        return self._position

    def get_rpy(self):
        """获取导弹的姿态角 (滚转角, 俯仰角, 偏航角)。"""
        return self._posture

    def get_velocity(self):
        """获取本地坐标系下的速度 (北向, 东向, 上向)。"""
        return self._velocity
    def launch(self, parent: 'Aircraft'):
        """记录发射者，并在发射者的列表中注册自己。"""
        self.parent = parent
        if self not in parent.launched_missiles:
            parent.launched_missiles.append(self)

    @classmethod
    def create(cls, missile_type: str, number: int, parent: 'Aircraft',
               target: Union['Aircraft', None]) -> 'Missile':
        """
        【修正】工厂方法：创建、初始化并建立导弹与飞机的完整连接。
        """
        missile = cls(missile_type, number)
        missile.launch(parent)
        missile.set_target(target)
        return missile

    def set_target(self, new_target: Union[Aircraft, None]):
        """
        【升级】智能地设置或切换目标。
        此方法会自动处理与旧目标的解绑和与新目标的绑定。
        """
        old_target = self.target

        # 如果新旧目标相同，则无需做任何事
        if old_target is new_target:
            return

        # 1. 如果存在旧目标，从它的被攻击列表中移除自己
        if old_target and self in old_target.under_missiles:
            old_target.under_missiles.remove(self)

        # 2. 更新自己内部的目标引用
        self.target = new_target

        # 3. 如果存在新目标，将自己添加到它的被攻击列表中
        if new_target and self not in new_target.under_missiles:
            new_target.under_missiles.append(self)

    def update(self, obs: dict):
        """
        根据传入的字典 obs 更新导弹的所有参数。

        :param obs: 一个包含导弹状态数据的字典。
        """
        for key, value in obs.items():
            # 检查类中是否存在与键同名的属性 (注意大小写敏感)
            if hasattr(self, key):
                setattr(self, key, value)
            # 在所有基础参数更新后，调用辅助函数来更新派生参数
        self._update_derived_parameters()

    def _update_derived_parameters(self):
        """
        使用基础参数计算并更新所有派生参数 (如 _geodetic, _velocity 等)。
        此方法在 self.update() 的末尾被调用，以确保数据同步。
        """
        FT_TO_M = 0.3048  # 单位换算常量: 英尺到米

        # 1. 更新大地坐标 (Geodetic)
        alt_m = self.Altitude * FT_TO_M
        self._geodetic = np.array([self.Longitude, self.Lattitude, alt_m])

        # 2. 更新本地坐标 (Position)
        # 注意: 此计算需要知道坐标系原点, 我们从发射飞机的原点获取
        if self.parent:
            origin_lon, origin_lat, origin_alt = self.parent.lon0, self.parent.lat0, self.parent.alt0
            self._position[:] = LLA2NEU(*self._geodetic, origin_lon, origin_lat, origin_alt)
        else:
            # 如果没有父飞机信息，则无法计算本地坐标
            self._position.fill(0)

            # 3. 更新姿态角 (RPY - Roll, Pitch, Yaw)
        # 原始数据均为度，需转为弧度
        roll_rad = math.radians(self.Roll)
        pitch_rad = math.radians(self.Pitch)
        yaw_rad = math.radians(self.Yaw)
        self._posture = np.array([roll_rad, pitch_rad, yaw_rad])

        # 4. 更新速度 (Velocity)
        speed_ms = self.Speed * FT_TO_M

        # 使用三角函数将总速度分解到北、东、上三个方向
        horizontal_speed = speed_ms * math.cos(pitch_rad)
        v_north = horizontal_speed * math.cos(yaw_rad)
        v_east = horizontal_speed * math.sin(yaw_rad)
        v_up = speed_ms * math.sin(pitch_rad)
        self._velocity = np.array([v_north, v_east, v_up])

    def get(self, key: str):
        """
        根据键获取对应的参数值。

        :param key: 状态参数的键 (例如 'Speed')。
        :return: 对应的参数值，如果键不存在则返回 None。
        """
        return getattr(self, key, None)

    @property
    def is_alive(self) -> bool:
        """判断导弹是否仍在正常飞行"""
        # Status 为 0 表示正常飞行
        return self.Status == 0

    @property
    def is_success(self) -> bool:
        """判断导弹是否成功击中目标"""
        # Status 为 1 表示命中
        return self.Status == 1

    @property
    def is_miss(self) -> bool:
        """判断导弹是否因各种原因失的 (未击中)"""
        # Status 为 2, 3, 4, 5 都表示飞行结束但未命中
        return self.Status in [2, 3, 4, 5]

    @property
    def is_done(self) -> bool:
        """判断导弹飞行是否已经结束 (无论命中或失的)"""
        # 只要 Status 不为 0 (正常飞行)，就代表飞行结束
        return self.Status != 0
