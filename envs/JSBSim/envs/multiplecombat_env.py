import csv
import os

import numpy as np
from typing import Tuple, Dict, Any
from .env_base import BaseEnv
from ..tasks.multiplecombat_task import HierarchicalMultipleCombatShootTask, HierarchicalMultipleCombatTask, MultipleCombatTask
import random
import math
from envs.JSBSim.utils.utils import get_AO_TA_R
from ..utils.shared_variable import GlobalVars


class MultipleCombatEnv(BaseEnv):
    """
    MultipleCombatEnv is an multi-player competitive environment.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        self._create_records = False
        self.record_data=[]

    @property
    def share_observation_space(self):
        return self.task.share_observation_space

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'multiplecombat':
            self.task = MultipleCombatTask(self.config)
        elif taskname == 'hierarchical_multiplecombat':
            self.task = HierarchicalMultipleCombatTask(self.config)
        elif taskname == 'hierarchical_multiplecombat_shoot':
            self.task = HierarchicalMultipleCombatShootTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            obs (dict): {agent_id: initial observation}
            share_obs (dict): {agent_id: initial state}
        """
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        share_obs = self.get_state()
        return self._pack(obs), self._pack(share_obs)


    def reset_simulators(self):
        # self.normal_reset_simulators()
        self.random_reset_simulators()
    def random_reset_simulators(self):
        # --- 常量定义 ---
        KM_PER_DEG_LAT = 111.132  # 每度纬度对应的公里数 (近似值)
        KM_PER_DEG_LON_AT_EQ = 111.320  # 赤道上每度经度对应的公里数 (近似值)
        FT_PER_METER = 3.28084

        # --- 基地和距离设置 ---
        red_base_lon_deg = 120.0
        red_base_lat_deg = 60.0
        inner_radius_km = 5.0  # 队伍内部散布半径
        min_base_separation_km = 10.0  # 队伍基地最小间距
        max_base_separation_km = 40.0  # (可选) 队伍基地最大间距，增加随机性

        # --- 计算红队纬度处的经度换算因子 ---
        # 注意：math.cos() 需要弧度
        km_per_deg_lon_red = KM_PER_DEG_LON_AT_EQ * math.cos(math.radians(red_base_lat_deg))

        # --- 计算蓝队基地的随机位置 ---
        # 1. 随机选择一个方向 (角度)
        angle_rad = random.uniform(0, 2 * math.pi)
        # 2. 随机选择一个距离 (大于等于最小间距)
        distance_km = random.uniform(min_base_separation_km, max_base_separation_km)

        # 3. 计算经纬度偏移量 (使用平面近似，对于几十公里通常足够)
        delta_lat_deg = (distance_km * math.cos(angle_rad)) / KM_PER_DEG_LAT
        # 使用红队基地的经度换算因子作为近似
        delta_lon_deg = (distance_km * math.sin(angle_rad)) / km_per_deg_lon_red

        # 4. 计算蓝队基准点
        blue_base_lon_deg = red_base_lon_deg + delta_lon_deg
        blue_base_lat_deg = red_base_lat_deg + delta_lat_deg

        # --- 计算蓝队纬度处的经度换算因子 ---
        km_per_deg_lon_blue = KM_PER_DEG_LON_AT_EQ * math.cos(math.radians(blue_base_lat_deg))

        # --- 计算内部散布的最大经纬度偏移量 ---
        #   (单位: 度)
        max_lat_offset_deg = inner_radius_km / KM_PER_DEG_LAT
        max_lon_offset_deg_red = inner_radius_km / km_per_deg_lon_red
        max_lon_offset_deg_blue = inner_radius_km / km_per_deg_lon_blue

        # --- 循环设置每个单位的初始条件 ---
        for sim_id, sim in self._jsbsims.items():
            # 为每个单位生成独立的随机属性
            altitude_m = random.randint(5000, 10000)  # 先用米，方便理解
            heading_deg = random.randint(0, 359)  # 0-359 更常用
            speed_mps = random.randint(100, 300)  # 先用米/秒
            phi_deg = random.randint(-180,180) # 滚转角
            theta_deg = random.randint(-90,90) # 俯仰角

            if sim_id.startswith('A'):  # 红队
                # 在基准点周围随机偏移 (允许负值)
                offset_lat = random.uniform(-max_lat_offset_deg, max_lat_offset_deg)
                offset_lon = random.uniform(-max_lon_offset_deg_red, max_lon_offset_deg_red)

                sim.reload({
                    "ic_long_gc_deg": red_base_lon_deg + offset_lon,
                    "ic_lat_geod_deg": red_base_lat_deg + offset_lat,
                    "ic_h_sl_ft": altitude_m * FT_PER_METER,
                    "ic_psi_true_deg": heading_deg,
                    "ic_u_fps": speed_mps * FT_PER_METER,  # 假设 ic_u_fps 是总速度标量
                    "ic_theta_deg": theta_deg,  # 俯仰
                    "ic_phi_deg": phi_deg, # 滚转

                })
            elif sim_id.startswith('B'):  # 蓝队
                # 在基准点周围随机偏移 (允许负值)
                offset_lat = random.uniform(-max_lat_offset_deg, max_lat_offset_deg)
                offset_lon = random.uniform(-max_lon_offset_deg_blue, max_lon_offset_deg_blue)

                sim.reload({
                    "ic_long_gc_deg": blue_base_lon_deg + offset_lon,
                    "ic_lat_geod_deg": blue_base_lat_deg + offset_lat,
                    "ic_h_sl_ft": altitude_m * FT_PER_METER,
                    "ic_psi_true_deg": heading_deg,
                    "ic_u_fps": speed_mps * FT_PER_METER,
                    "ic_theta_deg": theta_deg,  # 俯仰
                    "ic_phi_deg": phi_deg,  # 滚转
                })

        self._tempsims.clear()
    def normal_reset_simulators(self):
        # Assign new initial condition here!
        for sim in self._jsbsims.values():
            sim.reload()
        self._tempsims.clear()


    def record_aircraft_state(self,time,acmi_file_id,event):
        #event是用来表示事件：0初始化 1导弹此刻发射 2导弹此刻miss 3导弹此刻击中
        # 记录每架飞机的状态
        for agent_id, agent in self.agents.items():#遍历飞机
            if(agent_id != "A0200"):#不记录不发射导弹的我机，只让A0100携带导弹
                lon, lat, alt = agent.get_geodetic()  # 获取经度、纬度、高度
                roll, pitch, yaw = agent.get_rpy() * 180 / np.pi  # 获取滚转角、俯仰角、偏航角,从弧度变为角度
                speed = agent.get_speed()
                v_north, v_east, v_up = agent.get_velocity()
                ego_feature = np.hstack([agent.get_position(),
                                         agent.get_velocity()])#获得当前飞机的速度与坐标
                enm_feature = np.hstack([agent.enemies[0].get_position(),
                                         agent.enemies[0].get_velocity()])#获得当前飞机的对手飞机的速度与坐标
                AO, TA, D = get_AO_TA_R(ego_feature, enm_feature)  # 目标方位角θ，且θ为正，没有负值。D是两机间的距离
                AO = np.degrees(AO)#将弧度制转化为度数制
                TA = np.degrees(TA)

                if agent_id=="A0100":
                    if event == 0:
                        self.record_data.append({
                            "acmi_file_id": acmi_file_id, #来自哪一个文件
                            "time_step": time,
                            "agent": agent_id,
                            "longitude": lon,#经度
                            "latitude": lat,#维度
                            "altitude": alt,#高度
                            "roll": roll,#滚转
                            "pitch": pitch,#俯仰
                            "yaw": yaw,#航向
                            "speed": speed,#速度
                            "v_north":v_north,
                            "v_east":v_east,
                            "v_up":v_up,
                            "AO":AO,#目标方位角
                            "TA":TA,#目标进入角
                            "Distance":D,#两机距离
                            "event": event,# 0:未发射 \ 1:此刻发射 \ 2:此刻 miss \ 3:此刻击中
                            "target": "null"
                        })
                    else:
                        self.record_data.append({
                            "acmi_file_id": acmi_file_id, #来自哪一个文件
                            "time_step": time,
                            "agent": agent_id,
                            "longitude": lon,#经度
                            "latitude": lat,#维度
                            "altitude": alt,#高度
                            "roll": roll,#滚转
                            "pitch": pitch,#俯仰
                            "yaw": yaw,#航向
                            "speed": speed,#速度
                            "v_north":v_north,
                            "v_east":v_east,
                            "v_up":v_up,
                            "AO":AO,#目标方位角
                            "TA":TA,#目标进入角
                            "Distance":D,#两机距离
                            "event": event,# 0:未发射 \ 1:此刻发射 \ 2:此刻 miss \ 3:此刻击中
                            "target": self.agents['A0100'].launch_missiles[0].target_aircraft.uid
                        })
                else:
                    self.record_data.append({
                        "acmi_file_id": acmi_file_id, #来自哪一个文件
                        "time_step": time,
                        "agent": agent_id,
                        "longitude": lon,#经度
                        "latitude": lat,#维度
                        "altitude": alt,#高度
                        "roll": roll,#滚转
                        "pitch": pitch,#俯仰
                        "yaw": yaw,#航向
                        "speed": speed,#速度
                        "v_north":v_north,
                        "v_east":v_east,
                        "v_up":v_up,
                        "AO": AO,
                        "TA": TA,  # 目标进入角
                        "Distance": D,
                        "event": 0,# 0:未发射 \ 1:此刻发射 \ 2:此刻 miss \ 3:此刻击中
                        "target": "null"
                    })

    def save_record_data_to_csv(self, folder_name, file_name='record_data.csv'):
        # 确保有数据可以写入
        if not self.record_data:
            print("No record data to save .")
            return

        # 创建完整的文件路径
        file_path = os.path.join(folder_name, file_name)

        # 获取字典中的所有键作为 CSV 的列标题
        keys = self.record_data[0].keys() if self.record_data else []

        # 打开文件并写入数据
        with open(file_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=keys)
            # 写入表头
            writer.writeheader()
            # 写入所有记录
            writer.writerows(self.record_data)

        print(f"Data saved to {file_path}")


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's observation. Accepts an action and
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (dict): the agents' actions, each key corresponds to an agent_id

        Returns:
            (tuple):
                obs: agents' observation of the current environment
                share_obs: agents' share observation of the current environment
                rewards: amount of rewards returned after previous actions
                dones: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information
        """
        self.current_step += 1
        info = {"current_step": self.current_step}

        #记录初始化状态
        if self.current_step==1:
            self.record_aircraft_state(0,GlobalVars.shared_acmi_id,0)#

        # apply actions
        action = self._unpack(action)
        for agent_id in self.agents.keys():
            a_action = self.task.normalize_action(self, agent_id, action[agent_id])
            self.agents[agent_id].set_property_values(self.task.action_var, a_action)

        # run simulation
        # count防止多次记录攻击结果
        count = 0
        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():
                if sim.uid=="B0100" or sim.uid=="B0200":#敌机运动，我机保持态势不变
                    sim.run()
            for sim in self._tempsims.values():
                sim.run()

            # 判断是否有导弹done：
            if count == 0:
                for missile in self._tempsims.values():
                    if missile.is_success:  # 如果是击中状态，之后仿真结束，不会出现击中事件重复记录的问题
                        count += 1
                        self.record_aircraft_state(self.current_step, GlobalVars.shared_acmi_id, 3)
                        break
                    elif missile.is_miss:  # 如果 真miss(发射了，但是没击中)，仿真未结束，多个时间步都会重复记录该事件，需要在下面加判断：
                        count += 1
                        if self.record_data[-3]["event"] == 2:  # 说明导弹未击中已经被记录过了，无须重复记录
                            break
                        else:  # 说明导弹被击中没有被记录过，需要进行记录
                            self.record_aircraft_state(self.current_step, GlobalVars.shared_acmi_id, 2)
                            break
                    else:  # 既不是miss，也不是hit
                        break

        #创建并发射导弹
        self.task.step(self)

        #在进行下一个时刻的run之前，进行一步判断，看看导弹是否进入发射状态，即在run之前看看是否执行了create
        if GlobalVars.shared_missile_shootpoint==True:
            self.record_aircraft_state(self.current_step,GlobalVars.shared_acmi_id,1)

        obs = self.get_obs()
        share_obs = self.get_state()

        rewards = {}
        for agent_id in self.agents.keys():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]
        ego_reward = np.mean([rewards[ego_id] for ego_id in self.ego_ids])
        enm_reward = np.mean([rewards[enm_id] for enm_id in self.enm_ids])
        for ego_id in self.ego_ids:
            rewards[ego_id] = [ego_reward]
        for enm_id in self.enm_ids:
            rewards[enm_id] = [enm_reward]

        dones = {}
        for agent_id in self.agents.keys():
            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [done]

        shoot_result = {}  # 用于存放该时刻是否有导弹击中目标
        shoot_result["A0100"] = np.array([0, 0, 0, 0,self.current_step-1]) #导弹发射结果字典进行初始化
        shoot_result["B0100"] = np.array([0, 0, 0, 0,self.current_step-1]) #五个位置的含义：发射时间，导弹是否击中，导弹是否未命中，导弹是否仍存活,当前时间步

        for missle_id, missle in self._tempsims.items():
            if missle.hitFlag == 1 or missle.missFlag ==1:  # 如果有导弹击中目标
                dones["B0100"] = [True]  # 手动结束，因为导弹发完了
                dones["A0100"] = [True]
                dones["B0200"] = [True]
                dones["A0200"] = [True]


        return self._pack(obs), self._pack(share_obs), self._pack(rewards), self._pack(dones), info
