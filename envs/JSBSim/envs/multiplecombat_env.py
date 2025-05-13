import numpy as np
from typing import Tuple, Dict, Any
from .env_base import BaseEnv
from ..tasks.multiplecombat_task import HierarchicalMultipleCombatShootTask, HierarchicalMultipleCombatTask, MultipleCombatTask
import random
import math
class MultipleCombatEnv(BaseEnv):
    """
    MultipleCombatEnv is an multi-player competitive environment.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        self._create_records = False

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
                })

        self._tempsims.clear()

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

        # apply actions
        action = self._unpack(action)
        for agent_id in self.agents.keys():
            a_action = self.task.normalize_action(self, agent_id, action[agent_id])
            self.agents[agent_id].set_property_values(self.task.action_var, a_action)
        # run simulation
        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():
                sim.run()
            for sim in self._tempsims.values():
                sim.run()
        self.task.step(self)
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

        return self._pack(obs), self._pack(share_obs), self._pack(rewards), self._pack(dones), info
