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
        # self.normal_reset_simulators()
        self.random_reset_simulators()

    def random_reset_simulators(self, min_sep_km: float = 0.5):
        import math, random

        # --- 常量定义 ---
        KM_PER_DEG_LAT = 111.132  # 每度纬度≈公里
        KM_PER_DEG_LON_AT_EQ = 111.320  # 赤道每度经度≈公里
        FT_PER_METER = 3.28084

        # --- 基地与范围设置 ---
        red_base_lon_deg = 120.0
        red_base_lat_deg = 60.0
        inner_radius_km = 5.0
        min_base_separation_km = 20.0
        max_base_separation_km = 120.0

        # 红队经度换算（在红队纬度处）
        km_per_deg_lon_red = KM_PER_DEG_LON_AT_EQ * math.cos(math.radians(red_base_lat_deg))

        # 随机生成蓝队基地（相对红队）
        angle_rad = random.uniform(0, 2 * math.pi)
        distance_km = random.uniform(min_base_separation_km, max_base_separation_km)
        delta_lat_deg = (distance_km * math.cos(angle_rad)) / KM_PER_DEG_LAT
        delta_lon_deg = (distance_km * math.sin(angle_rad)) / km_per_deg_lon_red
        blue_base_lat_deg = red_base_lat_deg + delta_lat_deg
        blue_base_lon_deg = red_base_lon_deg + delta_lon_deg

        # 蓝队经度换算（在蓝队纬度处）
        km_per_deg_lon_blue = KM_PER_DEG_LON_AT_EQ * math.cos(math.radians(blue_base_lat_deg))

        # 工具函数：在半径为 R_km 的圆内均匀采样一个偏移（返回经纬度偏移，单位度）
        def sample_offset_deg(R_km: float, km_per_deg_lon: float):
            # 圆内均匀：r = R*sqrt(u), theta~U(0,2π)
            u = random.random()
            r = R_km * math.sqrt(u)
            theta = random.uniform(0.0, 2.0 * math.pi)
            dlat_km = r * math.cos(theta)
            dlon_km = r * math.sin(theta)
            return dlat_km / KM_PER_DEG_LAT, dlon_km / km_per_deg_lon

        # 工具函数：检查平面近似下的两点距离是否>=最小间距
        def far_enough(new_lat_deg, new_lon_deg, placed_list, km_per_deg_lon: float, min_sep: float):
            for (lat_deg, lon_deg) in placed_list:
                dlat_km = (new_lat_deg - lat_deg) * KM_PER_DEG_LAT
                dlon_km = (new_lon_deg - lon_deg) * km_per_deg_lon
                if (dlat_km * dlat_km + dlon_km * dlon_km) < (min_sep * min_sep):
                    return False
            return True

        # 按队伍分组
        red_ids = [sid for sid in self._jsbsims.keys() if sid.startswith('A')]
        blue_ids = [sid for sid in self._jsbsims.keys() if sid.startswith('B')]
        other_ids = [sid for sid in self._jsbsims.keys() if not (sid.startswith('A') or sid.startswith('B'))]

        # 为每个队在其基地圆内放置满足最小间距的点
        def place_team(team_ids, base_lat_deg, base_lon_deg, km_per_deg_lon, R_km, min_sep):
            placed = []  # 已放置的(纬度, 经度)
            # 逐个飞机放置
            for _ in team_ids:
                attempts = 0
                max_attempts = 2000
                cur_min_sep = min_sep
                # 连续数次放不下就小幅放宽（避免极端拥挤导致死循环）
                while True:
                    attempts += 1
                    off_lat_deg, off_lon_deg = sample_offset_deg(R_km, km_per_deg_lon)
                    cand_lat = base_lat_deg + off_lat_deg
                    cand_lon = base_lon_deg + off_lon_deg
                    if far_enough(cand_lat, cand_lon, placed, km_per_deg_lon, cur_min_sep):
                        placed.append((cand_lat, cand_lon))
                        break
                    if attempts >= max_attempts:
                        # 放宽 10% 再继续尝试
                        cur_min_sep *= 0.9
                        attempts = 0
            return placed  # 与 team_ids 顺序对应

        red_positions = place_team(red_ids, red_base_lat_deg, red_base_lon_deg, km_per_deg_lon_red, inner_radius_km,
                                   min_sep_km)
        blue_positions = place_team(blue_ids, blue_base_lat_deg, blue_base_lon_deg, km_per_deg_lon_blue,
                                    inner_radius_km, min_sep_km)

        # 将坐标回填到各飞机并随机其它属性
        # 注意：ic_u_fps 通常为机体前向速度分量，如果你需要总速度=真空速，且引擎/姿态初始化匹配，可继续使用此写法
        # 如需三轴速度，可额外设置 ic_v_fps / ic_w_fps = 0
        # 这里保持你原有的范围
        for idx, sid in enumerate(red_ids):
            sim = self._jsbsims[sid]
            altitude_m = random.randint(5000, 10000)
            heading_deg = random.randint(0, 359)
            speed_mps = random.randint(175, 300)
            lat_deg, lon_deg = red_positions[idx]
            sim.reload({
                "ic_long_gc_deg": lon_deg,
                "ic_lat_geod_deg": lat_deg,
                "ic_h_sl_ft": altitude_m * FT_PER_METER,
                "ic_psi_true_deg": heading_deg,
                "ic_u_fps": speed_mps * FT_PER_METER,
                # "ic_u_fps": 800,
            })

        for idx, sid in enumerate(blue_ids):
            sim = self._jsbsims[sid]
            altitude_m = random.randint(5000, 10000)
            heading_deg = random.randint(0, 359)
            speed_mps = random.randint(150, 300)
            lat_deg, lon_deg = blue_positions[idx]
            sim.reload({
                "ic_long_gc_deg": lon_deg,
                "ic_lat_geod_deg": lat_deg,
                "ic_h_sl_ft": altitude_m * FT_PER_METER,
                "ic_psi_true_deg": heading_deg,
                "ic_u_fps": speed_mps * FT_PER_METER,
                # "ic_u_fps": 800,
            })

        # 对于非 A/B 的 sim_id，给出明确提示（也可改成 raise）
        for sid in other_ids:
            raise ValueError(f"Unsupported sim_id prefix for {sid}. Use 'A' or 'B'.")

        self._tempsims.clear()

    def normal_reset_simulators(self):
        # Assign new initial condition here!
        for sim in self._jsbsims.values():
            sim.reload()
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
        taskname = getattr(self.config, 'task', None)
        if taskname != 'hierarchical_multiplecombat_shoot':
            for agent_id in self.agents.keys():
                reward, info = self.task.get_reward(self, agent_id, info)
                rewards[agent_id] = [reward]
            ego_reward = np.mean([rewards[ego_id] for ego_id in self.ego_ids])
            enm_reward = np.mean([rewards[enm_id] for enm_id in self.enm_ids])
            for ego_id in self.ego_ids:
                rewards[ego_id] = [ego_reward]
            for enm_id in self.enm_ids:
                rewards[enm_id] = [enm_reward]
        else:
            for agent_id in self.agents.keys():
                reward, info = self.task.get_reward(self, agent_id, info)
                rewards[agent_id] = [reward]
            team_reward_ego = 0
            team_reward_enm = 0
            for ego_id in self.ego_ids:
                team_reward_ego, info = self.task.get_team_reward(self, ego_id, info)
                break
            for enm_id in self.enm_ids:
                team_reward_enm, info = self.task.get_team_reward(self, enm_id, info)
                break

            ALPHA = 0.75  # 4v4 常用 0.7~0.85；越大越偏向个体

            # 我方
            for ego_id in self.ego_ids:
                indiv = float(rewards[ego_id][0])  # 你的个体奖励（保持列表结构）
                team = float(team_reward_ego)  # 你的团队奖励（按你现在的取法）
                rewards[ego_id][0] = ALPHA * indiv + (1 - ALPHA) * team

            # 敌方
            for enm_id in self.enm_ids:
                indiv = float(rewards[enm_id][0])
                team = float(team_reward_enm)
                rewards[enm_id][0] = ALPHA * indiv + (1 - ALPHA) * team


        dones = {}
        for agent_id in self.agents.keys():
            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [done]



        return self._pack(obs), self._pack(share_obs), self._pack(rewards), self._pack(dones), info
