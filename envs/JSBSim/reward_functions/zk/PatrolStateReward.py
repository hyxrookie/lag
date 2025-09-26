import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.core.catalog import Catalog as c


class PatrolStateReward(BaseRewardFunction):
    """
    当没有探测到敌机时，奖励无人机进入并维持一个理想的巡逻状态（能量优势）。
    - 当无人机在理想区间内时，给予一个稳定的正奖励。
    - 当无人机在理想区间外时，使用势函数法引导其向区间的最近边缘靠拢。
    """

    def __init__(self, config):
        super().__init__(config)
        # --- 目标区间定义 ---
        self.H_MIN = getattr(self.config, 'H_MIN', 9000.0)
        self.H_MAX = getattr(self.config, 'H_MAX', 10000.0)
        self.V_MIN = getattr(self.config, 'V_MIN', 269.55)  # 0.9 Mach
        self.V_MAX = getattr(self.config, 'V_MAX', 329.45)  # 1.1 Mach

        # --- 奖励和缩放系数 ---
        self.w_altitude = getattr(self.config, 'w_altitude', 1.0)
        self.w_velocity = getattr(self.config, 'w_velocity', 0.8)
        # 在舒适区内时给予的稳定奖励
        self.stay_in_zone_bonus = getattr(self.config, 'stay_in_zone_bonus', 1.0)

        # 存储上一时刻的状态
        self.previous_state = {}

    def reset(self, task, env):
        self.previous_state.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]

        if any(enm.is_alive for enm in agent.share_detected_enemies):
            return 0

        current_altitude = agent.get_position()[2]
        current_velocity = np.linalg.norm(agent.get_velocity())
        print("velocity:{}, current_velocity:{}, current_altitude".format(agent.get_velocity(), current_velocity, current_altitude))

        if agent_id not in self.previous_state:
            self.previous_state[agent_id] = {'altitude': current_altitude, 'velocity': current_velocity}
            return 0

        prev_state = self.previous_state[agent_id]
        R_altitude = 0.0
        R_velocity = 0.0

        # --- 1. 高度奖励计算 (容忍区间逻辑) ---
        if self.H_MIN <= current_altitude <= self.H_MAX:
            # 在区间内，给予稳定奖励
            R_altitude = self.stay_in_zone_bonus
        else:
            # 在区间外，使用势函数法引导至最近的边缘
            target_H = self.H_MIN if current_altitude < self.H_MIN else self.H_MAX
            prev_dist_H = abs(prev_state['altitude'] - target_H)
            curr_dist_H = abs(current_altitude - target_H)
            R_altitude = 5 if prev_dist_H > curr_dist_H else -7

        # --- 2. 速度奖励计算 (容忍区间逻辑) ---
        if self.V_MIN <= current_velocity <= self.V_MAX:
            # 在区间内，给予稳定奖励
            R_velocity = self.stay_in_zone_bonus
        else:
            # 在区间外，使用势函数法引导至最近的边缘
            target_V = self.V_MIN if current_velocity < self.V_MIN else self.V_MAX
            prev_dist_V = abs(prev_state['velocity'] - target_V)
            curr_dist_V = abs(current_velocity - target_V)
            R_velocity = 5 if prev_dist_V > curr_dist_V else -7

        new_reward = self.w_altitude * R_altitude + self.w_velocity * R_velocity

        self.previous_state[agent_id] = {'altitude': current_altitude, 'velocity': current_velocity}
        print("PatrolStateReward:{}".format(new_reward))
        return self._process(new_reward, agent_id)