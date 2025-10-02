import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.core.catalog import Catalog as c


class PatrolStateReward(BaseRewardFunction):
    """
    当没有探测到敌机时，引导无人机进入并维持一个理想的巡逻状态。
    - 使用平滑的势函数法（Potential-based Reward Shaping）引导飞机。
    - 对危险的飞行姿态（大攻角、极端姿态）进行软性惩罚。
    """

    def __init__(self, config):
        super().__init__(config)
        # --- 目标区间定义 (单位: 米, 米/秒) ---
        self.H_MIN = getattr(self.config, 'H_MIN', 9000.0)
        self.H_MAX = getattr(self.config, 'H_MAX', 10000.0)
        self.V_MIN = getattr(self.config, 'V_MIN', 269.55)  # 0.9 Mach
        self.V_MAX = getattr(self.config, 'V_MAX', 329.45)  # 1.1 Mach

        # --- 奖励和缩放系数 ---
        self.w_altitude = getattr(self.config, 'w_altitude', 0.1)  # 调整后的权重/缩放系数
        self.w_velocity = getattr(self.config, 'w_velocity', 0.08)  # 调整后的权重/缩放系数
        self.w_stability = getattr(self.config, 'w_stability', 0.2)  # 稳定性权重

        # 在舒适区内时给予的稳定奖励
        self.in_zone_bonus = getattr(self.config, 'in_zone_bonus', 0.1)

        # 存储上一时刻的状态
        self.previous_state = {}

    def reset(self, task, env):
        self.previous_state.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]
        FT_TO_M = 0.3048

        if not agent.is_alive or any(enm.is_alive for enm in agent.share_detected_enemies):
            return 0

        # --- 获取当前状态 ---
        current_altitude = agent.get('position/h-sl-ft') * FT_TO_M
        current_velocity = agent.get('velocities/ve-fps') * FT_TO_M

        if agent_id not in self.previous_state:
            self.previous_state[agent_id] = {'altitude': current_altitude, 'velocity': current_velocity}
            return 0

        prev_state = self.previous_state[agent_id]

        # --- 1. 高度势函数奖励 ---
        in_altitude_zone = self.H_MIN <= current_altitude <= self.H_MAX
        target_H = self.H_MIN if current_altitude < self.H_MIN else self.H_MAX
        prev_dist_H = abs(prev_state['altitude'] - target_H)
        curr_dist_H = abs(current_altitude - target_H)

        # 如果在区域内，progress恒为0，只拿bonus
        altitude_progress = 0 if in_altitude_zone else prev_dist_H - curr_dist_H
        R_altitude = self.w_altitude * altitude_progress
        if in_altitude_zone:
            R_altitude += self.in_zone_bonus

        # --- 2. 速度势函数奖励 ---
        in_velocity_zone = self.V_MIN <= current_velocity <= self.V_MAX
        target_V = self.V_MIN if current_velocity < self.V_MIN else self.V_MAX
        prev_dist_V = abs(prev_state['velocity'] - target_V)
        curr_dist_V = abs(current_velocity - target_V)

        velocity_progress = 0 if in_velocity_zone else prev_dist_V - curr_dist_V
        R_velocity = self.w_velocity * velocity_progress
        if in_velocity_zone:
            R_velocity += self.in_zone_bonus

        # --- 3. 稳定性惩罚 ---
        pitch_rad = agent.get('attitude/pitch-rad')
        aoa_rad = math.radians(agent.get('aero/alpha-deg'))

        R_stability = 0.0
        # 惩罚大攻角以防失速
        MAX_AOA_RAD = math.radians(20)
        if abs(aoa_rad) > MAX_AOA_RAD:
            R_stability -= (abs(aoa_rad) - MAX_AOA_RAD) * 1.0  # 对攻角的惩罚要敏感一些

        # 惩罚极端俯仰角
        MAX_PITCH_RAD = math.radians(60)
        if abs(pitch_rad) > MAX_PITCH_RAD:
            R_stability -= (abs(pitch_rad) - MAX_PITCH_RAD) * 0.5

        # --- 组合最终奖励 ---
        new_reward = R_altitude + R_velocity + self.w_stability * R_stability

        self.previous_state[agent_id] = {'altitude': current_altitude, 'velocity': current_velocity}
        return self._process(new_reward, agent_id)