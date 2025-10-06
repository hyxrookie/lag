import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.core.catalog import Catalog as c, JsbsimCatalog


class TwoPhasePatrolReward(BaseRewardFunction):
    """
    为F-16设计的两阶段巡逻奖励函数 (V2 - 优化版)。
    专为 dt=0.2s 的仿真环境优化，使用 a-pilot-x-ft_sec2 直接计算动能变化，确保奖励信号精确。
    目标状态: 35,000英尺, 0.9马赫。
    """

    def __init__(self, config):
        super().__init__(config)
        # --- 目标状态定义 ---
        self.H_TARGET_FT = 35000.0
        self.V_TARGET_FPS = 875.8

        # --- 阶段切换的容差范围 ---
        self.H_TOLERANCE_FT = 2000.0
        self.V_TOLERANCE_FPS = 50.0

        # --- 宽松的姿态限制 ---
        self.MAX_AOA_DEG = 20.0
        self.MAX_PITCH_RAD = math.radians(60.0)

        # --- 物理常数 ---
        self.G_FPS2 = 32.174

        # --- 奖励权重 ---
        self.w_energy_rate = getattr(self.config, 'w_energy_rate', 1e-6)
        self.w_sustain = getattr(self.config, 'w_sustain', 3.0)
        self.w_stability = getattr(self.config, 'w_stability', 0.3)

    def reset(self, task, env):
        # 此版本不需要存储历史状态来进行能量计算
        pass

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]

        if not agent.is_alive or any(enm.is_alive for enm in agent.share_detected_enemies):
            return 0

        # --- 1. 获取所有需要的状态 ---
        current_altitude_ft = agent.get("position/h-sl-ft")
        current_velocity_fps = agent.get_property_value("velocities/ve-fps")
        h_dot_fps = agent.get_property_value("velocities/h-dot-fps")

        # 直接获取机头方向的瞬时加速度，这是关键！
        tangential_accel_fps2 = agent.get_property_value("accelerations/a-pilot-x-ft_sec2")

        body_mass_slug = agent.get_property_value("inertia/mass-slugs")
        fuel_weight_lbs = agent.get_property_value("propulsion/tank/contents-lbs")
        total_mass_slug = body_mass_slug + (fuel_weight_lbs / self.G_FPS2)
        total_weight_lbs = total_mass_slug * self.G_FPS2

        pitch_rad = agent.get_property_value("attitude/pitch-rad")
        aoa_deg = agent.get_property_value("aero/alpha-deg")
        roll_rad = agent.get_property_value("attitude/roll-rad")

        # --- 2. 稳定性惩罚 ---
        R_stability = 0.0
        if abs(aoa_deg) > self.MAX_AOA_DEG: R_stability -= (abs(aoa_deg) - self.MAX_AOA_DEG) * 0.1
        if abs(pitch_rad) > self.MAX_PITCH_RAD: R_stability -= (abs(pitch_rad) - self.MAX_PITCH_RAD) * 0.05
        R_stability -= abs(roll_rad) * 0.01

        # --- 3. 核心奖励逻辑 ---
        alt_error = abs(current_altitude_ft - self.H_TARGET_FT)
        vel_error = abs(current_velocity_fps - self.V_TARGET_FPS)
        in_sustain_zone = (alt_error < self.H_TOLERANCE_FT) and (vel_error < self.V_TOLERANCE_FPS)

        R_main = 0.0
        if not in_sustain_zone:
            # 阶段一: 能量积累 (使用精确的瞬时加速度)
            dEp_dt = total_weight_lbs * h_dot_fps
            dEk_dt = total_mass_slug * current_velocity_fps * tangential_accel_fps2
            energy_rate = dEp_dt + dEk_dt
            R_main = self.w_energy_rate * energy_rate
        else:
            # 阶段二: 状态维持
            norm_reward_H = 1.0 - (alt_error / self.H_TOLERANCE_FT)
            norm_reward_V = 1.0 - (vel_error / self.V_TOLERANCE_FPS)
            sustain_reward = norm_reward_H + norm_reward_V
            R_main = self.w_sustain * sustain_reward

        # print("in_sustain_zone:{}, R_main:{}, R_stability:{}".format(in_sustain_zone, R_main, R_stability))
        # --- 4. 组合最终奖励 ---
        new_reward = R_main + self.w_stability * R_stability

        return self._process(new_reward, agent_id)