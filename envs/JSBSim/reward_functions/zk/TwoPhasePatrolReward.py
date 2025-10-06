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
        self.H_TARGET_FT = 30000.0
        self.V_TARGET_FPS = 895.8
        self.ENERGY_THRESHOLD_PCT = 0.95

        # --- “维持”状态的容差范围 ---
        self.H_SUSTAIN_TOLERANCE_FT = 2000.0
        self.V_SUSTAIN_TOLERANCE_FPS = 100.0

        # --- 姿态限制 ---
        self.MAX_AOA_DEG = 20.0
        self.MAX_PITCH_RAD = math.radians(60.0)

        # --- 物理常数 ---
        self.G_FPS2 = 32.174

        # --- 奖励权重 ---
        self.w_sustain = getattr(self, 'w_sustain', 3.0)
        self.w_transition = getattr(self, 'w_transition', 5.0)  # 状态转换奖励权重
        self.w_accumulate = getattr(self, 'w_accumulate', 1e-6)
        self.w_stability = getattr(self, 'w_stability', 0.3)

        self._target_energy = None

        self.previous_state = {}

    def reset(self, task, env):
        # 此版本不需要存储历史状态来进行能量计算
        self._target_energy = None
        self.previous_state.clear()
        return super().reset(task, env)

    def _get_total_energy(self, altitude_ft, velocity_fps, total_mass_slug, total_weight_lbs):
        """辅助函数：根据输入计算总能量"""
        potential_energy = total_weight_lbs * altitude_ft
        kinetic_energy = 0.5 * total_mass_slug * (velocity_fps ** 2)
        return potential_energy + kinetic_energy

    def _calculate_target_energy(self, agent):
        """计算并缓存目标状态下的总能量"""
        # 注意：目标能量会随燃油消耗而轻微变化，这里我们简化为使用当前质量计算
        body_mass_slug = agent.get_property_value("inertia/mass-slugs")
        fuel_weight_lbs = agent.get_property_value("propulsion/tank-contents-lbs")
        total_mass_slug = body_mass_slug + (fuel_weight_lbs / self.G_FPS2)
        total_weight_lbs = total_mass_slug * self.G_FPS2

        self._target_energy = self._get_total_energy(self.H_TARGET_FT, self.V_TARGET_FPS, total_mass_slug,
                                                     total_weight_lbs)
        return self._target_energy

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]

        # --- 1. 获取所有状态 ---
        current_altitude_ft = agent.get_property_value("position/h-sl-ft")
        current_velocity_fps = agent.get_property_value("velocities/ve-fps")
        # ... (获取其他所需参数) ...
        body_mass_slug = agent.get_property_value("inertia/mass-slugs")
        fuel_weight_lbs = agent.get_property_value("propulsion/tank-contents-lbs")
        total_mass_slug = body_mass_slug + (fuel_weight_lbs / self.G_FPS2)
        total_weight_lbs = total_mass_slug * self.G_FPS2

        # --- 2. 核心逻辑：状态机判断 ---
        in_sustain_zone = (abs(current_altitude_ft - self.H_TARGET_FT) < self.H_SUSTAIN_TOLERANCE_FT) and \
                          (abs(current_velocity_fps - self.V_TARGET_FPS) < self.V_SUSTAIN_TOLERANCE_FPS)

        R_main = 0.0
        if in_sustain_zone:
            # 状态一: 维持 (Sustain)
            alt_error = abs(current_altitude_ft - self.H_TARGET_FT)
            vel_error = abs(current_velocity_fps - self.V_TARGET_FPS)
            norm_reward_H = 1.0 - (alt_error / self.H_SUSTAIN_TOLERANCE_FT)
            norm_reward_V = 1.0 - (vel_error / self.V_SUSTAIN_TOLERANCE_FPS)
            sustain_reward = norm_reward_H + norm_reward_V
            R_main = self.w_sustain * sustain_reward
        else:
            # 不在维持区，需要判断能量
            current_energy = self._get_total_energy(current_altitude_ft, current_velocity_fps, total_mass_slug,
                                                    total_weight_lbs)
            if self._target_energy is None:
                self._calculate_target_energy(agent)

            if current_energy >= self._target_energy * self.ENERGY_THRESHOLD_PCT:
                # 状态二: 能量分配 / 状态转换 (Transition)
                # 奖励与目标状态“距离”的减小
                # 使用归一化误差来计算距离，避免单位影响
                H_SCALE = 50000.0  # 典型高度范围
                V_SCALE = 1500.0  # 典型速度范围

                current_dist = math.sqrt(((current_altitude_ft - self.H_TARGET_FT) / H_SCALE) ** 2 + \
                                         ((current_velocity_fps - self.V_TARGET_FPS) / V_SCALE) ** 2)

                prev_dist = self.previous_state.get(agent_id, {}).get('last_dist', current_dist)

                # 奖励 = (上一时刻的距离 - 当前距离)，即距离的减小量
                dist_reduction = prev_dist - current_dist
                R_main = self.w_transition * dist_reduction

                self.previous_state[agent_id] = {'last_dist': current_dist}
            else:
                # 状态三: 能量积累 (Accumulate)
                h_dot_fps = agent.get_property_value("velocities/h-dot-fps")
                tangential_accel_fps2 = agent.get_property_value("accelerations/a-pilot-x-ft_sec2")
                dEp_dt = total_weight_lbs * h_dot_fps
                dEk_dt = total_mass_slug * current_velocity_fps * tangential_accel_fps2
                energy_rate = dEp_dt + dEk_dt
                R_main = self.w_accumulate * energy_rate

        # --- 3. 稳定性惩罚 (始终生效) ---
        pitch_rad = agent.get_property_value("attitude/pitch-rad")
        aoa_deg = agent.get_property_value("aero/alpha-deg")
        roll_rad = agent.get_property_value("attitude/roll-rad")
        R_stability = 0.0
        if abs(aoa_deg) > self.MAX_AOA_DEG: R_stability -= (abs(aoa_deg) - self.MAX_AOA_DEG) * 0.1
        if abs(pitch_rad) > self.MAX_PITCH_RAD: R_stability -= (abs(pitch_rad) - self.MAX_PITCH_RAD) * 0.05
        # 在非维持状态下，不惩罚滚转
        if in_sustain_zone: R_stability -= abs(roll_rad) * 0.01

        # --- 4. 组合最终奖励 ---
        new_reward = R_main + self.w_stability * R_stability
        return self._process(new_reward, agent_id)