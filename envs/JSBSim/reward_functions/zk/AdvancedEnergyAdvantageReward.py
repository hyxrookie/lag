import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math


class AdvancedEnergyAdvantageReward(BaseRewardFunction):
    """
    高级空战奖励函数，包含智能目标选择逻辑。
    - 动态选择威胁最大的敌人作为目标。
    - 奖励获得并扩大相对于该目标的能量优势。
    """

    def __init__(self, config):
        super().__init__(config)
        # --- 物理常数 ---
        self.G_FPS2 = 32.174


        # --- 威胁评估权重 ---
        self.w_threat_dist = getattr(self.config, 'w_threat_dist', 1.0)  # 距离在威胁评估中的权重
        self.w_threat_energy = getattr(self.config, 'w_threat_energy', 0.5)  # 能量在威胁评估中的权重

        # --- 奖励权重 ---
        self.w_energy_advantage = getattr(self.config, 'w_energy_advantage', 5.0e-7)

        self.previous_state = {}

    def reset(self, task, env):
        self.previous_state.clear()
        return super().reset(task, env)

    def _get_total_energy(self, agent):
        """辅助函数：计算单个智能体的总能量"""
        # (此函数与上一版本完全相同)
        altitude_ft = agent.get_property_value("position/h-sl-ft")
        velocity_fps = agent.get_property_value("velocities/ve-fps")
        body_mass_slug = agent.get_property_value("inertia/mass-slugs")
        fuel_weight_lbs = agent.get_property_value("propulsion/tank/contents-lbs")
        total_mass_slug = body_mass_slug + (fuel_weight_lbs / self.G_FPS2)
        total_weight_lbs = total_mass_slug * self.G_FPS2
        potential_energy = total_weight_lbs * altitude_ft
        kinetic_energy = 0.5 * total_mass_slug * (velocity_fps ** 2)
        return potential_energy + kinetic_energy

    def _get_position_vector(self, agent):
        """辅助函数：获取智能体的位置向量"""
        # 注意：JSBSim通常使用大地坐标(lat, lon, alt)，为简化计算，
        # 我们假设有一个局部笛卡尔坐标系或使用经纬度直接计算。
        # 这里使用一个简化的例子，您可能需要根据环境替换为更精确的坐标。
        #
        # 更好的方式是使用 v-north, v-east, v-down 积分得到局部坐标，
        # 或者直接从环境中获取笛卡尔坐标x, y, z。
        # 假设环境提供了 get_position() 方法返回 [x, y, z]
        return agent.get_position()

    def _select_target(self, ego_agent, enemies):
        """智能选择威胁最大的敌人"""
        if not enemies:
            return None

        best_target = None
        max_threat = -1.0

        my_pos = self._get_position_vector(ego_agent)
        my_energy = self._get_total_energy(ego_agent)

        for enm in enemies:
            enm_pos = self._get_position_vector(enm)
            enm_energy = self._get_total_energy(enm)

            distance = np.linalg.norm(my_pos - enm_pos)
            # 加上一个很小的数epsilon，防止距离为0时除法错误
            distance = max(distance, 1e-6)

            # 计算威胁指数
            threat_from_dist = self.w_threat_dist / distance
            threat_from_energy = self.w_threat_energy * (enm_energy / my_energy)

            threat_score = threat_from_dist + threat_from_energy

            if threat_score > max_threat:
                max_threat = threat_score
                best_target = enm

        return best_target

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]

        # --- 1. 确认并选择目标 ---
        detected_enemies = [enm for enm in agent.share_detected_enemies if enm.is_alive]

        # 使用新的智能选择函数
        enemy_agent = self._select_target(agent, detected_enemies)

        if not enemy_agent:
            return 0  # 没有敌人或无法选择目标，不给奖励

        # (后续逻辑与之前版本完全相同，但现在是针对动态选择的'enemy_agent')
        # --- 2. 计算当前能量差距 ---
        my_energy = self._get_total_energy(agent)
        enemy_energy = self._get_total_energy(enemy_agent)
        current_energy_delta = my_energy - enemy_energy

        # --- 3. 计算能量差距的变化量作为奖励 ---
        if agent_id not in self.previous_state:
            self.previous_state[agent_id] = {}

        # 注意：如果目标切换了，需要重置'last_energy_delta'
        last_target_id = self.previous_state[agent_id].get('last_target_id', None)
        if last_target_id != enemy_agent.uid:
            prev_energy_delta = current_energy_delta
        else:
            prev_energy_delta = self.previous_state[agent_id].get('last_energy_delta', current_energy_delta)

        reward_change = current_energy_delta - prev_energy_delta
        R_main = self.w_energy_advantage * reward_change

        self.previous_state[agent_id]['last_energy_delta'] = current_energy_delta
        self.previous_state[agent_id]['last_target_id'] = enemy_agent.uid

        # --- 4. 稳定性惩罚 ---
        # (与上一版本相同)
        print("R_main:{}".format(R_main))
        # --- 5. 组合最终奖励 ---
        new_reward = R_main
        return self._process(new_reward, agent_id)