import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.utils.utils import get_AO_TA_R
from envs.JSBSim.core.catalog import Catalog as c


class LLRTacticalReward(BaseRewardFunction):
    """
    【最终修正版】-> 【AIM-120B专属 4v4 净化版】
    该版本修正了两个关键逻辑错误：
    1. 【动态能量计算】: 能量优势的计算将与“注意力机制”选出的【焦点敌人】动态绑定。
    2. 【补全WVR双雷达模式】: WVR奖励现在会同时考虑HUD扫描和垂直扫描(垂扫)模式。
    【二次开发说明】:
    这是逻辑最简化的最终版本。注意力机制永远聚焦于最近的敌机，
    并且移除了所有额外的、可能污染奖励信号的权重（tactical_value_weight），
    直接使用最纯粹的战术奖励值来驱动AI。
    """

    def __init__(self, config):
        super().__init__(config)

        # --- BVR模式参数与新增战术参数 ---
        self.danger_close_range_m = getattr(
            self.config, 'danger_close_range_m', 8000.0)
        self.max_engagement_range_m = getattr(
            self.config, 'max_engagement_range_m', 70000.0)

        self.penalty_too_close = getattr(
            self.config, 'penalty_too_close', -5.0)

        self.w_bvr_closing = getattr(
            self.config, 'w_bvr_closing', 1.0)
        self.w_bvr_disengaging_angle = getattr(
            self.config, 'w_bvr_disengaging_angle', 2.0)

        # --- BVR (AIM-120B) 模式权重与参数 (保留原始) ---
        self.w_bvr_pre_launch_range = getattr(
            self.config, 'w_bvr_pre_launch_range', 2.0)
        self.w_bvr_pre_launch_angle = getattr(
            self.config, 'w_bvr_pre_launch_angle', 1.5)
        self.w_bvr_crank_angle = getattr(
            self.config, 'w_bvr_crank_angle', 2.5)
        self.w_bvr_beam_angle = getattr(
            self.config, 'w_bvr_beam_angle', 4.0)
        self.optimal_bvr_launch_range_m = getattr(
            self.config, 'optimal_bvr_launch_range_m', 40000.0)
        self.bvr_range_sigma_m = getattr(
            self.config, 'bvr_range_sigma_m', 5000.0)
        self.CRM_LIMIT_DEG = getattr(
            self.config, 'CRM_LIMIT_DEG', 35.0)
        self.optimal_crank_ao_deg = getattr(
            self.config, 'optimal_crank_ao_deg', 50.0)

        self.min_defensive_duration_steps = getattr(
            self.config, 'min_defensive_duration_steps', 50)
        self.max_defensive_duration_steps = getattr(
            self.config, 'max_defensive_duration_steps', 200)

        # --- 通用能量和高度奖励参数 ---
        # ... (与上一版完全相同) ...
        self.w_energy = getattr(self.config, 'w_energy', 0.5)
        self.energy_vel_sq_factor = getattr(self.config, 'energy_vel_sq_factor', 0.05)
        self.MAX_ALTITUDE_METER = getattr(self.config, 'MAX_ALTITUDE_METER', 11000.0)
        self.penalty_hard_ceiling = getattr(self.config, 'penalty_hard_ceiling', -200.0)
        self.OPTIMAL_ALT_MAX_METER = getattr(self.config, 'OPTIMAL_ALT_MAX_METER', 10500.0)
        self.OPTIMAL_ALT_MIN_METER = getattr(self.config, 'OPTIMAL_ALT_MIN_METER', 5000.0)
        self.w_altitude_penalty = getattr(self.config, 'w_altitude_penalty', -0.1)
        self.previous_metrics = {}

    def reset(self, task, env):
        self.previous_metrics.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]
        alive_enemies = [enm for enm in agent.enemies if enm.is_alive]
        if not alive_enemies:
            if agent_id in self.previous_metrics: del self.previous_metrics[agent_id]
            return 0.0
        if agent_id not in self.previous_metrics:
            self.previous_metrics[agent_id] = {'agent_states': {}, 'enemy_states': {}}

        agent_history = self.previous_metrics[agent_id].get('agent_states', {})
        current_amraam_num = task._remaining_missiles[agent_id]
        prev_amraam_num = agent_history.get('amraam_num', current_amraam_num)
        just_launched = current_amraam_num < prev_amraam_num
        ego_feature = np.hstack([agent.get_position(), agent.get_velocity()])
        all_enemy_tactical_info = []

        for enemy in alive_enemies:
            enm_feature = np.hstack([enemy.get_position(), enemy.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
            ao_deg = abs(math.degrees(AO))
            ta_deg = abs(math.degrees(TA))

            # ... (BVR状态机逻辑不变) ...
            enemy_states_history = self.previous_metrics[agent_id].get('enemy_states', {})
            per_enemy_history = enemy_states_history.get(enemy.uid, {})
            bvr_state = per_enemy_history.get('bvr_state', 0)
            defensive_timer = per_enemy_history.get('defensive_timer', 0)

            if bvr_state == 0:
                if just_launched and R > self.danger_close_range_m: bvr_state = 1
            elif bvr_state == 1:
                is_high_entry_angle = ta_deg > 150
                threat_condition = is_high_entry_angle
                ego_energy = agent.get_position()[2] + self.energy_vel_sq_factor * np.linalg.norm(
                    agent.get_velocity()) ** 2
                enm_energy = enemy.get_position()[2] + self.energy_vel_sq_factor * np.linalg.norm(
                    enemy.get_velocity()) ** 2
                timeline_advantage = ego_energy > enm_energy
                decision_is_win = not threat_condition or timeline_advantage
                if decision_is_win:
                    bvr_state = 0
                else:
                    bvr_state = 2
                    defensive_timer = 1
            elif bvr_state == 2:
                defensive_timer += 1
                min_time_passed = defensive_timer > self.min_defensive_duration_steps
                timeout_reached = defensive_timer > self.max_defensive_duration_steps
                is_high_entry_angle = ta_deg > 150
                threat_is_gone = not is_high_entry_angle
                if (threat_is_gone and min_time_passed) or timeout_reached:
                    bvr_state = 0
                    defensive_timer = 0

            single_enemy_reward = 0.0

            # ... (三段式奖励计算逻辑不变) ...
            if R > self.optimal_bvr_launch_range_m:
                bvr_state = 0
                range_reward = 1.0 - (R - self.optimal_bvr_launch_range_m) / (
                            self.max_engagement_range_m - self.optimal_bvr_launch_range_m)
                range_reward = max(0, min(1, range_reward))
                angle_reward = -2.0 if ao_deg > self.CRM_LIMIT_DEG else 1.0 - (ao_deg / self.CRM_LIMIT_DEG)
                single_enemy_reward = self.w_bvr_closing * range_reward + self.w_bvr_pre_launch_angle * angle_reward
            elif R > self.danger_close_range_m:
                if bvr_state == 0:
                    range_diff = R - self.optimal_bvr_launch_range_m;
                    R_range = math.exp(-(range_diff ** 2) / (2 * self.bvr_range_sigma_m ** 2))
                    R_angle = -2.0 if ao_deg > self.CRM_LIMIT_DEG else 1.0 - (ao_deg / self.CRM_LIMIT_DEG)
                    single_enemy_reward = self.w_bvr_pre_launch_range * R_range + self.w_bvr_pre_launch_angle * R_angle
                elif bvr_state == 1:
                    angle_diff = ao_deg - self.optimal_crank_ao_deg;
                    R_angle = -2.0 if ao_deg > self.CRM_LIMIT_DEG else math.exp(-(angle_diff ** 2) / (2 * 15 ** 2))
                    single_enemy_reward = self.w_bvr_crank_angle * R_angle
                elif bvr_state == 2:
                    angle_diff = ao_deg - 90.0;
                    R_angle = math.exp(-(angle_diff ** 2) / (2 * 20 ** 2))
                    single_enemy_reward = self.w_bvr_beam_angle * R_angle
            else:
                bvr_state = 2
                range_penalty = self.penalty_too_close * (1.0 - (R / self.danger_close_range_m))
                angle_reward = ao_deg / 180.0
                single_enemy_reward = range_penalty + self.w_bvr_disengaging_angle * angle_reward

            # [核心修改] 彻底移除tactical_value_weight，直接使用纯粹的奖励值
            # tactical_value_weight = (1 / (R + 1e-6)) * (1 + math.cos(AO)) / 2.0
            # total_tactical_value = single_enemy_reward * tactical_value_weight

            all_enemy_tactical_info.append((single_enemy_reward, R, enemy))

            if enemy.uid not in self.previous_metrics[agent_id]['enemy_states']:
                self.previous_metrics[agent_id]['enemy_states'][enemy.uid] = {}
            self.previous_metrics[agent_id]['enemy_states'][enemy.uid].update(
                {'bvr_state': bvr_state, 'defensive_timer': defensive_timer})

        # [核心修改] 注意力机制：永远聚焦于最近的敌机
        if all_enemy_tactical_info:
            closest_enemy_info = min(all_enemy_tactical_info, key=lambda item: item[1])
            final_tactical_reward, _, focused_enemy = closest_enemy_info
        else:
            final_tactical_reward = 0.0
            focused_enemy = None

        # --- [核心修正1: 动态能量计算] ---
        R_energy = 0.0
        if focused_enemy:
            ego_alt = agent.get_position()[2]
            ego_vel = np.linalg.norm(agent.get_velocity())
            ego_energy = ego_alt + self.energy_vel_sq_factor * ego_vel ** 2
            enm_energy = focused_enemy.get_position()[2] + self.energy_vel_sq_factor * np.linalg.norm(
                focused_enemy.get_velocity()) ** 2
            R_energy = 0.5 if ego_energy > enm_energy else -0.5

        # 高度奖励 (逻辑不变)
        ego_alt = agent.get_position()[2]
        R_altitude = 0.0
        if ego_alt > self.MAX_ALTITUDE_METER:
            R_altitude = self.penalty_hard_ceiling
        elif ego_alt > self.OPTIMAL_ALT_MAX_METER:
            R_altitude = self.w_altitude_penalty * (ego_alt - self.OPTIMAL_ALT_MAX_METER) / 500.0
        elif ego_alt < self.OPTIMAL_ALT_MIN_METER:
            R_altitude = self.w_altitude_penalty * (self.OPTIMAL_ALT_MIN_METER - ego_alt) / 500.0

        final_reward = final_tactical_reward + self.w_energy * R_energy + R_altitude

        self.previous_metrics[agent_id]['agent_states']['amraam_num'] = current_amraam_num

        if 'enemy_states' in self.previous_metrics[agent_id]:
            current_enemy_ids = {enm.uid for enm in alive_enemies}
            obsolete_ids = [eid for eid in self.previous_metrics[agent_id]['enemy_states'] if
                            eid not in current_enemy_ids]
            for eid in obsolete_ids:
                del self.previous_metrics[agent_id]['enemy_states'][eid]
        # print("agentid:{},LLRTacticalReward奖励：{}".format(agent_id, final_tactical_reward))
        return self._process(final_reward, agent_id)