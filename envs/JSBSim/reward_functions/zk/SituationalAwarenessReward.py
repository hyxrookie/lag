import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import get_AO_TA_R, get_az_el_R


class SituationalAwarenessReward(BaseRewardFunction):
    """
    一个完全基于固定值的全场态势感知奖励函数。
    它动态切换BVR/WVR模式，并通过威胁加权聚合所有敌人的奖励信号。
    所有奖励项（角度、距离、能量）都基于当前状态，不使用差值计算。
    """

    def __init__(self, config):
        super().__init__(config)
        # --- 权重参数 ---
        self.w_energy = getattr(self.config, 'w_energy', 0.8)

        # --- BVR (超视距, >20km) 参数集 ---
        self.w_angle_bvr = getattr(self.config, 'w_angle_bvr', 1.5)
        self.w_range_bvr = getattr(self.config, 'w_range_bvr', 1.0)
        self.angle_limit_deg_bvr = getattr(self.config, 'angle_limit_deg_bvr', 35.0)
        self.reward_angle_in_bvr = getattr(self.config, 'reward_angle_in_bvr', 5.0)
        self.penalty_angle_out_bvr = getattr(self.config, 'penalty_angle_out_bvr', -4.0)
        self.reward_bvr_zone = getattr(self.config, 'reward_bvr_zone', 2.0)
        # [新] >50km时的固定值奖励
        self.reward_closing_far = getattr(self.config, 'reward_closing_far', 2)
        self.penalty_distancing_far = getattr(self.config, 'penalty_distancing_far', -1)

        # --- WVR (近距格斗, <=20km) 参数集 ---
        self.w_angle_wvr = getattr(self.config, 'w_angle_wvr', 2.0)
        self.w_range_wvr = getattr(self.config, 'w_range_wvr', 0.8)
        self.hud_scan_az_deg = getattr(self.config, 'hud_scan_az_deg', 15.0)
        self.hud_scan_el_deg = getattr(self.config, 'hud_scan_el_deg', 10.0)
        self.vsl_scan_az_deg = getattr(self.config, 'vsl_scan_az_deg', 5.0)
        self.vsl_scan_el_deg = getattr(self.config, 'vsl_scan_el_deg', 30.0)
        self.reward_angle_in_wvr = getattr(self.config, 'reward_angle_in_wvr', 5.0)
        self.penalty_angle_out_wvr = getattr(self.config, 'penalty_angle_out_wvr', -1.5)
        self.reward_wvr_zone = getattr(self.config, 'reward_wvr_zone', 1.0)
        self.penalty_too_close = getattr(self.config, 'penalty_too_close', -2.0)
        self.penalty_improper_range = getattr(self.config, 'penalty_improper_range', -10.0)

        # --- 通用能量奖励参数 (固定值) ---
        # 能量 E ≈ m*g*h + 0.5*m*v^2。为简化，我们忽略常数m,g,0.5，用 E' = h + k*v^2
        # k 的值决定了速度和高度哪个更重要。k = 1/(2g) ≈ 0.05
        self.energy_vel_sq_factor = getattr(self.config, 'energy_vel_sq_factor', 0.05)
        self.reward_energy_advantage = getattr(self.config, 'reward_energy_advantage', 0.5)
        self.penalty_energy_disadvantage = getattr(self.config, 'penalty_energy_disadvantage', -0.5)

        # 历史记录现在只需要存上一刻的距离R
        self.previous_metrics = {}

    def reset(self, task, env):
        self.previous_metrics.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]
        alive_enemies = [enm for enm in agent.share_detected_enemies if enm.is_alive]

        if not alive_enemies:
            if agent_id in self.previous_metrics:
                del self.previous_metrics[agent_id]
            return 0.0

        if agent_id not in self.previous_metrics:
            self.previous_metrics[agent_id] = {}
        agent_history = self.previous_metrics[agent_id]

        total_weighted_reward = 0
        total_threat_weight = 0

        ego_feature = np.hstack([agent.get_position(), agent.get_velocity()])
        ego_alt = agent.get_position()[2]
        ego_vel = np.linalg.norm(agent.get_velocity())
        ego_energy = ego_alt + self.energy_vel_sq_factor * (ego_vel ** 2)
        amraam_num = agent.get("AMRAAMCurrentNum")

        for enemy in alive_enemies:
            R_angle, R_range, R_energy = 0.0, 0.0, 0.0
            R_improper_range = 0.0
            single_enemy_w = {}

            enm_feature = np.hstack([enemy.get_position(), enemy.get_velocity()])
            _, TA, R = get_AO_TA_R(ego_feature, enm_feature)

            # --- a. 动态选择BVR/WVR模式 ---
            if R > 20000:
                # --- BVR 模式 ---
                AO, _, _ = get_AO_TA_R(ego_feature, enm_feature)
                ao_abs_deg = abs(math.degrees(AO))
                R_angle = self.reward_angle_in_bvr if ao_abs_deg <= self.angle_limit_deg_bvr else self.penalty_angle_out_bvr

                if R > 50000:
                    prev_R = agent_history.get(enemy.uid, {}).get('R', None)
                    if prev_R is not None:
                        R_range = self.reward_closing_far if R < prev_R else self.penalty_distancing_far
                    else:
                        R_range = 0  # 首次接触时中性
                else:  # 20km < R <= 50km
                    R_range = self.reward_bvr_zone
                single_enemy_w = {'angle': self.w_angle_bvr, 'range': self.w_range_bvr}

            else:
                # --- WVR 模式 ---
                az_deg, el_deg, _ = get_az_el_R(ego_feature, enm_feature)
                in_hud = (abs(az_deg) < self.hud_scan_az_deg) and (abs(el_deg) < self.hud_scan_el_deg)
                in_vsl = (abs(az_deg) < self.vsl_scan_az_deg) and (abs(el_deg) < self.vsl_scan_el_deg)
                R_angle = self.reward_angle_in_wvr if in_hud or in_vsl else self.penalty_angle_out_wvr

                if R > 2000:
                    R_range = self.reward_wvr_zone
                else:
                    R_range = self.penalty_too_close

                if amraam_num > 0 and R < 15000:
                    R_improper_range = self.penalty_improper_range
                single_enemy_w = {'angle': self.w_angle_wvr, 'range': self.w_range_wvr}

            # --- b. 计算通用的能量奖励 (固定值) ---
            enm_alt = enemy.get_position()[2]
            enm_vel = np.linalg.norm(enemy.get_velocity())
            enm_energy = enm_alt + self.energy_vel_sq_factor * (enm_vel ** 2)
            R_energy = self.reward_energy_advantage if ego_energy > enm_energy else self.penalty_energy_disadvantage

            # --- c. 聚合单体奖励 ---
            single_enemy_reward = (single_enemy_w['angle'] * R_angle +
                                   single_enemy_w['range'] * R_range +
                                   self.w_energy * R_energy +
                                   R_improper_range)

            # --- d. 威胁加权与累加 ---
            threat_weight = (1 / (R + 1e-6)) * (1 - math.cos(TA))
            total_weighted_reward += single_enemy_reward * threat_weight
            total_threat_weight += threat_weight

            # --- e. 更新历史记录 (现在只需要存R) ---
            if enemy.uid not in agent_history:
                agent_history[enemy.uid] = {}
            agent_history[enemy.uid]['R'] = R

        # --- f. 计算最终加权平均奖励 ---
        final_reward = total_weighted_reward / total_threat_weight if total_threat_weight > 0 else 0

        # --- g. 清理历史记录 ---
        current_enemy_ids = {enm.uid for enm in alive_enemies}
        obsolete_ids = [eid for eid in agent_history if eid not in current_enemy_ids]
        for eid in obsolete_ids:
            del agent_history[eid]

        return self._process(final_reward, agent_id)
