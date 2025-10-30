import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.utils.utils import get_AO_TA_R, get_az_el_R
from envs.JSBSim.core.catalog import Catalog as c


class LLRTacticalReward(BaseRewardFunction):
    """
    【最终修正版】
    该版本修正了两个关键逻辑错误：
    1. 【动态能量计算】: 能量优势的计算将与“注意力机制”选出的【焦点敌人】动态绑定。
    2. 【补全WVR双雷达模式】: WVR奖励现在会同时考虑HUD扫描和垂直扫描(垂扫)模式。
    """

    def __init__(self, config):
        super().__init__(config)

        # --- BVR (AIM-120B) 模式权重与参数 ---
        self.bvr_wvr_switch_range_m = getattr(
            self.config, 'bvr_wvr_switch_range_m', 19000.0)
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
            self.config, 'min_defensive_duration_steps', 50)  # 至少防御5秒 (假设10Hz)
        self.max_defensive_duration_steps = getattr(
            self.config, 'max_defensive_duration_steps', 200)  # 最多防御20秒

        # --- WVR (AIM-9M) 模式权重与参数 ---
        self.w_angle_wvr_tail_chase = getattr(
            self.config, 'w_angle_wvr_tail_chase', 2.5)
        self.w_range_wvr = getattr(
            self.config, 'w_range_wvr', 1.5)
        self.w_angle_wvr_in_scan = getattr(  # [新] 保持在扫描区内的奖励权重
            self.config, 'w_angle_wvr_in_scan', 1.0)

        self.optimal_wvr_launch_range_m = getattr(
            self.config, 'optimal_wvr_launch_range_m', 10000.0)
        self.wvr_range_sigma_m = getattr(
            self.config, 'wvr_range_sigma_m', 3000.0)
        self.too_close_range_m = getattr(
            self.config, 'too_close_range_m', 2000.0)
        self.penalty_too_close_wvr = getattr(
            self.config, 'penalty_too_close_wvr', -2.0)

        # [新] WVR双雷达模式参数
        self.ACM_HUD_AZ_DEG = getattr(self.config, 'ACM_HUD_AZ_DEG', 15.0)  # 30/2
        self.ACM_HUD_EL_DEG = getattr(self.config, 'ACM_HUD_EL_DEG', 10.0)  # 20/2
        self.ACM_VSL_AZ_DEG = getattr(self.config, 'ACM_VSL_AZ_DEG', 5.0)  # 10/2
        self.ACM_VSL_EL_DEG = getattr(self.config, 'ACM_VSL_EL_DEG', 30.0)  # 60/2

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
        current_amraam_num = agent.get("AMRAAMCurrentNum")
        prev_amraam_num = agent_history.get('amraam_num', current_amraam_num)

        just_launched = current_amraam_num < prev_amraam_num

        ego_feature = np.hstack([agent.get_position(), agent.get_velocity()])
        all_enemy_tactical_info = []  # [新] 用于存储 (价值, 敌机对象)

        for enemy in alive_enemies:
            enm_feature = np.hstack([enemy.get_position(), enemy.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
            ao_deg = abs(math.degrees(AO))
            # [新增] ta_deg用于BVR状态机判断
            ta_deg = abs(math.degrees(TA))

            # ... (BVR状态机逻辑不变) ...
            enemy_states_history = self.previous_metrics[agent_id].get('enemy_states', {})
            per_enemy_history = enemy_states_history.get(enemy.uid, {})
            bvr_state = per_enemy_history.get('bvr_state', 0)
            defensive_timer = per_enemy_history.get('defensive_timer', 0)
            is_spiked = agent.get("WarningNumber") >= 2

            if bvr_state == 0:  # 在发射前
                # [修改] 状态切换判断距离改为danger_close_range_m
                if just_launched and R > self.bvr_wvr_switch_range_m:
                    print("agentId:{}发射成功进入crank, enmId:{}".format(agent_id, enemy.uid))
                    bvr_state = 1  # 发射后，进入Crank

            elif bvr_state == 1:  # 在Crank状态下，进行决断
                print("agentId:{}crank, enmId:{}".format(agent_id, enemy.uid))
                is_high_entry_angle = ta_deg > 150
                threat_condition = is_spiked or is_high_entry_angle
                ego_energy = agent.get_position()[2] + self.energy_vel_sq_factor * np.linalg.norm(
                    agent.get_velocity()) ** 2
                enm_energy = enemy.get_position()[2] + self.energy_vel_sq_factor * np.linalg.norm(
                    enemy.get_velocity()) ** 2
                timeline_advantage = ego_energy > enm_energy

                decision_is_win = not threat_condition or timeline_advantage

                if decision_is_win:
                    # 【赢】的决策: 回转，重新进攻！
                    print("agentId:{}crank重新进攻, enmId:{}".format(agent_id, enemy.uid))
                    bvr_state = 0
                else:  # 判定为【输】
                    print("agentId:{}进入防御, enmId:{}".format(agent_id, enemy.uid))
                    bvr_state = 2  # 进入防御
                    defensive_timer = 1  # 启动计时器

            elif bvr_state == 2:  # 如果已经在防御状态
                print("agentId:{}防御, enmId:{}".format(agent_id, enemy.uid))
                defensive_timer += 1  # 计时器累加
                # 退出条件
                min_time_passed = defensive_timer > self.min_defensive_duration_steps
                timeout_reached = defensive_timer > self.max_defensive_duration_steps
                threat_is_gone = not is_spiked

                if (threat_is_gone and min_time_passed) or timeout_reached:
                    print("agentId:{}防御转进攻, enmId:{}".format(agent_id, enemy.uid))
                    bvr_state = 0  # 威胁解除或超时，返回发射前状态
                    defensive_timer = 0

            single_enemy_reward = 0.0
            if R > self.bvr_wvr_switch_range_m:
                # ... (BVR奖励计算逻辑不变) ...
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

            # 区域3: 危险规避/惩罚区 (R <= 危险距离) - 任务：严厉惩罚并引导安全脱离
            else:
                # --- [核心修正2: 补全WVR双雷达模式] ---
                az_deg, el_deg, _ = get_az_el_R(ego_feature, enm_feature)
                in_hud_scan = (abs(az_deg) < self.ACM_HUD_AZ_DEG) and (abs(el_deg) < self.ACM_HUD_EL_DEG)
                in_vsl_scan = (abs(az_deg) < self.ACM_VSL_AZ_DEG) and (abs(el_deg) < self.ACM_VSL_EL_DEG)

                R_angle_in_scan_zone = 1.0 if (in_hud_scan or in_vsl_scan) else -1.0  # 在任一扫描区内都有基础奖励
                R_angle_tail_chase = 1.0 - (TA / math.pi)
                range_diff = R - self.optimal_wvr_launch_range_m
                R_range = math.exp(-(range_diff ** 2) / (2 * self.wvr_range_sigma_m ** 2))
                if R < self.too_close_range_m: R_range += self.penalty_too_close_wvr

                single_enemy_reward = (self.w_angle_wvr_tail_chase * R_angle_tail_chase +
                                       self.w_range_wvr * R_range +
                                       self.w_angle_wvr_in_scan * R_angle_in_scan_zone)
                bvr_state, defensive_timer = 0, 0

            # tactical_value_weight = (1 / (R + 1e-6)) * (1 + math.cos(AO)) / 2.0
            # total_tactical_value = single_enemy_reward * tactical_value_weight
            if just_launched:
                single_enemy_reward += 50
            all_enemy_tactical_info.append((single_enemy_reward, R, enemy))  # 存储价值和敌机对象

            if enemy.uid not in self.previous_metrics[agent_id]['enemy_states']:
                self.previous_metrics[agent_id]['enemy_states'][enemy.uid] = {}
            self.previous_metrics[agent_id]['enemy_states'][enemy.uid].update(
                {'bvr_state': bvr_state, 'defensive_timer': defensive_timer})

        # 注意力机制: 找出价值最高的目标及其信息
        if all_enemy_tactical_info:
            closest_enemy_info = min(all_enemy_tactical_info, key=lambda item: item[1])
            final_tactical_reward, _, focused_enemy = closest_enemy_info

            enemy_states_history = self.previous_metrics[agent_id].get('enemy_states', {})
            per_enemy_history = enemy_states_history.get(focused_enemy.uid, {})
            bvr_state = per_enemy_history.get('bvr_state', 0)
            print("agentId:{}最终选择{}, enmId:{}".format(agent_id, bvr_state, focused_enemy.uid))
        else:
            final_tactical_reward = 0.0
            focused_enemy = None

        # --- [核心修正1: 动态能量计算] ---
        R_energy = 0.0
        if focused_enemy:  # 只与焦点敌人比较能量
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