
import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.utils.utils import get_AO_TA_R, get_az_el_R
from envs.JSBSim.core.catalog import Catalog as c


class NewLLRTacticalReward(BaseRewardFunction):
    """
    【最终重构版 V2 - 全状态趋势与塑形】
    本奖励函数的核心设计哲学是为AI提供最清晰、最直接的学习信号。
    1. 【固定值奖励】: 放弃复杂的连续函数，所有战术位置的好坏都由具体的、分层的固定数值来定义。
    2. 【趋势判断】: 不仅奖励“好的状态”，更奖励“正在进入好状态的过程”。例如，奖励正在接近最佳距离或正在绕后的行为。
    3. 【角速度塑形】: 在需要转弯机动时，对“转对方向”这个动作本身给予持续的微小奖励，解决AI因“目标太远”而不愿开始机动的问题。
    4. 【状态化权重】: 在关键战术机动（如防御）中，动态降低对能量损失的惩罚，确保AI优先执行战术动作而非单纯保存能量。
    """

    def __init__(self, config):
        super().__init__(config)

        # --- 通用参数 ---
        # BVR (超视距) 和 WVR (视距内) 作战模式的切换距离
        self.bvr_wvr_switch_range_m = getattr(self.config, 'bvr_wvr_switch_range_m', 19000.0)

        # --- BVR (AIM-120B) 模式固定奖励值 ---
        # 状态0: 发射前准备 (目标: 进入最佳发射区)
        self.reward_bvr_pre_launch_optimal = getattr(self.config, 'reward_bvr_pre_launch_optimal', 10.0)  # 最佳位置的奖励
        self.reward_bvr_pre_launch_acceptable = getattr(self.config, 'reward_bvr_pre_launch_acceptable',
                                                        5.0)  # 可接受位置的奖励
        self.penalty_bvr_pre_launch_bad = getattr(self.config, 'penalty_bvr_pre_launch_bad', -5.0)  # 糟糕位置的惩罚
        self.BVR_OPTIMAL_RANGE_MAX = getattr(self.config, 'BVR_OPTIMAL_RANGE_MAX', 38000.0)  # BVR最佳射程上限
        self.BVR_OPTIMAL_RANGE_MIN = getattr(self.config, 'BVR_OPTIMAL_RANGE_MIN', 24000.0)  # BVR最佳射程下限

        # 状态1: 曲柄机动 (Crank) (目标: 发射后侧转，保持锁定并规避)
        self.reward_bvr_crank_optimal = getattr(self.config, 'reward_bvr_crank_optimal', 10.0)
        self.reward_bvr_crank_acceptable = getattr(self.config, 'reward_bvr_crank_acceptable', 4.0)
        self.penalty_bvr_crank_bad = getattr(self.config, 'penalty_bvr_crank_bad', -8.0)

        # 状态2: 防御机动 (Beam) (目标: 侧身对敌，利用雷达特性摆脱锁定)
        self.reward_bvr_beam_optimal = getattr(self.config, 'reward_bvr_beam_optimal', 10.0)
        self.reward_bvr_beam_acceptable = getattr(self.config, 'reward_bvr_beam_acceptable', 5.0)
        self.penalty_bvr_beam_bad = getattr(self.config, 'penalty_bvr_beam_bad', -10.0)
        self.reward_bvr_beam_dist_increase = getattr(self.config, 'reward_bvr_beam_dist_increase', 2)  # 防御时成功拉开距离的奖励
        self.penalty_bvr_beam_dist_decrease = getattr(self.config, 'penalty_bvr_beam_dist_decrease',
                                                      -3.0)  # 防御时距离被拉近的惩罚

        # BVR 状态机参数
        self.min_defensive_duration_steps = getattr(self.config, 'min_defensive_duration_steps', 50)
        self.max_defensive_duration_steps = getattr(self.config, 'max_defensive_duration_steps', 200)

        # --- WVR (AIM-9M) 模式固定奖励值 (目标: 绕后咬尾，锁定发射) ---
        self.reward_wvr_optimal_tail_chase = getattr(self.config, 'reward_wvr_optimal_tail_chase', 10.0)  # 完美咬尾+距离合适
        self.reward_wvr_in_scan_zone = getattr(self.config, 'reward_wvr_in_scan_zone', 5.0)  # 仅在扫描区但位置不佳
        self.penalty_wvr_overshoot = getattr(self.config, 'penalty_wvr_overshoot', -8.0)  # 冲过头或距离过近
        self.penalty_wvr_bad_position = getattr(self.config, 'penalty_wvr_bad_position', -5.0)  # 位置非常差

        # --- 趋势与塑形奖励 ---
        # BVR 发射前趋势
        self.reward_bvr_dist_closing = getattr(self.config, 'reward_bvr_dist_closing', 1)  # 在BVR区外，朝最佳距离飞行的奖励
        self.reward_bvr_angle_improving = getattr(self.config, 'reward_bvr_angle_improving', 1)  # 在BVR中，机头朝向敌机的奖励
        # WVR 趋势
        self.reward_wvr_dist_closing = getattr(self.config, 'reward_wvr_dist_closing', 1.5)  # 在WVR中，朝最佳格斗距离逼近的奖励
        self.reward_wvr_tail_angle_closing = getattr(self.config, 'reward_wvr_tail_angle_closing',
                                                     2.0)  # 在WVR中，成功减小尾角的奖励（正在绕后）
        self.penalty_wvr_dist_opening = getattr(self.config, 'penalty_wvr_dist_opening', -2.0)  # 在WVR中，距离被拉远的惩罚
        # 通用塑形
        self.reward_shaping_turn_correctly = getattr(self.config, 'reward_shaping_turn_correctly', 1.0)  # 朝着正确角度转弯的过程奖励
        self.penalty_shaping_turn_wrongly = getattr(self.config, 'penalty_shaping_turn_wrongly', -1.0)  # 转弯方向错误的过程惩罚

        # WVR 雷达扫描参数
        self.ACM_HUD_AZ_DEG = getattr(self.config, 'ACM_HUD_AZ_DEG', 15.0)
        self.ACM_HUD_EL_DEG = getattr(self.config, 'ACM_HUD_EL_DEG', 10.0)
        self.ACM_VSL_AZ_DEG = getattr(self.config, 'ACM_VSL_AZ_DEG', 5.0)
        self.ACM_VSL_EL_DEG = getattr(self.config, 'ACM_VSL_EL_DEG', 30.0)

        # --- 通用能量和高度奖励参数 ---
        self.w_energy = getattr(self.config, 'w_energy', 0.5)
        self.energy_vel_sq_factor = getattr(self.config, 'energy_vel_sq_factor', 0.05)
        # 状态化能量权重：在防御等关键机动中，能量没那么重要，权重应降低
        self.defensive_energy_weight_multiplier = getattr(self.config, 'defensive_energy_weight_multiplier', 0.1)

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
        # 1. 初始化和获取基本信息
        agent = env.agents[agent_id]
        alive_enemies = [enm for enm in agent.enemies if enm.is_alive]
        if not alive_enemies:
            if agent_id in self.previous_metrics:
                del self.previous_metrics[agent_id]
            return 0.0
        if agent_id not in self.previous_metrics:
            self.previous_metrics[agent_id] = {'agent_states': {}, 'enemy_states': {}}

        agent_history = self.previous_metrics[agent_id].get('agent_states', {})
        current_amraam_num = agent.get("AMRAAMCurrentNum")
        prev_amraam_num = agent_history.get('amraam_num', current_amraam_num)
        just_launched = current_amraam_num < prev_amraam_num

        ego_feature = np.hstack([agent.get_position(), agent.get_velocity()])
        all_enemy_tactical_info = []

        # 2. 遍历所有敌机，计算单机战术奖励
        for enemy in alive_enemies:
            enm_feature = np.hstack([enemy.get_position(), enemy.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
            ao_deg = abs(math.degrees(AO))
            ta_deg = abs(math.degrees(TA))

            # 从历史记录中获取该敌机的状态
            enemy_states_history = self.previous_metrics[agent_id].get('enemy_states', {})
            per_enemy_history = enemy_states_history.get(enemy.uid, {})
            bvr_state = per_enemy_history.get('bvr_state', 0)
            defensive_timer = per_enemy_history.get('defensive_timer', 0)
            is_spiked = agent.get("WarningNumber") >= 2

            # 3. BVR 状态机：根据战况决定我机所处的战术状态
            if bvr_state == 0:  # 发射前
                if just_launched and R > self.bvr_wvr_switch_range_m: bvr_state = 1
            elif bvr_state == 1:  # Crank
                is_high_entry_angle = ta_deg > 150
                threat_condition = is_spiked or is_high_entry_angle
                ego_energy = agent.get_position()[2] + self.energy_vel_sq_factor * np.linalg.norm(
                    agent.get_velocity()) ** 2
                enm_energy = enemy.get_position()[2] + self.energy_vel_sq_factor * np.linalg.norm(
                    enemy.get_velocity()) ** 2
                timeline_advantage = ego_energy > enm_energy
                if not threat_condition or timeline_advantage:
                    bvr_state = 0  # 赢了，转回进攻
                else:
                    bvr_state = 2; defensive_timer = 1  # 输了，转入防御
            elif bvr_state == 2:  # 防御
                defensive_timer += 1
                min_time_passed = defensive_timer > self.min_defensive_duration_steps
                timeout_reached = defensive_timer > self.max_defensive_duration_steps
                threat_is_gone = not is_spiked
                if (threat_is_gone and min_time_passed) or timeout_reached: bvr_state = 0; defensive_timer = 0

            single_enemy_reward = 0.0
            # =================== 奖励计算核心区 (固定值 + 趋势 + 塑形) ===================
            if R > self.bvr_wvr_switch_range_m:
                # --- BVR 模式 ---
                if bvr_state == 0:  # 状态0: 发射前准备
                    # 主奖励 (R_main): 基于当前位置的好坏给予固定值
                    if ao_deg <= 20 and self.BVR_OPTIMAL_RANGE_MIN <= R <= self.BVR_OPTIMAL_RANGE_MAX:
                        R_main = self.reward_bvr_pre_launch_optimal
                    elif ao_deg <= 35 and (self.BVR_OPTIMAL_RANGE_MIN - 5000) <= R <= (
                            self.BVR_OPTIMAL_RANGE_MAX + 5000):
                        R_main = self.reward_bvr_pre_launch_acceptable
                    else:
                        R_main = self.penalty_bvr_pre_launch_bad

                    # 趋势奖励: 奖励正在改善位置的行为
                    R_dist_trend = 0
                    R_angle_trend = 0
                    prev_R = per_enemy_history.get('R', R)
                    dist_change = R - prev_R
                    # 如果在最佳射程之外，且正在接近，给予奖励
                    if R > self.BVR_OPTIMAL_RANGE_MAX and dist_change < -10:
                        R_dist_trend =  self.reward_bvr_dist_closing


                    prev_ao_deg = per_enemy_history.get('ao_deg', ao_deg)
                    ao_change = ao_deg - prev_ao_deg
                    # 如果角度在减小（机头在转向敌人），给予奖励
                    if ao_deg > 35 and ao_change < -0.5:
                        R_angle_trend = self.reward_bvr_angle_improving
                    elif ao_deg > 35 and ao_change > 0.5:
                        R_angle_trend = -self.reward_bvr_angle_improving

                    single_enemy_reward = R_main + R_dist_trend + R_angle_trend

                elif bvr_state == 1:  # 状态1: Crank
                    # 塑形奖励 (R_shaping): 奖励“正在朝正确方向转”这个动作本身
                    R_shaping = 0.0
                    prev_ao_deg = per_enemy_history.get('ao_deg', ao_deg)
                    if ao_deg > 65 and ao_deg < prev_ao_deg:
                        R_shaping = self.reward_shaping_turn_correctly
                    elif ao_deg > 65 and ao_deg > prev_ao_deg:
                        R_shaping = self.penalty_shaping_turn_wrongly
                    elif ao_deg < 35 and ao_deg < prev_ao_deg:
                        R_shaping = self.penalty_shaping_turn_wrongly
                    elif ao_deg < 35 and ao_deg > prev_ao_deg:
                        R_shaping = self.reward_shaping_turn_correctly

                    # 主奖励 (R_main): 基于当前角度的好坏给予固定值
                    if 45 <= ao_deg <= 65:
                        R_main = self.reward_bvr_crank_optimal
                    elif 35 <= ao_deg < 45:
                        R_main = self.reward_bvr_crank_acceptable
                    else:
                        R_main = self.penalty_bvr_crank_bad
                    single_enemy_reward = R_main + R_shaping

                elif bvr_state == 2:  # 状态2: Beam
                    # 塑形奖励 (R_shaping)
                    R_shaping = 0.0
                    prev_ao_deg = per_enemy_history.get('ao_deg', ao_deg)
                    if ao_deg > 105 and ao_deg < prev_ao_deg:
                        R_shaping = self.reward_shaping_turn_correctly
                    elif ao_deg > 105 and ao_deg > prev_ao_deg:
                        R_shaping = self.penalty_shaping_turn_wrongly
                    elif ao_deg < 45 and ao_deg < prev_ao_deg:
                        R_shaping = self.penalty_shaping_turn_wrongly
                    elif ao_deg < 45 and ao_deg > prev_ao_deg:
                        R_shaping = self.reward_shaping_turn_correctly

                    # 主奖励 - 角度部分
                    if 75 <= ao_deg <= 105:
                        R_angle = self.reward_bvr_beam_optimal
                    elif 45 <= ao_deg < 75 or 105 < ao_deg <= 135:
                        R_angle = self.reward_bvr_beam_acceptable
                    else:
                        R_angle = self.penalty_bvr_beam_bad

                    # 趋势奖励 - 距离部分
                    prev_R = per_enemy_history.get('R', R)
                    dist_change = R - prev_R
                    if dist_change > 10:
                        R_distance_trend = self.reward_bvr_beam_dist_increase
                    elif dist_change < -10:
                        R_distance_trend = self.penalty_bvr_beam_dist_decrease
                    else:
                        R_distance_trend = 0
                    single_enemy_reward = R_angle + R_distance_trend + R_shaping

            else:
                # --- WVR 模式 ---
                az_deg, el_deg, _ = get_az_el_R(ego_feature, enm_feature)
                in_scan_zone = (abs(az_deg) < self.ACM_HUD_AZ_DEG and abs(el_deg) < self.ACM_HUD_EL_DEG) or \
                               (abs(az_deg) < self.ACM_VSL_AZ_DEG and abs(el_deg) < self.ACM_VSL_EL_DEG)

                is_tail_chase = ta_deg <= 30
                is_good_range = 4000 <= R <= 12000
                is_too_close = R < 2000

                # 主奖励 (R_main): 基于当前格斗态势给予固定值
                if is_too_close:
                    R_main = self.penalty_wvr_overshoot
                elif is_tail_chase and is_good_range and in_scan_zone:
                    R_main = self.reward_wvr_optimal_tail_chase
                elif in_scan_zone:
                    R_main = self.reward_wvr_in_scan_zone
                else:
                    R_main = self.penalty_wvr_bad_position

                # WVR 趋势奖励
                prev_R = per_enemy_history.get('R', R)
                dist_change = R - prev_R
                # 如果没过近且在逼近，奖励
                if not is_too_close and dist_change < -10:
                    R_dist_trend =  self.reward_wvr_dist_closing
                elif dist_change > 10:
                    R_dist_trend = self.penalty_wvr_dist_opening
                else:
                    R_dist_trend = 0

                prev_ta_deg = per_enemy_history.get('ta_deg', ta_deg)
                ta_change = ta_deg - prev_ta_deg
                # 如果尾角在减小（正在绕后），奖励
                if ta_deg > 30 and ta_change < -0.5:
                    R_angle_trend = self.reward_wvr_tail_angle_closing
                elif ta_deg > 30 and ta_change > 0.5:
                    R_angle_trend = -self.reward_wvr_tail_angle_closing
                else:
                    R_angle_trend = 0

                single_enemy_reward = R_main + R_dist_trend + R_angle_trend
                bvr_state, defensive_timer = 0, 0
            # =================================================================================
            if just_launched:
                single_enemy_reward += 50
            all_enemy_tactical_info.append((single_enemy_reward, R, enemy))

            # 4. 更新历史记录，为下一帧的趋势/塑形计算做准备
            if enemy.uid not in self.previous_metrics[agent_id]['enemy_states']:
                self.previous_metrics[agent_id]['enemy_states'][enemy.uid] = {}
            self.previous_metrics[agent_id]['enemy_states'][enemy.uid].update({
                'bvr_state': bvr_state, 'defensive_timer': defensive_timer,
                'R': R, 'ao_deg': ao_deg, 'ta_deg': ta_deg  # 必须记录这些值
            })

        # 5. 注意力机制: 找出距离最近的敌人作为当前回合的焦点
        if all_enemy_tactical_info:
            closest_enemy_info = min(all_enemy_tactical_info, key=lambda item: item[1])
            # 最终战术奖励 = 针对最近敌人的奖励
            final_tactical_reward, _, focused_enemy = closest_enemy_info
        else:
            final_tactical_reward = 0.0; focused_enemy = None

        # 6. 计算通用奖励 (能量 & 高度)
        # 动态能量计算：只和焦点敌人比较能量
        R_energy = 0.0
        effective_w_energy = self.w_energy
        if focused_enemy:
            ego_energy = agent.get_position()[2] + self.energy_vel_sq_factor * np.linalg.norm(agent.get_velocity()) ** 2
            enm_energy = focused_enemy.get_position()[2] + self.energy_vel_sq_factor * np.linalg.norm(
                focused_enemy.get_velocity()) ** 2
            R_energy = 0.5 if ego_energy > enm_energy else -0.5

            # 状态化能量权重：如果焦点敌人正处于需要机动的状态，降低能量权重
            focused_enemy_history = self.previous_metrics[agent_id].get('enemy_states', {}).get(focused_enemy.uid, {})
            bvr_state = focused_enemy_history.get('bvr_state', 0)
            if bvr_state in [1, 2]:  # Crank 或 Beam 状态
                effective_w_energy = self.w_energy * self.defensive_energy_weight_multiplier

        # 高度奖励
        ego_alt = agent.get_position()[2]
        R_altitude = 0.0
        if ego_alt > self.MAX_ALTITUDE_METER:
            R_altitude = self.penalty_hard_ceiling
        elif ego_alt > self.OPTIMAL_ALT_MAX_METER:
            R_altitude = self.w_altitude_penalty * (ego_alt - self.OPTIMAL_ALT_MAX_METER) / 500.0
        elif ego_alt < self.OPTIMAL_ALT_MIN_METER:
            R_altitude = self.w_altitude_penalty * (self.OPTIMAL_ALT_MIN_METER - ego_alt) / 500.0

        # 7. 汇总最终奖励
        final_reward = final_tactical_reward + effective_w_energy * R_energy + R_altitude

        # 8. 更新和清理历史数据
        self.previous_metrics[agent_id]['agent_states']['amraam_num'] = current_amraam_num
        if 'enemy_states' in self.previous_metrics[agent_id]:
            current_enemy_ids = {enm.uid for enm in alive_enemies}
            obsolete_ids = [eid for eid in self.previous_metrics[agent_id]['enemy_states'] if
                            eid not in current_enemy_ids]
            for eid in obsolete_ids: del self.previous_metrics[agent_id]['enemy_states'][eid]

        return self._process(final_reward, agent_id)