import logging
import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.utils.utils import get_AO_TA_R


class NewMissileDodgeContinuousReward(BaseRewardFunction):
    """
    奖励智能体有效规避来袭导弹的行为（连续值版本）。
    - 该版本将离散奖惩转化为连续的奖励信号，以更好地引导学习过程。
    - 综合评估三个核心规避战术，并对最具威胁的导弹进行奖励：
      1. 最佳规避姿态 (Optimal Evasion Angle):
         - 使用函数的特性，同时奖励“横向机动 (Beaming)”和“同向逃逸 (Fleeing)”。
         - 当我机与导弹速度夹角接近 90度 或 0度 时，都会获得高分。
      2. 降低接近率 (Closing Speed Reduction):
         - 平滑地奖励任何能够降低导弹与自身接近速度的机动。
      3. 成功规避 (Successful Dodge):
         - 在导弹生命周期结束后，给予一个较大的一次性终局奖励。
    - 最终奖励基于对“最具威胁”导弹的综合规避分数。
    """

    def __init__(self, config):
        super().__init__(config)
        # --- 状态追踪 ---
        self.prev_missile_states = {}

        # --- 奖励参数 ---
        self.success_dodge_reward = getattr(self.config, 'success_dodge_reward', 100.0)

        # --- 战术权重 ---
        self.w_angle = getattr(self.config, 'w_angle', 0.5)  # 姿态控制权重
        self.w_closing_speed = getattr(self.config, 'w_closing_speed', 0.5)  # 降低接近率权重

        # --- 运动学参数 ---
        self.closing_speed_change_ref = getattr(self.config, 'closing_speed_change_ref', 50.0)
        self.threat_range_decay = getattr(self.config, 'threat_range_decay', 0.0001)

        # 最终奖励的放大系数
        self.reward_scale = getattr(self.config, 'dodge_reward_scale', 20.0)

    def reset(self, task, env):
        self.prev_missile_states.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]
        if not agent.is_alive:
            return 0.0

        total_reward = 0.0

        # ========================= 代码修改区域开始 =========================

        # [删除] 不再需要追踪单个最大威胁和最佳分数
        # max_threat_level = -1.0
        # best_evasion_score = 0.0

        # [新增] 初始化用于加权平均的累加器
        sum_of_weighted_scores = 0.0  # 分子: sum(threat_i * score_i)
        sum_of_threats = 0.0  # 分母: sum(threat_i)

        # ========================= 代码修改区域结束 =========================

        missile_sims = agent.check_all_missile_warning()

        # 检查并处理成功规避的导弹 (这部分逻辑不变)
        prev_uids = list(self.prev_missile_states.keys())
        current_uids = {m.uid for m in missile_sims if m.is_alive}
        for uid in prev_uids:
            if uid not in current_uids:
                total_reward += self.success_dodge_reward
                del self.prev_missile_states[uid]

        # 遍历当前所有来袭导弹
        for sim in missile_sims:
            if not sim.is_alive:
                continue

            # --- 1. 计算当前状态 (这部分逻辑不变) ---
            agent_pos = agent.get_position()
            agent_vel = agent.get_velocity()
            missile_pos = sim.get_position()
            missile_vel = sim.get_velocity()

            distance = np.linalg.norm(agent_pos - missile_pos)
            if distance < 1e-6: continue

            los_vector = (agent_pos - missile_pos) / distance
            closing_speed = np.dot(missile_vel, los_vector)

            # --- 2. 评估威胁等级 (这部分逻辑不变) ---
            threat_level = max(0, closing_speed) * math.exp(-self.threat_range_decay * distance)

            current_evasion_score = 0.0  # [修改] 初始化当前分数，以处理首次出现的情况
            if sim.uid in self.prev_missile_states:
                # --- 3. 计算规避分数 (Evasion Score) [0, 1] (这部分逻辑不变) ---

                # a. 最佳规避姿态分数 (Angle Score)
                dot_product = np.dot(agent_vel, missile_vel)
                norm_product = np.linalg.norm(agent_vel) * np.linalg.norm(missile_vel)
                cos_angle = np.clip(dot_product / (norm_product + 1e-6), -1.0, 1.0)

                beaming_score = 1 - cos_angle ** 2
                fleeing_score = max(0, cos_angle)
                angle_score = max(beaming_score, fleeing_score)

                # b. 降低接近率分数 (Closing Speed Reduction Score)
                prev_closing_speed = self.prev_missile_states[sim.uid]['closing_speed']
                delta_closing_speed = closing_speed - prev_closing_speed
                closing_speed_score = (math.tanh(-delta_closing_speed / self.closing_speed_change_ref) + 1.0) / 2.0

                # c. 综合规避分数
                current_evasion_score = (self.w_angle * angle_score +
                                         self.w_closing_speed * closing_speed_score)

            # ========================= 代码修改区域开始 =========================

            # [修改] 将原来的 "if-else" 更新最大威胁的逻辑，改为累加
            # 只有当存在实际威胁时（threat_level > 0），才将其计入加权平均
            if threat_level > 0:
                sum_of_weighted_scores += threat_level * current_evasion_score
                sum_of_threats += threat_level

            # [删除] 原始的更新最大威胁逻辑
            # if threat_level > max_threat_level:
            #     max_threat_level = threat_level
            #     best_evasion_score = current_evasion_score

            # ========================= 代码修改区域结束 =========================

            # --- 4. 更新上一时刻状态 (这部分逻辑不变) ---
            self.prev_missile_states[sim.uid] = {'closing_speed': closing_speed}

        # ========================= 代码修改区域开始 =========================

        # [新增] 计算最终的加权平均规避分数
        final_evasion_score = 0.0
        # 为防止除零错误，仅在总威胁大于零时进行计算
        if sum_of_threats > 1e-6:
            final_evasion_score = sum_of_weighted_scores / sum_of_threats

        # [修改] 使用新的加权平均分数来计算奖励
        total_reward += self.reward_scale * final_evasion_score

        # [删除] 原始的基于 best_evasion_score 的奖励计算
        # total_reward += self.reward_scale * best_evasion_score

        # ========================= 代码修改区域结束 =========================

        return self._process(total_reward, agent_id)