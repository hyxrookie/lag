import numpy as np

from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
from envs.JSBSim.utils.utils import get_AO_TA_R


class BVRAttackGeometryReward(BaseRewardFunction):
    """
    【攻击几何模块 - 进阶版】
    目标：优化 BVR 攻击阵位。不仅要求机头指向敌机 (AO)，还要求敌机处于“迎头”态势 (TA)，
    以获得最大的导弹射程和命中率。
    """

    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__ + '_total']
        self.max_attack_angle = np.radians(35)
        self.range = 60000

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]
        enemy, dist = self._get_target_and_dist(agent)
        if enemy is None or len(agent.check_all_missile_warning()) != 0:
            return 0.0

        # 获取状态
        ego_feature = np.hstack([agent.get_position(), agent.get_velocity()])
        enm_feature = np.hstack([enemy.get_position(), enemy.get_velocity()])

        # AO: Antenna Train Angle (我方机头 vs 视线) [0, PI]
        # TA: Target Aspect (敌机机头 vs 视线) [0, PI]。通常定义：0=尾追(敌机背对我)，PI=迎头(敌机冲向我)
        AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)

        # --- 1. 指向奖励 (Pointing Quality) ---
        # 越接近 0 越好
        # 使用高斯分布核心，只有在 AO 很小时才给高分，宽角度时衰减快
        r_pointing = 0.6 * np.exp(-2.0 * AO ** 2)
        if abs(AO) < self.max_attack_angle:
            r_pointing = 0.4 + r_pointing
        # --- 2. 态势奖励 (Aspect Quality) ---
        # 逻辑：BVR 中，迎头 (Head-On, TA -> PI) 优于 尾追 (Tail-Chase, TA -> 0)
        # 迎头意味着高闭合速度 (High Closing Speed)，导弹射程由于敌机迎面飞来而极大增加。
        # 尾追意味着敌机在逃，极大压缩射程。

        # 将 TA 映射到 [0, 1]，其中 迎头=1，尾追=0.2 (不完全为0，因为指准了也有价值)
        # Cosine 映射: TA=PI -> cos=-1 -> (1 - (-1))/2 = 1
        #              TA=0  -> cos=1  -> (1 - 1)/2 = 0
        r_aspect = 0.5 + 0.5 * ((1.0 - np.cos(TA)) / 2.0)

        # --- 3. 融合奖励 ---
        # 只有在“指准了”的前提下，态势才有意义，因此采用乘法而非加法
        total_reward = r_pointing * r_aspect
        return self._process(total_reward, agent_id)
    def _get_target_and_dist(self, agent):
        """获取目标和距离的辅助函数"""
        enemies = agent.enemies
        if not enemies: return None, None

        # 简单逻辑：取最近的存活单位
        # (进阶逻辑：应结合你之前定义的“锁定目标”逻辑)
        min_dist = float('inf')
        best_target = None
        ego_pos = agent.get_position()

        for enm in enemies:
            if hasattr(enm, 'is_alive') and not enm.is_alive: continue
            d = np.linalg.norm(ego_pos - enm.get_position())
            if d < min_dist:
                min_dist = d
                best_target = enm

        return best_target, min_dist