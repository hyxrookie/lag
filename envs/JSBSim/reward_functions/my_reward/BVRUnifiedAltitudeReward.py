import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction


class BVRUnifiedAltitudeReward(BaseRewardFunction):
    """
    【统一高度管理奖励】
    整合了巡航高度维持和空战高度优势压制的逻辑。

    逻辑分层：
    1. 安全层 (Safety Layer): 无论何时，低于最低安全高度(Hard Deck)直接给予重罚。
    2. 战术层 (Tactical Layer): 根据距离权重动态调整目标。
       - 远距离 (>60km): 巡航逻辑 (维持在最佳省油高度层)
       - 近距离 (<40km): 空战逻辑 (高于敌机，或维持基础BVR高度)
       - 过渡区 (40-60km): 线性融合
    """

    def __init__(self, config):
        super().__init__(config)

        # --- 1. 高度参数 (单位: 米) ---
        # 全局安全底线 (Hard Deck)
        self.h_min_safety = 3500.0  # 绝对不能低于这个高度，否则面临撞地风险/低空劣势
        self.h_ceiling = 13000.0  # 升限

        # 巡航阶段参数
        self.h_cruise_best = 8000.0  # 最佳巡航高度 (通常高空省油)
        self.h_cruise_tol = 1000.0  # 巡航高度的宽容度 (Sigma)

        # 空战阶段参数
        self.h_combat_base = 9000.0  # 空战基础高度 (太低了导弹射程不够)
        self.h_advantage = 2000.0  # 期望比敌机高多少

        # --- 2. 距离过渡参数 ---
        self.d_cruise_start = 60000.0  # > 60km: 100% 巡航
        self.d_combat_start = 50000.0  # < 40km: 100% 空战

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]

        # 1. 获取自身高度 (假设 NED 坐标系 Z 轴向下，取负转为正高度)
        # 请根据实际仿真环境确认 z 是否需要取负
        pos = agent.get_position()
        my_alt = pos[2]

        # --- 第一层：安全底线 (Safety Override) ---
        # 如果低于安全高度，无论远近，直接惩罚。
        # 这种设计防止了“为了战术目标而忽略撞地风险”。
        if my_alt < self.h_min_safety:
            # 线性惩罚: 高度越低，惩罚越重 (-1.0 ~ -2.0)
            penalty_factor = (self.h_min_safety - my_alt) / self.h_min_safety
            reward = -1.0 * (1.0 + penalty_factor)
            return self._process(reward, agent_id)

        # 如果高于升限，同样惩罚
        if my_alt > self.h_ceiling:
            reward = -0.5
            return self._process(reward, agent_id)

        # --- 第二层：计算战术权重 ---
        target_enm, dist = self._get_target_and_dist(agent)

        # 距离权重计算
        if target_enm is None:
            # 无目标时，默认巡航
            w_cruise = 1.0
            enemy_alt = 0.0
        else:
            enemy_alt = target_enm.get_position()[2]
            if dist >= self.d_cruise_start:
                w_cruise = 1.0
            elif dist <= self.d_combat_start:
                w_cruise = 0.0
            else:
                w_cruise = (dist - self.d_combat_start) / (self.d_cruise_start - self.d_combat_start)

        w_combat = 1.0 - w_cruise

        # --- 第三层：计算子奖励 ---

        # A. 巡航子奖励 (Cruise Score)
        # 目标: 维持在 h_cruise_best 附近
        # 使用高斯函数，偏离越多分越低
        r_cruise = np.exp(-((my_alt - self.h_cruise_best) ** 2) / (2 * self.h_cruise_tol ** 2))

        # B. 空战子奖励 (Combat Score)
        # 目标: 动态高度 = max(基础高度, 敌机高度 + 优势)
        target_h_combat = max(self.h_combat_base, enemy_alt + self.h_advantage)
        target_h_combat = min(target_h_combat, self.h_ceiling)  # 物理封顶

        r_combat = 0.0
        if my_alt >= target_h_combat:
            # 达到或超过目标高度 -> 满分
            r_combat = 1.0
        else:
            # 还在爬升/追赶中
            # 计算进度: (当前 - 底线) / (目标 - 底线)
            # 因为前面已经过滤了 < h_min_safety 的情况，这里 my_alt 一定 >= h_min_safety
            denominator = target_h_combat - self.h_min_safety + 1e-6
            progress = (my_alt - self.h_min_safety) / denominator

            # 基础分 0.3，越高分越多
            r_combat = 0.3 + 0.7 * progress

            # (可选) 额外激励：虽然没到目标，但只要比敌机高，就多给点分
            if my_alt > enemy_alt:
                r_combat += 0.1
            r_combat = min(1.0, r_combat)

        # --- 第四层：融合 ---
        total_reward = w_cruise * r_cruise + w_combat * r_combat
        return self._process(total_reward, agent_id)

    def _get_target_and_dist(self, agent):
        """ 复用选敌逻辑 """
        enemies = agent.enemies
        if not enemies: return None, float('inf')
        ego_pos = agent.get_position()

        # 过滤死亡单位
        valid_enemies = [e for e in enemies if getattr(e, 'is_alive', True)]
        if not valid_enemies: return None, float('inf')

        best_target = min(valid_enemies, key=lambda e: np.linalg.norm(ego_pos - e.get_position()))
        dist = np.linalg.norm(ego_pos - best_target.get_position())
        return best_target, dist