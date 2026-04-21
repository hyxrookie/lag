import numpy as np

from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction


class BVRZoneRangeReward(BaseRewardFunction):
    """
    【分段距离奖励】
    严格按照用户设定的战术区间进行奖励塑形：
    1. > 50km: 趋近趋势奖励 (Encourage Approach)
    2. 40-50km: 搜索/准备区 (Preparation)
    3. 25-40km: 有效射程区 (Effective Range)
    4. 15-25km: 不可逃逸区 (NEZ / Sweet Spot) -> 最高分
    5. < 15km:  惩罚区 (Penalty Zone)
    """

    def __init__(self, config):
        super().__init__(config)
        # --- 距离阈值设定 (单位: 米) ---
        self.d_far = 50000.0  # 50km
        self.d_enter = 40000.0  # 40km
        self.d_nez = 25000.0  # 25km
        self.d_min = 15000.0  # 15km

        self.reward_item_names = [self.__class__.__name__ + '_total']

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]

        # 使用智能选敌逻辑（避免震荡，优先计算锁定的或最近的）
        target, dist = self._get_target_and_dist(agent)

        if target is None or len(agent.check_all_missile_warning()) != 0:
            return 0.0  # 无目标时不给分

        reward = 0.0

        # --- 分段逻辑实现 ---

        # 1. 惩罚区 (< 15km)
        if dist < self.d_min:
            # 目标：保持连续性。
            # 在 d_min (15km) 处，我们要接上 Sweet Spot 的 1.0 分。
            # 在 0 km (撞击) 处，我们给一个惩罚值，比如 -1.0。

            # 公式推导：
            # dist = 15 -> Reward = 1.0
            # dist = 0  -> Reward = -1.0
            # 这是一个线性方程。

            # 计算比例 (0.0 ~ 1.0), 15km时为0, 0km时为1
            ratio = (self.d_min - dist) / self.d_min

            # Start_value - (Total_Drop * ratio)
            # 1.0 - (2.0 * ratio)
            # 解释：从1.0开始扣，最多扣掉2.0分，变成 -1.0
            reward = 1.0 - 2.0 * ratio
        elif dist < self.d_nez:
            # 这是最理想的区域，给满分
            reward = 1.0

        # 3. 有效射程区 (25km <= D < 40km)
        elif dist < self.d_enter:
            # 线性插值：鼓励从 40km 接近到 25km
            # 距离 40km 时 -> 0.4
            # 距离 25km 时 -> 1.0
            # 公式: 0.4 + 0.6 * 进度
            progress = (self.d_enter - dist) / (self.d_enter - self.d_nez)
            reward = 0.4 + 0.6 * progress

        # 4. 准备区 (40km <= D < 50km)
        elif dist < self.d_far:
            # 给予一个小的正向奖励，维持接敌
            # 可以是常数，也可以微小增长
            progress = (self.d_far - dist) / (self.d_far - self.d_enter)
            reward = 0.1 + 0.3 * progress

        # 5. 远距离区 (> 50km)
        else:
            # 奖励“趋近趋势”
            # 我们希望 100km 处的奖励比 150km 处高，引导它飞向 50km
            # 使用指数衰减，以 50km 为基准
            # 例如: dist=60km -> exp(-(10000)/30000) ≈ 0.7 * 0.2 ≈ 0.14
            over_dist = dist - self.d_far
            scale = 30000.0  # 衰减尺度
            trend_score = np.exp(-over_dist / scale)

            # 最大给 0.2 (不能超过准备区的奖励，否则飞机就不愿进圈了)
            reward = 0.2 * trend_score

        return self._process(reward, agent_id)

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