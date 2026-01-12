import numpy as np

from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction


class BVRCruiseStateReward(BaseRewardFunction):
    """
    【巡航状态奖励】 (> 50km)
    目标：引导飞机在远距离接敌时，维持最佳巡航状态，节省燃油并保持战术灵活性。

    逻辑：
    - 速度：奖励维持在高亚音速 (0.6 ~ 0.9 Mach)。惩罚超音速(耗油)和低速(失速风险)。
    - 高度：奖励维持在最佳巡航高度 (8000m ~ 11000m)。
    """

    def __init__(self, config):
        super().__init__(config)
        # 速度参数 (单位: m/s, 假设音速 ~340m/s)
        self.v_cruise_min = 0.6 * 340  # ~200 m/s
        self.v_cruise_best = 0.8 * 340  # ~270 m/s (最佳点)
        self.v_cruise_max = 0.95 * 340  # ~320 m/s (不建议开加力)

        # 高度参数 (单位: 米)
        self.h_cruise_min = 6000.0
        self.h_cruise_best = 7000.0  # 最佳巡航高度
        self.h_cruise_max = 10000.0

        self.reward_item_names = [self.__class__.__name__ + '_total']

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]

        # 获取状态
        vel = agent.get_velocity()
        speed = np.linalg.norm(vel)
        alt = -agent.get_position()[2]  # 假设 NED
        target, dist = self._get_target_and_dist(agent)
        if dist < 50000:
            return 0
        reward = 0.0

        # --- 1. 巡航速度奖励 ---
        r_spd = 0.0
        # A. 最佳区间 (0.6 ~ 0.95 Mach)
        if self.v_cruise_min <= speed <= self.v_cruise_max:
            # 使用钟形函数(高斯)，在 v_cruise_best 处达到 1.0
            # 允许一定的宽容度
            sigma = 30.0
            r_spd = np.exp(-((speed - self.v_cruise_best) ** 2) / (2 * sigma ** 2))
            # 修正：保证区间内最低也有 0.5，不要掉得太快
            r_spd = 0.5 + 0.5 * r_spd

        # B. 超速 (耗油区)
        elif speed > self.v_cruise_max:
            # 给予轻微惩罚或零分，引导它减速
            # 不要罚太重，万一它需要赶路呢？给 0 分即可，甚至 -0.1
            r_spd = -0.2 * (speed - self.v_cruise_max) / 100.0
            r_spd = max(r_spd, -0.5)

        # C. 低速 (危险区)
        else:  # speed < self.v_cruise_min
            # 严重惩罚，防止失速或飞得太慢
            r_spd = -1.0 * (1.0 - speed / self.v_cruise_min)

        # --- 2. 巡航高度奖励 ---
        r_alt = 0.0
        # A. 最佳区间 (6000 ~ 11000m)
        if self.h_cruise_min <= alt <= self.h_cruise_max:
            # 同样使用高斯函数引导至 9000m
            sigma = 1500.0
            r_alt = np.exp(-((alt - self.h_cruise_best) ** 2) / (2 * sigma ** 2))
            r_alt = 0.5 + 0.5 * r_alt

        # B. 太高 (易被发现/没必要)
        elif alt > self.h_cruise_max:
            # 线性衰减
            r_alt = 0.0

        # C. 太低 (地形风险/空气稠密耗油)
        else:
            # 惩罚
            r_alt = -0.5 * (self.h_cruise_min - alt) / self.h_cruise_min
            r_alt = max(r_alt, -1.0)

        # 综合评分 (各占一半)
        total = 0.5 * r_spd + 0.5 * r_alt
        return self._process(total, agent_id)
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