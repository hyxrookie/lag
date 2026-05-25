import numpy as np

from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction


class WVRZoneRangeReward(BaseRewardFunction):
    """
    【近距距离区间奖励】

    目标：
    - 鼓励进入近距格斗范围；
    - 最优距离不是 BVR 的 15~25km，而是 WVR 的 1~5km 左右；
    - 避免过近导致冲过头、相撞或失去攻击窗口。
    """

    def __init__(self, config):
        super().__init__(config)

        # 单位：米
        self.d_far = 15000.0       # 15 km，近距弹最大准备范围
        self.d_enter = 8000.0      # 8 km，进入较有效近距弹区
        self.d_good = 3000.0       # 3 km，优良攻击区
        self.d_min = 800.0         # 800 m，过近风险开始明显
        self.d_collision = 300.0   # 300 m，强惩罚

        self.reward_item_names = [self.__class__.__name__ + '_total']

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]
        target, dist = self._get_target_and_dist(agent)

        if target is None:
            return 0.0

        if len(agent.check_all_missile_warning()) != 0:
            return 0.0

        if dist is None:
            return 0.0

        # -----------------------------
        # 1. 极近危险区：< 300m
        # -----------------------------
        if dist < self.d_collision:
            reward = -1.0

        # -----------------------------
        # 2. 过近区：300m ~ 800m
        # -----------------------------
        elif dist < self.d_min:
            # 从 -0.5 平滑增长到 0.6
            progress = (dist - self.d_collision) / (self.d_min - self.d_collision)
            reward = -0.5 + 1.1 * progress

        # -----------------------------
        # 3. 最优近距攻击区：800m ~ 3km
        # -----------------------------
        elif dist < self.d_good:
            # 最高奖励区，适合尾追、机炮、近距弹
            reward = 1.0

        # -----------------------------
        # 4. 近距弹有效区：3km ~ 8km
        # -----------------------------
        elif dist < self.d_enter:
            # 3km 处 1.0，8km 处 0.6
            progress = (self.d_enter - dist) / (self.d_enter - self.d_good)
            reward = 0.6 + 0.4 * progress

        # -----------------------------
        # 5. 近距准备区：8km ~ 15km
        # -----------------------------
        elif dist < self.d_far:
            # 15km 处 0.2，8km 处 0.6
            progress = (self.d_far - dist) / (self.d_far - self.d_enter)
            reward = 0.2 + 0.4 * progress

        # -----------------------------
        # 6. 远距离：> 15km
        # -----------------------------
        else:
            # 鼓励接近，但奖励不能太高，否则不愿意继续进入格斗区
            over_dist = dist - self.d_far
            reward = 0.2 * np.exp(-over_dist / 10000.0)

        return self._process(reward, agent_id)

    def _get_target_and_dist(self, agent):
        enemies = agent.enemies
        if not enemies:
            return None, None

        ego_pos = agent.get_position()
        alive = [e for e in enemies if getattr(e, 'is_alive', True)]
        if not alive:
            return None, None

        target = min(alive, key=lambda e: np.linalg.norm(ego_pos - e.get_position()))
        dist = np.linalg.norm(ego_pos - target.get_position())
        return target, dist