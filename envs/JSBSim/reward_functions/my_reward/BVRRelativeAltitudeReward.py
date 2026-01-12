import numpy as np

from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction


class BVRRelativeAltitudeReward(BaseRewardFunction):
    """
    【相对高度奖励 - 能量压制版】
    目标：建立相对于敌机的势能优势，同时维持在最佳空战高度层。

    逻辑核心：
    1. 地面防撞区 (H < H_min): 严厉惩罚。
    2. 战术优势区:
       - 目标高度 Target_H = Max(基础最佳高度, 敌机高度 + 优势余量)
       - 也就是说，如果敌机低，我保持最佳高度；如果敌机高，我争取比他更高。
    3. 升限区 (H > H_max): 惩罚，防止失速或飞出大气层。
    """

    def __init__(self, config):
        super().__init__(config)
        # --- 参数配置 (单位: 米) ---
        self.h_min = getattr(self.config, 'h_min', 3500.0)  # 最低安全高度 (Hard Deck)
        self.h_best = getattr(self.config, 'h_best', 8500.0)  # 基础最佳BVR高度 (通常8km-10km空气稀薄适合发射)
        self.h_max = getattr(self.config, 'h_ceiling', 12000.0)  # 实用升限
        self.h_advantage = 2000.0  # 我们希望比敌机高多少 (2km)

        self.reward_item_names = [self.__class__.__name__ + '_total']

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]

        # 1. 获取自身高度 (假设 NED 坐标系，z 为负)
        # 务必确认仿真环境返回的是正高度还是负Z
        my_alt = -agent.get_position()[2]

        # 2. 获取敌机高度
        enemies = agent.enemies
        best_enemy = None
        min_dist = float('inf')
        ego_pos = agent.get_position()

        for enm in enemies:
            if hasattr(enm, 'is_alive') and not enm.is_alive: continue
            d = np.linalg.norm(ego_pos - enm.get_position())
            if d < min_dist:
                min_dist = d
                best_enemy = enm

        if best_enemy is None or len(agent.check_all_missile_warning()) != 0:
            return 0.0
        if min_dist > 50000:
            return 0.0
        enemy_alt = 0
        if best_enemy:
            enemy_alt = -best_enemy.get_position()[2]

        # 3. 计算“动态目标高度” (Dynamic Target Altitude)
        # 逻辑：我要比敌机高 2000米，但无论如何不能低于我的 h_best (8000米)
        # 这样防止被敌机把高度带到低空泥潭中
        target_h = max(self.h_best, enemy_alt + self.h_advantage)

        # 物理封顶
        target_h = min(target_h, self.h_max)

        reward = 0.0

        # --- 4. 分段奖励计算 ---

        # [区域 A]: 撞地危险区 (Min Alt 以下)
        if my_alt < self.h_min:
            # 严重惩罚 (-1.0 ~ -2.0)
            # 离地越近罚越重
            penalty = (self.h_min - my_alt) / self.h_min
            reward = -1.0 * (1.0 + penalty)

            # [区域 B]: 爬升/追赶区 (Min Alt ~ Target H)
        elif my_alt < target_h:
            # 线性插值：鼓励爬升
            # 在 h_min 处为 0.0
            # 在 target_h 处为 1.0
            progress = (my_alt - self.h_min) / (target_h - self.h_min + 1e-6)

            # 增加一个系数：如果我比敌机高(尽管没达到target)，给分高一点
            if my_alt < enemy_alt:
                reward = 0.4 * progress
            else:
                reward = 0.4 + 0.6 * progress

        # [区域 C]: 优势区 (Target H ~ H_max)
        elif my_alt <= self.h_max:
            # 已经达到或超过目标高度，给满分
            reward = 1.0

            # [可选微调]：如果高出太多，接近升限，可以稍微降一点分，引导不要贴着升限飞
            # 但在 BVR 中通常越高越好，只要不失速，这里保持 1.0 也可以

        # [区域 D]: 升限溢出区 (> H_max)
        else:
            # 超过升限，惩罚 (防止失速/解体)
            reward = -0.5

        return self._process(reward, agent_id)