import numpy as np
from envs.JSBSim.core.catalog import Catalog
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction


class BVRSpeedAltitudeEnergyReward_312(BaseRewardFunction):
    """
    【BVR 速度–高度能量感知奖励函数】

    设计原则：
    1. 速度是硬约束（生存与能量基础）
    2. 高度是战术资源（但必须有速度支撑）
    3. 用“速度门控 + 爬升惩罚”隐式约束能量守恒
    """

    def __init__(self, config):
        super().__init__(config)

        # ------------------------
        # 1. 速度参数（Mach）
        # ------------------------
        self.mach_cruise = 0.8
        self.mach_combat = 1.05

        self.mach_min_safe = 0.75          # 绝对不希望低于
        self.mach_gate_width = 0.12         # 速度门控平滑区

        # ------------------------
        # 2. 高度参数（m）
        # ------------------------
        self.h_min_safe = 3500.0
        self.h_cruise_best = 6000.0
        self.h_combat_base = 7000.0
        self.h_advantage = 1000.0

        # ------------------------
        # 3. 奖励权重（已调好量级）
        # ------------------------
        self.w_speed = 1.0
        self.w_height = 1.0
        self.w_energy = 0.3
        self.sigma_height = 2000.0  # <--- 就是这行，定义高度奖励的平滑范围
        self.sigma_speed = 0.2  # 速度奖励也建议用这个逻辑


    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]

        # ------------------------
        # 读取状态
        # ------------------------
        mach = agent.get_property_value(Catalog.velocities_mach)
        pos = agent.get_position()
        alt = pos[2]  # 假设 index 2 是高度 (Up)

        # 垂直速度 (根据用户注释：v_up > 0 表示上升)
        v_up = agent.get_velocity()[2]

        target, dist = self._get_target_and_dist(agent)

        # 被导弹锁定或警告时，生存逻辑由其他函数处理或返回中性值
        if len(agent.check_all_missile_warning()) != 0:
            return 0.0

        # ------------------------
        # 1. 动态目标设定
        # ------------------------
        if target is None or dist > 50000.0:
            target_mach = self.mach_cruise
            target_alt = self.h_cruise_best
        else:
            target_mach = self.mach_combat
            enemy_alt = target.get_position()[2]
            target_alt = max(self.h_combat_base, enemy_alt + self.h_advantage)

        # ------------------------
        # 2. 角度 1：高斯高度奖励 (Avoid Ceiling Effect)
        # ------------------------
        if alt <= self.h_min_safe:
            # 极低高度给予固定重罚，防止坠地
            r_height_base = -1.5 * (self.h_min_safe - alt) / 1000.0 - 0.5
        else:
            # 使用高斯分布：在 target_alt 处获得 1.0，偏离则平滑下降
            r_height_base = np.exp(- ((alt - target_alt) ** 2) / (2 * self.sigma_height ** 2))

        # ------------------------
        # 3. 速度奖励与高度门控 (Angle 3: Survival First)
        # ------------------------
        # 高度门控：如果高度接近危险线，速度奖励的效果大幅减弱，迫使智能体关注高度
        h_gate = np.clip((alt - self.h_min_safe) / 2000.0, 0.0, 1.0)

        if mach < self.mach_min_safe:
            r_speed_base = -1.0 * (self.mach_min_safe - mach) / 0.1
        else:
            # 同样使用高斯分布，不追求无限超速
            r_speed_base = np.exp(- ((mach - target_mach) ** 2) / (2 * self.sigma_speed ** 2))

        r_speed = r_speed_base * h_gate

        # ------------------------
        # 4. 角度 2：能量平衡惩罚 (Bidirectional Energy Penalty)
        # ------------------------
        r_energy = 0.0
        # 情况 A：低速强爬 (消耗动能换势能，但动能已不足)
        if v_up > 5.0 and mach < target_mach:
            r_energy -= self.w_energy * v_up * (target_mach - mach)

        # 情况 B：高速下钻 (卖掉势能换取不必要的动能)
        # 如果垂直下降速度超过 10m/s 且速度已经超过巡航速度，则惩罚
        if v_up < -10.0 and mach > (target_mach - 0.1):
            r_energy -= self.w_energy * abs(v_up) * 0.1

        # ------------------------
        # 5. 总奖励组合
        # ------------------------
        # 给高度奖励也加一个速度门控：如果完全没速度，维持高度也没有意义（可能失速）
        v_gate = np.clip((mach - 0.4) / 0.4, 0.2, 1.0)

        total_reward = (
                self.w_speed * r_speed
                + self.w_height * r_height_base * v_gate
                + r_energy
        )
        return self._process(total_reward, agent_id)

    def _get_target_and_dist(self, agent):
        enemies = agent.enemies
        if not enemies:
            return None, float('inf')

        ego_pos = agent.get_position()
        alive = [e for e in enemies if getattr(e, 'is_alive', True)]
        if not alive:
            return None, float('inf')

        target = min(alive, key=lambda e: np.linalg.norm(ego_pos - e.get_position()))
        dist = np.linalg.norm(ego_pos - target.get_position())
        return target, dist
