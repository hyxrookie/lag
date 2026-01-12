import numpy as np
from envs.JSBSim.core.catalog import Catalog
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction


class BVRSpeedAltitudeEnergyReward(BaseRewardFunction):
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
        self.mach_cruise = 0.9
        self.mach_combat = 1.2

        self.mach_min_safe = 0.75          # 绝对不希望低于
        self.mach_gate_width = 0.12         # 速度门控平滑区

        # ------------------------
        # 2. 高度参数（m）
        # ------------------------
        self.h_min_safe = 3500.0
        self.h_cruise_best = 7000.0
        self.h_combat_base = 9000.0
        self.h_advantage = 2000.0

        # ------------------------
        # 3. 奖励权重（已调好量级）
        # ------------------------
        self.w_speed = 1.0
        self.w_height = 1.0
        self.w_energy = 0.3


    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]

        # ------------------------
        # 读取状态
        # ------------------------
        mach = agent.get_property_value(Catalog.velocities_mach)
        pos = agent.get_position()
        alt = pos[2]

        # 垂直速度（NEU系，v_down > 0 表示上升）
        v_down = agent.get_velocity()[2]  # m/s
        target, dist = self._get_target_and_dist(agent)
        if len(agent.check_all_missile_warning()) != 0:
            return 0
        # ------------------------
        # 1. 阶段判定
        # ------------------------
        if target is None or dist > 50000.0:
            target_mach = self.mach_cruise
            target_alt = self.h_cruise_best
        else:
            target_mach = self.mach_combat
            enemy_alt = target.get_position()[2]
            target_alt = max(self.h_combat_base, enemy_alt + self.h_advantage)

        # ------------------------
        # 2. 速度奖励（最高优先级）
        # ------------------------
        if mach < self.mach_min_safe:
            # 接近失速：强惩罚
            r_speed = -2.0 * (self.mach_min_safe - mach)
        else:
            speed_err = mach - target_mach
            # 平滑速度奖励（不过度追求超速）
            r_speed = np.exp(- (speed_err ** 2) / (2 * 0.15 ** 2))

        # ------------------------
        # 3. 速度门控（Energy Gate）
        # ------------------------
        # Mach < target_mach - gate_width -> gate ≈ 0
        # Mach >= target_mach -> gate = 1
        v_gate = (mach - (target_mach - self.mach_gate_width)) / self.mach_gate_width
        v_gate = np.clip(v_gate, 0.05, 1.0)  # 保留最小梯度

        # ------------------------
        # 4. 高度奖励（受速度门控）
        # ------------------------
        if alt <= self.h_min_safe:
            r_height_base = -1.0
        elif alt >= target_alt:
            r_height_base = 1.0
        else:
            r_height_base = (alt - self.h_min_safe) / (target_alt - self.h_min_safe)

        r_height = v_gate * r_height_base

        # ------------------------
        # 5. 能量交换惩罚（低速强爬）
        # ------------------------
        # 上升 + 速度不足 -> 认为是在掏动能
        if v_down > 3.0 and mach < target_mach:
            r_energy = - self.w_energy * (v_down) * (target_mach - mach)
        else:
            r_energy = 0.0

        # ------------------------
        # 6. 总奖励
        # ------------------------
        total_reward = (
            self.w_speed * r_speed
            + self.w_height * r_height
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
