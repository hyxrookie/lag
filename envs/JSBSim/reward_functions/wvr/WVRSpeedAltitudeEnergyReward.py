import numpy as np

from envs.JSBSim.core.catalog import Catalog
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction


class WVRSpeedAltitudeEnergyReward(BaseRewardFunction):
    """
    【近距速度-高度-能量奖励】

    目标：
    - 不再鼓励 BVR 式高速高空；
    - 鼓励保持适合格斗的速度区间；
    - 防止低速失速和无意义爬升；
    - 高度主要作为安全约束，而不是持续爬升目标。
    """

    def __init__(self, config):
        super().__init__(config)

        # 近距格斗速度区间
        self.mach_wvr_best = 0.85
        self.mach_min_safe = 0.45
        self.mach_max_good = 1.1
        self.sigma_speed = 0.25

        # 高度安全约束
        self.h_min_safe = 3500.0
        self.h_prefer = 7000.0
        self.h_high_penalty = 9000.0
        self.sigma_height = 2500.0

        # 权重
        self.w_speed = 1.0
        self.w_height = 0.5
        self.w_energy_penalty = 0.25

        self.reward_item_names = [self.__class__.__name__ + '_total']

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]

        mach = agent.get_property_value(Catalog.velocities_mach)
        pos = agent.get_position()
        alt = pos[2]
        v_up = agent.get_velocity()[2]

        # 被导弹威胁时，由规避奖励主导
        if len(agent.check_all_missile_warning()) != 0:
            return 0.0

        target, dist = self._get_target_and_dist(agent)

        # -----------------------------
        # 1. 速度奖励
        # -----------------------------
        if mach < self.mach_min_safe:
            # 低速强惩罚，防止失速盘旋
            r_speed = -1.5 * (self.mach_min_safe - mach) / 0.1
        elif mach > self.mach_max_good:
            # 过高速轻惩罚，避免一直高速冲过
            r_speed = np.exp(-((mach - self.mach_wvr_best) ** 2) / (2 * self.sigma_speed ** 2))
            r_speed -= 0.5 * (mach - self.mach_max_good)
        else:
            # 最优格斗速度附近奖励最高
            r_speed = np.exp(-((mach - self.mach_wvr_best) ** 2) / (2 * self.sigma_speed ** 2))

        # -----------------------------
        # 2. 高度奖励
        # -----------------------------
        if alt < self.h_min_safe:
            r_height = -2.0 * (self.h_min_safe - alt) / 1000.0 - 0.5
        else:
            # 近距只需要安全高度，不强制越高越好
            r_height = np.exp(-((alt - self.h_prefer) ** 2) / (2 * self.sigma_height ** 2))

            # 过高不适合近距格斗，轻微惩罚
            if alt > self.h_high_penalty:
                r_height -= 0.3 * (alt - self.h_high_penalty) / 3000.0

        # -----------------------------
        # 3. 能量惩罚
        # -----------------------------
        r_energy_penalty = 0.0

        # 低速还强行爬升，容易失速
        if mach < 0.6 and v_up > 5.0:
            r_energy_penalty -= self.w_energy_penalty * v_up * (0.6 - mach)

        # 近距格斗中高速大幅俯冲，容易冲过目标
        if mach > 0.9 and v_up < -20.0:
            r_energy_penalty -= self.w_energy_penalty * abs(v_up) * 0.05

        # -----------------------------
        # 4. 距离门控
        # -----------------------------
        # 进入近距后，速度控制更重要
        if target is not None and dist < 15000.0:
            w_speed_dynamic = self.w_speed
        else:
            w_speed_dynamic = 0.6 * self.w_speed

        total_reward = (
            w_speed_dynamic * r_speed
            + self.w_height * r_height
            + r_energy_penalty
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