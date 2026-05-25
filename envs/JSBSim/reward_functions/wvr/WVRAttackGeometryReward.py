import numpy as np

from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
from envs.JSBSim.utils.utils import get_AO_TA_R


class WVRAttackGeometryReward(BaseRewardFunction):
    """
    【F-16 + AIM-9L 全向近距攻击几何奖励】

    适配 AIM-9L 近距格斗：
    1. AO -> 0：我方机头指向目标；
    2. TA -> 0：尾追攻击，最优；
    3. TA -> pi：迎头攻击，可发射，次优；
    4. TA -> pi/2：侧向穿越，较差；
    5. LOS rate 小：降低 PN 导弹过载需求；
    6. 距离处于 AIM-9L 近距窗口。
    """

    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__ + '_total']

        # AO 指向角
        self.good_attack_angle = np.radians(20.0)
        self.max_attack_angle = np.radians(45.0)

        # TA 双峰宽度
        self.tail_width = np.radians(45.0)
        self.headon_width = np.radians(55.0)

        # 迎头奖励系数：低于尾追，但不能太低
        self.headon_weight = 0.75

        # AIM-9L 攻击距离窗口
        self.r_min = 700.0
        self.r_best_min = 1500.0
        self.r_best_max = 5000.0
        self.r_max = 9000.0

        # LOS rate 参考尺度
        self.los_rate_ref = 0.08

        # 权重
        self.w_ao = 0.35
        self.w_ta = 0.30
        self.w_los = 0.25
        self.w_range = 0.10

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]

        if len(agent.check_all_missile_warning()) != 0:
            return 0.0

        target, dist = self._get_target_and_dist(agent)
        if target is None or dist is None:
            return 0.0

        ego_pos = agent.get_position()
        ego_vel = agent.get_velocity()
        enm_pos = target.get_position()
        enm_vel = target.get_velocity()

        ego_feature = np.hstack([ego_pos, ego_vel])
        enm_feature = np.hstack([enm_pos, enm_vel])

        AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)

        # =========================================================
        # 1. AO 指向奖励：AO 越小越好
        # =========================================================
        r_ao = np.exp(-3.0 * AO ** 2)

        if AO < self.good_attack_angle:
            r_ao = 0.5 + 0.5 * r_ao
        elif AO < self.max_attack_angle:
            r_ao = 0.2 + 0.8 * r_ao

        # =========================================================
        # 2. TA 全向攻击奖励：尾追最高，迎头次高，侧向较低
        # =========================================================
        # TA = 0：尾追
        tail_score = np.exp(- (TA / self.tail_width) ** 2)

        # TA = pi：迎头
        headon_score = np.exp(- ((np.pi - TA) / self.headon_width) ** 2)

        # 双峰奖励：尾追 1.0，迎头 0.75
        r_ta = max(tail_score, self.headon_weight * headon_score)

        # =========================================================
        # 3. LOS rate 奖励：发射窗口稳定
        # =========================================================
        los_rate = self._compute_los_rate_3d(ego_pos, ego_vel, enm_pos, enm_vel)
        r_los = np.exp(- (los_rate / self.los_rate_ref) ** 2)

        # =========================================================
        # 4. 距离窗口奖励
        # =========================================================
        r_range = self._range_reward(dist)

        # =========================================================
        # 5. 迎头过近高速闭合惩罚
        # =========================================================
        rel_pos = enm_pos - ego_pos
        rel_dist = np.linalg.norm(rel_pos) + 1e-6
        los_unit = rel_pos / rel_dist

        rel_vel = enm_vel - ego_vel

        # closing > 0 表示双方距离正在缩短
        closing = -np.dot(rel_vel, los_unit)

        r_close_penalty = 0.0

        # 迎头且过近时，防止正面硬冲
        if TA > np.radians(140.0) and dist < 2500.0 and closing > 250.0:
            r_close_penalty -= np.clip((closing - 250.0) / 300.0, 0.0, 1.0)

        # 尾追但过近且闭合太快，防止冲过头
        if TA < np.radians(50.0) and dist < 1200.0 and closing > 180.0:
            r_close_penalty -= np.clip((closing - 180.0) / 200.0, 0.0, 1.0)

        # =========================================================
        # 6. 融合
        # =========================================================
        quality = (
            self.w_ao * r_ao
            + self.w_ta * r_ta
            + self.w_los * r_los
            + self.w_range * r_range
        )

        # AO 是攻击前提，用 AO 门控整体质量
        total_reward = r_ao * quality + 0.3 * r_close_penalty

        return self._process(total_reward, agent_id)

    def _compute_los_rate_3d(self, ego_pos, ego_vel, target_pos, target_vel):
        """
        三维 LOS rate：
        LOS_rate = |r x v_rel| / |r|^2

        对你的 PN 制导 AIM-9L 来说，LOS rate 越小，导弹需要的制导过载越小。
        """
        r_vec = target_pos - ego_pos
        v_rel = target_vel - ego_vel

        r_norm = np.linalg.norm(r_vec) + 1e-6
        los_rate = np.linalg.norm(np.cross(r_vec, v_rel)) / (r_norm ** 2)

        return los_rate

    def _range_reward(self, dist):
        """
        AIM-9L 近距攻击距离奖励。
        """
        if dist < self.r_min:
            return -0.5

        elif dist < self.r_best_min:
            progress = (dist - self.r_min) / (self.r_best_min - self.r_min)
            return 0.3 + 0.7 * progress

        elif dist < self.r_best_max:
            return 1.0

        elif dist < self.r_max:
            progress = (self.r_max - dist) / (self.r_max - self.r_best_max)
            return 0.3 + 0.7 * progress

        else:
            return 0.1 * np.exp(-(dist - self.r_max) / 3000.0)

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