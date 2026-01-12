import numpy as np


class BVRRewardController:
    def __init__(self, config):
        # ====== 基础权重（你现有配置） ======
        self.w_attack_geom = 4.0
        self.w_zone_range  = 2.0
        self.w_altitude    = 2.0
        self.w_speed       = 2.0
        self.w_evasion     = 8.0

        # ====== threat 门控参数 ======
        self.energy_suppress = 0.5   # 威胁下能量奖励衰减比例
        self.min_attack_gate = 0.0   # 是否保留一点攻击梯度（一般 0 就好）

    # ------------------------------------------------------------------
    # 1. 统一导弹威胁评估（你可以之后单独精修这个函数）
    # ------------------------------------------------------------------
    def compute_threat_from_distance(self, dist):
        d_launch = 40000.0
        d_critical = 30000.0

        if dist >= d_launch:
            return 0.0
        if dist <= d_critical:
            return 1.0

        # 归一化
        x = (d_launch - dist) / (d_launch - d_critical)

        # 指数压缩（前慢后快）
        threat = 1.0 - np.exp(-4.0 * x)
        return np.clip(threat, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 2. 核心：动态权重调度
    # ------------------------------------------------------------------
    def compute_dynamic_weights(self, threat):
        """
        根据 threat 返回一个 dict：各 reward 的“动态权重”
        """

        # ---------- 进攻类：随威胁线性衰减 ----------
        attack_gate = max(self.min_attack_gate, 1.0 - threat)

        w_attack_geom = self.w_attack_geom * attack_gate
        w_zone_range  = self.w_zone_range  * attack_gate

        # ---------- 能量类：部分衰减（不完全关闭） ----------
        energy_gate = 1.0 - self.energy_suppress * threat

        w_altitude = self.w_altitude * energy_gate
        w_speed = self.w_speed * energy_gate

        # ---------- 防御类：随威胁增强 ----------
        w_evasion = self.w_evasion * threat

        return {
            "attack_geom": w_attack_geom,
            "zone_range":  w_zone_range,
            "altitude":    w_altitude,
            "speed":       w_speed,
            "evasion":     w_evasion,
        }

    # ------------------------------------------------------------------
    # 3. 对外统一接口（你在 step() 里只调用这个）
    # ------------------------------------------------------------------
    def compute_total_reward(
        self,
        agent,
        r_attack_geom,
        r_zone_range,
        r_altitude,
        r_speed,
        r_evasion,
    ):
        threat = self.compute_threat_from_distance(agent)
        w = self.compute_dynamic_weights(threat)

        total_reward = (
            w["attack_geom"] * r_attack_geom
            + w["zone_range"]  * r_zone_range
            + w["altitude"]    * r_altitude
            + w["speed"]       * r_speed
            + w["evasion"]     * r_evasion
        )

        return total_reward, threat
