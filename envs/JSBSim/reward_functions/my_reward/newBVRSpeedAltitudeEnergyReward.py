import numpy as np
from envs.JSBSim.core.catalog import Catalog
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction


class EnergyCentricReward(BaseRewardFunction):
    """
    【基于能量机动理论的奖励函数】

    核心机制：
    1. 不再单纯奖励速度或高度，而是奖励“比能量(Es)”的积累。
    2. 奖励“能量变化率(Ps)”，迫使智能体使用引擎推力而非重力来获得速度。
    """

    def __init__(self, config):
        super().__init__(config)

        # 目标参数
        self.target_mach = 1.0
        self.target_alt = 8000.0  # m
        self.g = 9.81

        # 状态记忆 (用于计算变化率)
        self.previous_energy = {}

        # 权重
        self.w_energy_state = 1.0  # 保持高能量状态
        self.w_energy_rate = 2.0  # 鼓励能量增长 (关键！用于学会平飞加速)
        self.w_stability = 0.5  # 姿态稳定奖励

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]

        # 1. 获取物理量
        mach = agent.get_property_value(Catalog.velocities_mach)
        velocities = agent.get_velocity()   # 真速 m/s (近似，严谨应用 get_velocity 模长)
        v_mps = np.linalg.norm(velocities)
        if v_mps < 10: v_mps = 340 * mach  # 简单估算防止冷启动bug

        pos = agent.get_position()
        alt = pos[2]

        # 2. 计算比能量 (Specific Energy, Es)
        # Es = H + V^2 / 2g (单位：米)
        energy_current = alt + (v_mps ** 2) / (2 * self.g)

        # 计算目标能量
        target_v = 340.0 * self.target_mach  # 粗略估算声速
        energy_target = self.target_alt + (target_v ** 2) / (2 * self.g)

        if agent_id not in self.previous_energy:
            # 如果字典里没它，说明是第一帧，初始化
            self.previous_energy[agent_id] = energy_current
            return 0.0

        # -----------------------------------------------------------
        # [关键逻辑 A] 能量增长率奖励 (Pseudo-Ps Reward)
        # -----------------------------------------------------------
        r_energy_rate = 0.0

        # 能量增量 (Delta E)
        delta_e = energy_current - self.previous_energy[agent_id]
        # print("agentid:{}delta_e:{}, energy_current:{},energy_target:{}, self.previous_energy:{} ".format(agent_id, delta_e, energy_current, energy_target, self.previous_energy))
        # 如果尚未达到目标能量，且能量在增加 -> 给大奖励
        # 这就是教它“平飞加速”的核心：只有引擎出力，delta_e 才是正的。
        # 俯冲时，delta_e 接近 0 (忽略阻力) 或为负 (考虑阻力)。
        if energy_current < energy_target:
            if delta_e > 0:
                r_energy_rate = delta_e / 5  # 放大系数，鼓励爬升或加速
            else:
                r_energy_rate = delta_e / 5  # 惩罚掉能量（如大过载转弯、开减速板）
        else:
            # 能量溢出（超速或太高），不需要再增加了
            r_energy_rate = 0.0

        # 更新上一帧能量
        self.previous_energy[agent_id] = energy_current

        # -----------------------------------------------------------
        # [关键逻辑 B] 能量状态保持奖励 (Energy State Reward)
        # -----------------------------------------------------------
        # 引导智能体接近目标能量水平
        err_energy = abs(energy_target - energy_current)
        # 归一化：假设能量差 10000米 对应 0分
        r_energy_state = np.exp(- (err_energy / 2000.0) ** 2)

        # -----------------------------------------------------------
        # [关键逻辑 C] 平飞/稳定修正 (Stability Constraint)
        # -----------------------------------------------------------
        # 为了更明确地告诉它“不要掉高度”，可以加一个弱约束：
        # 当速度不足时，垂直速度越接近 0，奖励越高
        v_vertical = agent.get_velocity()[2]
        r_stability = 0.0

        if mach < self.target_mach:
            # 正在加速阶段，鼓励垂直速度接近0 (平飞)
            # 或者鼓励垂直速度 > 0 (爬升加速)
            if v_vertical > -2.0:  # 允许爬升或微弱掉高
                r_stability = 0.5
            else:
                # 正在剧烈掉高度加速 -> 扣分
                r_stability = -0.1 * abs(v_vertical)

        # -----------------------------------------------------------
        # 总奖励
        # -----------------------------------------------------------
        total_reward = (
                self.w_energy_state * r_energy_state +
                self.w_energy_rate * r_energy_rate +
                self.w_stability * r_stability
        )

        return self._process(total_reward, agent_id)

    def reset(self, task, env):
        # 记得在 reset 时重置 history
        self.previous_energy = {}
        return super().reset(task, env)