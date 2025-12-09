import math
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c


class HeadingReward(BaseRewardFunction):
    """
    基于论文策略改进的奖励函数：
    1. 从几何平均 (乘法) 改为 加权求和 (加法)。
    2. 引入动态速度权重 (Dynamic Speed Weight)：航向误差大时，不关心速度；航向对准后，重点保持速度。
    """

    def __init__(self, config):
        super().__init__(config)
        # 记录各项的 reward 值，方便在 Tensorboard 中观察
        self.reward_item_names = [self.__class__.__name__ + item for item in
                                  ['', '_heading', '_alt', '_roll', '_speed', '_w_spd']]

    def get_reward(self, task, env, agent_id):
        # --- 1. 计算各项基础奖励 (保持你原有的高斯形式) ---

        # Heading Reward
        heading_error_scale = 5.0  # degrees
        delta_heading = env.agents[agent_id].get_property_value(c.delta_heading)
        heading_r = math.exp(-((delta_heading / heading_error_scale) ** 2))

        # Altitude Reward
        alt_error_scale = 15.24  # m
        alt_r = math.exp(-((env.agents[agent_id].get_property_value(c.delta_altitude) / alt_error_scale) ** 2))

        # Roll Reward
        # 注意：在加权求和模式下，即使这里得分为0，也不会“杀死”整个奖励。
        # 建议保留作为稳定性项，但权重可以给低一点。
        # roll_error_scale = 0.35  # radians
        # roll_r = math.exp(-((env.agents[agent_id].get_property_value(c.attitude_roll_rad) / roll_error_scale) ** 2))

        # Speed Reward
        speed_error_scale = 24  # mps
        speed_r = math.exp(-((env.agents[agent_id].get_property_value(c.delta_velocities_u) / speed_error_scale) ** 2))

        # --- 2. 核心修改：定义权重 (Weights) ---

        # [固定权重]
        # 航向是首要目标，给较高的固定权重
        w_heading = 0.3
        # 高度也比较重要，固定权重
        w_alt = 0.3
        # 滚转角作为辅助约束，给一个较小的权重。
        # 在大机动转弯时，roll_r 会变小，但因为权重只有 0.1，对总分影响有限，Agent 敢于牺牲这 0.1 分去换取 0.4 分的 Heading。
        w_roll = 0.1

        # [动态权重 - 速度] (复现论文公式 11)
        # 逻辑：当航向误差很大时，我们需要飞机全力转弯（通常伴随掉速），此时如果惩罚速度，Agent就不敢动了。
        # 所以：航向误差越大 -> 速度权重越小。

        # 论文中的 scaling factor 是 10 (度)。
        # w_speed = 0.4 / (1 + |error|/10)
        # 假设 delta_heading 单位是度 (根据你的 heading_error_scale=5.0 推断)
        abs_heading_err = abs(delta_heading)
        w_speed_max = 0.4  # 最大速度权重，与 heading 相当

        # 计算动态权重
        # 当 error = 0时, w_speed = 0.4
        # 当 error = 50度时, w_speed = 0.4 / 6 ≈ 0.06 (几乎忽略速度误差)
        w_speed = w_speed_max / (1.0 + abs_heading_err / 10.0)

        # --- 3. 计算总奖励 (加权求和) ---
        # 这种形式允许 Agent 进行"利益交换" (Trade-off)
        reward = (w_heading * heading_r) + \
                 (w_alt * alt_r) + \
                 (w_speed * speed_r)

        # 归一化一下 (可选)，让总分大概在 0~1 之间，方便观察
        # 当前最大权重和 = 0.4 + 0.3 + 0.4 + 0.1 = 1.2

        return self._process(reward, agent_id, (heading_r, alt_r, speed_r, w_speed))