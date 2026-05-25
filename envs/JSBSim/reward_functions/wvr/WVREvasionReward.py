import numpy as np

from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction


class WVREvasionReward(BaseRewardFunction):
    """
    【近距导弹规避奖励】

    和 BVR 规避不同：
    - 远距离可以背向拖距；
    - 中近距离更需要大 LOS 角变化；
    - 极近距离应避免直线飞，鼓励近似垂直于导弹视线方向的机动。
    """

    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__ + '_total']

        self.danger_dist = 12000.0
        self.mid_dist = 6000.0
        self.close_dist = 3000.0

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]

        missiles = agent.check_all_missile_warning() if hasattr(agent, 'check_all_missile_warning') else []
        if not missiles:
            return 0.0

        ego_pos = agent.get_position()
        missile_sim = min(missiles, key=lambda m: np.linalg.norm(ego_pos - m.get_position()))
        dist = np.linalg.norm(ego_pos - missile_sim.get_position())

        if dist > self.danger_dist:
            return 0.0

        aircraft_pos = agent.get_position()
        missile_pos = missile_sim.get_position()
        aircraft_v = agent.get_velocity()
        missile_v = missile_sim.get_velocity()

        v_norm = np.linalg.norm(aircraft_v) + 1e-6

        # LOS：从飞机指向导弹
        los_vec = missile_pos - aircraft_pos
        los_dist = np.linalg.norm(los_vec) + 1e-6
        los_unit = los_vec / los_dist

        # -----------------------------
        # 1. 背向逃逸奖励：适合较远距离
        # -----------------------------
        # cos_drag = 1 表示飞机速度方向完全背对导弹
        cos_drag = np.dot(aircraft_v, -los_unit) / v_norm
        cos_drag = np.clip(cos_drag, -1.0, 1.0)

        r_drag = (cos_drag + 1.0) / 2.0

        # -----------------------------
        # 2. 垂直 LOS 奖励：适合近距
        # -----------------------------
        # cos_los = 0 表示速度方向与 LOS 垂直，有利于制造 LOS 变化
        cos_los = np.dot(aircraft_v, los_unit) / v_norm
        cos_los = np.clip(cos_los, -1.0, 1.0)

        r_beam = 1.0 - abs(cos_los)

        # -----------------------------
        # 3. 闭合速度惩罚
        # -----------------------------
        rel_vel = missile_v - aircraft_v
        closing_speed = -np.dot(rel_vel, los_unit)
        # closing_speed > 0 表示导弹正在接近飞机
        r_closing = -np.clip(closing_speed / 1000.0, 0.0, 1.0)

        # -----------------------------
        # 4. 分距离融合
        # -----------------------------
        if dist > self.mid_dist:
            # 远一点：拖距为主
            reward_action = 0.8 * r_drag + 0.2 * r_beam
        elif dist > self.close_dist:
            # 中距离：拖距和制造 LOS 变化都重要
            reward_action = 0.45 * r_drag + 0.55 * r_beam
        else:
            # 极近距离：更强调垂直 LOS 机动
            reward_action = 0.2 * r_drag + 0.8 * r_beam

        total_reward = 6.0 * reward_action + 2.0 * r_closing

        return self._process(total_reward, agent_id)