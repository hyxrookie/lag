import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.core.catalog import Catalog as c


class MissileEvasionReward(BaseRewardFunction):
    """
    使用 agent.MissileAlert 作为总开关的导弹规避奖励函数。
    - 当告警消失时，给予带冷却的成功规`避奖励。
    - 当告警持续时，完全采用用户提供的精细机动评估逻辑来计算
      每一步的过程奖励/惩罚。
    """

    def __init__(self, config):
        super().__init__(config)
        # --- 宏观事件奖励 ---
        self.reward_on_dodge = getattr(self.config, 'reward_on_dodge', 100.0)
        self.dodge_cooldown_steps = getattr(self.config, 'dodge_cooldown_steps', 100)

        # --- 微观机动奖励/惩罚值 (来自您的代码，使其可配置) ---
        self.reward_angle_beaming = getattr(self.config, 'reward_angle_beaming', 10.0)
        self.penalty_angle_other = getattr(self.config, 'penalty_angle_other', -15.0)
        self.reward_accel_away = getattr(self.config, 'reward_accel_away', 10.0)
        self.penalty_accel_toward = getattr(self.config, 'penalty_accel_toward', -10.0)
        self.reward_v_comp_decrease = getattr(self.config, 'reward_v_comp_decrease', 10.0)
        self.penalty_v_comp_increase = getattr(self.config, 'penalty_v_comp_increase', -10.0)

        # --- 状态追踪记忆体 ---
        # {agent_id: {
        #   'prev_alert_status': bool,
        #   'last_dodge_reward_step': int,
        #   'missile_histories': {uid: {'relative_v': ..., 'v_component': ...}}
        # }}
        self.agent_memory = {}

    def reset(self, task, env):
        self.agent_memory.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]
        current_step = env.current_step
        total_reward = 0

        # --- 1. 初始化或获取记忆 ---
        if agent_id not in self.agent_memory:
            self.agent_memory[agent_id] = {
                'prev_alert_status': False,
                'last_dodge_reward_step': -float('inf'),
                'missile_histories': {}
            }
        memory = self.agent_memory[agent_id]
        prev_alert_status = memory['prev_alert_status']
        missile_histories = memory['missile_histories']

        # --- 2. 获取当前告警状态 ---
        current_alert_status = agent.get("MissileAlert")

        # --- 3. 处理告警状态转变：从 True -> False (成功规避) ---
        if prev_alert_status and not current_alert_status:
            last_reward_step = memory['last_dodge_reward_step']
            if current_step - last_reward_step > self.dodge_cooldown_steps:
                total_reward += self.reward_on_dodge
                memory['last_dodge_reward_step'] = current_step
            # 规避成功，清空所有导弹的历史记录
            missile_histories.clear()

        # --- 4. 处理告警持续为 True 的情况 (执行您的机动代码) ---
        if current_alert_status:
            missile_sims = agent.detected_missiles()
            current_missile_ids = {sim.uid for sim in missile_sims if sim.is_alive}

            for sim in missile_sims:
                if not sim.is_alive:
                    continue

                # --- a. 计算当前帧指标 ---
                R = np.linalg.norm(agent.get_position() - sim.get_position())
                if R < 1e-6: continue

                relative_velocity = sim.get_velocity() - agent.get_velocity()
                direction_to_aircraft = (agent.get_position() - sim.get_position()) / R
                velocity_component = np.dot(sim.get_velocity(), direction_to_aircraft)

                # 如果第一次见到，只记录历史
                if sim.uid not in missile_histories:
                    missile_histories[sim.uid] = {'relative_v': relative_velocity, 'v_component': velocity_component}
                    continue

                # --- b. 角度奖励 ---
                ego_vel, msl_vel = agent.get_velocity(), sim.get_velocity()
                cos_angle = np.clip(
                    np.dot(ego_vel, msl_vel) / (np.linalg.norm(ego_vel) * np.linalg.norm(msl_vel) + 1e-8), -1.0, 1.0)
                relative_angle_deg = np.degrees(np.arccos(cos_angle))
                if 60 <= relative_angle_deg <= 110:
                    total_reward += self.reward_angle_beaming
                else:
                    total_reward += self.penalty_angle_other

                # --- c. 相对加速度奖励 ---
                pre_relative_velocity = missile_histories[sim.uid]['relative_v']
                relative_acceleration = relative_velocity - pre_relative_velocity
                acceleration_component = np.dot(relative_acceleration, direction_to_aircraft)
                if acceleration_component < 0:
                    total_reward += self.reward_accel_away
                else:
                    total_reward += self.penalty_accel_toward

                # --- d. 速度分量奖励 ---
                pre_velocity_component = missile_histories[sim.uid]['v_component']
                if velocity_component < pre_velocity_component:
                    total_reward += self.reward_v_comp_decrease
                else:
                    total_reward += self.penalty_v_comp_increase

                # --- e. 更新历史 ---
                missile_histories[sim.uid]['relative_v'] = relative_velocity
                missile_histories[sim.uid]['v_component'] = velocity_component

            # 清理已经消失的导弹的历史
            obsolete_ids = [uid for uid in missile_histories if uid not in current_missile_ids]
            for uid in obsolete_ids:
                del missile_histories[uid]

        # --- 5. 更新记忆，为下一帧做准备 ---
        memory['prev_alert_status'] = current_alert_status

        return self._process(total_reward, agent_id)