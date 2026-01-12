# import logging
#
# import numpy as np
# from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
# import math
# from envs.JSBSim.core.catalog import Catalog as c, Catalog
# from envs.JSBSim.utils.utils import LLA2NEU, get_AO_TA_R
#
# class MissileDodgeReward(BaseRewardFunction):
#     def __init__(self, config):
#         super().__init__(config)
#         self.pre_missiles = {}
#         self.pre_missiles_v = {}
#         self.prev_velocity_component = {}
#         self.danger_dist = 40000.0  # 40km 开始预警
#         self.target_evade_mach = 1.5
#     def reset(self, task, env):
#         self.pre_missiles.clear()
#         self.pre_missiles_v.clear()
#         self.prev_velocity_component.clear()
#         return super().reset(task, env)
#
#     def get_reward(self, task, env, agent_id):
#         new_reward = 0
#         ego_feature = np.hstack([env.agents[agent_id].get_position(),
#                                  env.agents[agent_id].get_velocity()])
#         agent = env.agents[agent_id]
#         mach = agent.get_property_value(Catalog.velocities_mach)
#         missiles = agent.check_all_missile_warning() if hasattr(agent, 'check_missile_warning') else []
#         sim, threat_pos, threat_vel, dist = self._get_threat_info(agent, missiles)
#         if sim is None:
#             return new_reward
#         sim_feature = np.hstack([sim.get_position(), sim.get_velocity()])
#         AO, TA, R = get_AO_TA_R(ego_feature, sim_feature)
#
#         relative_velocity = sim.get_velocity() - agent.get_velocity()
#         direction_to_aircraft = (agent.get_position() - sim.get_position()) / R
#         velocity_component = np.dot(sim.get_velocity(), direction_to_aircraft)
#
#
#         dir_factor = np.degrees(np.arccos(np.dot(sim.get_velocity(), env.agents[agent_id].get_velocity()) /
#                                               (np.linalg.norm(sim.get_velocity()) * np.linalg.norm(env.agents[agent_id].get_velocity()))))
#
#         if dir_factor > 0:
#             # 正确方向：加上速度奖励
#             # 连续值优化：使用平滑插值
#             speed_score = np.clip((mach - 0.4) / (self.target_evade_mach - 0.4), 0.0, 1.0)
#             base_reward = dir_factor * (0.3 + 0.7 * speed_score)
#         else:
#             # 错误方向：直接惩罚
#             base_reward = dir_factor
#
#         new_reward += base_reward
#
#         missile_to_aircraft_direction = (agent.get_position() - sim.get_position()) / R
#         if sim.uid in self.pre_missiles_v:
#
#             pre_relative_velocity = self.pre_missiles_v[sim.uid]
#             relative_acceleration = relative_velocity - pre_relative_velocity
#
#             acceleration_component = np.dot(relative_acceleration, missile_to_aircraft_direction)
#             if acceleration_component < 0:
#                 new_reward += 0.2  # 加速远离导弹的奖励
#             else:
#                 new_reward -= 0.2
#
#
#
#         # 计算导弹速度在该方向上的分量
#
#         if sim.uid in self.prev_velocity_component:
#             pre_velocity_component = self.prev_velocity_component[sim.uid]
#             if velocity_component - pre_velocity_component < 0:
#                 new_reward += 0.4
#             else:
#                 new_reward -= 0.4
#         self.pre_missiles_v.update({sim.uid: relative_velocity})
#         self.prev_velocity_component.update({sim.uid: velocity_component})
#         self.pre_missiles.update({sim.uid: sim})
#         return self._process(new_reward, agent_id)
#     def _get_threat_info(self, agent, missiles):
#         # ... (保持原有的距离筛选逻辑) ...
#         if not missiles:
#             return None, None, None, float('inf')
#         ego_pos = agent.get_position()
#         nearest_m = min(missiles, key=lambda m: np.linalg.norm(ego_pos - m.get_position()))
#         dist = np.linalg.norm(ego_pos - nearest_m.get_position())
#         if dist > self.danger_dist:
#             return None, None, None, float('inf')
#         return nearest_m, nearest_m.get_position(), nearest_m.get_velocity(), dist
#
#
