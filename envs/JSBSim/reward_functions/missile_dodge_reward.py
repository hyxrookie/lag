import logging

import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import LLA2NEU, get_AO_TA_R

class MissileDodgeReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.pre_missiles = {}
        self.pre_missiles_v = {}
        self.prev_velocity_component = {}
    def reset(self, task, env):
        self.pre_missiles.clear()
        self.pre_missiles_v.clear()
        self.prev_velocity_component.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        new_reward = 0
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        missile_sims = env.agents[agent_id].check_all_missile_warning()
        agent = env.agents[agent_id]
        for sim in missile_sims:
            if not sim.is_alive :
                #成功躲避导弹给比较大的奖励
                if env.agents[agent_id].is_alive:
                    new_reward += 100
                continue

            sim_feature = np.hstack([sim.get_position(), sim.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, sim_feature)
            relative_velocity = sim.get_velocity() - agent.get_velocity()
            direction_to_aircraft = (agent.get_position() - sim.get_position()) / R
            velocity_component = np.dot(sim.get_velocity(), direction_to_aircraft)
            if sim.uid not in self.pre_missiles:
                self.pre_missiles.update({sim.uid: sim})
                self.pre_missiles_v.update({sim.uid: relative_velocity})
                self.prev_velocity_component.update({sim.uid: velocity_component})
                continue


            relative_angle = np.degrees(np.arccos(np.dot(sim.get_velocity(), env.agents[agent_id].get_velocity()) /
                                                  (np.linalg.norm(sim.get_velocity()) * np.linalg.norm(env.agents[agent_id].get_velocity()))))

            if 60 <= relative_angle <= 110:
                new_reward += 10
            else:
                new_reward -= 5


            missile_to_aircraft_direction = (agent.get_position() - sim.get_position()) / R
            if sim.uid in self.pre_missiles_v:

                pre_relative_velocity = self.pre_missiles_v[sim.uid]
                relative_acceleration = relative_velocity - pre_relative_velocity

                acceleration_component = np.dot(relative_acceleration, missile_to_aircraft_direction)
                if acceleration_component < 0:
                    new_reward += 10  # 加速远离导弹的奖励
                else:
                    new_reward -= 10



            # 计算导弹速度在该方向上的分量

            if sim.uid in self.prev_velocity_component:
                pre_velocity_component = self.prev_velocity_component[sim.uid]
                if velocity_component - pre_velocity_component < 0:
                    new_reward += 10
                else:
                    new_reward -= 10
            self.pre_missiles_v.update({sim.uid: relative_velocity})
            self.prev_velocity_component.update({sim.uid: velocity_component})
            self.pre_missiles.update({sim.uid: sim})
        return self._process(new_reward, agent_id)



