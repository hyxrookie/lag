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
    def reset(self, task, env):
        self.pre_missiles.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        new_reward = 0
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        missile_sims = env.agents[agent_id].check_all_missile_warning()
        for sim in missile_sims:
            if not sim.is_alive :
                #成功躲避导弹给比较大的奖励
                if env.agents[agent_id].is_alive:
                    new_reward += 100
                continue
            if sim.uid not in self.pre_missiles:
                self.pre_missiles.update({sim.uid: sim})
                continue
            pre_missile = self.pre_missiles[sim.uid]
            pre_missile_feature = np.hstack([pre_missile.get_position(), pre_missile.get_velocity()])
            sim_feature = np.hstack([sim.get_position(), sim.get_velocity()])
            preAO, preTA, preR = get_AO_TA_R(ego_feature, pre_missile_feature)
            AO, TA, R = get_AO_TA_R(ego_feature, sim_feature)
            relative_angle = np.degrees(np.arccos(np.dot(sim.get_velocity(), env.agents[agent_id].get_velocity()) /
                                                  (np.linalg.norm(sim.get_velocity()) * np.linalg.norm(env.agents[agent_id].get_velocity()))))
            # 距离变远给奖励，变近给惩罚
            if preR - R > 0:
                new_reward += 10
            else:
                new_reward -= 10
            if 60 <= relative_angle <= 110:
                new_reward += 15
            elif relative_angle < 60:
                if (180 - TA) - (180 - preTA) >= 0:
                    new_reward += 5
                else:
                    new_reward -= 5
            elif relative_angle > 110:
                if (180 - TA) - (180 - preTA) <= 0:
                    new_reward += 5
                else:
                    new_reward -= 5
        return self._process(new_reward, agent_id)



