import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import LLA2NEU, get_AO_TA_R

class DogdeAttackWindowReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.max_missile_attack_distance = getattr(self.config, 'max_missile_attack_distance')
        self.min_missile_attack_distance = getattr(self.config, 'min_missile_attack_distance')
        self.max_missile_attack_angle = getattr(self.config, 'max_missile_attack_angle', 60)
        self.max_missile_attack_angle = math.radians(self.max_missile_attack_angle)

    def get_reward(self, task, env, agent_id):
        new_reward = 0
        enm_attack = set()
        # feature: (north, east, down, vn, ve, vd)
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(),
                                     enm.get_velocity()])
            AO, _, R = get_AO_TA_R(ego_feature, enm_feature)
            enm_AO = math.radians(180) - AO
            if self.isAttacked(enm_AO, R):
                enm_attack.add(enm.uid)

        if len(enm_attack) > 1:
            new_reward -= 10
        elif len(enm_attack) == 1:
            new_reward -= 5
        return self._process(new_reward, agent_id)

    def isAttacked(self, AO, R):
        return AO < self.max_missile_attack_angle and self.min_missile_attack_distance < R < self.max_missile_attack_distance

