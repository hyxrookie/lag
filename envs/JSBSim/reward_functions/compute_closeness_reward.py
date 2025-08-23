import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import LLA2NEU, get_AO_TA_R

class ComputeClosenessReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.optimal_combat_range = getattr(self.config, 'optimal_combat_range')
        self.pre_ave_r = {}
    def reset(self, task, env):
        self.pre_ave_r.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        new_reward = 0
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        nums = 0
        sum_r = 0
        for enm in env.agents[agent_id].enemies:
            if not enm.is_alive:
                continue
            enm_feature = np.hstack([enm.get_position(),
                                     enm.get_velocity()])
            _, _, R = get_AO_TA_R(ego_feature, enm_feature)
            nums += 1
            sum_r += R
        if nums == 0:
            return 0
        ave_r = sum_r / nums
        if ave_r > self.optimal_combat_range:
            if agent_id not in self.pre_ave_r:
                self.pre_ave_r.update({agent_id: ave_r})
            pre_ave_r = self.pre_ave_r[agent_id]
            # 距离变大-20
            if ave_r - pre_ave_r > 0:
                new_reward -= 20
            else:
                new_reward += 20
            self.pre_ave_r.update({agent_id: ave_r})

        return self._process(new_reward, agent_id)



