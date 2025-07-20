import datetime
import os

import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.JSBSim.utils.shared_variable import GlobalVars
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor import PPOActor
import datetime
import logging
logging.basicConfig(level=logging.ERROR)

class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = True
    
def _t2n(x):
    return x.detach().cpu().numpy()

num_agents = 4
render = True
ego_policy_index = 193
enm_policy_index = 193
episode_rewards = 0
ego_run_dir = "D:/pythonProject/lag/scripts/results/MultipleCombat/2v2/shootMissile/MyHierarchySelfplay/mappo/v1/wandb/run-20250508_080520-54h9aorx/files"
enm_run_dir = "D:/pythonProject/lag/scripts/results/MultipleCombat/2v2/shootMissile/MyHierarchySelfplay/mappo/v1/wandb/run-20250508_080520-54h9aorx/files"
experiment_name = ego_run_dir.split('/')[-4]
# D:\pythonProject\lag\scripts\results\MultipleCombat\2v2\shootMissile\MyHierarchySelfplay\mappo\v1\wandb\run-20250508_080520-54h9aorx\files\actor_193.pt
env = MultipleCombatEnv("2v2/ShootMissile/MyHierarchySelfplay")
env.seed(0)
timestamp_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
experiment_name += timestamp_str

args = Args()

ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
enm_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
ego_policy.eval()
enm_policy.eval()
ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_policy_index}.pt"))
enm_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_{enm_policy_index}.pt"))


print("Start render")
obs, _ = env.reset()

current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M')
# 提取文件名
file_name = enm_run_dir.split('wandb/')[1].split('/')[0]
# 存放文件夹
folder_name = f'Date{current_time}.{ego_policy_index}vs{enm_policy_index}_in_({file_name}).{experiment_name}'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

start = 0
end = 30

GlobalVars.shared_acmi_id=0
GlobalVars.use_autoshoot_in_render=True
# 注意：接入模型发射，本文件两处use_autoshoot_in_render改为false
# 用逻辑控制自动发射，本文件两处use_autoshoot_in_render改为True


while start != end:
    start += 1
    #记录文件号
    GlobalVars.shared_acmi_id += 1

    # 生成文件路径
    file_path = os.path.join(folder_name,
                             f'{start}.txt.acmi')

    obs, _ = env.reset()
    if render:
        env.render(mode='txt', filepath=file_path)
    ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
    masks = np.ones((num_agents // 2, 1))
    enm_obs =  obs[num_agents // 2:, :]
    ego_obs =  obs[:num_agents // 2, :]
    enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)
    while True:
        # start = time.time()
        ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
        # end = time.time()
        # print(f"NN forward time: {end-start}")
        ego_actions = _t2n(ego_actions)
        ego_rnn_states = _t2n(ego_rnn_states)
        enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
        enm_actions = _t2n(enm_actions)
        enm_rnn_states = _t2n(enm_rnn_states)
        actions = np.concatenate((ego_actions, enm_actions), axis=0)
        # Obser reward and next obs
        # start = time.time()
        obs, _, rewards, dones, infos = env.step(actions)
        # end = time.time()
        # print(f"Env step time: {end-start}")
        # rewards = rewards[:num_agents // 2, ...]
        # episode_rewards += rewards
        if render:
            env.render(mode='txt', filepath=file_path)
        if dones.all():
            env._create_records = False
            # print(infos)
            break
        # bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
        # print(f"step:{env.current_step}, bloods:{bloods}")
        enm_obs =  obs[num_agents // 2:, ...]
        ego_obs =  obs[:num_agents // 2, ...]

    GlobalVars.use_autoshoot_in_render = True

    if GlobalVars.shared_acmi_id == end:#此时需要将记录数据写入到磁盘
        env.save_record_data_to_csv(folder_name)

# print(episode_rewards)