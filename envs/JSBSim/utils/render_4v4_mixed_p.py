import numpy as np
import torch
import time
import logging
import datetime
import matplotlib.pyplot as plt  # 新增：用于绘制图表

from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.JSBSim.utils.utils import parse_config
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c

# TODO 1: 导入你的 MAPPO 和 Transformer Actor
from algorithms.st.ppo_actor import PPOActorST
from algorithms.mappo.ppo_actor import PPOActor as PPOActorMAPPO, PPOActor

logging.basicConfig(level=logging.DEBUG)

# ================= 自定义设置 =================
NUM_EPISODES = 10  # TODO 0: 在这里自定义对战次数


# ============================================

class ArgsST:
    """Transformer 模型的配置"""

    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 8192
        self.recurrent_hidden_layers = 2
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cuda'))
        self.use_prior = True


class ArgsMAPPO:
    """MAPPO (MLP/GRU) 模型的配置"""

    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cuda'))
        self.use_prior = True


def _t2n(x):
    return x.detach().cpu().numpy()


# ================= 环境与路径配置 =================
scenario_name = "2v2/ShootMissile/MyHierarchySelfplay"
config = parse_config(scenario_name)
num_agents = len(config.aircraft_configs)
render = True

ego_policy_index = 559
enm_policy_index = 300

ego_run_dir = "/mnt/d/MyProject/lag/scripts/results/MultipleCombat/2v2/ShootMissile/MyHierarchySelfplay/st/v1/run16/"
enm_run_dir = "/mnt/d/MyProject/lag/scripts/results/MultipleCombat/2v2/ShootMissile/MyHierarchySelfplay/mappo/v1/run6/"
experiment_name = ego_run_dir.split('/')[-4] + "_ST_vs_MAPPO"

timestamp_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
experiment_name += "_" + timestamp_str

env = MultipleCombatEnv(scenario_name)
env.seed(0)

args_st = ArgsST()
args_mappo = ArgsMAPPO()

# ================= 策略初始化 =================
ego_policy = PPOActorST(args_st, env.observation_space, env.action_space, device=torch.device("cuda"))
enm_policy = PPOActor(args_mappo, env.observation_space, env.action_space, device=torch.device("cuda"))

ego_policy.eval()
enm_policy.eval()

ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_policy_index}.pt"))
enm_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_{enm_policy_index}.pt"))

print(f"Start render: ST(Ego) vs MAPPO(Enm) for {NUM_EPISODES} episodes")

# ================= 统计变量 =================
battle_stats = {'Red Win': 0, 'Blue Win': 0, 'Draw': 0}
history_bloods = []  # 记录每局结束时的血量分布

# ================= 对战主循环 =================
for episode in range(NUM_EPISODES):
    print(f"\n========== Starting Episode {episode + 1}/{NUM_EPISODES} ==========")
    obs, _ = env.reset()

    if render:
        # 为每局单独命名渲染文件
        env.render(mode='txt', filepath=f'{experiment_name}_ep{episode + 1}.txt.acmi')

    # 每局必须重置 RNN 状态
    ego_rnn_states = np.zeros(
        (1, args_st.recurrent_hidden_layers, args_st.recurrent_hidden_size),
        dtype=np.float32
    )
    enm_rnn_states = np.zeros(
        (num_agents // 2, args_mappo.recurrent_hidden_layers, args_mappo.recurrent_hidden_size),
        dtype=np.float32
    )
    masks = np.ones((num_agents // 2, 1))
    episode_rewards = 0

    while True:
        ego_obs = obs[:num_agents // 2, :]
        enm_obs = obs[num_agents // 2:, :]

        # --- Ego (Transformer) ---
        ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
        ego_actions = _t2n(ego_actions)
        ego_rnn_states = _t2n(ego_rnn_states)

        # --- Enm (MAPPO) ---
        enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
        enm_actions = _t2n(enm_actions)
        enm_rnn_states = _t2n(enm_rnn_states)

        actions = np.concatenate((ego_actions, enm_actions), axis=0)
        obs, _, rewards, dones, infos = env.step(actions)

        ego_rewards = rewards[:num_agents // 2, ...]
        episode_rewards += ego_rewards

        if render:
            env.render(mode='txt', filepath=f'{experiment_name}_ep{episode + 1}.txt.acmi')

        # 实时打印每步血量（如果觉得输出太多可以注释掉）
        # bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
        # print(f"step:{env.current_step}, bloods:{bloods}")

        if dones.all():
            # 获取终局血量
            final_bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]

            # 统计红蓝总血量进行胜负判定
            red_health = sum(final_bloods[:2])
            blue_health = sum(final_bloods[2:])

            if red_health > blue_health:
                winner = 'Red Win'
            elif blue_health > red_health:
                winner = 'Blue Win'
            else:
                winner = 'Draw'

            battle_stats[winner] += 1
            history_bloods.append(final_bloods)

            print(f"Episode {episode + 1} Finished! Result: {winner}")
            print(f"Final Bloods -> Red: {final_bloods[:2]} | Blue: {final_bloods[2:]}")
            print(f"Ego Episode Rewards: {episode_rewards}")
            break

# ================= 结果统计与可视化 =================
print("\n========== Final Battle Statistics ==========")
print(f"Total Episodes: {NUM_EPISODES}")
print(f"Red (ST) Wins  : {battle_stats['Red Win']} ({(battle_stats['Red Win'] / NUM_EPISODES) * 100:.1f}%)")
print(f"Blue (MAPPO) Wins: {battle_stats['Blue Win']} ({(battle_stats['Blue Win'] / NUM_EPISODES) * 100:.1f}%)")
print(f"Draws          : {battle_stats['Draw']} ({(battle_stats['Draw'] / NUM_EPISODES) * 100:.1f}%)")

# 绘制数据图表
labels = ['Red (ST) Win', 'Blue (MAPPO) Win', 'Draw']
sizes = [battle_stats['Red Win'], battle_stats['Blue Win'], battle_stats['Draw']]
colors = ['#ff6666', '#66b3ff', '#99ff99']

plt.figure(figsize=(12, 5))

# 子图1：胜率饼图
plt.subplot(1, 2, 1)
# 过滤掉为0的数据以防饼图报错或不美观
sizes_filtered = [s for s in sizes if s > 0]
labels_filtered = [l for s, l in zip(sizes, labels) if s > 0]
colors_filtered = [c for s, c in zip(sizes, colors) if s > 0]

plt.pie(sizes_filtered, labels=labels_filtered, colors=colors_filtered, autopct='%1.1f%%', startangle=90)
plt.title(f'Win Rate Statistics ({NUM_EPISODES} Episodes)')

# 子图2：每局剩余总血量柱状图
plt.subplot(1, 2, 2)
x = np.arange(1, NUM_EPISODES + 1)
red_remaining = [sum(b[:2]) for b in history_bloods]
blue_remaining = [sum(b[2:]) for b in history_bloods]

width = 0.35
plt.bar(x - width / 2, red_remaining, width, label='Red (Ego)', color='#ff6666')
plt.bar(x + width / 2, blue_remaining, width, label='Blue (Enm)', color='#66b3ff')
plt.xlabel('Episode')
plt.ylabel('Total Remaining Bloods')
plt.title('Team Remaining Bloods per Episode')
plt.xticks(x)
plt.legend()

plt.tight_layout()
# 保存图片到当前目录并展示
plt.savefig(f'{experiment_name}_results_chart.png')
print(f"\nVisualization saved as: {experiment_name}_results_chart.png")
plt.show()