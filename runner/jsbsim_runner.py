import time
import torch
import logging
import numpy as np
from typing import List
from .base_runner import Runner, ReplayBuffer


def _t2n(x):
    return x.detach().cpu().numpy()


class CurriculumScheduler:
    def __init__(self, total_steps, start_level=0.0, end_level=1.0):
        self.total_steps = total_steps
        # --- Sigmoid 参数 ---
        # 决定曲线的中点 (Midpoint)：在训练到 40% 时，难度达到 0.5
        self.midpoint = total_steps * 0.4
        # 决定曲线的陡峭程度 (Slope)：值越大，难度提升越快
        # 20.0 是一个经验值，能保证曲线平滑且两端饱和
        self.slope = 20.0 / total_steps

    def get_level(self, current_step):
        # 核心公式: 1 / (1 + e^(-slope * (x - midpoint)))
        x = current_step
        exponent = -self.slope * (x - self.midpoint)
        # 防止 exp 溢出
        exponent = np.clip(exponent, -20, 20)
        sigmoid = 1.0 / (1.0 + np.exp(exponent))

        # 确保值在 0.0 到 1.0 之间
        return float(np.clip(sigmoid, 0.0, 1.0))

class JSBSimRunner(Runner):

    def load(self):
        self.obs_space = self.envs.observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.num_agents
        self.use_selfplay = self.all_args.use_selfplay

        # policy & algorithm
        if self.algorithm_name == "ppo":
            from algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
            from algorithms.ppo.ppo_policy import PPOPolicy as Policy
        else:
            raise NotImplementedError
        self.policy = Policy(self.all_args, self.obs_space, self.act_space, device=self.device)
        self.trainer = Trainer(self.all_args, device=self.device)

        # buffer
        self.buffer = ReplayBuffer(self.all_args, self.num_agents, self.obs_space, self.act_space)

        if self.model_dir is not None:
            self.restore()

    def run(self):
        self.warmup()
        start = time.time()
        self.total_num_steps = 0
        episodes = self.num_env_steps // self.buffer_size // self.n_rollout_threads

        # [新增] 1. 初始化课程调度器
        curriculum_scheduler = CurriculumScheduler(self.num_env_steps)

        for episode in range(episodes):

            # [新增] 2. 计算当前难度等级 (0.0 ~ 1.0)
            # 使用 self.total_num_steps 作为当前进度
            current_level = curriculum_scheduler.get_level(self.total_num_steps)

            # [新增] 3. 将难度同步给所有并行环境
            # 假设 self.envs 是一个 VecEnv 或者类似的并行环境包装器
            # 我们需要调用环境的一个方法来更新 self.curriculum_level
            self.update_env_curriculum(current_level)

            heading_turns_list = []

            for step in range(self.buffer_size):
                # Sample actions
                values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.collect(step)

                # Obser reward and next obs
                # [注意] 当 step 触发 done 并自动 reset 时，
                # 环境里的 reset_simulators 就会使用刚才更新过的 current_level
                obs, rewards, dones, infos = self.envs.step(actions)

                # Extra recorded information
                for info in infos:
                    if 'heading_turn_counts' in info:
                        heading_turns_list.append(info['heading_turn_counts'])

                data = obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            self.total_num_steps = (episode + 1) * self.buffer_size * self.n_rollout_threads

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                logging.info(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                    .format(self.all_args.scenario_name,
                            self.algorithm_name,
                            self.experiment_name,
                            episode,
                            episodes,
                            self.total_num_steps,
                            self.num_env_steps,
                            int(self.total_num_steps / (end - start))))

                train_infos["average_episode_rewards"] = self.buffer.rewards.sum() / (self.buffer.masks == False).sum()

                # [新增] 记录当前课程难度，方便在 Tensorboard 查看
                train_infos["curriculum_level"] = current_level
                logging.info(f"Current Curriculum Level: {current_level:.4f}")  # 打印到日志

                logging.info("average episode rewards is {}".format(train_infos["average_episode_rewards"]))

                if len(heading_turns_list):
                    train_infos["average_heading_turns"] = np.mean(heading_turns_list)
                    logging.info("average heading turns is {}".format(train_infos["average_heading_turns"]))
                self.log_info(train_infos, self.total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and episode != 0 and self.use_eval:
                self.eval(self.total_num_steps)

            # save model
            if (episode % self.save_interval == 0) or (episode == episodes - 1):
                self.save(episode)

    def update_env_curriculum(self, level):
        """
        兼容 DummyVecEnv 和 SubprocVecEnv 的参数更新函数
        """
        # 情况 A: 单线程调试模式 (DummyVecEnv / ShareDummyVecEnv)
        # 这种模式下，self.envs.envs 就是环境实例的列表，直接循环调用即可
        if "Dummy" in self.envs.__class__.__name__:
            for env in self.envs.envs:
                if hasattr(env, 'set_curriculum_level'):
                    env.set_curriculum_level(level)

        # 情况 B: 多进程并行模式 (SubprocVecEnv / ShareSubprocVecEnv)
        # 这种模式下，环境在别的进程里，不能直接访问
        else:
            self.envs.set_curriculum_level(level)
    def warmup(self):
        # reset env
        obs = self.envs.reset()
        self.buffer.step = 0
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.policy.prep_rollout()
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic \
            = self.policy.get_actions(np.concatenate(self.buffer.obs[step]),
                                      np.concatenate(self.buffer.rnn_states_actor[step]),
                                      np.concatenate(self.buffer.rnn_states_critic[step]),
                                      np.concatenate(self.buffer.masks[step]))
        # split parallel data [N*M, shape] => [N, M, shape]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def insert(self, data: List[np.ndarray]):
        obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic = data

        dones_env = np.all(dones.squeeze(axis=-1), axis=-1)

        rnn_states_actor[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_actor.shape[1:]), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_critic.shape[1:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        self.buffer.insert(obs, actions, rewards, masks, action_log_probs, values, rnn_states_actor, rnn_states_critic)

    @torch.no_grad()
    def eval(self, total_num_steps):
        logging.info("\nStart evaluation...")
        total_episodes, eval_episode_rewards = 0, []
        eval_cumulative_rewards = np.zeros((self.n_eval_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=np.float32)

        eval_obs = self.eval_envs.reset()
        eval_masks = np.ones((self.n_eval_rollout_threads, *self.buffer.masks.shape[2:]), dtype=np.float32)
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)

        self.timestamp = 0 # use for tacview real time render 
        if self.render_mode == "real_time" and self.tacview: #reconnect tacview to clear the telemetry
            print("reconnect tacview.....")
            self.tacview.reconnect()
        
        while total_episodes < self.eval_episodes:

            self.policy.prep_rollout()
            eval_actions, eval_rnn_states = self.policy.act(np.concatenate(eval_obs),
                                                            np.concatenate(eval_rnn_states),
                                                            np.concatenate(eval_masks), deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)

            # real render with tacview
            if self.render_mode == "real_time" and self.tacview:
                render_data = [f"#{self.timestamp:.2f}\n"]
                for sim in self.eval_envs.envs[0]._jsbsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        render_data.append(log_msg + "\n")
                for sim in self.eval_envs.envs[0]._tempsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        render_data.append(log_msg + "\n")
                render_data_str = "".join(render_data)
                try:
                    self.tacview.send_data_to_client(render_data_str)
                except Exception as e:
                    logging.error(f"Tacview rendering error: {e}")
            self.timestamp += 0.2   # step 0.2s
            
            eval_cumulative_rewards += eval_rewards
            eval_dones_env = np.all(eval_dones.squeeze(axis=-1), axis=-1)
            total_episodes += np.sum(eval_dones_env)
            eval_episode_rewards.append(eval_cumulative_rewards[eval_dones_env == True])
            eval_cumulative_rewards[eval_dones_env == True] = 0

            eval_masks = np.ones_like(eval_masks, dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), *eval_masks.shape[1:]), dtype=np.float32)
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), *eval_rnn_states.shape[1:]), dtype=np.float32)

        eval_infos = {}
        eval_infos['eval_average_episode_rewards'] = np.concatenate(eval_episode_rewards).mean(axis=1)  # shape: [num_agents, 1]
        logging.info(" eval average episode rewards: " + str(np.mean(eval_infos['eval_average_episode_rewards'])))
        self.log_info(eval_infos, total_num_steps)
        logging.info("...End evaluation")

    @torch.no_grad()
    def render(self):
        logging.info("\nStart render, render mode is {self.render_mode} ... ...")
        render_episode_rewards = 0
        render_obs = self.envs.reset()
        render_masks = np.ones((1, *self.buffer.masks.shape[2:]), dtype=np.float32)
        render_rnn_states = np.zeros((1, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
        self.envs.render(mode=self.render_mode, filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi',tacview=self.tacview)
        while True:
            self.policy.prep_rollout()
            render_actions, render_rnn_states = self.policy.act(np.concatenate(render_obs),
                                                                np.concatenate(render_rnn_states),
                                                                np.concatenate(render_masks),
                                                                deterministic=True)
            render_actions = np.expand_dims(_t2n(render_actions), axis=0)
            render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)

            # Obser reward and next obs
            render_obs, render_rewards, render_dones, render_infos = self.envs.step(render_actions)
            if self.use_selfplay:
                render_rewards = render_rewards[:, :self.num_agents // 2, ...]
            render_episode_rewards += render_rewards
            self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
            if render_dones.all():
                break
        render_infos = {}
        render_infos['render_episode_reward'] = render_episode_rewards
        logging.info("render episode reward of agent: " + str(render_infos['render_episode_reward']))

    def save(self, episode):
        policy_actor_state_dict = self.policy.actor.state_dict()
        torch.save(policy_actor_state_dict, str(self.save_dir) + '/actor_latest.pt')
        policy_critic_state_dict = self.policy.critic.state_dict()
        torch.save(policy_critic_state_dict, str(self.save_dir) + '/critic_latest.pt')
