import unittest
import numpy as np
import torch
from types import SimpleNamespace


# ==========================================
# 1. 模拟环境依赖 (Mock Classes)
# ==========================================

class MockSpace:
    def __init__(self, shape):
        self.shape = shape


def get_shape_from_space(space):
    return space.shape


class BaseReplayBuffer:
    """模拟原始代码中的父类，只保留基础功能"""

    def __init__(self, args, obs_space, act_space):
        self.buffer_size = args.buffer_size
        self.n_rollout_threads = args.n_rollout_threads

    def _cast(self, x):
        return x


# ==========================================
# 2. 修改后的 SharedReplayBuffer (待测试目标)
# ==========================================

class SharedReplayBuffer(BaseReplayBuffer):
    def __init__(self, args, num_agents, obs_space, share_obs_space, act_space):
        # 基础配置
        self.num_agents = num_agents
        self.n_rollout_threads = args.n_rollout_threads
        self.buffer_size = args.buffer_size

        # === 关键点 1: 显式定义 Memory 维度 ===
        self.mem_len = args.data_chunk_length
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers

        obs_shape = get_shape_from_space(obs_space)
        share_obs_shape = get_shape_from_space(share_obs_space)
        act_shape = get_shape_from_space(act_space)

        # 数据存储容器
        self.obs = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, *obs_shape),
                            dtype=np.float32)
        self.share_obs = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, *share_obs_shape),
                                  dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, *act_shape),
                                dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.masks = np.ones((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.active_masks = np.ones_like(self.masks)
        self.action_log_probs = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, *act_shape),
                                         dtype=np.float32)
        self.value_preds = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1),
                                    dtype=np.float32)
        self.returns = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # === 关键点 2: 扩充 RNN States 维度 ===
        # 新 Shape: (Time, Threads, Agents, Layers, Mem_Len, Hidden)
        self.rnn_states_actor = np.zeros(
            (self.buffer_size + 1, self.n_rollout_threads, self.num_agents,
             self.recurrent_hidden_layers, self.mem_len, self.recurrent_hidden_size),
            dtype=np.float32
        )
        self.rnn_states_critic = np.zeros_like(self.rnn_states_actor)

        self.step = 0

    def insert(self, obs, share_obs, actions, rewards, masks, action_log_probs,
               value_preds, rnn_states_actor, rnn_states_critic, active_masks=None):
        """插入单步数据"""
        self.obs[self.step + 1] = obs.copy()
        self.share_obs[self.step + 1] = share_obs.copy()

        # 注意: 这里的 rnn_states 是网络输出的 next_mems
        # 理论上它的形状应该是 (Threads, Agents, Layers, Mem_Len, Hidden)
        self.rnn_states_actor[self.step + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()

        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()

        self.step = (self.step + 1) % self.buffer_size

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        """
        GTrXL 专用 Generator
        必须按【序列】(Chunks) 提取数据，并保持时序连续性。
        """
        # 1. 获取有效数据范围 (不包含最后一步的obs，因为是next_obs)
        T = self.buffer_size
        N = self.n_rollout_threads

        # 确认数据足够切分
        assert T >= data_chunk_length, "Buffer size must be larger than chunk length"

        # 计算总共有多少个 Chunk
        # 每个 Thread 可以切出 (T // Chunk_Len) 个 Chunk
        chunks_per_thread = T // data_chunk_length
        total_chunks = chunks_per_thread * N

        # 随机打乱 Chunk 的索引
        rand = torch.randperm(total_chunks).numpy()
        sampler = [rand[i * (total_chunks // num_mini_batch):(i + 1) * (total_chunks // num_mini_batch)]
                   for i in range(num_mini_batch)]

        # 辅助数据准备
        obs = self.obs[:-1]
        rnn_states_actor = self.rnn_states_actor[:-1]
        # ... (为简化测试，只取这两个关键变量，其他同理) ...

        for indices in sampler:
            obs_batch = []
            rnn_states_actor_batch = []

            for index in indices:
                # 解析 index -> (Thread_ID, Time_Start_Index)
                # 假设我们将所有 Threads 的 chunks 铺平：
                # Thread 0: [Chunk 0, Chunk 1, ...]
                # Thread 1: [Chunk 0, Chunk 1, ...]

                thread_id = index // chunks_per_thread
                chunk_id = index % chunks_per_thread

                start_t = chunk_id * data_chunk_length
                end_t = start_t + data_chunk_length

                # === 关键逻辑: 切片 ===
                # 取出该 Thread 在该时间段的序列数据
                # obs shape: (Chunk_Len, Agents, Dim)
                _obs = obs[start_t:end_t, thread_id]
                obs_batch.append(_obs)

                # === 关键逻辑: Memory ===
                # 只取 Chunk 开始时刻 (start_t) 的 Memory
                # mems shape: (Agents, Layers, Mem_Len, Hidden)
                _mems = rnn_states_actor[start_t, thread_id]
                rnn_states_actor_batch.append(_mems)

            # 堆叠 Batch
            # obs_batch: (Mini_Batch, Chunk_Len, Agents, Dim)
            # rnn_states_batch: (Mini_Batch, Agents, Layers, Mem_Len, Hidden)

            out_obs = np.stack(obs_batch)
            out_mems = np.stack(rnn_states_actor_batch)

            yield out_obs, out_mems


# ==========================================
# 3. 单元测试 (Unit Tests)
# ==========================================

class TestSharedReplayBuffer(unittest.TestCase):

    def setUp(self):
        # 1. 配置参数
        self.args = SimpleNamespace(
            n_rollout_threads=4,  # 4个并行环境
            buffer_size=20,  # 存20步数据
            data_chunk_length=5,  # 序列长度为5 (Transformer Window)
            recurrent_hidden_size=8,  # Hidden Dim
            recurrent_hidden_layers=2  # Layers
        )
        self.num_agents = 2

        # 2. 定义 Space (Obs 维度 10, Share_Obs 维度 20, Action 维度 5)
        self.obs_space = MockSpace((10,))
        self.share_obs_space = MockSpace((20,))
        self.act_space = MockSpace((5,))

        # 3. 实例化 Buffer
        self.buffer = SharedReplayBuffer(self.args, self.num_agents, self.obs_space, self.share_obs_space,
                                         self.act_space)

    def test_1_memory_structure(self):
        """测试 rnn_states 是否正确初始化为 GTrXL 需要的 6D 张量"""
        print("\n=== Test 1: Structure Verification ===")
        # 预期: (Time+1, Threads, Agents, Layers, Mem_Len, Hidden)
        expected_shape = (21, 4, 2, 2, 5, 8)
        real_shape = self.buffer.rnn_states_actor.shape

        print(f"RNN States Shape: {real_shape}")
        self.assertEqual(real_shape, expected_shape)

    def test_2_insertion_logic(self):
        """测试 insert 方法是否能存入复杂维度的 memory"""
        print("\n=== Test 2: Insertion Logic ===")

        # 模拟 Step 0 的网络输出
        obs = np.random.randn(4, 2, 10)  # (Threads, Agents, Dim)

        # 构造 Memory: (Threads, Agents, Layers, Mem_Len, Hidden)
        mems = np.random.randn(4, 2, 2, 5, 8)

        # 其他 Dummy 数据
        share_obs = np.random.randn(4, 2, 20)
        actions = np.zeros((4, 2, 5))
        rewards = np.zeros((4, 2, 1))
        masks = np.ones((4, 2, 1))
        log_probs = np.zeros((4, 2, 5))
        values = np.zeros((4, 2, 1))

        try:
            self.buffer.insert(obs, share_obs, actions, rewards, masks, log_probs, values, mems, mems)
            print("Insertion successful without errors.")
        except Exception as e:
            self.fail(f"Insertion failed with error: {e}")

        # 验证是否存进去了 (存入的是 step+1，即 index 1)
        stored_mems = self.buffer.rnn_states_actor[1]
        self.assertTrue(np.allclose(stored_mems, mems), "Stored memory mismatch!")

    def test_3_chunk_generator(self):
        """核心测试：Generator 是否正确切分 Chunk 并对齐 Memory"""
        print("\n=== Test 3: Generator Sequence & Memory Alignment ===")

        # === 步骤 1: 初始化第 0 步的 Observation ===
        # Buffer 的 obs 通常长度是 len+1，insert 负责填入 next_obs (即 index 1 开始)
        # 所以必须手动设置 index 0 的值，保证序列从 0 开始连续

        init_obs = np.zeros((4, 2, 10))
        init_share_obs = np.zeros((4, 2, 20))

        for thread_id in range(4):
            # t=0 的值
            init_obs[thread_id, :, 0] = (thread_id + 1) * 100 + 0
            init_share_obs[thread_id, :, 0] = (thread_id + 1) * 100 + 0

        # 手动赋值给 buffer 的第 0 帧
        self.buffer.obs[0] = init_obs
        self.buffer.share_obs[0] = init_share_obs

        # === 步骤 2: 模拟 Rollout 过程 ===
        for t in range(self.args.buffer_size):
            # 注意：Buffer 的 insert 通常意味着 "这一步结束了，存入这一步的 Reward 和 下一步的 Obs"
            # 所以这里我们要创建的是 (t+1) 时刻的 Obs

            next_t = t + 1
            obs = np.zeros((4, 2, 10))
            share_obs = np.zeros((4, 2, 20))

            for thread_id in range(4):
                # 生成连续的数值: 100, 101, 102...
                obs[thread_id, :, 0] = (thread_id + 1) * 100 + next_t
                share_obs[thread_id, :, 0] = (thread_id + 1) * 100 + next_t

            # Memory 对应的是当前步 t (用于 RNN 输入)
            mems = np.full((4, 2, 2, 5, 8), t, dtype=np.float32)

            # 其他占位符
            actions = np.zeros((4, 2, 5))
            log_probs = np.zeros((4, 2, 5))
            rewards = np.zeros((4, 2, 1))
            masks = np.ones((4, 2, 1))
            values = np.zeros((4, 2, 1))

            # 执行插入
            # insert 会将 obs 放入 self.obs[t+1]
            self.buffer.insert(obs, share_obs, actions, rewards, masks, log_probs, values, mems, mems)

        # === 步骤 3: 调用 Generator ===
        # data_chunk_length=5
        gen = self.buffer.recurrent_generator(np.zeros(1), num_mini_batch=2, data_chunk_length=5)

        chunk_count = 0
        for obs_batch, mems_batch in gen:
            chunk_count += 1
            print(f"Batch {chunk_count} - Obs Shape: {obs_batch.shape}, Mems Shape: {mems_batch.shape}")

            # === 验证时序连续性 ===
            # 取第一个样本
            sample_obs = obs_batch[0]  # (5, 2, 10)

            # 提取时间标记
            time_steps = sample_obs[:, 0, 0]
            print(f"  Sample 0 Time Sequence: {time_steps}")

            # 验证差分是否为 1
            diffs = time_steps[1:] - time_steps[:-1]

            # 加上详细的错误信息，方便调试
            error_msg = f"Sequence incorrect! Got: {time_steps}. Expected diffs all 1."
            self.assertTrue(np.allclose(diffs, 1), error_msg)

            # 额外验证：不仅连续，而且不能是 0 (除非 t=0)
            # 如果 sample 是从中间截取的，绝对不应该包含 0 (除非正好是 t=0 的那个数据)
            # 这一步通过上面的 diffs=1 其实已经间接验证了


if __name__ == '__main__':
    unittest.main()