import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from algorithms.transformer.ppo_policy import PPOPolicy


# ==========================================
# 1. Mock 工具 (模拟你的 utils)
# ==========================================
def check(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


class DummyActLayer(nn.Module):
    """模拟 ACTLayer"""

    def __init__(self, act_space, input_dim, hidden_size, activation_id, gain):
        super().__init__()
        self.fc = nn.Linear(input_dim, act_space)

    def forward(self, x, deterministic=False):
        logits = self.fc(x)
        probs = torch.softmax(logits, dim=-1)
        return torch.argmax(probs, dim=-1), logits

    def evaluate_actions(self, x, action, active_masks=None):
        logits = self.fc(x)
        return logits, torch.tensor(0.1)  # dummy entropy


# ==========================================
# 2. 模拟配置参数 (Args)
# ==========================================
class MockArgs:
    def __init__(self):
        # 基础配置
        self.hidden_size = 64
        self.act_hidden_size = '64'
        self.gain = 0.01
        self.activation_id = 1
        self.use_feature_normalization = True
        self.lr = 5e-4

        # RNN / Transformer 配置
        self.use_recurrent_policy = True
        self.recurrent_hidden_layers = 2  # 2层 Transformer
        self.recurrent_hidden_size = 64
        self.data_chunk_length = 16  # 训练时的序列长度

        # 多智能体配置
        self.num_agents = 4  # 假设有3个Agent

        # 实体配置 (对应 EntityEmbedder)
        # 假设 Obs 结构: [Own(10), Ally(2*10), Enemy(2*10), Missile(1*5)] = 55 dims
        self.entity_config = {
            'own_dim': 10,
            'ally_dim': 10,
            'ally_num': 1,
            'enemy_dim': 10,
            'enemy_num': 2,
            'missile_dim': 5,
            'missile_num': 1
        }


# ==========================================
# 3. 构造 Dummy Policy (依赖你之前的类)
# ==========================================
# 注意：这里假设你已经定义了 PPOPolicy, PPOActor, PPOCritic, EntityEmbedder, GatedTransformerBlock 等
# 为了代码可运行，你需要确保那些类在当前命名空间中可用。
# 如果是在同一个文件中，直接粘贴在上方即可。

# 这里我将模拟 PPOPolicy 的实例化过程
def test_pipeline():
    print("=== 开始集成测试 ===")
    args = MockArgs()
    device = torch.device("cpu")  # 测试用 CPU 即可

    # 1. 计算总 Obs 维度
    # Own(10) + Ally(20) + Enemy(20) + Missile(5) = 55
    obs_dim = 55
    # Share Obs 假设是所有 Agent obs 的拼接
    cent_obs_dim = obs_dim * args.num_agents
    act_space = spaces.MultiDiscrete([3, 5, 3])

    print(f"配置检查: Obs Dim={obs_dim}, Cent Obs Dim={cent_obs_dim}, Layers={args.recurrent_hidden_layers}")

    # 实例化 Policy
    # 注意：这里需要你确保 import 了 PPOPolicy
    # from your_module import PPOPolicy
    try:
        # 假设 PPOPolicy 已经正确导入或定义
        policy = PPOPolicy(args, obs_dim, cent_obs_dim, act_space, device)
        print("✅ PPOPolicy 实例化成功")
    except NameError:
        print("❌ 错误: 找不到 PPOPolicy 类。请确保测试脚本和类定义在同一文件中或正确 Import。")
        return

    # ==========================================
    # 测试场景 A: Rollout (单步推理)
    # ==========================================
    print("\n--- 测试 A: Rollout (环境交互模式) ---")
    batch_size = 4
    n_rollout_threads = 4

    # 模拟输入
    # Obs: (Batch, Obs_Dim)
    obs = torch.randn(batch_size, obs_dim)
    cent_obs = torch.randn(batch_size, cent_obs_dim)
    masks = torch.ones(batch_size, 1)  # 全 1 表示没有 Done

    # Memory 初始化 (关键！)
    # Shape: (Layers, Mem_Len, Batch, Hidden)
    # 初始 Mem_Len 可以是 0 (空) 或 max_len (全0)。通常 GTrXL 初始可以用全0填充
    mem_len = 96
    rnn_states_actor = torch.zeros(args.recurrent_hidden_layers, mem_len, batch_size, args.hidden_size)
    rnn_states_critic = torch.zeros(args.recurrent_hidden_layers, mem_len, batch_size, args.hidden_size)

    print(f"Input Obs Shape: {obs.shape}")
    print(f"Input Memory Shape: {rnn_states_actor.shape}")

    with torch.no_grad():
        values, actions, log_probs, new_rnn_actor, new_rnn_critic = policy.get_actions(
            cent_obs, obs, rnn_states_actor, rnn_states_critic, masks
        )

    print("✅ Rollout Forward Pass 成功")
    print(f"Output Actions Shape: {actions.shape} (预期: [{batch_size}, 1] 或 [{batch_size}, Act_Dim])")
    print(f"Output Values Shape: {values.shape}")
    print(f"Output New Memory Shape: {new_rnn_actor.shape}")

    # 验证 Memory 维度是否保持一致 (GTrXL 应该维持 Mem_Len 长度，或者滑窗更新)
    assert new_rnn_actor.shape == rnn_states_actor.shape, \
        f"Memory 维度变化不符合预期! 原: {rnn_states_actor.shape}, 新: {new_rnn_actor.shape}"

    # ==========================================
    # 测试场景 B: Train (序列训练)
    # ==========================================
    print("\n--- 测试 B: Training (序列训练模式) ---")
    # 模拟从 Buffer 采样的 Chunk
    chunk_len = 16
    batch_size = 8  # mini-batch size

    # 模拟输入 (Batch, Seq_Len, Dim)
    obs_batch = torch.randn(batch_size, chunk_len, obs_dim)
    cent_obs_batch = torch.randn(batch_size, chunk_len, cent_obs_dim)
    if isinstance(act_space, spaces.MultiDiscrete):
        # MultiDiscrete 情况: act_space.nvec = [3, 5, 3] -> 维度为 3
        action_dims = len(act_space.nvec)
        # 生成形状: (Batch, Seq, Action_Dims)
        # 这里简单起见，所有维度都随机生成 0-1 之间的数
        actions_batch = torch.randint(0, 2, (batch_size, chunk_len, action_dims))
    elif isinstance(act_space, spaces.Discrete):
        # Discrete 情况: 生成形状 (Batch, Seq, 1)
        actions_batch = torch.randint(0, act_space.n, (batch_size, chunk_len, 1))
    else:
        raise NotImplementedError("测试脚本暂不支持此 Action Space 类型")
    masks_batch = torch.ones(batch_size, chunk_len, 1)

    # Training 时，Memory 只需要 Chunk 开头的那个状态
    # Shape: (Layers, Mem_Len, Batch, Hidden)
    rnn_states_actor_batch = torch.zeros(args.recurrent_hidden_layers, mem_len, batch_size, args.hidden_size)
    rnn_states_critic_batch = torch.zeros(args.recurrent_hidden_layers, mem_len, batch_size, args.hidden_size)

    print(f"Train Obs Batch Shape: {obs_batch.shape}")

    # 调用 evaluate_actions
    values, action_log_probs, dist_entropy = policy.evaluate_actions(
        cent_obs_batch, obs_batch,
        rnn_states_actor_batch, rnn_states_critic_batch,
        actions_batch, masks_batch
    )

    print("✅ Train Forward Pass 成功")
    print(f"Train Values Shape: {values.shape} (预期: [{batch_size}, {chunk_len}, 1])")
    print(f"Train Log Probs Shape: {action_log_probs.shape} (预期: [{batch_size}, {chunk_len}, 1])")

    assert values.shape == (batch_size, chunk_len, 1), "Value 输出维度错误"
    assert action_log_probs.shape == (batch_size, chunk_len, 1), "LogProb 输出维度错误"

    print("\n=== 🎉 所有测试通过！集成代码逻辑正常。 ===")


if __name__ == "__main__":
    test_pipeline()