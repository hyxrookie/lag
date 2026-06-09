import torch
import torch.nn as nn

from .agent_attention import AgentLevelAttention
from .entity_embedding import EntityEmbedding, EntityEmbeddingCritic
from .gru import GRULayer
from .mlp import MLPBase
from .spatial_attention import SpatialAttention
from .gtrxl import GTrXL  # 假设上面的GTrXL代码保存在这里

class SpatialTemporalBaseMLPT(nn.Module):
    """
    时空注意力基础模块 (Spatial-Temporal Base)
    
    架构流程:
    1. Entity Embedding: 将原始 obs [N, dim] 分解为 [N, num_entities, embed_dim]
    2. Spatial Attention: 处理实体间关系 [N, num_entities, embed_dim] -> [N, embed_dim]
    3. Temporal Attention (GTrXL): 处理时间序列 [N, embed_dim] -> [N, hidden_size]
    
    支持:
    - 任意前置维度 (L*B 或 B)
    - Actor (Ego-centric) 和 Critic (Global) 模式
    """
    def __init__(self, 
                 obs_space,
                 mlp_hidden_size=128,
                 hidden_size=128, 
                 embed_dim=128, 
                 
                 # 空间参数
                 ego_dim=9, 
                 relative_dim=6, 
                 num_friendly=0, 
                 num_enemy=0, 
                 num_missiles=0,
                 num_spatial_heads=4,
                 
                 # 时间参数 (GTrXL)
                 num_temporal_heads=4, 
                 num_temporal_layers=1,
                 memory_length=32,
                 
                 # 通用参数
                 activation_id=1, 
                 use_type_embedding=True, 
                 use_feature_normalization=True,
                 dropout=0.0,
                 is_critic=False, 
                 num_agents=None):
        
        super(SpatialTemporalBaseMLPT, self).__init__()
        self.is_critic = is_critic
        self.hidden_size = hidden_size
        self.mlp_hidden_size = mlp_hidden_size
        self.embed_dim = embed_dim
        self.memory_length = memory_length
        self.num_agents = num_agents
        self.activation_id = activation_id
        self.use_feature_normalization = use_feature_normalization
        # 1. 计算输入维度
        # 假设 obs_space 是 gym.spaces.Box 或类似对象
        if hasattr(obs_space, 'shape'):
            obs_dim = obs_space.shape[0]
        else:
            obs_dim = obs_space # 兼容直接传数字
            
        # 2. 特征归一化 (LayerNorm)
        if use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)
        else:
            self.feature_norm = nn.Identity()
            
        # 3. 实体嵌入层 (Entity Embedding)

        # self.entity_embed = EntityEmbedding(
        #     ego_dim=ego_dim,
        #     relative_dim=relative_dim,
        #     embed_dim=embed_dim,
        #     activation_id=activation_id,
        #     use_type_embedding=use_type_embedding,
        #     num_missiles=num_missiles
        # )
        #
        # # 4.1 实体级空间注意力 (Spatial Attention)
        # # Actor 使用 Ego-centric 空间注意力
        # self.spatial_attn = SpatialAttention(
        #     embed_dim=embed_dim,
        #     num_heads=num_spatial_heads,
        #     dropout=dropout
        # )
        self.mlp = MLPBase(obs_space, self.mlp_hidden_size, self.activation_id, self.use_feature_normalization)
        # 4.2 智能体级注意力
        if self.is_critic:
            assert num_agents is not None, "Critic 模式必须指定 num_agents"
            self.agent_attn = AgentLevelAttention(
                embed_dim=embed_dim,
                num_heads=num_spatial_heads,
                dropout=dropout
            )
            self.agent_fusion_mlp = nn.Sequential(
                nn.Linear(num_agents * embed_dim, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, embed_dim)
            )
            
        # 5. 时间注意力 (GTrXL)
        self.temporal_attn = GTrXL(
            input_size=embed_dim,
            hidden_size=hidden_size, # 如果 embed_dim != hidden_size，GTrXL内部会投影
            num_layers=num_temporal_layers,
            num_heads=num_temporal_heads,
            memory_len=memory_length
        )
        
        # 记录所需的参数供外部调用
        self.num_friendly = num_friendly
        self.num_enemy = num_enemy
        self.num_missiles = num_missiles

    def forward(self, obs, rnn_states, masks, agent_index=None):
        """
        Args:
            obs: Actor传入局部obs [L*B, obs_dim], Critic传入全局share_obs [L*B, num_agents * obs_dim]
            rnn_states: [Batch, Layers, Mem_Len * Embed_Dim] (GTrXL 的扁平化 Memory)
            masks: [L*B, 1] 存活/终止掩码
            agent_index: 当前 Critic 正在评估的智能体索引 (用于精准的 Agent-level Attention)

        Returns:
            features: [L*B, hidden_size]
            new_rnn_states_flat: [Batch, Layers, Mem_Len * Embed_Dim]
        """
        x_spatial = self.mlp(obs)
        # ============================================================
        # 3. 时间序列特征提取 (Temporal Attention - GTrXL)
        # ============================================================
        batch_size, n_layers, flat_dim = rnn_states.shape
        # A. Reshape 恢复 Memory 维度
        # [Batch, Layers, Flat_Dim] -> [Batch, Layers, Mem_Len, Hidden]
        rnn_states_view = rnn_states.view(batch_size, n_layers, self.memory_length, self.embed_dim)

        # B. Permute 置换维度以适应 GTrXL 内部逻辑
        # GTrXL 期待 memory 格式为 [Layers, Batch, Mem_Len, Hidden]
        rnn_states_view = rnn_states_view.permute(1, 0, 2, 3)
        # C. 执行 GTrXL 时间注意力
        # features: [Batch, embed_dim]
        # new_rnn_states: [Layers, Batch, Mem_Len, Hidden]
        features, new_rnn_states = self.temporal_attn(x_spatial, rnn_states_view, masks)

        # print("rnn_states after", new_rnn_states.shape)

        # D. 输出还原 (Layers First -> Batch First -> Flattened)
        # [Layers, Batch, Mem_Len, Embed_Dim] -> [Batch, Layers, Mem_Len, Embed_Dim]
        new_rnn_states = new_rnn_states.permute(1, 0, 2, 3)
        # 变回 [Batch, Layers, Flat_Dim] 存入 Buffer
        new_rnn_states_flat = new_rnn_states.flatten(2, 3)

        return features, new_rnn_states_flat

    @property
    def output_size(self):
        return self.hidden_size