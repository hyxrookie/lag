import torch
import torch.nn as nn
from .entity_embedding import EntityEmbedding, EntityEmbeddingCritic
from .spatial_attention import SpatialAttention
from .gtrxl import GTrXL  # 假设上面的GTrXL代码保存在这里

class SpatialTemporalBase(nn.Module):
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
        
        super(SpatialTemporalBase, self).__init__()
        self.is_critic = is_critic
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.memory_length = memory_length
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
        if is_critic:
            assert num_agents is not None, "Critic 模式必须指定 num_agents"
            self.entity_embed = EntityEmbeddingCritic(
                obs_dim=obs_dim,
                num_agents=num_agents,
                embed_dim=embed_dim,
                activation_id=activation_id,
                use_type_embedding=use_type_embedding
            )
        else:
            self.entity_embed = EntityEmbedding(
                ego_dim=ego_dim,
                relative_dim=relative_dim,
                embed_dim=embed_dim,
                activation_id=activation_id,
                use_type_embedding=use_type_embedding,
                num_missiles=num_missiles
            )
            
        # 4. 空间注意力 (Spatial Attention)
        if is_critic:
            # Critic 处理 share_obs，通常不需要 Ego-centric 视角
            # 这里简单使用 Identity，后续通过 mean pooling 聚合
            # 如果需要更强能力，可以用普通的 nn.MultiheadAttention
            self.spatial_attn = nn.Identity()
        else:
            # Actor 使用 Ego-centric 空间注意力
            self.spatial_attn = SpatialAttention(
                embed_dim=embed_dim,
                num_heads=num_spatial_heads,
                dropout=dropout
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

    def forward(self, obs, rnn_states, masks, available_actions=None):
        """
        Args:
            obs: [L*B, obs_dim]  (支持任意 N 行)
            rnn_states: [Layers, B, Mem_Len, Hidden] (GTrXL 的 Memory)
            masks: [L*B, 1]
            available_actions: 预留接口，暂未使用
        
        Returns:
            features: [L*B, hidden_size]
            new_rnn_states: [Layers, B, Mem_Len, Hidden]
        """
        # 1. 预处理
        if self.feature_norm is not None:
            obs = self.feature_norm(obs)
            
        # 2. 实体嵌入
        # Input: [N, obs_dim]
        # Output: [N, num_entities, embed_dim]
        valid_masks = None
        if self.is_critic:
            x_embed = self.entity_embed(obs)
        else:
            x_embed, valid_masks = self.entity_embed(
                obs,
                num_friendly=self.num_friendly,
                num_enemy=self.num_enemy,
                num_missiles=self.num_missiles
            )
            
        # 3. 空间注意力
        # Actor Output: [N, embed_dim] (已聚合)
        # Critic Output: [N, num_agents, embed_dim] (需聚合)
        x_spatial = self.spatial_attn(x_embed, valid_masks)
        
        if self.is_critic:
            # Critic 聚合: [N, num_agents, dim] -> [N, dim]
            # 这里使用 Mean Pooling
            x_spatial = x_spatial.mean(dim=1)

        # ============================================================
        # 【关键修改点 1】：拆包 (Flattened -> Matrix)
        # ============================================================
        # rnn_states 原始维度: [Layers, B, Mem_Len * Embed_Dim]
        batch_size, n_layers, flat_dim = rnn_states.shape

        # 1. Reshape 恢复 Memory 维度
        # [Batch, Layers, Flat_Dim] -> [Batch, Layers, Mem_Len, Hidden]
        rnn_states_view = rnn_states.view(batch_size, n_layers, self.memory_length, self.embed_dim)

        # 2. Permute 置换维度以适应 GTrXL 内部逻辑
        # GTrXL 的 forward 通常期待 hxs 为 [Layers, Batch, Mem_Len, Hidden]
        # 这样 forward 里的 `layer_mem = hxs[i]` 才能正确取到第 i 层的 memory
        rnn_states_view = rnn_states_view.permute(1, 0, 2, 3)

        # ------------------------------------------------------------
        # 4. 时间注意力 (GTrXL)
        # ------------------------------------------------------------
        # Input x_spatial: [N, embed_dim]
        # Input rnn_states: [Layers, Batch, Mem, Hidden] (经过 permute)
        # Output new_rnn_states: [Layers, Batch, Mem, Hidden] (通常 stack 也是层优先)
        features, new_rnn_states = self.temporal_attn(x_spatial, rnn_states_view, masks)

        # ============================================================
        # 【关键修改点 2】：输出还原 (Layers First -> Batch First)
        # ============================================================
        # new_rnn_states 目前是: [Layers, Batch, Mem_Len, Embed_Dim]

        # 1. Permute 回来：变回 [Batch, Layers, Mem_Len, Embed_Dim]
        new_rnn_states = new_rnn_states.permute(1, 0, 2, 3)
        # 2. Flatten 压扁：变回 [Batch, Layers, Flat_Dim] 存入 Buffer
        new_rnn_states_flat = new_rnn_states.flatten(2, 3)

        return features, new_rnn_states_flat

    @property
    def output_size(self):
        return self.hidden_size