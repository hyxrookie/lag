"""
空间注意力机制模块
修改说明：
1. 移除对 Batch/Sequence 的显式 reshape，支持任意前置维度 [..., num_entities, dim]
2. 逻辑改为：我机 (Index 0) 作为 Query，关注所有实体 (Keys/Values)
3. 输出自动降维为 [..., embed_dim]
"""
import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import math


class SpatialAttention(nn.Module):
    """
    空间注意力机制：我机(Index 0)作为Query，关注所有实体
    采用 Pre-Norm 架构
    """

    def __init__(self, embed_dim=128, num_heads=8, dropout=0.1):
        super(SpatialAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Pre-Norm 的 LayerNorm 层
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, entity_embeddings, mask=None):
        batch_dims = entity_embeddings.shape[:-2]
        num_entities = entity_embeddings.shape[-2]

        # ==========================================
        # Pre-Norm 核心修改点 1：先进行 LayerNorm
        # ==========================================
        # 1. 保存原始的“我机”特征，用于最后的残差连接 (不经过LayerNorm)
        ego_feature_raw = entity_embeddings[..., 0:1, :]

        # 2. 对所有输入实体进行归一化
        normed_embeddings = self.layer_norm(entity_embeddings)

        # 3. 基于归一化后的特征生成 Q, K, V
        # Query: 取归一化后的我机特征
        ego_feature_norm = normed_embeddings[..., 0:1, :]
        Q = self.w_q(ego_feature_norm)

        # Key, Value: 取归一化后的所有实体特征
        K = self.w_k(normed_embeddings)
        V = self.w_v(normed_embeddings)

        # ==========================================
        # 后续注意力计算逻辑保持不变
        # ==========================================
        Q = Q.view(*batch_dims, 1, self.num_heads, self.head_dim).transpose(-3, -2)
        K = K.view(*batch_dims, num_entities, self.num_heads, self.head_dim).transpose(-3, -2)
        V = V.view(*batch_dims, num_entities, self.num_heads, self.head_dim).transpose(-3, -2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # 建议加上这个安全补丁，防止我机被意外 mask 导致全 -inf 报错 NaN
            mask = mask.clone()  # 避免原地修改影响外部数据
            mask[..., 0] = 1.0

            mask_expanded = mask.unsqueeze(-2).unsqueeze(-2)
            scores = scores.masked_fill(mask_expanded == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attended = torch.matmul(attn_weights, V)

        attended = attended.transpose(-3, -2).contiguous()
        attended = attended.view(*batch_dims, 1, self.embed_dim)

        output = self.out_proj(attended)
        output = self.dropout(output)

        # ==========================================
        # Pre-Norm 核心修改点 2：纯残差连接
        # ==========================================
        # 直接与未归一化的原始 ego_feature_raw 相加，外部不再包裹 LayerNorm
        output = output + ego_feature_raw

        return output.squeeze(-2)

    # def forward_critic(self, entity_embeddings, mask=None):
    #     """
    #     Critic 模式：所有实体作为Query (Self-Attention)，最后取平均
    #     用于评估全局状态 value
    #
    #     Args:
    #         entity_embeddings: [..., num_entities, embed_dim]
    #         mask: [..., num_entities] (0表示无效实体，计算平均时需排除)
    #
    #     Returns:
    #         output: [..., embed_dim] (全局平均后的特征)
    #     """
    #     batch_dims = entity_embeddings.shape[:-2]
    #     num_entities = entity_embeddings.shape[-2]
    #
    #     # 1. 生成 Q, K, V
    #     # Critic 区别点：Q 是所有实体，而不仅仅是 Index 0
    #     Q = self.w_q(entity_embeddings)  # [..., num_entities, dim]
    #     K = self.w_k(entity_embeddings)  # [..., num_entities, dim]
    #     V = self.w_v(entity_embeddings)  # [..., num_entities, dim]
    #
    #     # 2. 拆分多头
    #     # shape: [..., Heads, Num_Entities, Head_Dim]
    #     Q = Q.view(*batch_dims, num_entities, self.num_heads, self.head_dim).transpose(-3, -2)
    #     K = K.view(*batch_dims, num_entities, self.num_heads, self.head_dim).transpose(-3, -2)
    #     V = V.view(*batch_dims, num_entities, self.num_heads, self.head_dim).transpose(-3, -2)
    #
    #     # 3. 计算注意力分数 (Standard Self-Attention)
    #     # (..., N, D) @ (..., D, N) -> (..., N, N)
    #     scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
    #
    #     # 4. 应用 Mask
    #     if mask is not None:
    #         # Mask 需要遮蔽 Key 维度的无效实体
    #         # mask: [..., N] -> [..., 1, 1, N]
    #         # 这里的 mask 逻辑是：任何 Query 都不应该关注无效的 Key
    #         mask_expanded = mask.unsqueeze(-2).unsqueeze(-2)
    #         scores = scores.masked_fill(mask_expanded == 0, float('-inf'))
    #
    #     # 5. Softmax & Dropout
    #     attn_weights = torch.softmax(scores, dim=-1)
    #     attn_weights = self.dropout(attn_weights)
    #
    #     # 6. 加权求和
    #     # (..., N, N) @ (..., N, D) -> (..., N, D)
    #     attended = torch.matmul(attn_weights, V)
    #
    #     # 7. 合并多头
    #     attended = attended.transpose(-3, -2).contiguous()
    #     attended = attended.view(*batch_dims, num_entities, self.embed_dim)
    #
    #     # 8. 输出投影
    #     output = self.out_proj(attended)
    #     output = self.dropout(output)
    #
    #     # 9. 残差连接 & LayerNorm
    #     # Critic 区别点：加的是所有实体的原始特征
    #     output = self.layer_norm(output + entity_embeddings)
    #     # 当前 shape: [..., num_entities, embed_dim]
    #
    #     # 10. 全局平均池化 (Global Average Pooling)
    #     # Critic 区别点：对所有实体取平均，代表当前战场的整体态势
    #     if mask is not None:
    #         # 如果有 mask，不能简单 mean，因为 padding 的部分会拉低均值
    #         # 扩展 mask: [..., N] -> [..., N, 1] 以匹配 embed_dim
    #         mask_for_avg = mask.unsqueeze(-1)
    #
    #         # 将无效实体置为 0
    #         masked_output = output * mask_for_avg
    #
    #         # 求和
    #         sum_output = masked_output.sum(dim=-2)  # [..., embed_dim]
    #
    #         # 计算有效实体数量 (避免除以0)
    #         valid_count = mask_for_avg.sum(dim=-2).clamp(min=1e-9)
    #
    #         return sum_output / valid_count
    #     else:
    #         # 没有 mask，直接对实体维度取平均
    #         return output.mean(dim=-2)  # [..., embed_dim]