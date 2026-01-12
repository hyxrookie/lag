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


class SpatialAttention(nn.Module):
    """
    空间注意力机制：我机(Index 0)作为Query，关注所有实体
    """
    def __init__(self, embed_dim=128, num_heads=8, dropout=0.1):
        super(SpatialAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 多头注意力投影层
        # w_q 只处理我机，w_k, w_v 处理所有实体
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, entity_embeddings, mask=None):
        """
        Args:
            entity_embeddings: [..., num_entities, embed_dim]
                               其中 Index 0 是我机
            mask: [..., num_entities] (可选)
                  1表示有效实体，0表示无效/填充实体

        Returns:
            output: [..., embed_dim] (聚合后的特征，已去除实体维度)
        """
        # 获取维度信息
        # batch_dims 保存前面的所有维度 (L*B 或 T, B 等)
        # 即使不知道具体有几维，view(*batch_dims) 也能工作
        batch_dims = entity_embeddings.shape[:-2]
        num_entities = entity_embeddings.shape[-2]

        # 1. 生成 Q, K, V
        # Query: 只取我机 [..., 0:1, dim] -> [..., 1, dim]
        # 注意：使用切片 0:1 而不是索引 0，是为了保持维度，方便后续计算
        ego_feature = entity_embeddings[..., 0:1, :]
        Q = self.w_q(ego_feature)

        # Key, Value: 所有实体 [..., N, dim]
        K = self.w_k(entity_embeddings)
        V = self.w_v(entity_embeddings)

        # 2. 拆分为多头 (Multi-head Split)
        # 变换: [..., Length, Dim] -> [..., Length, Heads, Head_Dim] -> [..., Heads, Length, Head_Dim]
        # view(*batch_dims, ...) 自动适配任意前置维度
        Q = Q.view(*batch_dims, 1, self.num_heads, self.head_dim).transpose(-3, -2)
        K = K.view(*batch_dims, num_entities, self.num_heads, self.head_dim).transpose(-3, -2)
        V = V.view(*batch_dims, num_entities, self.num_heads, self.head_dim).transpose(-3, -2)

        # 当前形状:
        # Q: [..., num_heads, 1, head_dim]
        # K: [..., num_heads, num_entities, head_dim]

        # 3. 计算注意力分数 (Scaled Dot-Product Attention)
        # Matmul: (..., 1, D) @ (..., D, N) -> (..., 1, N)
        # 这里的 ... 包含 batch_dims 和 num_heads，会自动并行处理
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 4. 应用 Mask (如果提供)
        if mask is not None:
            # mask 输入形状: [..., num_entities]
            # 我们需要将其扩展为: [..., 1, 1, num_entities] 以匹配 scores 的形状
            # [..., num_heads, 1, num_entities]

            # unsqueeze(-2) 增加 Query 维度 (1)
            # unsqueeze(-2) 增加 Head 维度 (1) - 利用广播机制适配 num_heads
            mask_expanded = mask.unsqueeze(-2).unsqueeze(-2)

            # 将 mask 为 0 的位置填为负无穷
            scores = scores.masked_fill(mask_expanded == 0, float('-inf'))

        # 5. Softmax & Dropout
        # 在最后一个维度 (实体维度) 进行归一化
        attn_weights = torch.softmax(scores, dim=-1) # [..., num_heads, 1, num_entities]
        attn_weights = self.dropout(attn_weights)

        # 6. 加权求和
        # (..., 1, N) @ (..., N, D) -> (..., 1, D)
        attended = torch.matmul(attn_weights, V) # [..., num_heads, 1, head_dim]

        # 7. 合并多头
        # transpose: [..., 1, num_heads, head_dim]
        attended = attended.transpose(-3, -2).contiguous()
        # flatten: [..., 1, embed_dim]
        attended = attended.view(*batch_dims, 1, self.embed_dim)

        # 8. 输出投影
        output = self.out_proj(attended)
        output = self.dropout(output)

        # 9. 残差连接 & LayerNorm
        # 注意：残差加的是原始的 Ego 特征，保证不丢失自我信息
        output = self.layer_norm(output + ego_feature)

        # 10. 降维
        # 去掉中间的 1: [..., 1, embed_dim] -> [..., embed_dim]
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