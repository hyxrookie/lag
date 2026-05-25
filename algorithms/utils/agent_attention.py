import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentLevelAttention(nn.Module):
    """
    极速版智能体级注意力 (Fast Agent-Level Attention)
    完全摒弃 nn.MultiheadAttention 的沉重 wrapper，直接调用底层 SDPA
    """

    def __init__(self, embed_dim=128, num_heads=2, dropout=0.1):
        super(AgentLevelAttention, self).__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.layer_norm = nn.LayerNorm(embed_dim)

        # 用一个线性层同时算出 Q, K, V，极大减少 GPU Kernel 调用次数
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, agent_features, agent_mask=None, agent_index=None):
        B, N, E = agent_features.shape

        # 1. Pre-Norm
        normed_features = self.layer_norm(agent_features)

        # 2. 一次性计算 Q, K, V 并拆分多头
        # [B, N, 3*E] -> [B, N, 3, num_heads, head_dim]
        qkv = self.qkv_proj(normed_features).view(B, N, 3, self.num_heads, self.head_dim)

        # 置换维度以适应 SDPA: [3, B, num_heads, N, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # 3. 处理精准打击 (Agent Index)
        if agent_index is not None:
            # 切片提取当前智能体的 Query: [B, num_heads, 1, head_dim]
            Q = Q[:, :, agent_index:agent_index + 1, :]
            q_len = 1
        else:
            q_len = N

        # 4. 构建极速版布尔掩码 (SDPA 规则：True 为保留，False 为屏蔽)
        if agent_mask is not None:
            # 原本的 mask: 1为存活，0为阵亡。转化为布尔值
            attn_mask = (agent_mask == 1).bool()

            # 扩展维度进行广播: [B, 1, 1, N]
            # (适配 [Batch, Heads, Query_Len, Key_Len])
            attn_mask = attn_mask.view(B, 1, 1, N)

            # 【防 NaN 补丁】：如果全军覆没，强制全设为 True 避免全 False 报错
            all_dead = (~attn_mask).all(dim=-1, keepdim=True)
            attn_mask = attn_mask.masked_fill(all_dead, True)
        else:
            attn_mask = None

        # 5. 调用底层 C++ 极速算子 (FlashAttention)
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0
        )

        # 6. 还原张量形状
        # [B, num_heads, q_len, head_dim] -> [B, q_len, E]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, q_len, E)
        attn_out = self.out_proj(attn_out)

        # 7. 残差连接与输出融合
        if agent_index is not None:
            # 提取对应的原始特征做残差
            original_q = normed_features[:, agent_index:agent_index + 1, :]
            output = original_q + attn_out
            return output.squeeze(1)  # 返回 [B, E]
        else:
            output = normed_features + attn_out

            # 全局模式的 Masked Mean Pooling
            if agent_mask is not None:
                mask_float = agent_mask.unsqueeze(-1).float()
                sum_output = (output * mask_float).sum(dim=1)
                valid_count = mask_float.sum(dim=1).clamp(min=1e-9)
                return sum_output / valid_count
            else:
                return output.mean(dim=1)