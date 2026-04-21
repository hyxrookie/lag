import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =====================================================================
# [修改 1] 移除旧的绝对位置编码，引入旋转位置编码 (RoPE) 的核心逻辑
# =====================================================================
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """预计算 RoPE 的 cos 和 sin 频率"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    # 【关键修复】：将频率复制一份拼接，使其最后一个维度从 dim//2 变为 dim
    freqs = torch.cat([freqs, freqs], dim=-1)

    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    """应用 RoPE 到 Query 和 Key 上"""
    T = xq.size(2)
    M_plus_T = xk.size(2)
    M = M_plus_T - T

    # xq 对应的是当前时间步 [M : M+T]
    cos_q = freqs_cos[M:M + T].view(1, 1, T, -1).to(xq.device)
    sin_q = freqs_sin[M:M + T].view(1, 1, T, -1).to(xq.device)

    # xk 对应的是历史 + 当前时间步 [0 : M+T]
    cos_k = freqs_cos[:M + T].view(1, 1, M_plus_T, -1).to(xk.device)
    sin_k = freqs_sin[:M + T].view(1, 1, M_plus_T, -1).to(xk.device)

    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    xq_out = (xq * cos_q) + (rotate_half(xq) * sin_q)
    xk_out = (xk * cos_k) + (rotate_half(xk) * sin_k)
    return xq_out, xk_out


class GTrXLBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super(GTrXLBlock, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # =====================================================================
        # [修改 2] 门控初始化：偏置初始化为负数，使初始门控接近 0 (倾向走残差)
        # =====================================================================
        self.gate1_w = nn.Linear(embed_dim, embed_dim)
        self.gate1_b = nn.Parameter(torch.full((embed_dim,), -2.0))
        self.gate2_w = nn.Linear(embed_dim, embed_dim)
        self.gate2_b = nn.Parameter(torch.full((embed_dim,), -2.0))

        # Feed Forward
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def gating(self, x, y):
        """ GTrXL 核心门控: g * y + (1-g) * x """
        gate = torch.sigmoid(self.gate1_w(x) + self.gate2_w(y) + self.gate1_b)
        return gate * y + (1 - gate) * x

    def forward(self, x, memory, attn_mask, freqs_cos, freqs_sin):
        T, B, D = x.shape

        # =====================================================================
        # [修改 3] 缓存当前层的输入作为未来的 Memory，解决特征层级错位问题
        # =====================================================================
        input_x = x

        if memory is not None and memory.size(1) > 0:
            mem_transposed = memory.transpose(0, 1)  # [Mem, B, D]
            kv_input = torch.cat([mem_transposed, x], dim=0)  # [Mem+T, B, D]
        else:
            kv_input = x

        residual = x
        x_norm = self.ln1(x)
        kv_norm = self.ln1(kv_input)

        Q = self.q_proj(x_norm)  # [T, B, D]
        K = self.k_proj(kv_norm)  # [Mem+T, B, D]
        V = self.v_proj(kv_norm)  # [Mem+T, B, D]

        Q = Q.view(T, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        K = K.view(-1, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        V = V.view(-1, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        # =====================================================================
        # [修改 1 接续] 在计算 Attention Score 之前，对 Q 和 K 应用 RoPE
        # =====================================================================
        Q, K = apply_rotary_emb(Q, K, freqs_cos, freqs_sin)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.flatten(0, 1)

        if attn_mask is not None:
            if attn_mask.size(0) == B:
                attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        V = V.flatten(0, 1)
        attn_out = torch.matmul(attn_weights, V)

        attn_out = attn_out.view(B, self.num_heads, T, self.head_dim).permute(2, 0, 1, 3).flatten(2, 3)
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout(attn_out)

        x = self.gating(residual, attn_out)

        residual = x
        x_norm = self.ln2(x)
        ff_out = self.ff(x_norm)
        x = self.gating(residual, ff_out)

        # =====================================================================
        # [修改 3 接续] 返回 input_x 而不是处理后的 x
        # =====================================================================
        new_memory = input_x.detach().transpose(0, 1)  # [T, B, D] -> [B, T, D]

        return x, new_memory


class GTrXL(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads=4, memory_len=64):
        super(GTrXL, self).__init__()
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._memory_len = memory_len
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.input_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()

        self.blocks = nn.ModuleList([
            GTrXLBlock(hidden_size, num_heads, hidden_size * 4, dropout=0.0)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, hxs: torch.Tensor, masks: torch.Tensor):
        B = hxs.size(1)
        N = x.size(0)
        T = int(N / B)

        x = x.view(T, B, -1)
        masks = masks.view(T, B)

        x = self.input_proj(x)

        # =====================================================================
        # [修改 1 接续] 动态生成 RoPE 频率
        # =====================================================================
        mem_len = hxs.size(2)
        freqs_cos, freqs_sin = precompute_freqs_cis(self.head_dim, end=mem_len + T)

        new_hxs_list = []

        # ================= T=1 (推理/单步) =================
        if T == 1:
            mask_reset = masks.transpose(0, 1).unsqueeze(2)
            attn_mask = None

            out = x
            for i, block in enumerate(self.blocks):
                layer_mem = hxs[i]
                layer_mem = layer_mem * mask_reset

                # 传入 RoPE 频率
                out, current_out_as_mem = block(out, layer_mem, attn_mask, freqs_cos, freqs_sin)

                cat_mem = torch.cat([layer_mem, current_out_as_mem], dim=1)
                new_mem = cat_mem[:, -self._memory_len:, :]
                new_hxs_list.append(new_mem)

        # ================= T>1 (训练/序列) =================
        else:
            causal_mask = torch.tril(torch.ones(T, T, device=x.device))
            dones = 1.0 - masks
            episode_ids = torch.cumsum(dones, dim=0)

            id_q = episode_ids.permute(1, 0).unsqueeze(2)
            id_k = episode_ids.permute(1, 0).unsqueeze(1)
            term_mask = (id_q == id_k).float()
            seq_mask = term_mask * causal_mask.unsqueeze(0)

            if mem_len > 0:
                mem_valid = masks[0].view(B, 1, 1)
                id_0 = episode_ids[0].view(B, 1, 1)
                t_same_as_0 = (id_q == id_0).float()
                mem_mask_t = mem_valid * t_same_as_0
                mem_mask = mem_mask_t.repeat(1, 1, mem_len)

                full_mask = torch.cat([mem_mask, seq_mask], dim=2)
            else:
                full_mask = seq_mask

            # =====================================================================
            # [修改 4] 计算 valid_history_mask，清除已经结束的 Episode 的遗留记忆
            # =====================================================================
            prior_id = (episode_ids[0] - dones[0]).unsqueeze(1)  # [B, 1]
            mem_ids = prior_id.repeat(1, mem_len)  # [B, Mem]
            curr_ids = episode_ids.transpose(0, 1)  # [B, T]
            all_ids = torch.cat([mem_ids, curr_ids], dim=1)  # [B, Mem+T]

            last_id = episode_ids[-1].unsqueeze(1)  # [B, 1]
            valid_history_mask = (all_ids == last_id).float().unsqueeze(-1)  # [B, Mem+T, 1]

            out = x
            for i, block in enumerate(self.blocks):
                layer_mem = hxs[i]

                # 传入 RoPE 频率
                out, current_out_as_mem = block(out, layer_mem, full_mask, freqs_cos, freqs_sin)

                cat_mem = torch.cat([layer_mem, current_out_as_mem], dim=1)

                # 应用清理 Mask：属于上一个死掉 Episode 的记忆全部置 0
                cat_mem = cat_mem * valid_history_mask

                new_mem = cat_mem[:, -self._memory_len:, :]
                new_hxs_list.append(new_mem)

        out = self.norm(out)
        out = out.view(N, -1)
        new_hxs = torch.stack(new_hxs_list, dim=0)

        return out, new_hxs

    @property
    def output_size(self):
        return self._hidden_size