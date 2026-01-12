import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 预计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1)) # [Max_Len, 1, Dim]

    def forward(self, x):
        # x: [T, B, D]
        # 自动切片匹配 T
        return x + self.pe[:x.size(0)]

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
        
        # Gating (GRU-like gating)
        self.gate1_w = nn.Linear(embed_dim, embed_dim)
        self.gate1_b = nn.Parameter(torch.zeros(embed_dim))
        self.gate2_w = nn.Linear(embed_dim, embed_dim)
        self.gate2_b = nn.Parameter(torch.zeros(embed_dim))

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

    def forward(self, x, memory, attn_mask):
        """
        x: [T, B, D]
        memory: [B, Mem_Len, D]  <-- 注意这里输入前会调整为 [Mem_Len, B, D] 以便拼接
        attn_mask: [B*Heads, T, Mem+T] 或 [B, T, Mem+T]
        """
        T, B, D = x.shape
        
        # 1. 拼接 Memory (Key/Value 的来源)
        # memory: [B, Mem, D] -> [Mem, B, D]
        if memory is not None and memory.size(1) > 0:
            mem_transposed = memory.transpose(0, 1) # [Mem, B, D]
            kv_input = torch.cat([mem_transposed, x], dim=0) # [Mem+T, B, D]
        else:
            kv_input = x
            
        # 2. Attention
        residual = x
        x_norm = self.ln1(x)
        kv_norm = self.ln1(kv_input)
        
        Q = self.q_proj(x_norm)      # [T, B, D]
        K = self.k_proj(kv_norm)     # [Mem+T, B, D]
        V = self.v_proj(kv_norm)     # [Mem+T, B, D]
        
        # Multi-head reshape: [T, B, Heads, D_h] -> [B, Heads, T, D_h]
        Q = Q.view(T, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        K = K.view(-1, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        V = V.view(-1, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        
        # Score: (B, Heads, T, D_h) @ (B, Heads, D_h, M+T) -> (B, Heads, T, M+T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Flatten for masking: [B*Heads, T, M+T]
        scores = scores.flatten(0, 1)
        
        # Apply Mask
        if attn_mask is not None:
            # 广播 mask 以匹配 scores
            # attn_mask 可能是 [B, T, M+T]，需要 repeat 给每个 head
            if attn_mask.size(0) == B:
                attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted Sum
        V = V.flatten(0, 1) # [B*Heads, M+T, D_h]
        attn_out = torch.matmul(attn_weights, V) # [B*Heads, T, D_h]
        
        # Restore: [B*Heads, T, D_h] -> [T, B, D]
        attn_out = attn_out.view(B, self.num_heads, T, self.head_dim).permute(2, 0, 1, 3).flatten(2, 3)
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout(attn_out)
        
        # Gating 1
        x = self.gating(residual, attn_out)
        
        # 3. FFN
        residual = x
        x_norm = self.ln2(x)
        ff_out = self.ff(x_norm)
        
        # Gating 2
        x = self.gating(residual, ff_out)
        
        # 4. 生成新 Memory (detach!)
        # 新的 memory 是本次的输入 (kv_input的后半部分，即x)
        # 注意：这里我们返回 x 而不是 x_norm，GTrXL 论文通常将 transform 后的层输出作为下一层的 memory，
        # 但也有实现直接滑动 kv_input。这里为了稳定性，我们把当前层的**输出**作为下一时刻该层的 Memory。
        new_memory = x.detach().transpose(0, 1) # [T, B, D] -> [B, T, D]
        
        return x, new_memory


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GTrXL(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads=4, memory_len=64):
        super(GTrXL, self).__init__()
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._memory_len = memory_len

        # 输入维度对齐
        self.input_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        self.pos_encoder = PositionalEncoding(hidden_size)
        self.blocks = nn.ModuleList([
            GTrXLBlock(hidden_size, num_heads, hidden_size * 4, dropout=0.0)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, hxs: torch.Tensor, masks: torch.Tensor):
        """
        Args:
            x: [B, input_size] (当L=1)  或  [L*B, input_size] (当L>1)
            hxs: [Layers, B, Mem_Len, Hidden]
            masks: [B, 1] (当L=1)       或  [L*B, 1] (当L>1)
        """
        # ==============================================================
        # 1. 维度推导 (核心修改)
        # ==============================================================
        # 我们从 hxs 中获取真实的 Batch Size，这是最可靠的
        # hxs: [Layers, Batch, Mem_Len, Hidden]
        B = hxs.size(1)

        # 获取输入数据的总行数 N
        # 如果 L=1, N = B
        # 如果 L>1, N = L * B
        N = x.size(0)

        # 计算时间步长 T
        T = int(N / B)

        # ==============================================================
        # 2. 统一 View 成 [T, B, ...] 方便后续处理
        # ==============================================================

        # 处理 x
        # [N, Dim] -> [T, B, Dim]
        x = x.view(T, B, -1)

        # 处理 masks
        # [N, 1] -> [T, B]
        # 如果输入是 [B, 1] (L=1)，这里变成了 [1, B]
        # 如果输入是 [L*B, 1] (L>1)，这里变成了 [L, B]
        masks = masks.view(T, B)

        # 投影和位置编码
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        new_hxs_list = []

        # ================= T=1 (推理/单步) =================
        if T == 1:
            # 这里的 mask 表示上一帧到这一帧是否 done
            # masks 现在是 [1, B]。转为 [B, 1, 1] 用于广播乘法
            # 如果 masks=0，说明刚 reset，历史记忆应该清空
            mask_reset = masks.transpose(0, 1).unsqueeze(2)  # [1, B] -> [B, 1] -> [B, 1, 1]

            attn_mask = None

            out = x
            for i, block in enumerate(self.blocks):
                layer_mem = hxs[i]  # [B, Mem, D]

                # 应用 Reset Mask: 如果 done，memory 变全 0
                layer_mem = layer_mem * mask_reset

                out, current_out_as_mem = block(out, layer_mem, attn_mask)

                # 更新 Memory
                cat_mem = torch.cat([layer_mem, current_out_as_mem], dim=1)
                new_mem = cat_mem[:, -self._memory_len:, :]
                new_hxs_list.append(new_mem)

        # ================= T>1 (训练/序列) =================
        else:
            # 1. Causal Mask
            causal_mask = torch.tril(torch.ones(T, T, device=x.device))

            # 2. Episode Mask
            dones = 1.0 - masks  # [T, B]
            episode_ids = torch.cumsum(dones, dim=0)  # [T, B]

            id_q = episode_ids.permute(1, 0).unsqueeze(2)  # [B, T, 1]
            id_k = episode_ids.permute(1, 0).unsqueeze(1)  # [B, 1, T]
            term_mask = (id_q == id_k).float()

            seq_mask = term_mask * causal_mask.unsqueeze(0)  # [B, T, T]

            # 3. Memory Mask (是否能看 Memory)
            mem_len = hxs.size(2)
            if mem_len > 0:
                # 只有 masks[0] == 1 (Chunk开头没死) 且当前 t 和 0 同 ID 时，才能看 Memory
                mem_valid = masks[0].view(B, 1, 1)  # [B, 1, 1]

                # id[t] == id[0]
                id_0 = episode_ids[0].view(B, 1, 1)
                t_same_as_0 = (id_q == id_0).float()  # [B, T, 1]

                mem_mask_t = mem_valid * t_same_as_0
                mem_mask = mem_mask_t.repeat(1, 1, mem_len)  # [B, T, Mem]

                full_mask = torch.cat([mem_mask, seq_mask], dim=2)
            else:
                full_mask = seq_mask

            out = x
            for i, block in enumerate(self.blocks):
                layer_mem = hxs[i]
                out, current_out_as_mem = block(out, layer_mem, full_mask)

                cat_mem = torch.cat([layer_mem, current_out_as_mem], dim=1)
                new_mem = cat_mem[:, -self._memory_len:, :]
                new_hxs_list.append(new_mem)

        # 3. 整理输出
        out = self.norm(out)

        # 还原回输入的形式 [N, D]
        # 如果是 L=1, N=B, [1, B, D] -> [B, D]
        # 如果是 L>1, N=L*B, [T, B, D] -> [T*B, D]
        out = out.view(N, -1)

        new_hxs = torch.stack(new_hxs_list, dim=0)

        return out, new_hxs

    @property
    def output_size(self):
        return self._hidden_size