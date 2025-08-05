# ----- 1. 位置编码（标准实现） -----
import math

from torch import nn

import torch
import torch.nn as nn
from .utils import check
from typing import List

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, device=torch.device("cpu")):
        super().__init__()
        pe = torch.zeros(max_len, d_model, device=device) # (max_len, D)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float, device=device)  # ← 指定 device
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(1)  # (max_len, 1, D)

    def forward(self, x):
        # x: (L, N, D)

        return x + self.pe[:x.size(0)]

# ----- 2. TransformerEncoder -----
class SimpleTransformer(nn.Module):
    def __init__(self, args, input_dim, device=torch.device("cpu")):
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.n_head = args.transformer_n_head
        self.n_layer = args.transformer_n_layer
        self.d_model = input_dim
        self.dropout = args.transformer_dropout
        self.num_layers = args.transformer_n_layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=self.d_model * 4,
            batch_first=False  # 因为输入是 (L, N, D)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.pos_encoder = PositionalEncoding(self.d_model, device=device)

    @property
    def output_size(self):
        return self.d_model

    def forward(self, src, src_key_padding_mask=None):
        """
        前向传播。
        src: 输入序列, 形状为 (L, N, D) -> (8, 1200, 128)
        src_done_mask: 标志 Episode 结束的遮罩, 形状为 (L, N) -> (8, 1200)
                       约定：1 = 继续, 0 = 终止/结束。
        """
        # 假设 src 已经是正确的 tensor
        # src = check(src).to(**self.tpdv)
        src = src.to(**self.tpdv)

        # 1. 添加位置编码
        src = self.pos_encoder(src)
        L = src.size(0)
        N = src.size(1)
        attn_mask = None
        if src_key_padding_mask is not None:
            # 确保遮罩在正确的设备上
            src_done_mask = src_key_padding_mask.to(device=src.device)
            # 2. 构建批处理的块状因果遮罩
            attn_mask_NLL = self.build_batch_block_causal_mask(src_done_mask) # 得到 (N, L, L)
            # 手动将遮罩扩展 (N, L, L) -> (N * n_head, L, L)
            # 1. 增加一个维度: (N, L, L) -> (N, 1, L, L)
            # 2. 沿着新维度复制 n_head 次: -> (N, n_head, L, L)
            # 3. 合并前两个维度: -> (N * n_head, L, L)
            attn_mask = attn_mask_NLL.unsqueeze(1).repeat(1, self.n_head, 1, 1)
            attn_mask = attn_mask.view(N * self.n_head, L, L)

        else:
            # 如果没有提供 done_mask，默认使用标准的因果遮罩，防止看到未来
            L = src.size(0)
            attn_mask = nn.Transformer.generate_square_subsequent_mask(L, device=src.device)

        # 3. 将数据送入 Transformer 编码器
        # attn_mask 的形状是 (N, L, L) 或 (L, L)，符合 PyTorch 的要求
        output = self.encoder(src, mask=attn_mask)

        if torch.isnan(output).any():
            print("警告：Transformer 输出中出现 NaN！")

        return output

    def build_batch_block_causal_mask(self, done_mask_LN: torch.Tensor) -> torch.Tensor:
        """
        为批处理数据构建块状因果遮罩。

        参数:
            done_mask_LN: 形状为 (L, N) 的张量。1 表示继续, 0 表示 episode 终止。

        返回:
            一个形状为 (N, L, L) 的注意力遮罩。
            其中 -inf 表示不能关注，0 表示可以关注。
        """
        L, N = done_mask_LN.shape

        # 1. 创建一个标准的下三角因果遮罩 (L, L)
        #    值为 0 表示可以关注, -inf 表示不可以关注
        causal_mask = torch.triu(
            torch.full((L, L), float("-inf"), device=done_mask_LN.device),
            diagonal=1
        )

        # 2. 找到每个序列中新 episode 的起始点
        #    一个新 episode 在 t=0 时开始，或者在 t-1 时刻 done_mask 为 0 时开始
        is_new_episode = torch.zeros_like(done_mask_LN, dtype=torch.bool)
        is_new_episode[0, :] = True
        is_new_episode[1:, :] = (done_mask_LN[:-1, :] == 0)

        # 3. 为每个 (时间步, 批次) 分配一个 episode ID
        #    通过累加 is_new_episode 来实现。同一 episode 内的 ID 相同。
        #    形状: (L, N)
        episode_ids = torch.cumsum(is_new_episode.long(), dim=0)

        # 4. 创建 episode 隔离遮罩
        #    只有当查询 (query) 和键 (key) 的 episode ID 相同时, 才允许关注。
        #    使用广播机制进行高效比较：
        #    (L, 1, N) vs (1, L, N) -> (L, L, N)
        query_episode_ids = episode_ids.unsqueeze(1)
        key_episode_ids = episode_ids.unsqueeze(0)
        same_episode_mask = (query_episode_ids == key_episode_ids)

        # 将形状从 (L, L, N) 转换为 (N, L, L) 以匹配 Transformer 的要求
        same_episode_mask_NLL = same_episode_mask.permute(2, 0, 1)

        # 5. 合并遮罩
        #    最终的遮罩是因果遮罩和 episode 隔离遮罩的组合
        #    扩展因果遮罩以匹配批处理维度 (1, L, L) -> (N, L, L)
        final_mask = causal_mask.unsqueeze(0).expand(N, -1, -1).clone()

        # 在不属于同一 episode 的位置填充 -inf
        final_mask[~same_episode_mask_NLL] = float("-inf")

        return final_mask  # 形状: (N, L, L)