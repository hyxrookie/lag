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
    def forward(self, src, src_key_padding_mask = None ):

        src = check(src).to(**self.tpdv)
        # src: (L, N, D)
        src = self.pos_encoder(src)
        attn_mask = None
            # 确保掩码是布尔类型并且在正确的设备上
        if src_key_padding_mask is not None:
            src_key_padding_mask = check(src_key_padding_mask).to(device=src.device, dtype=torch.bool)
            print("src_key_padding_mask shape", src_key_padding_mask.shape)
            # 1) Episode 边界
            bounds = self.episode_boundaries_TN(src_key_padding_mask)

            # 2) Block-Causal Mask
            attn_mask = self.build_block_causal_mask(bounds,
                                                device=src.device,
                                                dtype=src.dtype)
            # key_pad_mask = (masks_TN.transpose(0, 1) == 0)  # True=被遮掉

        # 生成 causal mask, 防止看到未来
        L = src.size(0)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=src.device)

        output = self.encoder(src, mask = attn_mask)

        if torch.isnan(output).any():
            print("警告：Transformer 输出中出现 NaN！")
            # 在这里可以设置断点进行调试
        return output  # (L, N, D)

    def episode_boundaries_TN(self, masks_TN: torch.Tensor) -> List[int]:
        """
        masks_TN : (T, N)，1=继续，0=终止
        返回     : [0, b1, b2, ..., T]
        """
        T, _ = masks_TN.shape
        # 看 t=1..T-1 哪些帧出现任何 agent 终止
        has_zeros = ((masks_TN[1:] == 0.0)
                     .any(dim=-1)  # 沿 batch 维
                     .nonzero(as_tuple=False)
                     .squeeze(-1)
                     .cpu())
        return [0] + (has_zeros + 1).tolist() + [T]
    def build_block_causal_mask(self, boundaries: List[int],
                                device, dtype=torch.float32) -> torch.Tensor:
        """
        生成 (L, L) 的 attn_mask，满足：
        - 因果性：token 只能看过去
        - 不同 Episode 之间完全互不可见
        """
        L = boundaries[-1]
        # 先做普通因果
        mask = torch.triu(torch.full((L, L), float("-inf"),
                                     device=device, dtype=dtype),
                          diagonal=1)

        # 再把跨 Episode 的位置全部置 -inf
        for s, e in zip(boundaries[:-1], boundaries[1:]):
            mask[s:e, :s] = float("-inf")
        return mask        # (L, L)
