#
# 文件: algorithms/utils/transformer.py
# 描述: 定义一个因果Transformer编码器层，用于PPO策略网络。
#
import torch
import torch.nn as nn
from .utils import check


class CausalTransformerEncoder(nn.Module):
    """
    A Causal Transformer Encoder module.
    It ensures that the output for a given timestep does not depend on future timesteps.
    """

    def __init__(self, args, input_dim, device=torch.device("cpu")):
        super(CausalTransformerEncoder, self).__init__()

        self.tpdv = dict(dtype=torch.float32, device=device)

        # Transformer parameters from args
        self.n_head = args.transformer_n_head
        self.n_layer = args.transformer_n_layer
        self.d_model = input_dim  # The input and output dimension of the transformer
        self.dropout = args.transformer_dropout

        # LayerNorm for the input
        self.pre_transformer_ln = nn.LayerNorm(self.d_model)

        # Create the Transformer Encoder Layer with causality
        # is_causal=True automatically creates the attention mask to prevent attending to future tokens.
        # batch_first=False expects input shape: (sequence_length, batch_size, feature_dim)
        try:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.n_head,
                dim_feedforward=self.d_model * 4,  # A common practice
                dropout=self.dropout,
                activation='relu',
                batch_first=False  # This is the key parameter for causality
            )
        except TypeError:
            # Fallback for older PyTorch versions that do not support is_causal
            # NOTE: This implementation requires manually creating the mask in forward pass.
            # For simplicity in this example, we will assume a modern PyTorch version.
            # If needed, the manual mask creation can be added here.
            # For now, we'll just raise a more informative error.
            raise ImportError("Your PyTorch version might not support `is_causal=True`. "
                              "Please upgrade PyTorch or implement manual causal masking.")

        # Stack the encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layer
        )

    @property
    def output_size(self):
        return self.d_model

    # def forward(self, src, src_key_padding_mask=None):
    #     """
    #     Args:
    #         src (torch.Tensor): The sequence to the encoder.
    #             Shape: (sequence_length, batch_size, d_model).
    #         src_key_padding_mask (torch.Tensor): The mask for the src keys.
    #             Shape: (batch_size, sequence_length).
    #             True values indicate positions that should be ignored.
    #     Returns:
    #         torch.Tensor: The encoded sequence.
    #             Shape: (sequence_length, batch_size, d_model).
    #     """
    #     src = check(src).to(**self.tpdv)
    #     if src_key_padding_mask is not None:
    #         src_key_padding_mask = check(src_key_padding_mask).to(device=self.tpdv['device'], dtype=torch.bool)
    #
    #     # Apply LayerNorm before passing to the transformer
    #     src = self.pre_transformer_ln(src)
    #
    #     seq_len = src.size(0)
    #     causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, src.device, dtype=torch.bool)
    #
    #     # The nn.TransformerEncoder with is_causal=True handles the causal mask internally.
    #     # We only need to provide the padding mask.
    #     output = self.transformer_encoder(src,mask =causal_mask, src_key_padding_mask=src_key_padding_mask, is_causal=True)
    #
    #     return output
    def forward(self, src, src_key_padding_mask=None):
        """
        src: (S,B,D)  或 (B,S,D) 任选，只要前后一致
        src_key_padding_mask: (B,S) bool  True=padding
        """
        src = check(src).to(**self.tpdv)

        if src_key_padding_mask is not None:
            src_key_padding_mask = (
                check(src_key_padding_mask)
                .to(device=self.tpdv["device"], dtype=torch.bool)
            )

        # ---- Pre‑LN ----------
        src = self.pre_transformer_ln(src)

        # ---- 因果掩码 (float32, 0 / -1e9) ----------
        T = src.shape[1] if self.batch_first else src.shape[0]
        causal_mask = torch.triu(
            torch.full((T, T), float("-1e9"), device=src.device),
            diagonal=1
        )

        # ---- padding 掩码 float ----------
        pad_mask_float = None
        if src_key_padding_mask is not None:
            pad_mask_float = src_key_padding_mask.float().masked_fill(
                src_key_padding_mask, float("-1e9")
            )

        # ---- Transformer ----------
        out = self.transformer_encoder(
            src,
            mask=causal_mask,
            src_key_padding_mask=pad_mask_float
        )

        # ---- 清零 padding 行，避免污染后续 ----
        if src_key_padding_mask is not None:
            if out.shape[0] == src_key_padding_mask.shape[0]:  # batch_first
                out = out.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
            else:  # seq_first
                out = out.masked_fill(src_key_padding_mask.t().unsqueeze(-1), 0.0)

        # ---- 调试期保险 ----
        assert torch.isfinite(out).all(), "NaN/Inf detected post‑Transformer"

        return out
