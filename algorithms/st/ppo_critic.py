import torch
import torch.nn as nn

from ..utils.gtrxl import GTrXL
from ..utils.mlp import MLPBase, MLPLayer
from ..utils.gru import GRULayer
from ..utils.utils import check


class PPOCriticST(nn.Module):
    def __init__(self, args, obs_space, device=torch.device("cpu")):
        super(PPOCriticST, self).__init__()
        # network config
        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.tpdv = dict(dtype=torch.float32, device=device)
        # (1) feature extraction module
        self.base = MLPBase(obs_space, self.hidden_size, self.activation_id, self.use_feature_normalization)
        # (2) rnn module 
        input_size = self.base.output_size
        # 5. 时间注意力 (GTrXL)
        self.embed_dim = getattr(args, 'embed_dim', 128)
        self.num_spatial_heads = getattr(args, 'num_spatial_heads', 8)
        self.num_temporal_heads = getattr(args, 'num_temporal_heads', 8)
        self.temporal_ff_dim = getattr(args, 'temporal_ff_dim', 512)
        self.num_temporal_layers = getattr(args, 'num_temporal_layers', 2)
        self.memory_length = getattr(args, 'memory_length', 64)

        self.temporal_attn = GTrXL(
            input_size=self.embed_dim,
            hidden_size=self.embed_dim, # 如果 embed_dim != hidden_size，GTrXL内部会投影
            num_layers=self.num_temporal_layers,
            num_heads=self.num_temporal_heads,
            memory_len=self.memory_length
        )
        input_size = self.embed_dim
        # (3) value module
        if len(self.act_hidden_size) > 0:
            self.mlp = MLPLayer(input_size, self.act_hidden_size, self.activation_id)
        self.value_out = nn.Linear(input_size, 1)

        self.to(device)

    def forward(self, obs, rnn_states, masks):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(obs)

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
        features, new_rnn_states = self.temporal_attn(critic_features, rnn_states_view, masks)

        # ============================================================
        # 【关键修改点 2】：输出还原 (Layers First -> Batch First)
        # ============================================================
        # new_rnn_states 目前是: [Layers, Batch, Mem_Len, Embed_Dim]

        # 1. Permute 回来：变回 [Batch, Layers, Mem_Len, Embed_Dim]
        new_rnn_states = new_rnn_states.permute(1, 0, 2, 3)
        # 2. Flatten 压扁：变回 [Batch, Layers, Flat_Dim] 存入 Buffer
        new_rnn_states_flat = new_rnn_states.flatten(2, 3)

        if len(self.act_hidden_size) > 0:
            critic_features = self.mlp(critic_features)

        values = self.value_out(critic_features)

        return values, new_rnn_states_flat
