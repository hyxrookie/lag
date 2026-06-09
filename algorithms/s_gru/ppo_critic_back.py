import torch
import torch.nn as nn

from ..utils.mlp import MLPBase, MLPLayer
from ..utils.gru import GRULayer
from ..utils.spatial_temporal_base import SpatialTemporalBase
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
        # (1) 时空注意力配置
        self.embed_dim = getattr(args, 'embed_dim', 128)
        self.num_spatial_heads = getattr(args, 'num_spatial_heads', 8)
        self.num_temporal_heads = getattr(args, 'num_temporal_heads', 8)
        self.temporal_ff_dim = getattr(args, 'temporal_ff_dim', 512)
        self.num_temporal_layers = getattr(args, 'num_temporal_layers', 2)
        self.memory_length = getattr(args, 'memory_length', 64)

        self.dropout = getattr(args, 'dropout', 0.1)
        self.ego_dim = getattr(args, 'ego_dim', 9)
        self.relative_dim = getattr(args, 'relative_dim', 6)
        self.num_friendly = getattr(args, 'num_friendly', 3)
        self.num_enemy = getattr(args, 'num_enemy', 4)
        self.num_missiles = getattr(args, 'num_missiles', 1)
        self.use_type_embedding = getattr(args, 'use_type_embedding', True)

        # (1) 时空注意力base模块（Actor模式）
        self.base = SpatialTemporalBase(
            obs_space=obs_space,
            activation_id=self.activation_id,
            use_feature_normalization=self.use_feature_normalization,
            embed_dim=self.embed_dim,
            num_spatial_heads=self.num_spatial_heads,
            num_temporal_heads=self.num_temporal_heads,
            num_temporal_layers=self.num_temporal_layers,
            memory_length=self.memory_length,
            dropout=self.dropout,
            ego_dim=self.ego_dim,
            relative_dim=self.relative_dim,
            num_friendly=self.num_friendly,
            num_enemy=self.num_enemy,
            num_missiles=self.num_missiles,
            use_type_embedding=self.use_type_embedding,
            is_critic=True,  # Actor模式
            num_agents=4
        )
        input_size = self.base.output_size
        # (3) value module
        if len(self.act_hidden_size) > 0:
            self.mlp = MLPLayer(input_size, self.act_hidden_size, self.activation_id)
        self.value_out = nn.Linear(input_size, 1)

        self.to(device)

    def forward(self, obs, rnn_states, masks):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features, rnn_states = self.base(obs, rnn_states, masks)

        if len(self.act_hidden_size) > 0:
            critic_features = self.mlp(critic_features)

        values = self.value_out(critic_features)

        return values, rnn_states
