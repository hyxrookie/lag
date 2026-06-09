import torch
import torch.nn as nn

from ..utils.gtrxl import GTrXL
from ..utils.mlp import MLPBase, MLPLayer
from ..utils.gru import GRULayer
from ..utils.spatial_temporal_base import SpatialTemporalBase
from ..utils.spatial_temporal_base_gru import SpatialTemporalBaseGRU
from ..utils.utils import check


class PPOCriticSGRU(nn.Module):
    def __init__(self, args, obs_space, device=torch.device("cpu")):
        super(PPOCriticSGRU, self).__init__()
        # network config
        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.tpdv = dict(dtype=torch.float32, device=device)


        self.base = SpatialTemporalBaseGRU(
            obs_space=obs_space,  # 这里传入的是 share_obs_space
            activation_id=self.activation_id,
            use_feature_normalization=self.use_feature_normalization,
            embed_dim=getattr(args, 'embed_dim', 128),
            num_spatial_heads=getattr(args, 'num_spatial_heads', 8),
            num_temporal_heads=getattr(args, 'num_temporal_heads', 8),
            num_temporal_layers=getattr(args, 'num_temporal_layers', 2),
            memory_length=getattr(args, 'memory_length', 64),
            dropout=getattr(args, 'dropout', 0.1),
            ego_dim=getattr(args, 'ego_dim', 14),
            relative_dim=getattr(args, 'relative_dim', 7),
            num_friendly=getattr(args, 'num_friendly', 1),
            num_enemy=getattr(args, 'num_enemy', 2),
            num_missiles=getattr(args, 'num_missiles', 1),
            use_type_embedding=getattr(args, 'use_type_embedding', True),
            is_critic=True,  # 开启 Critic 模式
            num_agents=getattr(args, 'num_agents', 4)  # 必须传入智能体数量！
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


        critic_features, new_rnn_states_flat = self.base(
            obs=obs,
            rnn_states=rnn_states,
            masks=masks,
            agent_index=None
        )

        if len(self.act_hidden_size) > 0:
            critic_features = self.mlp(critic_features)

        values = self.value_out(critic_features)

        return values, new_rnn_states_flat
