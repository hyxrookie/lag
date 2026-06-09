import torch
import torch.nn as nn

from ..utils.mlp import MLPBase
from ..utils.gru import GRULayer
from ..utils.act import ACTLayer
from ..utils.spatial_temporal_base import SpatialTemporalBase
from ..utils.spatial_temporal_base_gru import SpatialTemporalBaseGRU
from ..utils.utils import check


class PPOActorSGRU(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(PPOActorSGRU, self).__init__()
        # network config
        self.gain = args.gain
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
        self.ego_dim = getattr(args, 'ego_dim', 14)
        self.relative_dim = getattr(args, 'relative_dim', 7)
        self.num_friendly = getattr(args, 'num_friendly', 1)
        self.num_enemy = getattr(args, 'num_enemy', 2)
        self.num_missiles = getattr(args, 'num_missiles', 1)
        self.use_type_embedding = getattr(args, 'use_type_embedding', True)

        # (1) 时空注意力base模块（Actor模式）
        self.base = SpatialTemporalBaseGRU(
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
            is_critic=False  # Actor模式
        )
        # (3) act module
        input_size = self.base.output_size
        self.act = ACTLayer(act_space, input_size, self.act_hidden_size, self.activation_id, self.gain)

        self.to(device)

    def forward(self, obs, rnn_states, masks, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features, rnn_states = self.base(obs, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features, rnn_states = self.base(obs, rnn_states, masks)


        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, active_masks)

        return action_log_probs, dist_entropy
