#
# 文件: PPOCritic.py (修改后)
#
import torch
import torch.nn as nn

from ..utils.mlp import MLPBase, MLPLayer
from ..utils.gru import GRULayer
# 新增导入
from ..utils.transformer import CausalTransformerEncoder
from ..utils.at import SimpleTransformer
from ..utils.utils import check


class PPOCritic(nn.Module):
    def __init__(self, args, obs_space, device=torch.device("cpu")):
        super(PPOCritic, self).__init__()
        # network config
        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization

        # recurrent & transformer config
        self.use_recurrent_policy = args.use_recurrent_policy
        self.use_transformer_policy = args.use_transformer_policy  # 新增参数
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.data_chunk_length = args.data_chunk_length  # 需要此参数来reshape

        self.tpdv = dict(dtype=torch.float32, device=device)

        # (1) feature extraction module (MLP)
        self.base = MLPBase(obs_space, self.hidden_size, self.activation_id, self.use_feature_normalization)

        input_size = self.base.output_size

        # (2) NEW: transformer module
        if self.use_transformer_policy:
            self.transformer = SimpleTransformer(args, input_size, device)
            input_size = self.transformer.output_size

        # (3) rnn module
        if self.use_recurrent_policy:
            self.rnn = GRULayer(input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
            input_size = self.rnn.output_size

        # (4) value module
        if len(self.act_hidden_size) > 0:
            self.mlp = MLPLayer(input_size, self.act_hidden_size, self.activation_id)
        self.value_out = nn.Linear(input_size, 1)

        self.to(device)

    def forward(self, obs, rnn_states, masks):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(obs)

        # Pass through Transformer if enabled
        # This forward is used for both training and inference.
        # We need to distinguish between T=1 and T>1 cases.
        if self.use_transformer_policy:
            if critic_features.shape[0] == rnn_states.shape[0]:  # Inference case, T=1
                N = critic_features.shape[0]
                critic_features = critic_features.unsqueeze(0)
                critic_features = self.transformer(critic_features, src_key_padding_mask=None)
                critic_features = critic_features.squeeze(0)
            else:  # Training case, T > 1
                T = self.data_chunk_length
                N = critic_features.shape[0] // T

                critic_features = critic_features.view(T, N, -1)

                padding_mask = (masks.view(T, N) == 0).contiguous()

                critic_features = self.transformer(critic_features, src_key_padding_mask=padding_mask)

                critic_features = critic_features.view(T * N, -1)

        if self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        if len(self.act_hidden_size) > 0:
            critic_features = self.mlp(critic_features)

        values = self.value_out(critic_features)

        return values, rnn_states