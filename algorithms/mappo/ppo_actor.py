#
# 文件: PPOActor.py (修改后)
#
import torch
import torch.nn as nn

from ..utils.mlp import MLPBase
from ..utils.gru import GRULayer
# 新增导入
from ..utils.transformer import SimpleTransformer
from ..utils.act import ACTLayer
from ..utils.utils import check


class PPOActor(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(PPOActor, self).__init__()
        # network config
        self.gain = args.gain
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
            # self.transformer = CausalTransformerEncoder(args, input_size, device)
            self.transformer = SimpleTransformer(args, input_size, device)

            # Transformer output dim is the same as input dim
            input_size = self.transformer.output_size


        # (3) rnn module
        if self.use_recurrent_policy:
            self.rnn = GRULayer(input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
            input_size = self.rnn.output_size

        # (4) act module
        self.act = ACTLayer(act_space, input_size, self.act_hidden_size, self.activation_id, self.gain)

        self.to(device)

    def forward(self, obs, rnn_states, masks, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.base(obs)

        # Pass through Transformer if enabled
        if self.use_transformer_policy:
            # For inference (T=1), reshape to (1, N, dim)
            # This allows the same code path for both training and inference
            actor_features = actor_features.unsqueeze(0)  # (1, N, dim)

            # During inference, there is no padding, so padding_mask is None
            actor_features = self.transformer(actor_features, src_key_padding_mask=None)

            actor_features = actor_features.squeeze(0)  # (N, dim)

        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)
        # print("actor_features shape:", actor_features.shape)

        # --- 诊断代码 ---
        # print(f"Features before Transformer: shape={actor_features.shape}")
        # print(f"  mean: {actor_features.mean().item():.4f}, std: {actor_features.std().item():.4f}")
        # print(f"  min: {actor_features.min().item():.4f}, max: {actor_features.max().item():.4f}")
        # --- 诊断结束 ---

        # Pass through Transformer if enabled
        if self.use_transformer_policy:
            # For training (T > 1), reshape to (T, N, dim)
            # T = sequence length, N = batch size
            T = self.data_chunk_length
            N = actor_features.shape[0] // T
            # print("transformer T:{T}, N:{N}".format(T=T, N=actor_features.shape[0]))

            actor_features = actor_features.view(T, N, -1)

            # print("transformer actor_features shape:", actor_features.shape)

            # Create padding mask from `masks`.
            # `masks` has shape (T*N, 1). A value of 0 means the state is terminal.
            # The padding mask for transformer should be (N, T) with True for padded positions.
            # We assume a 0 in `masks` means that and all subsequent steps are padding.
            padding_mask = (masks.view(T, N) == 0).contiguous()
            # print("padding_mask 全 True 行:", padding_mask.all(dim=1).nonzero(as_tuple=False).flatten())
            # print("Before Transformer NaN?", torch.isnan(actor_features).any())
            actor_features = self.transformer(actor_features, src_key_padding_mask=padding_mask)

            # print(f"After Transformer: any NaN? {torch.isnan(actor_features).any()}")

            actor_features = actor_features.view(T * N, -1)

        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, active_masks)
        # print(f"After GRU: any NaN? {torch.isnan(actor_features).any()}")

        return action_log_probs, dist_entropy