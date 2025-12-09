import torch
import torch.nn as nn
from ..utils.utils import check
from .spatio_temporal import CriticEmbedder, SpatialAttention, GatedTransformerBlock


class PPOCritic(nn.Module):
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(PPOCritic, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.tpdv = dict(dtype=torch.float32, device=device)

        # --- 1. Critic Embedder ---
        # 假设 cent_obs_dim 是拼接后的总维度，num_agents 需传入
        self.embedder = CriticEmbedder(cent_obs_space, 8, self.hidden_size)

        # --- 2. Spatial Attention ---
        # Critic 同样需要聚合所有 Agent 的信息
        self.spatial_attn = SpatialAttention(self.hidden_size)

        # --- 3. GTrXL ---
        self.use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_layers = args.recurrent_hidden_layers
        if self.use_recurrent_policy:
            self.gtrxl_blocks = nn.ModuleList([
                GatedTransformerBlock(self.hidden_size, nhead=4)
                for _ in range(self.recurrent_layers)
            ])

        # --- 4. Value Head ---
        self.value_out = nn.Linear(self.hidden_size, 1)
        self.to(device)

    def forward(self, cent_obs, mems, masks):
        cent_obs = check(cent_obs).to(**self.tpdv)
        mems = check(mems).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if cent_obs.dim() == 2:
            cent_obs = cent_obs.unsqueeze(1)
            seq_len = 1
        else:
            seq_len = cent_obs.shape[1]

        batch_size = cent_obs.shape[0]

        # 1. Embed & Spatial
        obs_flat = cent_obs.view(-1, cent_obs.shape[-1])
        # CriticEmbedder 输出: (Batch*Seq, Num_Agents, Hidden)
        agent_tokens, agent_masks = self.embedder(obs_flat)

        # 聚合信息 -> (Batch*Seq, Hidden)
        spatial_feat = self.spatial_attn(agent_tokens, agent_masks)

        # 2. Temporal (GTrXL)
        x = spatial_feat.view(batch_size, seq_len, self.hidden_size)

        new_mems = []
        if self.use_recurrent_policy:
            x = x.permute(1, 0, 2)

            # Rollout mask 处理
            if seq_len == 1:
                mems = mems * masks.view(1, 1, batch_size, 1)

            for i, block in enumerate(self.gtrxl_blocks):
                layer_mems = mems[i] if mems is not None else None
                x = block(x, layer_mems)

                # Update Mem logic (same as Actor)
                if layer_mems is not None:
                    cat_mem = torch.cat([layer_mems, x.detach()], dim=0)
                else:
                    cat_mem = x.detach()
                new_mems.append(cat_mem[-96:])  # Keep 96

            new_mems = torch.stack(new_mems)
            x = x.permute(1, 0, 2)
        else:
            new_mems = mems

        # 3. Value Head
        values = self.value_out(x)  # (Batch, Seq, 1)

        if seq_len == 1:
            values = values.view(batch_size, -1)

        return values, new_mems