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
        self.transformer_size = 128
        # --- 1. Critic Embedder ---
        # 假设 cent_obs_dim 是拼接后的总维度，num_agents 需传入
        self.embedder = CriticEmbedder(cent_obs_space.shape[0], 4, self.transformer_size)

        # --- 2. Spatial Attention ---
        # Critic 同样需要聚合所有 Agent 的信息
        self.spatial_attn = SpatialAttention(self.transformer_size)

        # --- 3. GTrXL ---
        self.use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_layers = args.recurrent_hidden_layers
        if self.use_recurrent_policy:
            self.gtrxl_blocks = nn.ModuleList([
                GatedTransformerBlock(self.transformer_size, nhead=4)
                for _ in range(self.recurrent_layers)
            ])

        # --- 4. Value Head ---
        self.value_out = nn.Linear(self.transformer_size, 1)
        self.to(device)

    def forward(self, cent_obs, mems, masks):
        cent_obs = check(cent_obs).to(**self.tpdv)
        mems = check(mems).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # ===================================================================
        # 1. 维度清洗 (核心修复)
        # 区分 Training (4D) 和 Rollout (2D/3D) 模式
        # ===================================================================

        # Training 模式输入:
        # cent_obs: [Batch_Chunks, Seq_Len, Agents, Dim]
        # mems:     [Batch_Chunks, Agents, Layers, Mem, Hidden]
        if cent_obs.dim() == 4:
            batch_chunks, seq_len, n_agents, dim = cent_obs.shape

            # 1. 合并 Batch 和 Agents -> [Total_Batch, Seq, Dim]
            # [25, 128, 4, Dim] -> [25, 4, 128, Dim] -> [100, 128, Dim]
            cent_obs = cent_obs.permute(0, 2, 1, 3).reshape(batch_chunks * n_agents, seq_len, dim)

            # 2. 处理 Masks 同理
            masks = masks.permute(0, 2, 1, 3).reshape(batch_chunks * n_agents, seq_len, -1)

            # 3. 处理 Mems (最关键)
            # [25, 4, Layers, Mem, Hidden] -> [100, Layers, Mem, Hidden]


            mems = mems.reshape(batch_chunks * n_agents, *mems.shape[2:])

            # 调整为 GTrXL 格式: [Layers, Mem, Total_Batch, Hidden]
            # [100, Layers, Mem, Hidden] -> [Layers, Mem, 100, Hidden]
            mems = mems.permute(1, 2, 0, 3)

        # Rollout 模式输入:
        # cent_obs: [Batch, Dim]
        # mems:     [Batch, Layers, Mem, Hidden] (可能需要调整顺序)
        elif cent_obs.dim() == 2:
            cent_obs = cent_obs.unsqueeze(1)  # [Batch, 1, Dim]
            seq_len = 1

            # Rollout 时 mems 通常是 [Batch, Layers, Mem, Hidden]
            # 我们需要调整为 [Layers, Mem, Batch, Hidden]
            # 注意：这里假设输入的 mems 还没调整过
            if mems.shape[0] == cent_obs.shape[0]:  # 简单检查 Batch 维是否在第一位
                mems = mems.permute(1, 2, 0, 3)

        else:
            seq_len = cent_obs.shape[1]

        # ===================================================================
        # 2. 获取参数 (必须在 Reshape 之后)
        # ===================================================================
        batch_size = cent_obs.shape[0]  # 这里的 Batch 已经是 100 了

        # 只有在 Reshape 且 Permute 之后，mems 的维度才是 [Layers, Mem, Batch, Hidden]
        # 所以 mems.shape[1] 才是 Mem_Len
        # 为了安全，建议直接用 args 里的参数，或者动态获取
        mem_len = mems.shape[1]

        # ===================================================================
        # 3. 网络前向传播
        # ===================================================================

        # 1. Embed & Spatial
        obs_flat = cent_obs.reshape(-1, cent_obs.shape[-1])

        # CriticEmbedder -> (Batch*Seq, Num_Agents, Hidden)
        agent_tokens, agent_masks = self.embedder(obs_flat)

        # Spatial Attn -> (Batch*Seq, Hidden)
        spatial_feat = self.spatial_attn(agent_tokens, agent_masks)

        # 2. Temporal (GTrXL)
        x = spatial_feat.view(batch_size, seq_len, self.transformer_size)

        new_mems = []
        if self.use_recurrent_policy:
            x = x.permute(1, 0, 2)  # [Seq, Batch, Hidden]

            # Rollout mask 处理 (训练时依靠 GTrXL 内部的 Causal Mask，不需要置零 memory)
            if seq_len == 1:
                # mems: [Layers, Mem, Batch, Hidden]
                # masks: [Batch, 1, 1] -> [1, 1, Batch, 1]
                mems = mems * masks.view(1, 1, -1, 1)

            for i, block in enumerate(self.gtrxl_blocks):
                layer_mems = mems[i] if mems is not None else None
                x = block(x, layer_mems)

                # Update Mem logic
                if layer_mems is not None:
                    cat_mem = torch.cat([layer_mems, x.detach()], dim=0)
                else:
                    cat_mem = x.detach()
                # 保持最新的 mem_len 长度
                new_mems.append(cat_mem[-mem_len:])

            new_mems = torch.stack(new_mems)  # [Layers, Mem, Batch, Hidden]
            x = x.permute(1, 0, 2)  # [Batch, Seq, Hidden]
        else:
            new_mems = mems

        # 3. 还原 Memory 维度 (给 Runner 存储用)
        # [Layers, Mem, Batch, Hidden] -> [Batch, Layers, Mem, Hidden]
        new_mems = new_mems.permute(2, 0, 1, 3)

        # 4. Value Head
        values = self.value_out(x)  # (Batch, Seq, 1)

        if seq_len == 1:
            values = values.view(batch_size, -1)

        return values, new_mems