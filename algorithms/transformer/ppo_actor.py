import torch
import torch.nn as nn
from ..utils.act import ACTLayer
from ..utils.utils import check
# 假设之前的模块保存在这里
from .spatio_temporal import EntityEmbedder, SpatialAttention, GatedTransformerBlock


class PPOActor(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(PPOActor, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.transformer_size = 128

        # --- 1. Entity Embedder (替代 MLPBase) ---
        # 注意: 这里需要从 args 中获取实体维度的配置 config

        self.entity_config = {
            'own_dim': 9,  # 自身特征维度 (例如: x,y,z,vx,vy,vz,hp...)
            'ally_dim': 6,  # 盟友特征维度
            'ally_num': 3,  # 盟友数量
            'enemy_dim': 6,  # 敌人特征维度
            'enemy_num': 4,  # 敌人数量
            'missile_dim': 6,  # 导弹特征维度
            'missile_num': 1  # 导弹数量
        }
        self.embedder = EntityEmbedder(obs_space, self.transformer_size, self.entity_config)

        # --- 2. Spatial Attention ---
        self.spatial_attn = SpatialAttention(self.transformer_size)

        # --- 3. GTrXL (替代 GRU) ---
        self.use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_layers = args.recurrent_hidden_layers

        if self.use_recurrent_policy:
            self.gtrxl_blocks = nn.ModuleList([
                GatedTransformerBlock(self.transformer_size, nhead=4)
                for _ in range(self.recurrent_layers)
            ])

        # --- 4. Action Head ---
        self.act = ACTLayer(act_space, self.transformer_size, self.act_hidden_size, args.activation_id, args.gain)

        self.to(device)

    def forward(self, obs, mems, masks, deterministic=False):
        """
        obs: (Batch, Obs_Dim) during rollout OR (Batch, Seq_Len, Obs_Dim) during training
        mems: (Layer, Mem_Len, Batch, Hidden)
        masks: (Batch, 1) or (Batch, Seq_Len, 1)
        """
        obs = check(obs).to(**self.tpdv)
        mems = check(mems).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        mem_len = mems.shape[2]  # 这个应该是超参数

        # 1. 维度统一化处理
        # 目标: internal_obs 变为 (Batch, Seq_Len, Dim)
        if obs.dim() == 2:
            # Rollout 模式: (Batch, Dim) -> (Batch, 1, Dim)
            obs = obs.unsqueeze(1)

            seq_len = 1
        else:
            # Training 模式: (Batch, Seq_Len, Dim)
            seq_len = obs.shape[1]

        batch_size = obs.shape[0]


        # 2. 实体编码与空间注意力 (处理每一帧)
        # Flatten: (Batch * Seq, Dim)
        obs_flat = obs.view(-1, obs.shape[-1])

        # Embed: -> (Batch * Seq, Entities, Hidden)
        entity_embeds, entity_masks = self.embedder(obs_flat)

        # Spatial Attn: -> (Batch * Seq, Hidden)
        # 提取 Ownship 特征作为核心特征
        spatial_features = self.spatial_attn(entity_embeds, entity_masks)

        # 3. GTrXL 时序处理
        x = spatial_features.view(batch_size, seq_len, self.transformer_size)  # (Batch, Seq, Hidden)

        new_mems = []
        if self.use_recurrent_policy:
            # 转换为 (Seq, Batch, Hidden) 适应 Transformer
            x = x.permute(1, 0, 2)

            # 处理 Memory 重置: 如果 mask=0 (Done)，则对应的 memory 应该视为无效
            # 注意：在 chunk 训练中，只需要处理 chunk 开头的 mask。
            # 在 rollout 中，mask 对应当前步。
            # 简单起见，这里假设外部已经处理好了 memory 的重置，或者在这里用 mask * mems
            # (但在 Transformer 中，memory 结构复杂，通常建议在 Buffer 存取时如果不连续则置零 memory)
            if seq_len == 1 and mems is not None:
                # 使用 mems 自身的 batch 维度 (index 2) 来确保安全，防止 8 vs 128 错误
                current_mem_batch = mems.shape[2]
                # masks shape: (Batch, 1) -> view -> (1, 1, Batch, 1)
                # 广播乘法: (Layers, Len, Batch, Hidden) * (1, 1, Batch, 1)
                mems = mems * masks.view(-1, 1, 1, 1)

                # === 修复 2: 维度排列 (Batch First -> Layer First) ===
                # 输入 mems: (Batch, Layer, Mem, Hidden)
                # 目标 mems: (Layer, Mem, Batch, Hidden)
                # permute(1, 2, 0, 3)
                mems = mems.permute(1, 2, 0, 3)

            for i, block in enumerate(self.gtrxl_blocks):
                # mems shape: (Layers, Mem_Len, Batch, Hidden)
                layer_mems = mems[i] if mems is not None else None

                # Forward
                x = block(x, layer_mems)  # Output x is (Seq, Batch, Hidden)

                # Update Memory (Detached for TBPTT)
                # 新的 memory 是 原始 memory + 当前输入 (concat)
                # block 内部不返回 memory，我们需要手动维护 memory 逻辑
                # GTrXL 论文逻辑: next_mem = cat([old_mem, input_x]).detach()
                # 这里为了简化，假设 block forward 已经利用了 memory。
                # 我们需要构建下一次的 memory。
                if layer_mems is not None:
                    cat_mem = torch.cat([layer_mems, x.detach()], dim=0)
                else:
                    cat_mem = x.detach()

                # 截断 Memory 长度 (例如保持最近 96 步)
                new_layer_mem = cat_mem[-mem_len:]
                new_mems.append(new_layer_mem)

            new_mems = torch.stack(new_mems)  # (Layer, Mem_Len, Batch, Hidden)

            # === 修复 3: 维度还原 (Layer First -> Batch First) ===
            # 我们需要返回给 Runner 存储，格式必须是 (Batch, Layer, Mem, Hidden)
            # permute(2, 0, 1, 3)
            # (Layer, Mem, Batch, Hidden) -> (Batch, Layer, Mem, Hidden)
            new_mems = new_mems.permute(2, 0, 1, 3)

            # 转回 (Batch, Seq, Hidden)
            x = x.permute(1, 0, 2)
        else:
            new_mems = mems  # Pass through

        # 4. Action Head
        # Flatten back to (Batch * Seq, Hidden)
        x_flat = x.reshape(-1, self.transformer_size)
        actions, action_log_probs = self.act(x_flat, deterministic)

        # 如果输入是 (Batch, 1, D)，输出也该 squeeze 回去，保持接口一致
        if seq_len == 1:
            actions = actions.view(batch_size, -1)
            action_log_probs = action_log_probs.view(batch_size, -1)
        else:
            actions = actions.view(batch_size, seq_len, -1)
            action_log_probs = action_log_probs.view(batch_size, seq_len, -1)

        return actions, action_log_probs, new_mems

    def evaluate_actions(self, obs, mems, action, masks, active_masks=None):
        """
        修正后的 evaluate_actions，专门适配 Recurrent Generator 的输出
        """
        # 0. 基础检查
        obs = check(obs).to(**self.tpdv)
        mems = check(mems).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        # ------------------------------------------------------------------
        # 1. 维度解析与重塑 (Reshape)
        # 输入 Obs 形状: [Batch_Chunks (25), Seq_Len (128), Agents (4), Dim (57)]
        # 输入 Mems 形状: [Batch_Chunks (25), Agents (4), Layers (1), Mem (128), Hidden (128)]
        # ------------------------------------------------------------------

        # 我们需要把 Batch_Chunks 和 Agents 合并，作为 Transformer 的 "Batch"
        # 目标 Obs:  [Batch_Chunks * Agents, Seq_Len, Dim]
        # 目标 Mems: [Layers, Mem, Batch_Chunks * Agents, Hidden]

        if obs.dim() == 4:
            batch_chunks, seq_len, n_agents, obs_dim = obs.shape

            # --- 处理 Obs ---
            # [25, 128, 4, 57] -> [25, 4, 128, 57] -> [100, 128, 57]
            obs = obs.permute(0, 2, 1, 3).reshape(batch_chunks * n_agents, seq_len, obs_dim)

            # --- 处理 Actions ---
            # [25, 128, 4, Act] -> [100, 128, Act]
            action = action.permute(0, 2, 1, 3).reshape(batch_chunks * n_agents, seq_len, -1)

            # --- 处理 Masks ---
            masks = masks.permute(0, 2, 1, 3).reshape(batch_chunks * n_agents, seq_len, -1)

            if active_masks is not None:
                active_masks = active_masks.permute(0, 2, 1, 3).reshape(batch_chunks * n_agents, seq_len, -1)

            # --- 处理 Mems (关键修正) ---
            # 输入: [25, 4, 1, 128, 128]
            # 注意：Generator 输出的 mems 已经是 Chunk 的起始状态，没有 Time 维度，不要取 mems[0]！

            # 1. 合并 Batch 和 Agents
            # -> [100, 1, 128, 128] (Batch_Total, Layers, Mem, Hidden)
            mems = mems.reshape(batch_chunks * n_agents, *mems.shape[2:])

            # 2. 调整为 GTrXL 要求的顺序 (Layers, Mem, Batch, Hidden)
            # permute(1, 2, 0, 3)
            # -> [1, 128, 100, 128]
            mems = mems.permute(1, 2, 0, 3)

        # ------------------------------------------------------------------
        # 2. 进入网络
        # ------------------------------------------------------------------

        # 此时:
        # obs.shape[0] = 100 (Total Batch)
        # mems.shape[2] = 100 (Total Batch)
        # 维度完全对齐，可以运行了。

        batch_size, seq_len = obs.shape[0], obs.shape[1]

        # Flatten: (Batch * Seq, Dim)
        obs_flat = obs.reshape(-1, obs.shape[-1])

        entity_embeds, entity_masks = self.embedder(obs_flat)
        spatial_features = self.spatial_attn(entity_embeds, entity_masks)

        # GTrXL Input
        x = spatial_features.view(batch_size, seq_len, self.transformer_size)

        if self.use_recurrent_policy:
            x = x.permute(1, 0, 2)  # (Seq, Batch, Hidden)

            for i, block in enumerate(self.gtrxl_blocks):
                layer_mems = mems[i] if mems is not None else None
                x = block(x, layer_mems)

            x = x.permute(1, 0, 2)  # (Batch, Seq, Hidden)

        # Action Head
        x_flat = x.reshape(-1, self.transformer_size)
        action_flat = action.reshape(-1, action.shape[-1])

        if active_masks is not None:
            active_masks_flat = active_masks.reshape(-1, 1)
        else:
            active_masks_flat = None

        action_log_probs, dist_entropy = self.act.evaluate_actions(x_flat, action_flat, active_masks_flat)

        action_log_probs = action_log_probs.view(batch_size, seq_len, -1)

        return action_log_probs, dist_entropy