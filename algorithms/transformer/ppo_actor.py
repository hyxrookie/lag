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

        # --- 1. Entity Embedder (替代 MLPBase) ---
        # 注意: 这里需要从 args 中获取实体维度的配置 config

        self.entity_config = {
            'own_dim': 9,  # 自身特征维度 (例如: x,y,z,vx,vy,vz,hp...)
            'ally_dim': 6,  # 盟友特征维度
            'ally_num': 3,  # 盟友数量
            'enemy_dim': 6,  # 敌人特征维度
            'enemy_num': 4,  # 敌人数量
            'missile_dim': 6,  # 导弹特征维度
            'missile_num': 4  # 导弹数量
        }

        self.embedder = EntityEmbedder(obs_space, self.hidden_size, self.entity_config)

        # --- 2. Spatial Attention ---
        self.spatial_attn = SpatialAttention(self.hidden_size)

        # --- 3. GTrXL (替代 GRU) ---
        self.use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_layers = args.recurrent_hidden_layers

        if self.use_recurrent_policy:
            self.gtrxl_blocks = nn.ModuleList([
                GatedTransformerBlock(self.hidden_size, nhead=4)
                for _ in range(self.recurrent_layers)
            ])

        # --- 4. Action Head ---
        self.act = ACTLayer(act_space, self.hidden_size, self.act_hidden_size, args.activation_id, args.gain)

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
        x = spatial_features.view(batch_size, seq_len, self.hidden_size)  # (Batch, Seq, Hidden)

        new_mems = []
        if self.use_recurrent_policy:
            # 转换为 (Seq, Batch, Hidden) 适应 Transformer
            x = x.permute(1, 0, 2)

            # 处理 Memory 重置: 如果 mask=0 (Done)，则对应的 memory 应该视为无效
            # 注意：在 chunk 训练中，只需要处理 chunk 开头的 mask。
            # 在 rollout 中，mask 对应当前步。
            # 简单起见，这里假设外部已经处理好了 memory 的重置，或者在这里用 mask * mems
            # (但在 Transformer 中，memory 结构复杂，通常建议在 Buffer 存取时如果不连续则置零 memory)
            if seq_len == 1:
                # Rollout 时，如果 mask 为 0，清空历史记忆
                # masks: (Batch, 1)
                mems = mems * masks.view(1, 1, batch_size, 1)

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
                mem_len = 96  # 这个应该是超参数
                new_layer_mem = cat_mem[-mem_len:]
                new_mems.append(new_layer_mem)

            new_mems = torch.stack(new_mems)  # (Layer, Mem_Len, Batch, Hidden)

            # 转回 (Batch, Seq, Hidden)
            x = x.permute(1, 0, 2)
        else:
            new_mems = mems  # Pass through

        # 4. Action Head
        # Flatten back to (Batch * Seq, Hidden)
        x_flat = x.reshape(-1, self.hidden_size)
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
        obs: (Batch, Seq_Len, Dim) 
        action: (Batch, Seq_Len, Act_Dim)
        """
        obs = check(obs).to(**self.tpdv)
        mems = check(mems).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        # 1. Forward logic same as above
        batch_size, seq_len = obs.shape[0], obs.shape[1]

        # ... Entity Embed + Spatial ...
        obs_flat = obs.view(-1, obs.shape[-1])
        entity_embeds, entity_masks = self.embedder(obs_flat)
        spatial_features = self.spatial_attn(entity_embeds, entity_masks)

        # ... GTrXL ...
        x = spatial_features.view(batch_size, seq_len, self.hidden_size)
        if self.use_recurrent_policy:
            x = x.permute(1, 0, 2)  # (Seq, Batch, Hidden)
            for i, block in enumerate(self.gtrxl_blocks):
                layer_mems = mems[i]
                x = block(x, layer_mems)
            x = x.permute(1, 0, 2)  # (Batch, Seq, Hidden)

        # 2. Evaluate Actions
        x_flat = x.reshape(-1, self.hidden_size)
        action_flat = action.reshape(-1, action.shape[-1])

        # 处理 active_masks
        if active_masks is not None:
            active_masks_flat = active_masks.reshape(-1, 1)
        else:
            active_masks_flat = None

        action_log_probs, dist_entropy = self.act.evaluate_actions(x_flat, action_flat, active_masks_flat)

        # Reshape back to (Batch, Seq)
        action_log_probs = action_log_probs.view(batch_size, seq_len, -1)
        if dist_entropy is not None:
            # entropy 通常是一个标量或 (Batch,)，视实现而定
            pass

        return action_log_probs, dist_entropy