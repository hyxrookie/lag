import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CriticEmbedder(nn.Module):
    """
    专门处理 hstack 拼接的 share_obs。
    将 (Batch, Num_Agents * Obs_Dim) -> (Batch, Num_Agents, d_model)
    """

    def __init__(self, full_share_obs_dim, num_agents, hidden_size):
        super().__init__()
        # 自动推导单个 agent 的 obs 维度
        assert full_share_obs_dim % num_agents == 0, "share_obs维度必须能被agent数量整除"
        self.single_obs_dim = full_share_obs_dim // num_agents
        self.num_agents = num_agents
        self.d_model = hidden_size

        # 我们用一个 MLP 来压缩单个 Agent 的巨大观测向量
        self.agent_encoder = nn.Sequential(
            nn.Linear(self.single_obs_dim, self.d_model * 2),
            nn.LayerNorm(self.d_model * 2),
            nn.Tanh(),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.Tanh()
        )

        # 可选：加入 Agent ID Embedding，让 Critic 知道哪个 Token 是谁
        self.id_embed = nn.Embedding(num_agents, self.d_model)

    def forward(self, share_obs):
        # share_obs: (Batch, Num_Agents * Obs_Dim)
        batch_size = share_obs.shape[0]

        # 1. Reshape: 还原回 (Batch, Num_Agents, Obs_Dim)
        x = share_obs.view(batch_size, self.num_agents, self.single_obs_dim)

        # 2. Encode: 对每个 Agent 的观测进行编码
        # (Batch, Num_Agents, Obs_Dim) -> (Batch, Num_Agents, d_model)
        tokens = self.agent_encoder(x)

        # 3. Add ID Embedding (位置编码的一种变体)
        ids = torch.arange(self.num_agents).to(share_obs.device).unsqueeze(0).repeat(batch_size, 1)
        tokens = tokens + self.id_embed(ids)

        # 生成 Mask (Critic通常看到所有Agent，除非有死掉的)
        # 这里简单起见，假设所有Agent都存在。如果Agent死了，obs通常是0，可以据此生成Mask
        # mask shape: (Batch, Num_Agents)
        # 如果 obs全是0，认为是死掉的 (根据你的环境逻辑调整)
        mask = (torch.abs(x).sum(dim=-1) == 0)

        return tokens, mask

class EntityEmbedder(nn.Module):
    """
    4.1 输入层设计：实体词元化
    将扁平的 obs 拆解为 Ownship, Allies, Enemies, Missiles 并进行 Embedding
    """

    def __init__(self, obs_shape, hidden_size, config):
        super().__init__()
        self.d_model = hidden_size

        # 解析配置 (你需要根据你的环境实际维度修改这些数字)
        # 假设 obs 是一个拼接的向量: [Own, Ally*N, Enemy*M, Missile*K]
        self.own_dim = config['own_dim']
        self.ally_dim = config['ally_dim']
        self.ally_num = config['ally_num']
        self.enemy_dim = config['enemy_dim']
        self.enemy_num = config['enemy_num']
        self.missile_dim = config['missile_dim']
        self.missile_num = config['missile_num']

        # 实体映射网络 (参数共享)
        self.own_mlp = nn.Sequential(nn.Linear(self.own_dim, self.d_model), nn.Tanh())
        if self.ally_num > 0:
            self.ally_mlp = nn.Sequential(nn.Linear(self.ally_dim, self.d_model), nn.Tanh())
        self.enemy_mlp = nn.Sequential(nn.Linear(self.enemy_dim, self.d_model), nn.Tanh())
        if self.missile_num > 0:
            self.missile_mlp = nn.Sequential(nn.Linear(self.missile_dim, self.d_model), nn.Tanh())

        # 位置/类型编码 (可选，帮助区分实体类型)
        self.type_embedding = nn.Embedding(4, self.d_model)  # 0:Own, 1:Ally, 2:Enemy, 3:Missile

    def forward(self, obs):
        # obs shape: (Batch, Total_Flat_Dim)
        batch_size = obs.shape[0]

        # 1. 切片 (Slicing) - 这是一个硬编码的切分，需要根据你obs的具体结构调整
        idx = 0
        own_feat = obs[:, idx:idx + self.own_dim]
        idx += self.own_dim

        ally_feats = []
        for _ in range(self.ally_num):
            ally_feats.append(obs[:, idx:idx + self.ally_dim])
            idx += self.ally_dim

        enemy_feats = []
        for _ in range(self.enemy_num):
            enemy_feats.append(obs[:, idx:idx + self.enemy_dim])
            idx += self.enemy_dim

        missile_feats = []
        for _ in range(self.missile_num):
            missile_feats.append(obs[:, idx:idx + self.missile_dim])
            idx += self.missile_dim

        # 2. Embedding & Mask Generation
        # Ownship
        tokens = [self.own_mlp(own_feat) + self.type_embedding(torch.tensor(0).to(obs.device))]
        masks = [torch.zeros(batch_size, dtype=torch.bool).to(obs.device)]  # Ownship 永远存在

        # Allies
        if self.ally_num > 0:
            ally_stack = torch.stack(ally_feats, dim=1)  # (B, N, D)
            # 生成 Mask: 如果特征全为0 (Padding)，则mask为True
            ally_mask = (torch.abs(ally_stack).sum(dim=-1) == 0)
            tokens.append(self.ally_mlp(ally_stack) + self.type_embedding(torch.tensor(1).to(obs.device)))
            masks.append(ally_mask)

        # Enemies
        if self.enemy_num > 0:
            enemy_stack = torch.stack(enemy_feats, dim=1)
            enemy_mask = (torch.abs(enemy_stack).sum(dim=-1) == 0)
            tokens.append(self.enemy_mlp(enemy_stack) + self.type_embedding(torch.tensor(2).to(obs.device)))
            masks.append(enemy_mask)

        # Missiles
        if self.missile_num > 0:
            missile_stack = torch.stack(missile_feats, dim=1)
            missile_mask = (torch.abs(missile_stack).sum(dim=-1) == 0)
            tokens.append(self.missile_mlp(missile_stack) + self.type_embedding(torch.tensor(3).to(obs.device)))
            masks.append(missile_mask)

        # 3. 拼接
        # tokens shape 最终变为: (Batch, Total_Entities, d_model)
        # key_padding_mask shape: (Batch, Total_Entities) -> True表示该位置被Mask

        final_tokens = []
        final_tokens.append(tokens[0].unsqueeze(1))  # Ownship
        if len(tokens) > 1:
            for t in tokens[1:]:
                final_tokens.append(t)

        final_tokens = torch.cat(final_tokens, dim=1)

        final_mask = []
        final_mask.append(masks[0].unsqueeze(1))
        if len(masks) > 1:
            for m in masks[1:]:
                final_mask.append(m)
        key_padding_mask = torch.cat(final_mask, dim=1)

        return final_tokens, key_padding_mask


class SpatialAttention(nn.Module):
    """
    4.2 第一阶段：空间注意力
    使用 Pre-LN Transformer Encoder
    """

    def __init__(self, d_model, nhead=4):
        super().__init__()
        # 使用 Pre-LayerNorm (norm_first=True) 对 RL 训练极其重要
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
                                           dropout=0.0, activation='gelu', norm_first=True, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, x, key_padding_mask):
        # x: (Batch, Seq_Len, d_model)
        # key_padding_mask: (Batch, Seq_Len)
        output = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # 提取 Ownship Token (Index 0) 作为聚合后的空间特征
        # H_t_spatial
        return output[:, 0, :]


class GatedTransformerBlock(nn.Module):
    """
    GTrXL 核心 Block: Identity Map + Gating
    """

    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)  # Keep Torch default dim
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

        # 门控参数
        self.gate1 = nn.Linear(d_model, d_model)
        self.gate2 = nn.Linear(d_model, d_model)

        # 初始化 Bias 使其初始状态接近 Identity (g -> 1)
        nn.init.constant_(self.gate1.bias, 2.0)
        nn.init.constant_(self.gate2.bias, 2.0)

    def forward(self, x, mems=None):
        # x: (Seq, Batch, Dim) - 注意这里是 Seq First，为了方便处理 Memory
        # mems: (Mem_Len, Batch, Dim)

        # 1. 拼接 Memory
        if mems is not None:
            cat_input = torch.cat([mems, x], dim=0)
        else:
            cat_input = x

        # 2. Attention Part
        u = self.norm1(x)
        # 注意: 我们只对当前的 x 进行 query，但 key 和 value 来自 cat_input (包含历史)
        # 需要生成 causal mask 保证 x 只能看自己之前的
        attn_out, _ = self.attn(query=u, key=cat_input, value=cat_input)

        # Gating 1
        g1 = torch.sigmoid(self.gate1(u))
        x = x + g1 * attn_out  # 这里也可以是 g * x + (1-g) * attn_out，具体看论文变体，这里用残差门控

        # 3. FFN Part
        v = self.norm2(x)
        ffn_out = self.ffn(v)

        # Gating 2
        g2 = torch.sigmoid(self.gate2(v))
        x = x + g2 * ffn_out

        return x


class GatedTransformerBlock(nn.Module):
    """
    GTrXL 核心 Block (修复版: 包含因果 Mask)
    """

    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.gate1 = nn.Linear(d_model, d_model)
        self.gate2 = nn.Linear(d_model, d_model)
        nn.init.constant_(self.gate1.bias, 2.0)
        nn.init.constant_(self.gate2.bias, 2.0)

    def forward(self, x, mems=None):
        # x: (Seq_Len, Batch, Dim)
        # mems: (Mem_Len, Batch, Dim)

        seq_len = x.shape[0]

        # 1. 拼接 Memory
        if mems is not None:
            cat_input = torch.cat([mems, x], dim=0)
            mem_len = mems.shape[0]
        else:
            cat_input = x
            mem_len = 0

        total_len = cat_input.shape[0]  # M + L

        # 2. === 关键: 生成因果 Mask ===
        # 我们需要一个形状为 (Seq_Len, Total_Len) 的 Mask
        # Query 长度是 Seq_Len (当前步)
        # Key 长度是 Total_Len (历史 + 当前)
        # 对于 Query 中的第 i 个元素 (在 x 中的位置 i)，它只能看到：
        #   - 所有的 Memory (0 ~ mem_len-1)
        #   - x 中位置 <= i 的元素 (mem_len ~ mem_len + i)

        # 方法: 生成全1矩阵 -> 上三角置为 -inf
        # attn_mask shape: (Seq_Len, Total_Len)
        # PyTorch 的 triu 逻辑: 保留上三角。我们需要把“未来” mask 掉。
        # 这里的逻辑是:attn_mask[i, j] = -inf 表示 i 不能关注 j

        # 创建一个全 0 矩阵 (可见)
        attn_mask = torch.zeros(seq_len, total_len).to(x.device)

        # 将 "未来" 部分填为 -inf
        # "未来" 的定义是: j > i + mem_len
        # 我们可以生成一个标准的 square causal mask (Total, Total)，然后切片

        full_mask = torch.triu(torch.ones(total_len, total_len), diagonal=1).to(x.device)
        full_mask = full_mask.masked_fill(full_mask == 1, float('-inf'))

        # 我们只关心 Query 是 x 的部分 (最后 seq_len 行)
        attn_mask = full_mask[-seq_len:, :]

        # 3. Attention Part
        u = self.norm1(x)

        # 传入 attn_mask
        attn_out, _ = self.attn(query=u, key=cat_input, value=cat_input, attn_mask=attn_mask)

        # Gating 1
        g1 = torch.sigmoid(self.gate1(u))
        x = x + g1 * attn_out

        # 4. FFN Part
        v = self.norm2(x)
        ffn_out = self.ffn(v)

        # Gating 2
        g2 = torch.sigmoid(self.gate2(v))
        x = x + g2 * ffn_out

        return x