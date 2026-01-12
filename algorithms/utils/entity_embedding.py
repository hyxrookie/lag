import torch
import torch.nn as nn

class EntityEmbedding(nn.Module):
    def __init__(self, ego_dim=9, relative_dim=6, embed_dim=128, activation_id=1, 
                 use_type_embedding=True, num_missiles=0):
        super(EntityEmbedding, self).__init__()
        self.ego_dim = ego_dim
        self.relative_dim = relative_dim
        self.embed_dim = embed_dim
        self.use_type_embedding = use_type_embedding
        self.num_missiles = num_missiles
        
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        
        # 1. 定义嵌入维度分配
        if use_type_embedding:
            self.type_embed_dim = embed_dim // 4
            self.feature_embed_dim = embed_dim - self.type_embed_dim
            self.type_embedding = nn.Embedding(4, self.type_embed_dim)
        else:
            self.type_embed_dim = 0
            self.feature_embed_dim = embed_dim
        
        # 2. 定义各个实体的处理网络
        # MLP会自动处理 [L*B, num_entity, dim] 中的最后一维 dim
        def make_mlp(input_dim):
            return nn.Sequential(
                nn.Linear(input_dim, self.feature_embed_dim),
                nn.LayerNorm(self.feature_embed_dim),
                active_func,
                nn.Linear(self.feature_embed_dim, self.feature_embed_dim),
                nn.LayerNorm(self.feature_embed_dim),
                active_func
            )

        self.ego_embed = make_mlp(ego_dim)
        self.friendly_embed = make_mlp(relative_dim)
        self.enemy_embed = make_mlp(relative_dim)
        self.missile_embed = make_mlp(relative_dim) if num_missiles > 0 else None
    
    def forward(self, obs, num_friendly=0, num_enemy=0, num_missiles=None):
        """
        Args:
            obs: [N, total_dim]  (N可以是 L*B，也可以是任何 batch 维度)
        Returns:
            entity_embeddings: [N, num_entities, embed_dim]
        """
        if num_missiles is None:
            num_missiles = self.num_missiles

        # 1. 处理我机 (Ego)
        # obs: [N, total_dim] -> ego_obs: [N, ego_dim]
        ego_obs = obs[..., :self.ego_dim] 
        ego_feat = self.ego_embed(ego_obs) # [N, feature_dim]
        
        # 增加一个维度，变成 [N, 1, feature_dim]，代表"1个"我机实体
        ego_feat = ego_feat.unsqueeze(-2) 

        # 处理类型嵌入
        if self.use_type_embedding:
            # 创建形状为 [N, 1] 的 type 索引
            ego_type = torch.zeros(ego_feat.shape[:-1], dtype=torch.long, device=obs.device)
            ego_emb = torch.cat([ego_feat, self.type_embedding(ego_type)], dim=-1)
        else:
            ego_emb = ego_feat

        entity_list = [ego_emb]

        # 2. 处理相对观测 (Friendly/Enemy/Missile)
        # 剩下的数据全部取出来
        relative_obs_flat = obs[..., self.ego_dim:] 
        total_relative = num_friendly + num_enemy + num_missiles
        
        if total_relative > 0:
            # 【核心逻辑】：直接把剩下的 flat 数据 reshape 成 [N, 实体数, 相对特征维]
            # view(*obs.shape[:-1], ...) 保证了前面不管是 L*B 还是 L,B 都不受影响
            relative_obs = relative_obs_flat.view(*obs.shape[:-1], total_relative, self.relative_dim)
            
            offset = 0
            
            # --- 友军 ---
            if num_friendly > 0:
                # 切片: [N, num_friendly, rel_dim]
                part_obs = relative_obs[..., offset : offset+num_friendly, :]
                part_feat = self.friendly_embed(part_obs) # MLP 自动保持中间的 num_friendly 维度
                
                if self.use_type_embedding:
                    part_type = torch.ones(part_feat.shape[:-1], dtype=torch.long, device=obs.device) # 1=友军
                    part_emb = torch.cat([part_feat, self.type_embedding(part_type)], dim=-1)
                else:
                    part_emb = part_feat
                
                entity_list.append(part_emb)
                offset += num_friendly

            # --- 敌机 ---
            if num_enemy > 0:
                part_obs = relative_obs[..., offset : offset+num_enemy, :]
                part_feat = self.enemy_embed(part_obs)
                
                if self.use_type_embedding:
                    part_type = torch.full(part_feat.shape[:-1], 2, dtype=torch.long, device=obs.device) # 2=敌机
                    part_emb = torch.cat([part_feat, self.type_embedding(part_type)], dim=-1)
                else:
                    part_emb = part_feat
                    
                entity_list.append(part_emb)
                offset += num_enemy

            # --- 导弹 ---
            if num_missiles > 0 and self.missile_embed is not None:
                part_obs = relative_obs[..., offset : offset+num_missiles, :]
                part_feat = self.missile_embed(part_obs)
                
                if self.use_type_embedding:
                    part_type = torch.full(part_feat.shape[:-1], 3, dtype=torch.long, device=obs.device) # 3=导弹
                    part_emb = torch.cat([part_feat, self.type_embedding(part_type)], dim=-1)
                else:
                    part_emb = part_feat
                    
                entity_list.append(part_emb)

        # 3. 拼接所有实体
        # 结果维度: [N, 1 + n_friend + n_enemy + n_missile, embed_dim]
        return torch.cat(entity_list, dim=-2)


class EntityEmbeddingCritic(nn.Module):
    """
    Critic 专用嵌入层
    功能：将 share_obs [Batch, Total_Dim] 拆解为 [Batch, Num_Agents, Agent_Dim] 并嵌入
    """

    def __init__(self, obs_dim, num_agents, embed_dim=128, activation_id=1, use_type_embedding=True):
        super(EntityEmbeddingCritic, self).__init__()
        self.num_agents = num_agents
        self.total_obs_dim = obs_dim

        # 【核心修正】：自动计算单个智能体的维度
        # 例如：456 / 4 = 114
        if self.total_obs_dim % num_agents != 0:
            raise ValueError(f"share_obs维度 ({self.total_obs_dim}) 无法被智能体数量 ({num_agents}) 整除！")

        self.agent_obs_dim = self.total_obs_dim // num_agents

        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]

        # 针对“单个智能体”的 MLP
        self.agent_embed = nn.Sequential(
            nn.Linear(self.agent_obs_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            active_func,
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            active_func
        )

    def forward(self, share_obs):
        """
        Args:
            share_obs: [L*B, 456]
        Returns:
            agents_emb: [L*B, 4, 128]  <-- 这就有了 4 个实体
        """
        # 1. 动态获取 Batch 维度
        batch_dims = share_obs.shape[:-1]

        # 2. 【关键 Reshape】：拆分维度
        # [..., 456] -> [..., 4, 114]
        agents_obs = share_obs.view(*batch_dims, self.num_agents, self.agent_obs_dim)

        # 3. 对每个智能体进行嵌入
        # MLP 会作用在最后一维 (114 -> 128)
        agents_emb = self.agent_embed(agents_obs)  # [..., 4, 128]

        return agents_emb