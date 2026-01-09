# 时空注意力机制使用说明

## 概述

本实现添加了空间注意力+时间注意力（GTrXL）机制，用于替代原有的MLP+GRU架构。新架构包含以下组件：

1. **实体嵌入层** (`EntityEmbedding`): 将不同实体类型（我机、友军、敌机、导弹）的观察统一到相同维度
   - 不同类型的实体使用不同的MLP（友军、敌机、导弹分别有独立的嵌入层）
   - 支持类型嵌入（Type Embedding），帮助模型区分不同类型的实体
   - Actor和Critic使用不同的嵌入层（Critic处理share_obs，包含所有智能体的信息）
2. **空间注意力** (`SpatialAttention`): 处理多实体之间的空间关系
3. **时间注意力** (`GTrXLLayer`): 使用GTrXL进行时间建模，替代GRU

## 架构流程

```
观察 (obs) 
  -> 实体嵌入 (EntityEmbedding)
  -> 空间注意力 (SpatialAttention) 
  -> 时间注意力 (GTrXL)
  -> MAPPO (Actor/Critic)
```

## 使用方法

### 1. 配置参数

在训练脚本中添加以下参数：

```bash
--use-spatial-temporal-attention  # 启用时空注意力机制
--embed-dim 128                   # 嵌入维度
--num-spatial-heads 8             # 空间注意力头数
--num-temporal-heads 8            # 时间注意力头数
--temporal-ff-dim 512            # 时间注意力前馈网络维度
--num-temporal-layers 2           # 时间注意力层数
--memory-length 128                # GTrXL记忆长度
--dropout 0.1                     # Dropout比率
--ego-dim 9                       # 我机观察维度
--relative-dim 6                  # 相对观察维度（每个实体）
--num-friendly 1                  # 友军数量（根据实际情况设置）
--num-enemy 1                     # 敌机数量（根据实际情况设置）
--num-missiles 0                  # 导弹数量（默认0）
--use-type-embedding              # 使用类型嵌入（默认True，推荐使用）
--data-chunk-length-st 32         # GTrXL的块大小（可选，建议32或64）
```

### 2. 修改Policy导入

在runner或训练脚本中，将：
```python
from algorithms.mappo.ppo_policy import PPOPolicy
```

改为：
```python
from algorithms.mappo.ppo_policy_st import PPOPolicyST as PPOPolicy
```

或者直接使用：
```python
from algorithms.mappo.ppo_policy_st import PPOPolicyST
policy = PPOPolicyST(args, obs_space, cent_obs_space, act_space, device)
```

### 3. 实体嵌入说明

#### 3.1 不同类型的MLP
- **我机**: 使用`ego_embed` MLP（输入维度：ego_dim）
- **友军**: 使用`friendly_embed` MLP（输入维度：relative_dim）
- **敌机**: 使用`enemy_embed` MLP（输入维度：relative_dim）
- **导弹**: 使用`missile_embed` MLP（输入维度：relative_dim，如果num_missiles > 0）

不同类型的实体使用不同的MLP，可以更好地学习各自的特征分布。

#### 3.2 类型嵌入（Type Embedding）
- 类型嵌入维度：`embed_dim // 4`（默认）
- 类型ID：0=我机, 1=友军, 2=敌机, 3=导弹
- 类型嵌入与特征嵌入拼接：`[feature_embed, type_embed]` -> `embed_dim`

类型嵌入帮助模型区分不同类型的实体，提高注意力机制的效果。

#### 3.3 Actor vs Critic的嵌入差异

**Actor** (`EntityEmbedding`):
- 输入：单个智能体的观察 `obs`（包含我机信息 + 相对信息）
- 输出：`[我机, 友军1, ..., 友军N, 敌机1, ..., 敌机M, 导弹1, ..., 导弹K]` 的嵌入

**Critic** (`EntityEmbeddingCritic`):
- 输入：所有智能体的共享观察 `share_obs`（包含所有智能体的完整信息）
- 输出：`[智能体1, 智能体2, ..., 智能体N]` 的嵌入
- 所有智能体共享同一个MLP（因为它们都是同一类型）
- 空间注意力后使用平均池化聚合所有智能体的特征

### 4. Buffer兼容性说明

**✅ 已解决**: 通过`BufferSTAdapter`实现了自动格式转换。

GTrXL使用memories（列表格式）而不是GRU的rnn_states（tensor格式）。`PPOPolicyST`会自动处理格式转换：

- **输入**: buffer的numpy数组格式 `[n_rollout_threads, num_agents, num_layers, memory_length, embed_dim]`
- **内部**: 转换为memories列表 `List of [memory_length, batch_size, embed_dim]`
- **输出**: 返回memories列表（在runner层需要转换回numpy格式）

**注意**: 如果直接使用`PPOActorST`或`PPOCriticST`，需要手动处理格式转换。建议使用`PPOPolicyST`，它会自动处理。

### 5. 实体数量配置

根据观察空间的结构设置实体数量：

- `ego_dim`: 我机观察维度（通常为9）
- `relative_dim`: 每个友军/敌机的相对观察维度（通常为6）
- `num_friendly`: 友军数量
- `num_enemy`: 敌机数量

观察空间总维度 = `ego_dim + (num_friendly + num_enemy) * relative_dim`

例如，对于2v2场景：
- `ego_dim = 9`
- `num_friendly = 1` (1个友军)
- `num_enemy = 2` (2个敌机)
- `relative_dim = 6`
- 总维度 = 9 + (1 + 2) * 6 = 27

### 6. 示例代码

```python
# 在训练脚本中
from algorithms.mappo.ppo_policy_st import PPOPolicyST

# 设置参数
args.use_spatial_temporal_attention = True
args.embed_dim = 128
args.num_spatial_heads = 8
args.num_temporal_heads = 8
args.temporal_ff_dim = 512
args.num_temporal_layers = 2
args.memory_length = 128
args.dropout = 0.1
args.ego_dim = 9
args.relative_dim = 6
args.num_friendly = 1  # 根据实际情况设置
args.num_enemy = 2     # 根据实际情况设置
args.num_missiles = 0  # 根据实际情况设置
args.use_type_embedding = True  # 推荐使用类型嵌入
args.data_chunk_length_st = 32  # 可选，建议使用更大的chunk

# 创建policy
policy = PPOPolicyST(args, obs_space, cent_obs_space, act_space, device)
```

## 文件结构

新增的文件：
- `algorithms/utils/entity_embedding.py`: 实体嵌入层
- `algorithms/utils/spatial_attention.py`: 空间注意力机制
- `algorithms/utils/gtrxl.py`: GTrXL时间注意力
- `algorithms/utils/spatial_temporal_base.py`: 整合的Base模块
- `algorithms/utils/buffer_st_adapter.py`: Buffer格式转换适配器（推荐使用）
- `algorithms/utils/memory_adapter.py`: Memory格式转换工具（函数式接口，备用）
- `algorithms/mappo/ppo_actor_st.py`: 使用时空注意力的Actor
- `algorithms/mappo/ppo_critic_st.py`: 使用时空注意力的Critic
- `algorithms/mappo/ppo_policy_st.py`: 支持新旧架构的Policy

## 注意事项

1. **Buffer兼容性**: ✅ 已修复！通过`BufferSTAdapter`实现了memories和buffer格式之间的自动转换。Policy层会自动处理格式转换。

2. **时间掩码**: ✅ 已实现！GTrXL正确使用了时间掩码，会mask掉无效（done）的时间步。

3. **空间掩码**: ✅ 已实现！空间注意力会正确处理`entity_masks`，mask掉死亡的实体。

4. **块大小**: ✅ 有效！`--data-chunk-length-st`参数是有效的，即使buffer是用小的chunk创建的。因为`data_chunk_length`是在**训练时**从buffer中取数据时使用的，不是在buffer创建时使用的。buffer只是存储原始数据，训练时会根据`data_chunk_length`重新切分数据。GTrXL建议使用更大的chunk（如32或64）来更好地利用注意力机制。

5. **性能**: GTrXL比GRU计算开销更大，训练时间可能更长。

6. **内存**: GTrXL需要存储memories，内存占用可能更大。

7. **超参数**: 需要根据实际情况调整注意力头数、层数等超参数。

8. **实体数量**: 必须正确设置`num_friendly`和`num_enemy`，否则观察空间解析会出错。

## 已完成的改进

1. ✅ **Buffer兼容性**: 通过`BufferSTAdapter`实现了memories和buffer格式的自动转换
2. ✅ **时间掩码**: GTrXL正确实现了时间掩码，mask掉无效时间步
3. ✅ **空间掩码**: 空间注意力支持`entity_masks`，处理实体死亡等情况
4. ✅ **块大小配置**: 添加了`--data-chunk-length-st`参数，可以为GTrXL单独设置更大的块大小

## 技术细节

### GTrXL时间注意力

- 使用门控机制（Gated Transformer）提高训练稳定性
- 支持memory机制，可以访问历史信息
- 正确处理时间掩码，避免关注无效时间步

### 空间注意力

- 多头注意力机制处理多实体交互
- 支持实体掩码，处理实体死亡等情况
- 残差连接和层归一化保证训练稳定性

### 实体嵌入

- 分别处理我机和相对信息
- 统一映射到相同维度便于后续处理
- 支持不同数量的友军和敌机

## 后续改进建议

1. 优化GTrXL实现，提高计算效率
2. 添加更多注意力机制选项（如Transformer-XL的其他变体）
3. 在runner层自动处理memories格式转换，使使用更加透明
4. 添加对variable-length sequences的更好支持
5. 添加更多的超参数调优建议和实验经验

## 常见问题

**Q: 如何选择embed_dim？**
A: 通常设置为128或256，根据观察空间大小和计算资源调整。

**Q: 如何选择memory_length？**
A: 建议设置为与data_chunk_length相同或稍大，如32、64或128。

**Q: 如何选择注意力头数？**
A: 通常设置为8，确保embed_dim能被num_heads整除。

**Q: 训练速度慢怎么办？**
A: 可以减小embed_dim、num_temporal_layers或memory_length，或者使用更小的data_chunk_length_st。

**Q: 内存占用大怎么办？**
A: 可以减小memory_length、num_temporal_layers或batch_size。
