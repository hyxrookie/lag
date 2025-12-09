import torch
import numpy as np
import unittest

# 假设你的代码保存在 spatio_temporal.py 中
from spatio_temporal import EntityEmbedder, SpatialAttention, TemporalGTrXL


class TestSpatioTemporal(unittest.TestCase):

    def setUp(self):
        """初始化测试所需的配置和模型"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 4
        self.hidden_size = 64  # d_model

        # 模拟环境配置 (对应你蓝图中的结构)
        self.config = {
            'own_dim': 15,
            'ally_dim': 10,
            'ally_num': 2,
            'enemy_dim': 12,
            'enemy_num': 3,  # 假设有3架敌机
            'missile_dim': 8,
            'missile_num': 4  # 假设有4枚导弹
        }

        # 计算总的扁平输入维度 (MLP输入层的大小)
        self.total_input_dim = (
                self.config['own_dim'] +
                self.config['ally_dim'] * self.config['ally_num'] +
                self.config['enemy_dim'] * self.config['enemy_num'] +
                self.config['missile_dim'] * self.config['missile_num']
        )

        # 实例化模型
        self.embedder = EntityEmbedder(None, self.hidden_size, self.config).to(self.device)
        self.spatial = SpatialAttention(d_model=self.hidden_size, nhead=4).to(self.device)

        # GTrXL 配置: 2层, 记忆长度 16
        self.mem_len = 16
        self.n_layers = 2
        self.temporal = TemporalGTrXL(d_model=self.hidden_size, n_layers=self.n_layers, nhead=4,
                                      mem_len=self.mem_len).to(self.device)

    def test_1_entity_embedding_shapes(self):
        """测试实体嵌入层的输出形状"""
        print("\n=== Test 1: Entity Embedding Shapes ===")

        # 构造随机输入 (Batch, Total_Flat_Dim)
        dummy_obs = torch.randn(self.batch_size, self.total_input_dim).to(self.device)

        tokens, mask = self.embedder(dummy_obs)

        # 预期实体数量 = 1 (Own) + 2 (Ally) + 3 (Enemy) + 4 (Missile) = 10
        expected_entities = 1 + self.config['ally_num'] + self.config['enemy_num'] + self.config['missile_num']

        print(f"Input Shape: {dummy_obs.shape}")
        print(f"Tokens Shape: {tokens.shape}")  # Should be (Batch, 10, 64)
        print(f"Mask Shape:   {mask.shape}")  # Should be (Batch, 10)

        self.assertEqual(tokens.shape, (self.batch_size, expected_entities, self.hidden_size))
        self.assertEqual(mask.shape, (self.batch_size, expected_entities))
        self.assertFalse(mask[0, 0].item())  # Ownship 永远不应该被 mask (False 表示不 mask)

    def test_2_masking_logic(self):
        """测试当实体为0时，Mask是否正确生成 (True表示被Mask)"""
        print("\n=== Test 2: Masking Logic ===")

        # 构造全0输入
        zero_obs = torch.zeros(self.batch_size, self.total_input_dim).to(self.device)

        # 只有 Ownship 部分给值，其他部分保持 0
        own_dim = self.config['own_dim']
        zero_obs[:, 0:own_dim] = torch.randn(self.batch_size, own_dim)

        tokens, mask = self.embedder(zero_obs)

        print(f"Mask Sample (First Batch): {mask[0]}")

        # Ownship (Index 0) 应该是 False (有效)
        self.assertEqual(mask[0, 0].item(), False, "Ownship should not be masked")

        # Ally 1 (Index 1) 应该是 True (被Mask，因为输入全是0)
        self.assertEqual(mask[0, 1].item(), True, "Zero-input Ally should be masked")

        # 验证 Mask 对 Spatial Attention 的影响
        # 如果 Mask 工作正常，输出不应包含 NaN，且计算应该忽略被 Mask 的部分
        output = self.spatial(tokens, mask)
        print(f"Spatial Output Shape: {output.shape}")
        self.assertFalse(torch.isnan(output).any(), "Spatial output contains NaNs")

    def test_3_temporal_memory_flow(self):
        """测试 GTrXL 的记忆传递与形状 (修正版: 适应固定长度Memory)"""
        print("\n=== Test 3: Temporal Memory Flow (GTrXL) ===")

        # 模拟 Spatial 的输出 (Batch, Dim) -> 扩充为 (Batch, Seq=1, Dim)
        spatial_out = torch.randn(self.batch_size, 1, self.hidden_size).to(self.device)

        # --- Step 1: Cold Start (No Memory) ---
        # 网络内部会将其初始化为全0的 (Layers, Mem_Len, Batch, Dim)
        out_t1, mems_t1 = self.temporal(spatial_out, mems=None)

        print(f"Step 1 Output: {out_t1.shape}")
        print(f"Step 1 Mems:   {mems_t1.shape}")

        self.assertEqual(out_t1.shape, (self.batch_size, 1, self.hidden_size))

        # 修正：现在的实现是固定长度初始化，所以应该是 mem_len (16)，而不是 1
        self.assertEqual(mems_t1.shape, (self.n_layers, self.mem_len, self.batch_size, self.hidden_size))

        # --- Step 2: Warm Start (With Memory) ---
        # 再次输入 (模拟下一帧)
        spatial_out_2 = torch.randn(self.batch_size, 1, self.hidden_size).to(self.device)
        out_t2, mems_t2 = self.temporal(spatial_out_2, mems=mems_t1)

        print(f"Step 2 Mems:   {mems_t2.shape}")
        # 长度依然保持 mem_len (16)
        self.assertEqual(mems_t2.shape[1], self.mem_len)

    def test_4_full_pipeline_simulation(self):
        """模拟 Actor 的一次完整 Forward"""
        print("\n=== Test 4: Full Pipeline Simulation ===")

        obs = torch.randn(self.batch_size, self.total_input_dim).to(self.device)
        # 初始化空 Memory (模拟第一步)
        # 注意：在实际代码中，None会被处理为 zeros，这里我们手动给 None
        mems = None

        # 1. Embed
        x_emb, x_mask = self.embedder(obs)

        # 2. Spatial
        h_spatial = self.spatial(x_emb, x_mask)

        # 3. Temporal (需要增加 seq 维度)
        h_spatial_seq = h_spatial.unsqueeze(1)  # (Batch, 1, Dim)
        h_temporal, next_mems = self.temporal(h_spatial_seq, mems)

        # 4. Result
        actor_features = h_temporal.squeeze(1)

        print(f"Final Feature Shape: {actor_features.shape}")
        print(f"Next Memory Shape:   {next_mems.shape}")

        self.assertEqual(actor_features.shape, (self.batch_size, self.hidden_size))
        # 验证梯度流是否通畅 (简单的 check)
        loss = actor_features.sum()
        loss.backward()
        print("Backward pass successful.")


if __name__ == '__main__':
    unittest.main()