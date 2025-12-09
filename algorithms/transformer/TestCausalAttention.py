import torch
import unittest
import numpy as np

# 引用你的模块
from spatio_temporal import GatedTransformerBlock


class TestCausalAttention(unittest.TestCase):

    def setUp(self):
        self.d_model = 32
        self.nhead = 4
        self.model = GatedTransformerBlock(self.d_model, self.nhead).eval()

    def test_future_leakage(self):
        """
        核心测试：未来的扰动不应影响现在的输出
        """
        seq_len = 5
        batch_size = 1

        # 1. 创建初始输入 (Seq, Batch, Dim)
        x_orig = torch.randn(seq_len, batch_size, self.d_model)

        # 2. 获取基准输出
        with torch.no_grad():
            y_orig = self.model(x_orig, mems=None)

        # 3. 扰动未来：修改最后一个时间步的数据
        x_perturbed = x_orig.clone()
        # 将 t=4 (最后一个) 的数据加上巨大的噪声
        x_perturbed[-1] = x_perturbed[-1] + 100.0

        # 4. 获取扰动后的输出
        with torch.no_grad():
            y_perturbed = self.model(x_perturbed, mems=None)

        # 5. 检查 t=0 (第一个时间步) 的输出是否发生变化
        # 理论上：t=0 的输出只取决于 x[0] 和 mems。x[4] 的变化不应传导到 y[0]。

        diff = torch.abs(y_orig[0] - y_perturbed[0]).max().item()

        print(f"\n[Causality Test]")
        print(f"Difference at t=0 after changing t={seq_len - 1}: {diff:.8f}")

        # 允许极小的浮点误差，但在 float32 下应该是 0.0 或 1e-7 级别
        self.assertTrue(diff < 1e-6, f"Causality Broken! Past changed by {diff} when Future changed.")

        # 同时也检查一下 t=last 应该变了 (Sanity Check)
        diff_last = torch.abs(y_orig[-1] - y_perturbed[-1]).max().item()
        print(f"Difference at t={seq_len - 1} (Should be large): {diff_last:.8f}")
        self.assertTrue(diff_last > 1e-3, "Model ignored input entirely?")

    def test_with_memory_mask(self):
        """
        测试带有 Memory 情况下的 Mask 形状逻辑
        """
        mem_len = 10
        seq_len = 5
        batch_size = 1

        mems = torch.randn(mem_len, batch_size, self.d_model)
        x = torch.randn(seq_len, batch_size, self.d_model)

        # 只要能跑通且不报错，且通过上面的 leakage 测试，说明 Memory Mask 拼接正确
        try:
            out = self.model(x, mems=mems)
            print("\n[Memory Test] Forward with memory successful.")
        except Exception as e:
            self.fail(f"Forward pass with memory failed: {e}")


if __name__ == '__main__':
    unittest.main()