"""
多头注意力（Multi-Head Attention）
==================================

面试考点：
  1. 拆头 / 合头：为什么要 reshape + transpose，为什么是 head 维在中间
  2. 参数量：4 * d * d（W_Q, W_K, W_V, W_O 各 d*d）
  3. 为什么多头不增加计算量：每个头 d/h 维，h 个头加起来等价于单头 d 维
  4. W_O 的作用：把各头独立的结果重新混合

常见追问：
  - "参数量是多少？" → 4d^2，与头数 h 无关。
  - "多头和单头比好在哪？" → 不同子空间学不同模式（局部/句法/指代）。
  - "h 越大越好吗？" → 不是。h 太大时每个头维度 d/h 太小，表达能力下降，
    通常 d_k >= 32~64。

运行：python code/attention/mha.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from self_attention import causal_mask


class MultiHeadAttention(nn.Module):
    """
    标准 MHA。用 nn.Linear 显式实现投影，方便数参数、方便讲清楚。

    输入输出都是 (batch, seq_len, d_model)。
    """

    def __init__(self, d_model, num_heads, bias=False, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """(B, N, D) -> (B, H, N, d_k)"""
        b, n, _ = x.shape
        # 先切成 (B, N, H, d_k)，再转置成 head 维度在前
        return x.view(b, n, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x):
        """(B, H, N, d_k) -> (B, N, D)"""
        b, h, n, d_k = x.shape
        return x.transpose(1, 2).reshape(b, n, h * d_k).contiguous()

    def forward(self, x, mask=None):
        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))

        out = torch.matmul(attn, v)          # (B, H, N, d_k)
        out = self._merge_heads(out)          # (B, N, D)
        return self.w_o(out)


def _test():
    torch.manual_seed(0)
    b, n, d_model, h = 2, 5, 32, 4
    mha = MultiHeadAttention(d_model, h).eval()
    x = torch.randn(b, n, d_model)

    out = mha(x)
    assert out.shape == (b, n, d_model), out.shape

    # --- 参数量应为 4 * d^2（bias=False）---
    n_params = sum(p.numel() for p in mha.parameters())
    assert n_params == 4 * d_model * d_model, f"参数应为 {4*d_model*d_model}, 实际 {n_params}"
    print(f"d_model={d_model}: 参数量 {n_params} = 4 * {d_model}^2（与头数无关）")

    # --- 和单头对比：同样 d_model 下计算量相当 ---
    single = MultiHeadAttention(d_model, 1).eval()
    assert sum(p.numel() for p in single.parameters()) == n_params

    # --- causal mask：位置 0 只能看到自己 ---
    m = causal_mask(n).unsqueeze(0).unsqueeze(0)      # (1,1,N,N) 广播到 (B,H,N,N)
    out_c = mha(x, m)
    assert out_c.shape == (b, n, d_model)

    # 验证因果性：改动后面的 token，前面的输出不应变化
    # 注意只比较位置 0..2 —— 位置 3 之后本就应该受影响
    x2 = x.clone()
    x2[:, 3:] += 100.0
    out_c2 = mha(x2, m)
    assert torch.allclose(out_c[:, :3], out_c2[:, :3], atol=1e-5), \
        "未来 token 影响了历史输出，mask 没生效"

    print("mha.py 全部通过")


if __name__ == "__main__":
    _test()
