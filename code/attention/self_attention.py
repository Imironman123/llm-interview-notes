"""
缩放点积注意力（Scaled Dot-Product Attention）
================================================

面试考点：
  1. 公式：softmax(QK^T / sqrt(d_k) + M) V
  2. 为什么要除 sqrt(d_k)？
  3. 时间复杂度 O(n^2 d)，空间复杂度 O(n^2)
  4. causal mask 怎么加，为什么用 -inf 而不是 0

常见追问：
  - "如果不除 sqrt(d_k) 会怎样？" → 点积方差是 d_k，softmax 饱和成 one-hot，
    梯度趋近 0，训练不收敛。
  - "为什么 mask 用 -inf 不用 0？" → softmax 前加 0 不起作用（e^0 = 1，
    等于让被 mask 的位置参与归一化）；-inf 经 exp 后为 0，权重才真正归零。
  - "能省掉 mask 吗？" → 训练时不能，因为要并行处理整条序列，未来 token 的信息
    会泄漏；推理时若逐 token 生成则天然满足因果性。

运行：python code/attention/self_attention.py
"""

import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Args:
        q: (..., n_q, d_k)   查询
        k: (..., n_k, d_k)   键（不走 mask 时 n_k == n_q）
        v: (..., n_k, d_v)   值
        mask: (..., n_q, n_k) 布尔张量，True 表示保留，False 表示被屏蔽。
              也可传 0/1 的整型 mask。

    Returns:
        (output, attn_weights)
        output: (..., n_q, d_v)
        attn_weights: (..., n_q, n_k)，每行和为 1
    """
    d_k = q.size(-1)

    # 1) 打分。除以 sqrt(d_k) 是数值稳定的关键，见模块注释。
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # 2) mask：被屏蔽的位置填 -inf，softmax 后自动变 0
    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask.bool()
        scores = scores.masked_fill(~mask, float("-inf"))

    # 3) 归一化 + 4) 加权求和
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out, attn


def causal_mask(n_q, n_k=None, device=None):
    """
    下三角掩码：位置 i 只能看到 j <= i。

    返回布尔张量，True = 可见。
    n_k != n_q 时按右下角对齐（用于 KV Cache 解码：只有 1 个 query，
    但要看到全部历史的 n_k 个 key）。
    """
    n_k = n_q if n_k is None else n_k
    q_idx = torch.arange(n_q, device=device).unsqueeze(1)      # (n_q, 1)
    k_idx = torch.arange(n_k, device=device).unsqueeze(0)      # (1, n_k)
    # 右下角对齐：query 的绝对位置 = 已生成长度 - n_q + 自身下标
    offset = n_k - n_q
    return k_idx <= (q_idx + offset)


def _test():
    torch.manual_seed(0)
    b, n, d_k, d_v = 2, 6, 8, 4
    q = torch.randn(b, n, d_k)
    k = torch.randn(b, n, d_k)
    v = torch.randn(b, n, d_v)

    # --- 基本形状与归一化 ---
    out, attn = scaled_dot_product_attention(q, k, v)
    assert out.shape == (b, n, d_v), out.shape
    assert attn.shape == (b, n, n)
    assert torch.allclose(attn.sum(-1), torch.ones(b, n), atol=1e-6), "每行权重和应为 1"

    # --- 用 torch 官方实现做交叉验证 ---
    ref = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, ref, atol=1e-6), "与 F.scaled_dot_product_attention 不一致"

    # --- causal mask 的因果性 ---
    m = causal_mask(n)
    out_c, attn_c = scaled_dot_product_attention(q, k, v, m)
    # 第 i 行只能看到 0..i，后面的权重必须是 0
    for i in range(n):
        assert torch.allclose(attn_c[:, i, i + 1:], torch.zeros(b, n - i - 1)), \
            f"位置 {i} 泄漏了未来信息"

    # --- 不除 sqrt(d_k) 会怎样：演示 softmax 饱和 ---
    big = 64
    q2 = torch.randn(1, 1, big) * 3
    k2 = torch.randn(1, 1, big) * 3
    raw = torch.matmul(q2, k2.transpose(-2, -1))
    scaled = raw / math.sqrt(big)
    print(f"d_k={big} 时点积绝对值: 未缩放 {raw.abs().mean():.1f}, 缩放后 {scaled.abs().mean():.2f}")

    print("self_attention.py 全部通过")


if __name__ == "__main__":
    _test()
