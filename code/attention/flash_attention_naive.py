"""
FlashAttention 的朴素实现：分块 + 在线 softmax
================================================

FlashAttention 的核心不是"少算"，而是"少读写显存"：
标准注意力要把 N×N 的分数矩阵写回 HBM 再读出来，显存占用 O(N²)；
FlashAttention 把 Q/K/V 切块，在 SRAM 里边算边归一化，**从不完整物化 N×N 矩阵**。

关键技巧是 **online softmax**：
    普通 softmax 需要先知道全局最大值才能算 exp，
    但分块时你只看到一部分分数。于是维护运行最大值 m 和运行和 l，
    每处理一个新块就用 rescale 校正之前的累积结果：

        m_new = max(m, max(s_block))
        l     = l * exp(m - m_new) + sum(exp(s_block - m_new))
        acc   = acc * exp(m - m_new) + exp(s_block - m_new) @ V_block

    处理完所有块后 out = acc / l，结果与一次性 softmax **完全等价**。

面试考点：
  1. 标准注意力的显存瓶颈是 N×N 的分数矩阵（N=8K 时 8192²×2B = 128MB/头/层）
  2. FlashAttention 把 N×N 的显存降到 O(N·d)，但 FLOPs 并没有减少，
     甚至因为重算略有增加 —— 收益来自 **减少 HBM 读写**
  3. 它是 IO-aware 的优化，属于 memory-bound 优化，与稀疏注意力（减少计算量）正交

常见追问：
  - "训练时反向怎么处理？" → 不保存中间激活，反向时用保存的 (m, l) 重新计算
  - "为什么实际实现比这个快得多？" → 这个 Python 版本仍然在 HBM 上操作，
    真正加速需要把块放进 SRAM（CUDA kernel），并用 warp 级并行
  - "FlashAttention-2/3 改进什么？" → 2 减少非矩阵乘的 FLOP、优化 warp 分工；
    3 针对 Hopper 的 TMA / 异步流水和 FP8

运行：python code/attention/flash_attention_naive.py
"""

import math

import torch
import torch.nn.functional as F


def flash_attention_naive(q, k, v, block_size=64, causal=False):
    """
    分块计算注意力，结果与标准实现一致。

    Args:
        q, k, v: (B, H, N, D)
        block_size: 分块大小
        causal: 是否使用因果掩码

    Returns:
        (B, H, N, D)

    注意：这里为了可读性仍然用 Python 循环 + HBM 张量操作，
    它演示的是**算法**，不是**性能**。
    """
    b, h, n, d = q.shape
    scale = 1.0 / math.sqrt(d)
    out = torch.empty_like(q)

    for i in range(0, n, block_size):
        q_end = min(i + block_size, n)
        q_block = q[:, :, i:q_end]
        bs = q_end - i

        # 运行统计量：最大值 m、归一化和 l、加权累积 acc
        m = torch.full((b, h, bs, 1), float("-inf"))
        l = torch.zeros((b, h, bs, 1))
        acc = torch.zeros((b, h, bs, d))

        for j in range(0, n, block_size):
            # 因果情况下，当整个 K 块都在所有 Q 之后，可以直接停止
            if causal and j > i + bs - 1:
                break

            k_end = min(j + block_size, n)
            k_block = k[:, :, j:k_end]
            v_block = v[:, :, j:k_end]

            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

            if causal:
                qi = torch.arange(i, q_end).unsqueeze(1)
                kj = torch.arange(j, k_end).unsqueeze(0)
                scores = scores.masked_fill(~(kj <= qi), float("-inf"))

            # ---- online softmax 的三行核心 ----
            m_new = torch.maximum(m, scores.max(dim=-1, keepdim=True).values)
            p = torch.exp(scores - m_new)                    # 本块未归一化权重
            l = l * torch.exp(m - m_new) + p.sum(dim=-1, keepdim=True)
            acc = acc * torch.exp(m - m_new) + torch.matmul(p, v_block)
            m = m_new

        out[:, :, i:q_end] = acc / l

    return out


def _test():
    torch.manual_seed(0)
    b, h, n, d = 1, 2, 130, 16          # 130 不是 block_size 的整数倍，顺带测边界
    q = torch.randn(b, h, n, d)
    k = torch.randn(b, h, n, d)
    v = torch.randn(b, h, n, d)

    for causal in [False, True]:
        for bs in [16, 32, 64]:
            out = flash_attention_naive(q, k, v, block_size=bs, causal=causal)
            ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            err = (out - ref).abs().max().item()
            assert torch.allclose(out, ref, atol=1e-4), \
                f"causal={causal}, block={bs}: 最大误差 {err:.2e}"
        print(f"causal={causal}: 各分块大小结果均与标准实现一致 ✓")

    # --- 展示分块的好处：工作集显存从 O(N²) 降到 O(block²) ---
    n = 8192
    print(f"\nN={n} 时单个注意力头的分数矩阵:")
    print(f"  标准实现 N×N      : {n*n*2/1024**2:8.1f} MB")
    for bs in [128, 256]:
        print(f"  Flash 分块 {bs:4d}   : {bs*n*2/1024**2:8.1f} MB  (每个 Q 块的工作集)")

    print("\nflash_attention_naive.py 全部通过")


if __name__ == "__main__":
    _test()
