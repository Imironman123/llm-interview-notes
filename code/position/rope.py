"""
旋转位置编码（Rotary Position Embedding, RoPE）
================================================

核心思想：
    不把位置信息「加」到 embedding 上，而是把 q/k 向量按位置旋转一个角度。
    由于旋转矩阵的正交性：
        <R(m)q, R(n)k> = <q, R(n-m)k>
    内积只依赖相对距离 n-m，天然编码相对位置。

    而且是**乘性**注入，不改变向量长度，不引入额外参数。

面试考点：
  1. 为什么是相对位置编码？（推导上面那个内积恒等式）
  2. 为什么是「旋转」而不是加法？—— 乘性保持范数，且相对位置性质严格成立
  3. 二维情形下就是把 (x1,x2) 旋转 theta*m 弧度，theta_i = 10000^(-2i/d)
  4. 远程衰减：高频分量随距离快速震荡抵消，内积随距离下降
  5. 外推问题：超出训练长度后性能下降，因此有 NTK-aware / YaRN / 线性内插等扩展方法

常见追问：
  - "RoPE 加在哪几层？" → 每一层 attention 前，对 q 和 k 都旋转（v 不旋转）
  - "怎么扩展到更长上下文？" → 调低 theta 底数（NTK 缩放）、位置内插、
    YaRN 对不同频率分量分段处理
  - "和 ALiBi 比？" → ALiBi 直接在 attention score 上加减距离偏置，更简单、
    外推更好；RoPE 在长上下文扩展方法上生态更成熟

运行：python code/position/rope.py
"""

import torch


def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """
    预计算每个位置的旋转角度，用复数表示 e^{i * m * theta_j}。

    theta_j = theta^(-2j/dim), j = 0, 1, ..., dim/2 - 1
    维度越高，旋转频率越低。

    Returns:
        complex tensor, shape (max_seq_len, dim // 2)
    """
    # 只取偶数下标：0, 2, 4, ... —— 每两个相邻维度组成一个平面
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    pos = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, freqs)                      # (max_seq_len, dim//2)
    return torch.polar(torch.ones_like(freqs), freqs)    # 半径 1 的复数


def apply_rotary_emb(x, freqs_cis):
    """
    对 q/k 应用旋转。

    Args:
        x: (B, N, H, D)，最后一维 D 必须是偶数
        freqs_cis: (N, D//2) 或 (1, N, 1, D//2)

    实现方式：把相邻两个维度 (x[2i], x[2i+1]) 看成复数，
    乘上 e^{i*m*theta_i} 即完成旋转。
    """
    # (B, N, H, D) -> (B, N, H, D/2) 的复数
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    if freqs_cis.dim() == 2:
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)   # (1, N, 1, D/2)
    out = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return out.to(x.dtype)


def _rope_at(x, pos, freqs_cis):
    """把长度为 D 的向量 x 按绝对位置 pos 旋转，便于做单向量实验。"""
    return apply_rotary_emb(
        x.view(1, 1, 1, -1), freqs_cis[pos:pos + 1]
    ).view(-1)


def _test():
    torch.manual_seed(0)
    dim, max_len = 16, 64
    freqs_cis = precompute_freqs_cis(dim, max_len)
    assert freqs_cis.shape == (max_len, dim // 2)

    # --- 形状与数值稳定性 ---
    x = torch.randn(2, 8, 4, dim)
    out = apply_rotary_emb(x, freqs_cis[:8])
    assert out.shape == x.shape
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5), \
        "旋转应保持向量范数不变"

    # --- 位置 0 旋转 0 角度，应当恒等 ---
    assert torch.allclose(_rope_at(x[0, 0, 0], 0, freqs_cis), x[0, 0, 0]), \
        "位置 0 应当不做任何旋转"

    # --- 核心性质：内积只依赖相对距离 ---
    q = torch.randn(dim)
    k = torch.randn(dim)

    def score(m, n):
        return torch.dot(_rope_at(q, m, freqs_cis), _rope_at(k, n, freqs_cis)).item()

    for m, n in [(3, 7), (10, 14), (20, 24)]:
        for shift in [1, 5, 13]:
            if max(m, n) + shift >= max_len:
                continue
            a, b = score(m, n), score(m + shift, n + shift)
            assert abs(a - b) < 1e-4, f"平移不变性破坏: s({m},{n})={a:.6f} vs ({m+shift},{n+shift})={b:.6f}"

    print(f"相对位置性质验证通过：s(3,7)={score(3,7):.6f}, "
          f"s(10,14)={score(10,14):.6f}, s(20,24)={score(20,24):.6f}")

    # --- 远程衰减 ---
    # <R(m)x, R(n)x> = sum_j |x_j|^2 * cos((n-m) * theta_j)
    # 不同频率的余弦随距离错开相位、相互抵消，内积总体上向 0 衰减。
    # 注意是「总体趋势」而非单调 —— 各频率相位交错，实际曲线会振荡。
    # 这里必须用「同一向量与自身」—— 随机独立的 q,k 基向量内积本身就大，
    # 会把位置效应完全淹没。
    x = torch.randn(dim)
    print(f"\n远程衰减：向量范数平方 = {x.norm()**2:.2f}")
    print("  距离 -> 与自身的旋转内积（总体衰减，有振荡，并非单调）:")
    for dist in [0, 1, 4, 16, 32, 63]:
        if dist >= max_len:
            continue
        s = torch.dot(_rope_at(x, 0, freqs_cis), _rope_at(x, dist, freqs_cis)).item()
        print(f"    Δ={dist:3d}: {s:7.4f}")

    print("\nrope.py 全部通过")


if __name__ == "__main__":
    _test()
