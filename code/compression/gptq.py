"""
GPTQ：基于 Hessian 的逐列误差补偿量化
==================================

核心：量化第 j 列产生的误差 e = q_j - w_j，不要浪费，而是把它
按 Hessian 逆的比例补偿到**还没量化的列**上：

    w_rest += e_j * Hinv[j, rest] / Hinv[j, j]

这是 OBS（Optimal Brain Surgeon）在大规模线性层上的工程化。
目标函数是层输出误差 ||WX - W_hat X||^2，等价于逐行最小化
(w - w_hat)^T H (w - w_hat)，其中 H = 2 X X^T。

面试考点：
  1. 为什么用 Hessian 而不是逐元素 QAT？→ 层输出误差，不是权重误差
  2. H = 2XX^T 的形状是 [in, in]，与输出维度无关，行与行独立
  3. dampening 的作用：H 奇异时（calibration 样本 < 输入维度）保证可逆
  4. group size 与 act-order 的取舍（生产上常二选一）

常见追问：
  - "GPTQ 和 OBS 什么关系？" → OBS 是删权重，GPTQ 是量化权重，
    补偿公式完全一样，只是约束不同
  - "为什么逐层做？" → 全局 Hessian 不可行；层独立性假设足够好
  - "INT4 为什么必须 per-group？" → 见 quantize.py

运行：python code/compression/gptq.py
"""

import torch


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def rtn_quantize(W, bits=4, group_size=None):
    """Round-to-nearest 基线：每个权重独立量化，不做任何补偿。"""
    qmax = 2 ** (bits - 1) - 1

    if group_size is None:
        scale = (W.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-8)
        return torch.round(W / scale).clamp(-qmax, qmax) * scale

    Wg = W.reshape(W.shape[0], -1, group_size)
    scale = (Wg.abs().amax(dim=-1, keepdim=True) / qmax).clamp_min(1e-8)
    Q = torch.round(Wg / scale).clamp(-qmax, qmax) * scale
    return Q.reshape(W.shape)


def _group_scales(W, bits, group_size):
    qmax = 2 ** (bits - 1) - 1
    if group_size is None:
        return (W.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-8)
    Wg = W.reshape(W.shape[0], -1, group_size)
    return (Wg.abs().amax(dim=-1, keepdim=True) / qmax).clamp_min(1e-8)


def gptq_quantize(W, X, bits=4, group_size=None, damp=0.01):
    """
    GPTQ 的核心循环（清晰版，非生产级加速）。

    W: [out, in]
    X: [in, n_samples]  该层的校准输入
    """
    qmax = 2 ** (bits - 1) - 1
    W = W.clone().float()
    n = W.shape[1]

    # Hessian：层输出误差的二阶近似
    H = 2.0 * X @ X.T
    # dampening：避免校准样本不足导致的奇异
    H += damp * torch.diag(H).mean() * torch.eye(n, dtype=H.dtype)
    Hinv = torch.linalg.inv(H)

    # scale 从**原始**权重统计，逐个 group 固定
    scales = _group_scales(W, bits, group_size)

    Q = torch.zeros_like(W)
    for j in range(n):
        w = W[:, j]

        if group_size is None:
            s = scales[:, 0]
        else:
            s = scales[:, j // group_size, 0]

        q = torch.round(w / s).clamp(-qmax, qmax) * s
        Q[:, j] = q

        # 把量化误差补偿到剩余列
        err = (w - q) / Hinv[j, j]
        if j + 1 < n:
            W[:, j + 1:] -= err.unsqueeze(1) * Hinv[j, j + 1:].unsqueeze(0)

    return Q


def output_error(W, Q, X):
    """层输出误差 ||WX - QX||_F"""
    return ((W - Q) @ X).norm().item()


def main():
    torch.manual_seed(0)

    out_features, in_features, n_samples = 32, 64, 512
    W = torch.randn(out_features, in_features) * 0.02
    X = torch.randn(in_features, n_samples)

    for group_size in (None, 32):
        name = "per-channel" if group_size is None else f"per-group(g={group_size})"
        Q_rtn = rtn_quantize(W, bits=4, group_size=group_size)
        Q_gptq = gptq_quantize(W, X, bits=4, group_size=group_size)

        e_rtn = output_error(W, Q_rtn, X)
        e_gptq = output_error(W, Q_gptq, X)

        _check(e_gptq < e_rtn, f"{name}: GPTQ 应优于 RTN，实际 {e_gptq:.4f} vs {e_rtn:.4f}")
        print(f"[{name}] 输出误差：RTN={e_rtn:.4f} → GPTQ={e_gptq:.4f} "
              f"（降低 {100 * (1 - e_gptq / e_rtn):.1f}%）  OK")

    # 分组更细时误差应更小
    e_ch = output_error(W, gptq_quantize(W, X, 4, None), X)
    e_g32 = output_error(W, gptq_quantize(W, X, 4, 32), X)
    _check(e_g32 < e_ch, "per-group 应优于 per-channel")
    print(f"[g=32 对比] per-channel={e_ch:.4f} > per-group={e_g32:.4f}  OK")

    # 8 bit 时 GPTQ 的优势应缩小（误差本来就小）
    e8_rtn = output_error(W, rtn_quantize(W, 8, None), X)
    e8_gptq = output_error(W, gptq_quantize(W, X, 8, None), X)
    print(f"[8bit] 输出误差：RTN={e8_rtn:.5f} → GPTQ={e8_gptq:.5f}")
    _check(e8_gptq <= e8_rtn * 1.05, "8bit 下 GPTQ 不应明显变差")
    print("[8bit 对比] GPTQ 不劣于 RTN（补偿收益随比特增加而缩小）  OK")

    print("\n全部通过。")


if __name__ == "__main__":
    main()
