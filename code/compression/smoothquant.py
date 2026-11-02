"""
SmoothQuant：把激活的量化难度迁移到权重
====================================

线性层满足缩放不变性：

    Y = X W = (X / s) (W * s)

SmoothQuant 逐输入通道选 s：

    s_j = max|X_j|^alpha / max|W_j|^(1-alpha)

alpha = 0.5 时两边动态范围的几何平均相等：

    max|X_j| / s_j = max|W_ij| * s_j = sqrt(max|X_j| * max|W_j|)

直观：激活有 outlier 通道，权重相对平滑。把 outlier 的「难度」
搬一部分给权重，让两者都变得能塞进 INT8 的 127 个码点。

工程上 s 可以融合进前置的 LayerNorm（diag 缩放可吸收），推理零开销。

对比 AWQ：
  - SmoothQuant 缩放激活（W8A8 场景，激活也要量化）
  - AWQ 缩放权重（W4A16 场景，只量化权重）

运行：python code/compression/smoothquant.py
"""

import torch
import torch.nn.functional as F


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def smooth_scale(W, X, alpha=0.5):
    """W: [out, in], X: [in, n] -> s: [in]"""
    a = X.abs().amax(dim=1).clamp_min(1e-8)
    b = W.abs().amax(dim=0).clamp_min(1e-8)
    return a.pow(alpha) / b.pow(1.0 - alpha)


def smooth(W, X, alpha=0.5):
    s = smooth_scale(W, X, alpha)
    return W * s.unsqueeze(0), X / s.unsqueeze(1), s


def quant_per_tensor_sym(x, bits=8):
    qmax = 2 ** (bits - 1) - 1
    scale = (x.abs().amax() / qmax).clamp_min(1e-8)
    return torch.round(x / scale).clamp(-qmax, qmax) * scale


def quant_per_channel_sym(x, bits=8):
    qmax = 2 ** (bits - 1) - 1
    scale = (x.abs().amax(dim=-1, keepdim=True) / qmax).clamp_min(1e-8)
    return torch.round(x / scale).clamp(-qmax, qmax) * scale


def mse(x, y):
    return (x - y).float().pow(2).mean().item()


def main():
    torch.manual_seed(0)

    out_f, in_f, n = 32, 64, 256
    W = torch.randn(out_f, in_f) * 0.05
    X = torch.randn(in_f, n) * 0.5
    # 构造 3 个 outlier 输入通道：比正常通道大 100 倍
    X[:3] *= 100.0

    # ---------- 1. 数学等价 ----------
    W2, X2, s = smooth(W, X, alpha=0.5)
    y0 = W @ X
    y1 = W2 @ X2
    _check(torch.allclose(y0, y1, atol=1e-4), "缩放后输出必须完全等价")
    print("[1] (W*s) @ (X/s) == W @ X  OK")

    # ---------- 2. alpha=0.5 让两侧动态范围相等 ----------
    x_max = X2.abs().amax(dim=1)          # 每个输入通道
    w_max = W2.abs().amax(dim=0)          # 每个输入通道
    _check(torch.allclose(x_max, w_max, rtol=1e-4),
           f"alpha=0.5 应使两侧 amax 相等，最大偏差 {(x_max - w_max).abs().max():.2e}")
    print("[2] alpha=0.5 时 max|X/s| == max|W*s|（几何平均平衡）  OK")

    # ---------- 3. 激活量化误差显著下降 ----------
    q_before = quant_per_tensor_sym(X, bits=8)
    err_before = mse(X, q_before)

    q_after = quant_per_tensor_sym(X2, bits=8) * s.unsqueeze(1)
    err_after = mse(X, q_after)

    _check(err_after < err_before * 0.2,
           f"平滑后激活量化误差应大幅下降：{err_before:.5f} -> {err_after:.5f}")
    print(f"[3] 激活量化 MSE：{err_before:.5f} → {err_after:.5f} "
          f"（下降 {100 * (1 - err_after / err_before):.1f}%）  OK")

    # ---------- 4. W8A8 的端到端输出误差下降 ----------
    # 未平滑：权重 per-channel，激活 per-tensor
    y_q_before = quant_per_channel_sym(W, 8) @ quant_per_tensor_sym(X, 8)
    err_out_before = (y0 - y_q_before).norm().item()

    # 平滑后：同样 W8A8
    y_q_after = quant_per_channel_sym(W2, 8) @ quant_per_tensor_sym(X2, 8)
    err_out_after = (y0 - y_q_after).norm().item()

    _check(err_out_after < err_out_before,
           f"W8A8 输出误差应下降：{err_out_before:.3f} -> {err_out_after:.3f}")
    print(f"[4] W8A8 输出误差：{err_out_before:.3f} → {err_out_after:.3f}  OK")

    # ---------- 5. alpha 的两个极端都会失衡 ----------
    def imbalance(alpha):
        Wa, Xa, _ = smooth(W, X, alpha)
        wmax = Wa.abs().amax().item()
        xmax = Xa.abs().amax().item()
        return max(wmax, xmax) / min(wmax, xmax)

    imb_0 = imbalance(0.0)
    imb_half = imbalance(0.5)
    imb_1 = imbalance(1.0)

    _check(imb_0 > 3 and imb_1 > 3, f"alpha 极端值应失衡：{imb_0:.1f}, {imb_1:.1f}")
    _check(imb_half < 1.01, f"alpha=0.5 应平衡，实际比值 {imb_half:.3f}")
    print(f"[5] 范围失衡比：alpha=0 → {imb_0:.1f}，alpha=0.5 → {imb_half:.2f}，"
          f"alpha=1 → {imb_1:.1f}（只有 0.5 是平衡点）  OK")

    # ---------- 6. 无 outlier 时不应明显变差 ----------
    X_clean = torch.randn(in_f, n) * 0.5
    W_c, X_c, _ = smooth(W, X_clean, alpha=0.5)
    e0 = (W @ X_clean - quant_per_channel_sym(W, 8) @ quant_per_tensor_sym(X_clean, 8)).norm().item()
    e1 = (W @ X_clean - quant_per_channel_sym(W_c, 8) @ quant_per_tensor_sym(X_c, 8)).norm().item()
    _check(e1 < e0 * 1.5, f"无 outlier 时平滑不应显著变差：{e0:.4f} -> {e1:.4f}")
    print(f"[6] 无 outlier 时误差 {e0:.4f} → {e1:.4f}，未显著劣化  OK")

    print("\n全部通过。")


if __name__ == "__main__":
    main()
