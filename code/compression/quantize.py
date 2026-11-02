"""
量化基础：对称 / 非对称、per-tensor / per-channel / per-group
==========================================================

量化的本质是把浮点映射到有限码点：

    q = round(x / s) + z,      x_hat = s * (q - z)

三个正交的维度：
  1. 有没有零点 z（对称 vs 非对称）
  2. scale 的作用域（per-tensor / per-channel / per-group）
  3. 比特数 b

面试考点：
  1. 对称量化的误差上界：|x - Q(x)| <= s/2 = amax / (2^b - 2)
  2. 为什么 INT4 必须 per-group？
     → 4 bit 只有 15 个正码点，一个 outlier 就会抬高整段 scale，
       小值全部塌缩到 0。per-group 把 outlier 的影响限制在组内。
  3. 元数据开销 = 16/g bit/权重（FP16 scale），g=128 时约 3% 访存。
  4. 非对称量化适合全正 / 偏斜分布（如 ReLU 后的激活）。

常见追问：
  - "per-group 的 g 怎么选？" → 128 是工业默认；g 越小精度越好但元数据线性上升
  - "为什么对称量化不把 -128 用上？" → 保持对称，避免 int8 乘法溢出
  - "量化到全零张量会怎样？" → scale 必须 clamp，否则除零出 NaN

运行：python code/compression/quantize.py
"""

import torch


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def quantize_symmetric(x, bits=4, granularity="group", group_size=128):
    """
    对称量化（zero-point = 0）。

    x: [out, in] 或任意 [..., in]
    返回 (q, scale)：
      - per-tensor: scale 形状 [1, 1]
      - per-channel: scale 形状 [out, 1]
      - per-group:   scale 形状 [out, in/g, 1]
    q 与 x 同形状，dtype=int8（为了演示用 int8 存 4bit 的值）。
    """
    qmax = 2 ** (bits - 1) - 1  # INT4 -> 7, INT8 -> 127

    if granularity == "tensor":
        amax = x.abs().amax()
        scale = (amax / qmax).clamp_min(1e-8).reshape(1, 1)
        q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)

    elif granularity == "channel":
        # 每个输出通道（每一行）一个 scale
        amax = x.abs().amax(dim=-1, keepdim=True)
        scale = (amax / qmax).clamp_min(1e-8)
        q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)

    elif granularity == "group":
        # 沿输入维每 group_size 个元素一组
        _check(x.shape[-1] % group_size == 0, "输入维度必须能被 group_size 整除")
        xg = x.reshape(*x.shape[:-1], -1, group_size)
        amax = xg.abs().amax(dim=-1, keepdim=True)
        scale = (amax / qmax).clamp_min(1e-8)
        q = torch.round(xg / scale).clamp(-qmax, qmax).to(torch.int8)
        q = q.reshape(x.shape)

    else:
        raise ValueError(f"未知粒度: {granularity}")

    return q, scale


def dequantize_symmetric(q, scale, group_size=None):
    """反量化：q * scale。group_size 为 None 表示 per-tensor 或 per-channel。"""
    if group_size is None:
        return q.float() * scale
    qg = q.reshape(*q.shape[:-1], -1, group_size)
    return (qg.float() * scale).reshape(q.shape)


def quantize_asymmetric(x, bits=8, granularity="channel"):
    """
    非对称量化：q = round(x/s) + z，q ∈ [0, 2^b - 1]。
    适合全正或偏斜分布（ReLU 激活）。
    """
    qmin, qmax = 0, 2 ** bits - 1

    if granularity == "channel":
        xmin = x.amin(dim=-1, keepdim=True)
        xmax = x.amax(dim=-1, keepdim=True)
    elif granularity == "tensor":
        xmin = x.amin().reshape(1, 1)
        xmax = x.amax().reshape(1, 1)
    else:
        raise ValueError("本实现只演示 tensor / channel 两种粒度")

    scale = ((xmax - xmin) / (qmax - qmin)).clamp_min(1e-8)
    zero_point = torch.round(-xmin / scale).clamp(qmin, qmax)

    q = (torch.round(x / scale) + zero_point).clamp(qmin, qmax).to(torch.int32)
    return q, scale, zero_point


def dequantize_asymmetric(q, scale, zero_point):
    return (q.float() - zero_point) * scale


def mse(x, x_hat):
    return (x - x_hat).float().pow(2).mean().item()


def main():
    torch.manual_seed(0)

    # ---------- 1. 对称量化的误差上界 ----------
    x = torch.randn(8, 128) * 0.5
    q, s = quantize_symmetric(x, bits=4, granularity="group", group_size=32)
    x_hat = dequantize_symmetric(q, s, group_size=32)
    max_err = (x - x_hat).abs().max().item()
    bound = (s.max() / 2).item() + 1e-6
    _check(max_err <= bound, f"误差 {max_err:.5f} 超过上界 {bound:.5f}")
    print(f"[1] 对称量化最大误差 {max_err:.5f} <= s/2 = {bound:.5f}  OK")

    # 反量化后每个元素都落在码点上：x_hat / s 应为整数
    ratio = (x_hat / s.reshape(8, 4, 1)[:, :, 0].repeat_interleave(32, dim=1))
    _check(torch.allclose(ratio, ratio.round(), atol=1e-5), "反量化结果不在码点上")
    print("[2] 反量化结果落在整数码点上  OK")

    # ---------- 2. outlier 下 per-group 的优势 ----------
    xo = torch.randn(16, 256) * 0.1
    xo[:, 0] = 10.0  # 一个巨大的 outlier

    q_t, s_t = quantize_symmetric(xo, 4, "tensor")
    mse_t = mse(xo, dequantize_symmetric(q_t, s_t))

    q_c, s_c = quantize_symmetric(xo, 4, "channel")
    mse_c = mse(xo, dequantize_symmetric(q_c, s_c))

    q_g, s_g = quantize_symmetric(xo, 4, "group", 32)
    mse_g = mse(xo, dequantize_symmetric(q_g, s_g, group_size=32))

    # 注意：outlier 只集中在某一列时，per-channel（按行）救不了它，
    # 所以 channel ≈ tensor；只有 per-group 能把 outlier 关进一个小盒子里。
    _check(mse_g < mse_c, f"per-group 应显著优于 per-channel，实际 {mse_g:.4f}/{mse_c:.4f}")
    _check(mse_c <= mse_t * 1.05, "per-channel 不应比 per-tensor 更差")
    print(f"[3] outlier 下的 MSE：tensor={mse_t:.4f} ≈ channel={mse_c:.4f} >> group={mse_g:.4f}  OK")

    # ---------- 3. 非对称量化在偏斜分布上更好 ----------
    xp = torch.rand(8, 64) * 2.0  # 全正

    sym_q, sym_s = quantize_symmetric(xp, bits=4, granularity="channel")
    sym_hat = dequantize_symmetric(sym_q, sym_s)

    asym_q, asym_s, asym_z = quantize_asymmetric(xp, bits=4, granularity="channel")
    asym_hat = dequantize_asymmetric(asym_q, asym_s, asym_z)

    _check(mse(xp, asym_hat) < mse(xp, sym_hat),
           "全正数据上非对称量化应该更好")
    print(f"[4] 全正数据：对称 MSE={mse(xp, sym_hat):.5f} > 非对称 MSE={mse(xp, asym_hat):.5f}  OK")

    # ---------- 4. 全零张量不能出 NaN ----------
    z = torch.zeros(4, 64)
    qz, sz = quantize_symmetric(z, bits=4, granularity="group", group_size=32)
    zh = dequantize_symmetric(qz, sz, group_size=32)
    _check(torch.isfinite(zh).all(), "全零张量出现了 NaN/Inf")
    print("[5] 全零张量的 scale 被 clamp，无 NaN  OK")

    print("\n全部通过。")


if __name__ == "__main__":
    main()
