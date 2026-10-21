"""
RMSNorm 与 LayerNorm
=====================

    LayerNorm: (x - mean) / sqrt(var + eps) * γ + β
    RMSNorm:              x / sqrt(mean(x²) + eps) * γ

RMSNorm 砍掉了两样东西：**去均值** 和 **偏置 β**。

面试考点：
  1. RMSNorm 为什么能省掉去均值？→ 实验上发现 LayerNorm 的重参数化不变性
     主要来自缩放，去均值贡献很小，砍掉后效果基本不掉
  2. 省了什么计算？→ 不用算均值、不用减均值，少一次 pass，也少 β 参数
  3. 关键性质：**尺度不变性** —— RMSNorm(c·x) = RMSNorm(x)（c>0）
     ⚠️ 严格成立要求 eps=0；实际 eps>0 时只是近似（sqrt(c²m+eps) ≠ c·sqrt(m+eps)）
     但对整体平移 c+x **不** 不变（LayerNorm 则两者都满足）
  4. LLaMA / Qwen / 绝大多数现代 LLM 都用 RMSNorm + Pre-Norm

常见追问：
  - "Pre-LN 还是 Post-LN？" → 现在基本都 Pre-LN（先 norm 再进子层），
    训练更稳定，可以不用 warmup；Post-LN 是原始 Transformer 的做法
  - "eps 一般取多少？" → 1e-5 / 1e-6
  - "为什么大模型偏好 RMSNorm？" → 更快、更省显存带宽，
    GPU 上这类逐元素归约的算子常常是瓶颈

运行：python code/norm/rmsnorm.py
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm。没有 bias，没有去均值。"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # 只有缩放，没有偏置

    def forward(self, x):
        # 在最后一维上求均方根
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class LayerNormReference(nn.Module):
    """手写 LayerNorm，用于对照（nn.LayerNorm 行为一致）。"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


def _test():
    torch.manual_seed(0)
    dim = 16
    rms = RMSNorm(dim).eval()
    ln = LayerNormReference(dim).eval()
    x = torch.randn(4, dim)

    # --- 与 torch.nn.LayerNorm 对齐（gamma=1, beta=0）---
    ref_ln = nn.LayerNorm(dim, elementwise_affine=True).eval()
    ref_ln.weight.data = ln.weight.data
    ref_ln.bias.data = ln.bias.data
    assert torch.allclose(ln(x), ref_ln(x), atol=1e-5), "手写 LayerNorm 与官方实现不一致"

    # --- 关键性质 1：尺度不变 ---
    # eps > 0 时只是近似成立；eps = 0 时才严格成立。这一点面试里值得主动提。
    rms0 = RMSNorm(dim, eps=0.0).eval()
    ln0 = LayerNormReference(dim, eps=0.0).eval()
    for c in [0.1, 10.0]:
        assert torch.allclose(rms0(x * c), rms0(x), atol=1e-6), "eps=0 时 RMSNorm 应严格尺度不变"
        assert torch.allclose(ln0(x * c), ln0(x), atol=1e-6), "eps=0 时 LayerNorm 应严格尺度不变"

    dev = (rms(x * 0.1) - rms(x)).abs().max().item()
    assert dev < 1e-3, "eps 带来的偏差应当很小"
    print(f"尺度不变性：eps=0 时严格成立 ✓；eps=1e-6 时 x*0.1 偏差 {dev:.2e}（近似）")

    # --- 关键性质 2：对整体平移的处理不同 ---
    shift = torch.full((dim,), 5.0)
    ln_shifted = ln(x + shift)
    rms_shifted = rms(x + shift)
    print(f"\n整体加 5.0 之后：")
    print(f"  LayerNorm 输出变化量 (最大值): {(ln_shifted - ln(x)).abs().max():.3e}  → 平移不变")
    print(f"  RMSNorm   输出变化量 (最大值): {(rms_shifted - rms(x)).abs().max():.3e}  → 会变")
    assert (ln_shifted - ln(x)).abs().max() < 1e-5
    assert (rms_shifted - rms(x)).abs().max() > 1e-3

    # --- 参数量对比 ---
    print(f"\n参数量：RMSNorm {sum(p.numel() for p in rms.parameters())}, "
          f"LayerNorm {sum(p.numel() for p in ln.parameters())}（多了 bias）")

    # --- 输出均值：RMSNorm 不保证零均值 ---
    print(f"输出均值：RMSNorm {rms(x).mean():.4f}，LayerNorm {ln(x).mean():.4f}")

    # --- 输出 RMS 恒为 1（gamma=1 时）---
    out_rms = torch.sqrt(rms(x).pow(2).mean(-1))
    assert torch.allclose(out_rms, torch.ones(4), atol=1e-4), "gamma=1 时输出 RMS 应为 1"
    print("gamma=1 时输出 RMS 恒为 1 ✓")

    print("\nrmsnorm.py 全部通过")


if __name__ == "__main__":
    _test()
