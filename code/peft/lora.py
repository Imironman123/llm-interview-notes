"""
LoRA：低秩适配（Low-Rank Adaptation）
=====================================

核心假设：微调带来的权重变化 ΔW 是**低秩**的，可以分解为 ΔW = B·A，
其中 A: (r, d_in)，B: (d_out, r)，r << min(d_in, d_out)。

    h = W₀x + ΔW·x = W₀x + (α/r)·B·A·x

    训练时冻结 W₀，只优化 A 和 B。

面试考点：
  1. 参数量：r·(d_in + d_out)，相比全量 d_in·d_out 的降幅
  2. 为什么 B 初始化为 0、A 随机？
     → 保证训练开始时 ΔW = 0，模型等价于原始模型，不会一开始就把预训练能力破坏掉
  3. 为什么推理时可以 merge？→ 训练完把 ΔW 加回 W₀，**零额外推理延迟**
  4. α/r 这个缩放的作用 → 调 r 时不必重调学习率

常见追问：
  - "r 怎么选？" → 常见 8~64。任务与预训练分布差异越大、数据越多，r 可以越大
  - "LoRA 加在哪些层？" → 通常所有线性层（q/k/v/o + FFN），只加 q/v 是老做法
  - "LoRA 和 Adapter 的区别？" → Adapter 是串行插入新层，有推理延迟；
    LoRA 是并行旁路，可 merge 掉
  - "QLoRA 在 LoRA 基础上多了什么？" → 4bit 量化基座 + 双重量化 + paged optimizer

运行：python code/peft/lora.py
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    给一个已有的 nn.Linear 套上 LoRA 旁路。

    前向：y = base(x) + scaling * B(A(x))
    """

    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False        # 冻结原权重

        self.r = r
        self.scaling = alpha / r

        d_out, d_in = base.weight.shape
        # A 用随机初始化（kaiming），B 初始化为 0
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)
        self.merged = False

    def forward(self, x):
        out = self.base(x)
        if not self.merged:
            out = out + self.scaling * F_linear(x, self.lora_A, self.lora_B, self.dropout, self.training)
        return out

    @torch.no_grad()
    def merge(self):
        """把 ΔW 合并进原权重，之后推理零开销。"""
        if self.merged:
            return
        self.base.weight += self.scaling * (self.lora_B @ self.lora_A)
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if not self.merged:
            return
        self.base.weight -= self.scaling * (self.lora_B @ self.lora_A)
        self.merged = False

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def F_linear(x, A, B, dropout, training):
    h = dropout(x) if training else x
    return (h @ A.T) @ B.T


def _test():
    torch.manual_seed(0)
    d_in, d_out, r = 256, 128, 8
    base = nn.Linear(d_in, d_out, bias=True)
    lora = LoRALinear(base, r=r, alpha=16).eval()

    x = torch.randn(4, d_in)

    # --- 初始化时 B=0，输出应当和基座完全一致 ---
    assert torch.allclose(lora(x), base(x), atol=1e-6), "初始化时 LoRA 不应改变输出"
    print("初始化时 ΔW = 0，输出与基座一致 ✓")

    # --- 参数量对比 ---
    base_params = sum(p.numel() for p in base.parameters())
    lora_params = lora.trainable_params()
    print(f"\n基座参数量      : {base_params:>8,}  (d_in*d_out + d_out)")
    print(f"LoRA 可训练参数 : {lora_params:>8,}  (r*(d_in+d_out) = {r}*({d_in}+{d_out}))")
    print(f"占比            : {lora_params / base_params:.2%}")
    assert lora_params == r * (d_in + d_out)

    # --- 随机化 B 后验证 merge 的数值等价性 ---
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.02)
    out_before = lora(x).clone()

    lora.merge()
    assert lora.merged
    out_merged = lora(x)
    assert torch.allclose(out_before, out_merged, atol=1e-5), \
        f"merge 前后输出不一致，最大误差 {(out_before-out_merged).abs().max():.2e}"
    print("\nmerge 前后输出一致 ✓（推理零额外延迟）")

    # --- unmerge 能还原 ---
    lora.unmerge()
    assert not lora.merged
    # 注意：unmerge 只还原权重，lora_B 仍是随机值，所以输出应与 merge 前一致
    assert torch.allclose(lora(x), out_before, atol=1e-5)
    print("unmerge 可还原 ✓")

    # --- scaling = alpha/r 的作用：换 r 时 alpha/r 保持学习率等价 ---
    print(f"\nscaling = alpha/r = {lora.scaling}")
    l1 = LoRALinear(nn.Linear(d_in, d_out), r=8, alpha=16)
    l2 = LoRALinear(nn.Linear(d_in, d_out), r=16, alpha=32)
    assert l1.scaling == l2.scaling == 2.0
    print("r 从 8 加到 16 时，alpha 同步从 16 加到 32，scaling 不变")

    print("\nlora.py 全部通过")


if __name__ == "__main__":
    _test()
