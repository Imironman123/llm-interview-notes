"""
分组查询注意力（Grouped-Query Attention, GQA）
================================================

MHA / MQA / GQA 的关系：
    MHA: 每个 Q 头配一个独立的 KV 头     num_kv_heads = num_heads
    MQA: 所有 Q 头共享同一个 KV 头      num_kv_heads = 1
    GQA: 每 G 个 Q 头共享一个 KV 头     num_kv_heads = G

面试考点：
  1. GQA 主要省的是 **KV Cache 显存** 和 **KV Cache 的访存带宽**，
     不是省计算量（计算量基本不变）
  2. 质量介于 MHA 和 MQA 之间，是现在的主流选择
     （LLaMA-2 70B / LLaMA-3 / Qwen2 都用了 GQA）

常见追问：
  - "GQA 省了多少显存？" → KV 头数从 H 降到 G，KV Cache 降为 G/H。
    LLaMA-2 70B：64 个 Q 头、8 个 KV 头 → KV Cache 降到 1/8。
  - "GQA 怎么训练？" → 有两种：从头初始化（Uptraining），或直接把
    MHA 的 KV 头按组平均得到初始权重，再继续预训练/微调。
  - "为什么不直接用 MQA？" → MQA 质量下降明显，GQA 用少量 KV 头换回大部分质量。

运行：python code/attention/gqa.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def repeat_kv(x, n_rep):
    """
    把 KV 头按组复制，使其数量与 Q 头对齐。

    (B, G, N, d) --repeat_interleave--> (B, G*n_rep, N, d)

    注意第 h 个 Q 头对应的 KV 头是 h // n_rep（连续分组），
    所以用 repeat_interleave 而不是简单 tile。
    """
    if n_rep == 1:
        return x
    b, g, n, d = x.shape
    # 在 head 维度后面插一个 n_rep 维，expand 后展平
    return x[:, :, None, :, :].expand(b, g, n_rep, n, d).reshape(b, g * n_rep, n, d)


class GroupedQueryAttention(nn.Module):
    """
    GQA 实现。num_kv_heads=num_heads 退化为 MHA，=1 退化为 MQA。

    输入输出 (batch, seq_len, d_model)。
    """

    def __init__(self, d_model, num_heads, num_kv_heads, bias=False):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0, "Q 头数必须能被 KV 头数整除"

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.n_rep = num_heads // num_kv_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, num_heads * self.d_k, bias=bias)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=bias)   # 注意比 Q 少
        self.w_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=bias)
        self.w_o = nn.Linear(num_heads * self.d_k, d_model, bias=bias)

    def forward(self, x):
        b, n, _ = x.shape

        q = self.w_q(x).view(b, n, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(b, n, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(b, n, self.num_kv_heads, self.d_k).transpose(1, 2)

        k = repeat_kv(k, self.n_rep)    # (B, H, N, d)
        v = repeat_kv(v, self.n_rep)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)                      # (B, H, N, d)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.w_o(out)


def _test():
    torch.manual_seed(0)
    b, n, d_model, h = 2, 5, 64, 8
    x = torch.randn(b, n, d_model)

    # --- GQA 形状正确 ---
    gqa = GroupedQueryAttention(d_model, h, num_kv_heads=2).eval()
    out = gqa(x)
    assert out.shape == (b, n, d_model)

    # --- n_rep=1 时退化为 MHA ---
    mha_like = GroupedQueryAttention(d_model, h, num_kv_heads=h).eval()
    assert mha_like.n_rep == 1
    assert repeat_kv(torch.randn(1, 4, 3, 2), 1).shape == (1, 4, 3, 2)

    # --- KV 投影参数量随 KV 头数线性下降 ---
    def kv_params(m):
        return m.w_k.weight.numel() + m.w_v.weight.numel()

    print(f"num_kv_heads={h} (MHA): KV 参数 {kv_params(mha_like)}")
    print(f"num_kv_heads=2 (GQA): KV 参数 {kv_params(gqa)}")
    assert kv_params(gqa) * (h // 2) == kv_params(mha_like)

    # --- repeat_kv 的分组语义：第 h 个 Q 头用的是第 h//n_rep 个 KV 头 ---
    kv = torch.arange(1 * 2 * 1 * 1, dtype=torch.float32).reshape(1, 2, 1, 1)  # [[0],[1]]
    rep = repeat_kv(kv, 4)  # 每组 4 个，应为 [0,0,0,0,1,1,1,1]
    assert rep.flatten().tolist() == [0, 0, 0, 0, 1, 1, 1, 1], rep.flatten().tolist()

    # --- KV Cache 显存对比 ---
    seq, layers, d_k, bytes_per = 8192, 80, 128, 2
    def kv_cache_mb(kv_heads):
        return layers * 2 * kv_heads * d_k * bytes_per * seq / 1024**2

    print(f"\n同样 80 层、128 维头、8192 上下文、fp16：")
    for name, kh in [("MHA  (64 个 KV 头)", 64), ("GQA  (8 个 KV 头)", 8), ("MQA  (1 个 KV 头)", 1)]:
        print(f"  {name}: {kv_cache_mb(kh):8.0f} MB")

    print("\ngqa.py 全部通过")


if __name__ == "__main__":
    _test()
