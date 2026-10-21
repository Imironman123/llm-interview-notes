"""
KV Cache 增量解码与显存计算
============================

为什么需要 KV Cache：
    自回归生成第 t 个 token 时，需要用到前 t-1 个 token 的 K/V。
    如果不缓存，每步都要重算整条序列 —— 计算量 O(n^3)；
    缓存后每步只算新 token 的 K/V —— 总计算量 O(n^2)。

    代价是显存：要为每一层、每个 KV 头、每个位置存 K 和 V。

面试考点：
  1. 写出 KV Cache 显存公式并现场算
     bytes = 2 * layers * num_kv_heads * head_dim * seq_len * batch * dtype_bytes
     （前面的 2 是 K 和 V 各一份）
  2. 为什么 decode 阶段是 memory-bound？
     → 每生成一个 token 只需一次矩阵乘，但要把整个 KV Cache 读一遍，
       算术强度极低，卡在显存带宽上
  3. GQA / PagedAttention 怎么优化

常见追问：
  - "LLaMA-2 70B 在 A100 80G 上一张卡能放多少上下文？"
    → 单 token 约 320 KB（fp16），8K 上下文约 2.5 GB，
      剩余显存还要放权重（70B * 2B ≈ 140 GB，需要多卡 TP）。
  - "KV Cache 和 PagedAttention 什么关系？"
    → PagedAttention 解决的是 KV Cache 的**碎片化**：按页分配，
      逻辑连续的 KV 物理上可以不连续。

运行：python code/inference/kv_cache.py
"""

import math

import torch
import torch.nn.functional as F


class KVCache:
    """
    一个预分配的最大长度 KV Cache。

    真实实现（vLLM）会按需分配 block，这里为了讲清楚原理用最朴素的写法。
    """

    def __init__(self, batch_size, num_heads, head_dim, max_seq_len, dtype=torch.float32):
        self.max_seq_len = max_seq_len
        self.k = torch.zeros(batch_size, num_heads, max_seq_len, head_dim, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.length = 0

    def append(self, k_new, v_new):
        """k_new / v_new: (B, H, n, d)。返回截至目前完整的 K 和 V。"""
        n = k_new.size(2)
        assert self.length + n <= self.max_seq_len, "超出预分配长度"
        end = self.length + n
        self.k[:, :, self.length:end] = k_new
        self.v[:, :, self.length:end] = v_new
        self.length = end
        # 只返回已使用的部分
        return self.k[:, :, :end], self.v[:, :, :end]

    def memory_bytes(self):
        """当前实际占用的显存（不含预分配的空闲部分）。"""
        used = 2 * self.k.size(0) * self.k.size(1) * self.length * self.k.size(-1)
        return used * self.k.element_size()


def attend_step(q, k, v):
    """
    单步注意力。q: (B, H, 1, d)，k/v: (B, H, t, d)。

    decode 阶段不需要 mask —— q 是当前 token，它所能看到的历史
    恰好就是 cache 里的全部内容。
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def attention_full(q, k, v):
    """全序列带 causal mask 的注意力（作为基准）。"""
    n_q, n_k = q.size(-2), k.size(-2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    mask = torch.tril(torch.ones(n_q, n_k, device=q.device, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.matmul(F.softmax(scores, dim=-1), v)


def kv_cache_gib(seq_len, layers, num_kv_heads, head_dim,
                 batch=1, dtype_bytes=2):
    """KV Cache 显存公式，返回 GiB。"""
    total_bytes = 2 * layers * num_kv_heads * head_dim * seq_len * batch * dtype_bytes
    return total_bytes / 1024**3


def _test():
    torch.manual_seed(0)
    b, h, d, n = 2, 4, 8, 10

    q_all = torch.randn(b, h, n, d)
    k_all = torch.randn(b, h, n, d)
    v_all = torch.randn(b, h, n, d)

    # --- 基准：一次性带 mask 算完 ---
    ref = attention_full(q_all, k_all, v_all)

    # --- 增量：逐 token 追加 cache 并计算 ---
    cache = KVCache(b, h, d, max_seq_len=n)
    outs = []
    for t in range(n):
        k_cur, v_cur = cache.append(k_all[:, :, t:t + 1], v_all[:, :, t:t + 1])
        outs.append(attend_step(q_all[:, :, t:t + 1], k_cur, v_cur))
    inc = torch.cat(outs, dim=2)

    assert inc.shape == ref.shape, (inc.shape, ref.shape)
    assert torch.allclose(inc, ref, atol=1e-5), \
        f"增量解码与全量注意力不一致，最大误差 {(inc-ref).abs().max():.2e}"

    # --- 缓存占用随长度线性增长 ---
    assert cache.memory_bytes() == 2 * b * h * n * d * 4
    print(f"增量解码与基准一致；cache 占用 {cache.memory_bytes()/1024:.1f} KB")

    # --- LLaMA-2 70B 实算 ---
    print("\nKV Cache 显存（LLaMA-2 70B: 80 层 / 8 KV 头 / head_dim 128 / fp16）:")
    print(f"  单 token   : {kv_cache_gib(1, 80, 8, 128) * 1024**2:.1f} KiB")
    for ctx in [2048, 8192, 32768, 131072]:
        print(f"  {ctx:6d} 上下文: {kv_cache_gib(ctx, 80, 8, 128):6.2f} GiB / 每请求")

    # --- 若用 MHA（64 个 KV 头）对比 ---
    print(f"\n  同样 8K 上下文，若用 MHA(64 KV 头): "
          f"{kv_cache_gib(8192, 80, 64, 128):.1f} GiB —— 这就是 GQA 的意义")

    print("\nkv_cache.py 全部通过")


if __name__ == "__main__":
    _test()
