"""
解码采样策略：temperature / top-k / top-p
==========================================

三种手段解决的是同一个问题：让输出既有多样性又不跑偏。

    temperature：整体缩放 logits，改变分布的「陡峭程度」
        T < 1 → 更锋利，趋向贪心；T > 1 → 更平坦，趋向均匀；T → 0 等价 greedy
    top-k：只保留概率最高的 k 个候选（数量固定）
    top-p / nucleus：保留累积概率达到 p 的最小集合（数量自适应）

面试考点：
  1. top-k 和 top-p 的本质差别
     → top-k 固定数量，分布平坦时会砍掉有意义的候选，分布尖锐时又保留太多垃圾；
       top-p 按概率质量裁剪，能自适应分布形状 —— 这是现在更常用的原因
  2. 为什么必须先 filter 再 softmax（或在 logits 上做 mask）
     → 如果用 -inf 屏蔽，softmax 时 exp(-inf)=0，概率自动归零并重新归一化
  3. temperature 和 top-p 的顺序：先 temperature 再 top-p

常见追问：
  - "T=0 怎么实现？" → 数学上除零，工程上直接 argmax
  - "top-p 里那个 p 一般取多少？" → 0.9~0.95 常见；代码生成偏低，创作偏高
  - "repetition penalty 原理？" → 对已出现 token 的 logit 做惩罚（除以或减去），
    属于启发式，会影响语义正确性

运行：python code/inference/sampling.py
"""

import torch
import torch.nn.functional as F


def apply_temperature(logits, temperature):
    """T<=0 时退化为贪心（避免除零）。"""
    if temperature <= 0:
        raise ValueError("temperature 必须 > 0，T=0 请直接用 argmax")
    return logits / temperature


def top_k_filter(logits, k):
    """
    只保留 top-k 个 logit，其余置 -inf。

    注意 k 要 clip 到词表大小，否则 topk 会报错。
    """
    k = min(k, logits.size(-1))
    # 第 k 大的值作为阈值
    threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits, p):
    """
    nucleus sampling：保留累积概率刚好达到 p 的最小前缀集合。

    实现要点：在**排序后**的概率上算累积和，从第一个超过 p 的位置之后全部屏蔽。
    允许把越过 p 的那个 token 保留下来（否则可能一个都不剩）。
    """
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)

    # 累积概率「已经超过 p」的位置才需要去掉，
    # 减去自身概率后仍 > p 才算（保住第一个跨过 p 的 token）
    remove = (cumulative - probs) > p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))

    # 散射回原始词表顺序
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(-1, sorted_idx, sorted_logits)
    return out


def sample(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    """完整流程：temperature → top-k → top-p → softmax → multinomial。"""
    if temperature != 1.0:
        logits = apply_temperature(logits, temperature)
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    # 归一化让「剩下的候选」概率和为 1
    probs = probs / probs.sum(-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def _test():
    torch.manual_seed(0)
    vocab = 10
    logits = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.0, -0.5, -1.0, -2.0])

    # --- temperature 改变分布陡峭度 ---
    for T in [0.5, 1.0, 2.0]:
        p = F.softmax(logits / T, dim=-1)
        print(f"T={T}: 最大概率 {p.max():.3f}, 熵 {-torch.sum(p*torch.log(p)):.3f}")

    # --- top-k 恰好保留 k 个 ---
    for k in [1, 3, 5]:
        f = top_k_filter(logits, k)
        assert (f > float("-inf")).sum().item() == k, f"top-k={k} 保留了 {(f > -float('inf')).sum()}"

    # --- top-p 保留的是「最小充分集合」---
    f = top_p_filter(logits, 0.9)
    kept = (f > float("-inf")).sum().item()
    print(f"\ntop-p=0.9 保留了 {kept} 个候选")
    # 保留集合的原始概率和应当 >= p，且去掉最后一个后 < p
    sorted_p = F.softmax(torch.sort(logits, descending=True).values, dim=-1)
    assert sorted_p[:kept].sum() >= 0.9
    assert sorted_p[:kept - 1].sum() < 0.9

    # --- top-p 的自适应：分布越平坦，保留越多 ---
    flat = torch.zeros(vocab)              # 均匀分布
    sharp = torch.tensor([20.0] + [0.0] * 9)
    n_flat = (top_p_filter(flat, 0.9) > float("-inf")).sum().item()
    n_sharp = (top_p_filter(sharp, 0.9) > float("-inf")).sum().item()
    print(f"自适应：均匀分布保留 {n_flat} 个，尖锐分布保留 {n_sharp} 个")
    assert n_flat > n_sharp, "top-p 应当随分布平坦度自适应"

    # --- 采样分布正确性：T=0.1 应几乎总是选第一个 ---
    g = torch.Generator().manual_seed(42)
    picks = [sample(logits.unsqueeze(0), temperature=0.1, generator=g).item()
             for _ in range(200)]
    assert picks.count(0) > 190, f"低温采样不够集中在 argmax，实际 {picks.count(0)}/200"
    print(f"T=0.1 采样 200 次，选中 argmax 的次数: {picks.count(0)}")

    print("\nsampling.py 全部通过")


if __name__ == "__main__":
    _test()
