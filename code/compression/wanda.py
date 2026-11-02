"""
Wanda 剪枝 与 2:4 半结构化掩码
==========================

Wanda 的显著性：

    S_ij = |W_ij| * ||X_j||_2

其中 ||X_j|| 是第 j 个输入通道在校准集上的 L2 范数。

两个关键点：
  1. **per-row 比较**：每一行独立选 top-k。全局比较会把幅值小的行整行删光。
  2. **激活感知**：权重再大，如果对应输入通道从不激活，也不重要。

理论联系：H = 2XX^T 的对角元 H_jj = 2||X_j||^2，
所以 Wanda 的分数 ≈ sqrt(对角 OBS 显著性) —— 它是「不补偿的
对角 OBS」，因此比纯 magnitude 好，但高稀疏度下不如 SparseGPT。

2:4 稀疏：
  - 每 4 个**连续**元素（沿输入/K 维）保留 2 个
  - 必须在每个 4 元组内部选 top-2，不能先全局 top-k 再分组
  - 索引元数据 2 bit / 4 元素，GPU Sparse Tensor Core 原生支持

运行：python code/compression/wanda.py
"""

import torch


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def wanda_scores(W, X):
    """W: [out, in], X: [in, n] -> 分数 [out, in]"""
    act_norm = X.norm(p=2, dim=1)          # [in]
    return W.abs() * act_norm.unsqueeze(0)


def magnitude_scores(W):
    return W.abs()


def topk_mask(scores, sparsity):
    """逐行保留 (1-sparsity) 比例的元素。"""
    k = int(scores.shape[1] * (1 - sparsity))
    _check(k >= 1, "稀疏度太高，每行至少保留 1 个")
    idx = scores.topk(k, dim=1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def mask_2_4(scores):
    """把任意分数投影到 2:4：每 4 个连续元素保留分数最大的 2 个。"""
    out_f, in_f = scores.shape
    _check(in_f % 4 == 0, "2:4 要求输入维度能被 4 整除")

    groups = scores.reshape(out_f, in_f // 4, 4)
    idx = groups.topk(2, dim=-1).indices

    mask = torch.zeros_like(scores, dtype=torch.bool).reshape(out_f, in_f // 4, 4)
    mask.scatter_(-1, idx, True)
    return mask.reshape(out_f, in_f)


def apply_mask(W, mask):
    return W * mask


def main():
    # ---------- 1. 激活感知：Wanda 保住「权重小但输入大」的连接 ----------
    W = torch.tensor([[1.0, 0.1]])
    # 通道 0 的输入几乎为 0；通道 1 的输入很大
    X = torch.tensor([[0.01, 0.0],
                      [0.0, 10.0]])

    wanda = topk_mask(wanda_scores(W, X), sparsity=0.5)
    mag = topk_mask(magnitude_scores(W), sparsity=0.5)

    _check(wanda[0, 1].item(), "Wanda 应保留第二个权重（输入活跃）")
    _check(mag[0, 0].item(), "magnitude 会保留第一个权重（数值大）")
    print("[1] Wanda 保留「权重小但输入活跃」的连接，magnitude 不会  OK")

    # ---------- 2. per-row 稀疏度一致性 ----------
    torch.manual_seed(0)
    W = torch.randn(8, 128)
    X = torch.randn(128, 64)
    mask = topk_mask(wanda_scores(W, X), sparsity=0.5)

    per_row = mask.sum(dim=1)
    _check((per_row == 64).all(), f"每行应恰好保留 64 个，实际 {per_row.tolist()}")
    print(f"[2] 每行保留数一致：{per_row.tolist()}  OK")

    # ---------- 3. 2:4 掩码的约束 ----------
    scores = torch.randn(6, 64)
    m24 = mask_2_4(scores)
    per_group = m24.reshape(6, 16, 4).sum(dim=-1)
    _check((per_group == 2).all(), "每个 4 元组必须恰好 2 个非零")
    _check(m24.sum() == 6 * 64 // 2, "2:4 的总稀疏度应为 50%")
    print("[3] 2:4：每个 4 元组恰好 2 个 True，整体 50% 稀疏  OK")

    # 组内选的是分数最大的两个
    g = scores.reshape(6, 16, 4)
    top2 = g.topk(2, dim=-1).indices
    got = m24.reshape(6, 16, 4)
    for i in range(6):
        for j in range(16):
            kept = set(got[i, j].nonzero().flatten().tolist())
            _check(kept == set(top2[i, j].tolist()), "2:4 组内选择的不是 top-2")
    print("[4] 2:4 组内保留的确实是分数最大的两个  OK")

    # ---------- 4. Wanda 的误差小于 magnitude（构造相关输入） ----------
    torch.manual_seed(1)
    W = torch.randn(16, 64)
    # 输入通道的尺度差异巨大：后半部分能量高 10 倍
    scale = torch.ones(64)
    scale[32:] = 10.0
    X = torch.randn(64, 128) * scale.unsqueeze(1)

    m_w = topk_mask(wanda_scores(W, X), 0.5)
    m_m = topk_mask(magnitude_scores(W), 0.5)

    err_w = (W - apply_mask(W, m_w)).norm().item()
    err_m = (W - apply_mask(W, m_m)).norm().item()
    print(f"[5] 权重误差（仅作参考，越小不代表输出误差越小）："
          f"magnitude={err_m:.2f}, wanda={err_w:.2f}")

    # 用输出误差衡量：Wanda 应当更小
    out_w = (W @ X - apply_mask(W, m_w) @ X).norm().item()
    out_m = (W @ X - apply_mask(W, m_m) @ X).norm().item()
    _check(out_w < out_m, f"Wanda 的输出误差应更小：{out_w:.2f} vs {out_m:.2f}")
    print(f"[6] 输出误差：magnitude={out_m:.2f} > wanda={out_w:.2f}  OK")

    print("\n全部通过。")


if __name__ == "__main__":
    main()
