"""
GRPO（Group Relative Policy Optimization）
===========================================

GRPO 是 DeepSeek 在 R1 / V3 里用的 RL 方法，核心改动只有一条：

    **去掉 Critic 模型，用「同一 prompt 下的一组采样」的组内统计量当基线。**

PPO 需要 Critic 估计 V(s) 来算优势 A = R - V(s)，而 Critic 和策略一样大，
显存和训练成本翻倍。GRPO 的做法是对每个 prompt 采样 G 个回答，
用这 G 个回答的奖励均值/标准差归一化：

    A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)

    L = -E[ min( ratio·A,  clip(ratio, 1-ε, 1+ε)·A ) ] + β·KL(π || π_ref)

    其中 ratio = π_θ(y_i) / π_θ_old(y_i)

面试考点：
  1. 为什么可以不要 Critic？→ 同一 prompt 的多个采样天然构成基线，
     组内均值就是 V(s) 的无偏估计，还省掉了价值网络的拟合误差
  2. 组内归一化的作用 → 让优势有统一的尺度，跨 prompt 可比
  3. std 为 0 时（组内奖励全相同）怎么办？→ 加 eps 防除零；
     实际实现中这组样本不提供梯度（优势全为 0）
  4. 和 PPO 的区别只有基线怎么来 + 是否用 clip

常见追问：
  - "GRPO 的 KL 项怎么算？" → 常用无偏估计 k3：
    exp(logπ_ref - logπ) - (logπ_ref - logπ) - 1
  - "G 取多少？" → R1 里用 8~16；太小方差大，太大显存吃不消
  - "为什么可验证奖励（RLVR）效果好？" → 数学/代码题答案可以规则校验，
    奖励稀疏但无噪声，避免了 RM 的 reward hacking

运行：python code/loss/grpo.py
"""

import torch
import torch.nn.functional as F


def group_advantages(rewards, group_size, eps=1e-4):
    """
    组内归一化优势。

    Args:
        rewards: (B*G,) 每个回答的标量奖励，按组连续排列
        group_size: 每个 prompt 采样多少个回答

    Returns:
        与 rewards 同形状的优势
    """
    assert rewards.numel() % group_size == 0
    r = rewards.view(-1, group_size)                      # (num_prompts, G)
    mean = r.mean(dim=-1, keepdim=True)
    std = r.std(dim=-1, keepdim=True, unbiased=False)
    adv = (r - mean) / (std + eps)
    return adv.reshape(-1)


def grpo_loss(logps, old_logps, advantages, clip_eps=0.2,
              ref_logps=None, beta=0.0):
    """
    带 clip 的策略损失 + 可选 KL 惩罚。

    Args:
        logps:     (N,) 当前策略下各回复的序列 log 概率
        old_logps: (N,) 采样时的旧策略 log 概率
        advantages:(N,) 组内归一化优势
        ref_logps: (N,) 参考模型 log 概率（算 KL 用）
        beta:      KL 系数
    """
    ratio = torch.exp(logps - old_logps)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

    # 取 min → 悲观下界，防止策略更新过猛
    surrogate = torch.min(ratio * advantages, clipped * advantages)
    loss = -surrogate.mean()

    if ref_logps is not None and beta > 0:
        # k3 估计器：exp(Δ) - Δ - 1，其中 Δ = logπ_ref - logπ
        log_ratio = ref_logps - logps
        kl = torch.exp(log_ratio) - log_ratio - 1
        loss = loss + beta * kl.mean()

    return loss


def _test():
    torch.manual_seed(0)
    G, num_prompts = 8, 4
    rewards = torch.randn(num_prompts * G)

    # --- 组内归一化后，每组均值≈0、标准差≈1 ---
    adv = group_advantages(rewards, G)
    adv_g = adv.view(num_prompts, G)
    assert torch.allclose(adv_g.mean(-1), torch.zeros(num_prompts), atol=1e-4)
    assert torch.allclose(adv_g.std(-1, unbiased=False), torch.ones(num_prompts), atol=1e-3)
    print("组内归一化：每组优势均值≈0、标准差≈1 ✓")

    # --- 全部奖励相同时，优势应为 0（这组不提供梯度）---
    flat = torch.full((G,), 2.5)
    flat_adv = group_advantages(flat, G)
    assert flat_adv.abs().max() < 1e-2, flat_adv
    print(f"组内奖励全相同时优势 {flat_adv.max():.4f} ≈ 0（不产生梯度）✓")

    # --- 组大小对优势的影响 ---
    print("\n同样一组奖励 [1, 0, 0, 1] 在不同 G 下的优势:")
    for g in [4]:
        r = torch.tensor([1.0, 0.0, 0.0, 1.0])
        print(f"  {g} 个样本: {group_advantages(r, g).tolist()}")

    # --- clip 的作用：ratio 超出区间时梯度被截断 ---
    old = torch.zeros(6)
    logps = torch.log(torch.tensor([0.5, 1.0, 1.5, 0.5, 1.0, 1.5]))
    advs = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])

    ratios = torch.exp(logps - old)
    clipped = torch.clamp(ratios, 0.8, 1.2)
    print("\nratio 与裁剪后的 ratio:")
    for r_raw, r_clip, a in zip(ratios.tolist(), clipped.tolist(), advs.tolist()):
        chosen = min(r_raw * a, r_clip * a)
        print(f"  ratio={r_raw:.2f}  adv={a:+.0f}  ->  采用 {chosen:+.3f}")

    loss = grpo_loss(logps, old, advs, clip_eps=0.2)
    print(f"\nclip 后的 loss = {loss.item():.4f}")

    # --- ratio == 1 时，目标恰好是 -mean(advantage) ---
    same = grpo_loss(old, old, advs)
    expected = -advs.mean().item()
    assert abs(same.item() - expected) < 1e-6, (same.item(), expected)
    print(f"ratio=1 时 loss = -mean(adv) = {same.item():.4f}")

    # --- clip 确实限制了激进更新 ---
    # ratio=5、adv=+1 时，surrogate 被压到 1+ε=1.2，loss = -1.2
    l_big = grpo_loss(torch.log(torch.tensor([5.0])), torch.zeros(1),
                      torch.tensor([1.0]), clip_eps=0.2)
    assert abs(l_big.item() + 1.2) < 1e-5, l_big.item()
    print(f"ratio=5, adv=+1 → 被裁剪到 1.2，loss = {l_big.item():.2f}")

    # --- KL 惩罚会让 loss 变大 ---
    ref = torch.zeros(6)
    l_no_kl = grpo_loss(logps, old, advs)
    l_kl = grpo_loss(logps, old, advs, ref_logps=ref, beta=0.1)
    assert l_kl.item() > l_no_kl.item(), "KL 惩罚应当增加 loss"
    print(f"加 KL 惩罚后 loss: {l_no_kl.item():.4f} -> {l_kl.item():.4f}")

    print("\ngrpo.py 全部通过")


if __name__ == "__main__":
    _test()
