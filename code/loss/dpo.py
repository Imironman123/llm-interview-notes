"""
DPO（Direct Preference Optimization）损失
===========================================

一句话：把「训好一个奖励模型再跑 PPO」简化成「直接在偏好数据上做分类」。

推导的核心结果：
    RLHF 的目标可以解析地解出最优策略
        π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)
    反解出隐式奖励
        r(x,y) = β · log(π(y|x)/π_ref(y|x)) + β·log Z(x)
    奖励的差值里 Z(x) 消掉了，于是 Bradley-Terry 偏好模型直接变成：

        L = -log σ( β·[ (log π(y_w) - log π_ref(y_w))
                        - (log π(y_l) - log π_ref(y_l)) ] )

    其中 y_w 是 chosen，y_l 是 rejected，β 控制离参考模型多远。

面试考点：
  1. 为什么 DPO 需要参考模型 π_ref？→ 它是 KL 约束的落地点，
     没有它就退化成"只把 chosen 概率拉高、rejected 压低的普通分类"，
     会严重遗忘、跑偏
  2. β 的作用 → 越大越保守（贴近参考模型），越小越激进
  3. DPO 的劣势 → 是离线方法，偏好数据分布固定；
     容易出现 "chosen 概率同时下降" 的问题（两边都在掉，只是差拉开）

常见追问：
  - "DPO 和 PPO 比？" → DPO 简单稳定、只需一次前向就能训；
    PPO 是在线采样、能探索，效果上限通常更高但工程复杂
  - "DPO 的梯度长什么样？" → 权重是 σ(-margin)，模型越确信时权重越小，
    相当于自动难负样本挖掘
  - "IPO / KTO / ORPO 区别？" → 都是对这一框架的修补：
    IPO 处理过拟合，KTO 只需二分类信号，ORPO 把 SFT 和偏好合成一步

运行：python code/loss/dpo.py
"""

import torch
import torch.nn.functional as F


def sequence_logprob(logits, labels, mask=None):
    """
    计算整条序列的 log 概率（各 token log 概率之和）。

    Args:
        logits: (B, T, V)
        labels: (B, T) token id
        mask:   (B, T) 1 表示计入（通常去掉 padding）
    """
    logps = F.log_softmax(logits.float(), dim=-1)
    token_logps = logps.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    if mask is not None:
        token_logps = token_logps * mask
    return token_logps.sum(dim=-1)


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    返回 (loss, chosen_rewards, rejected_rewards)。

    reward 是隐式奖励 β·(logπ - logπ_ref)，仅用于监控，不参与反传。
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()

    logits = beta * (
        (policy_chosen_logps - ref_chosen_logps)
        - (policy_rejected_logps - ref_rejected_logps)
    )
    loss = -F.logsigmoid(logits).mean()
    return loss, chosen_rewards, rejected_rewards


def dpo_accuracy(chosen_rewards, rejected_rewards):
    """偏好准确率：chosen 的隐式奖励高于 rejected 的比例。"""
    return (chosen_rewards > rejected_rewards).float().mean()


def _test():
    torch.manual_seed(0)
    b, t, vocab = 3, 6, 50

    policy_logits = torch.randn(b, t, vocab)
    ref_logits = torch.randn(b, t, vocab)
    chosen = torch.randint(0, vocab, (b, t))
    rejected = torch.randint(0, vocab, (b, t))

    # --- 情况 1：policy == ref 时，margin 为 0，loss 应恰为 log(2) ---
    pc = sequence_logprob(policy_logits, chosen)
    pr = sequence_logprob(policy_logits, rejected)
    loss, cr, rr = dpo_loss(pc, pr, pc, pr, beta=0.1)
    assert abs(loss.item() - torch.log(torch.tensor(2.0)).item()) < 1e-5, loss.item()
    print(f"policy == ref 时 loss = log2 = {loss.item():.6f} ✓")

    # --- 情况 2：policy 越偏向 chosen，loss 越小 ---
    ref_c = sequence_logprob(ref_logits, chosen)
    ref_r = sequence_logprob(ref_logits, rejected)

    losses = []
    for boost in [0.0, 1.0, 2.0, 4.0]:
        # 人为抬高 policy 对 chosen 的 logprob
        l, _, _ = dpo_loss(pc + boost, pr, ref_c, ref_r, beta=0.1)
        losses.append(l.item())
    print(f"\nchosen 的 logprob 抬高 0/1/2/4 后的 loss: "
          f"{' -> '.join(f'{x:.4f}' for x in losses)}")
    assert losses == sorted(losses, reverse=True), "loss 应随 margin 增大而单调下降"

    # --- 情况 3：β 的作用 ---
    # 固定 margin = 1.0，观察 β 如何影响 loss。
    # β 越大，同样的 logprob 差就足以支撑「确信的偏好」→ loss 更快饱和 →
    # 模型不需要偏离参考模型太多 → 越保守。
    # β → 0 时参考模型被忽略，等价于普通的偏好分类，最激进。
    print("\n固定 margin = 1.0 时，β 对 loss 的影响:")
    for beta in [0.01, 0.1, 0.5, 2.0]:
        l = -F.logsigmoid(torch.tensor(beta * 1.0)).item()
        print(f"  beta={beta:5.2f}: loss = {l:.4f}"
              f"{'   ← 接近 log2，几乎不约束' if beta < 0.05 else ''}")
    print("  β 越大 → loss 越快饱和 → 策略越贴近参考模型（越保守）")

    # --- 情况 4：隐式奖励与偏好准确率 ---
    # 用确定性数值构造：policy 明确偏向 chosen，参考模型为 0
    pc_det = torch.tensor([10.0, 10.0, 10.0])
    pr_det = torch.tensor([-10.0, -10.0, -10.0])
    ref_zero = torch.zeros(3)
    _, cr, rr = dpo_loss(pc_det, pr_det, ref_zero, ref_zero, beta=0.1)
    acc = dpo_accuracy(cr, rr)
    print(f"\n构造偏好后的隐式奖励: chosen {cr.mean():.2f}, rejected {rr.mean():.2f}")
    print(f"偏好准确率: {acc.item():.1%}")
    assert acc.item() == 1.0
    assert torch.allclose(cr, torch.full((3,), 1.0))     # 0.1 * 10
    assert torch.allclose(rr, torch.full((3,), -1.0))    # 0.1 * -10

    # --- 情况 5：梯度权重随 margin 变化（难样本挖倔）---
    print("\n梯度权重 σ(-margin)（margin 越大权重越小）:")
    for margin in [-2.0, 0.0, 2.0, 5.0]:
        w = torch.sigmoid(torch.tensor(-margin)).item()
        print(f"  margin={margin:5.1f}: {w:.4f}")

    print("\ndpo.py 全部通过")


if __name__ == "__main__":
    _test()
