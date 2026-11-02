"""
知识蒸馏损失：Forward KL / Reverse KL / JSD
========================================

标准 KD：

    L = (1-a) * CE(y, p_s) + a * T^2 * KL(p_t^T || p_s^T)

T^2 不是拍脑袋的系数。对 student logit 求导：

    d/dz_i [T^2 KL(p_t^T || p_s^T)] = T * (p_s,i^T - p_t,i^T)

若没有 T^2，梯度会带 1/T，温度一变 a 的语义就变了。

Forward vs Reverse KL：
  - forward: KL(p_t || p_s)  → mode-covering，必须覆盖 teacher 的所有模式
  - reverse: KL(p_s || p_t)  → mode-seeking，专攻 teacher 的某个模式

生成任务里 student 容量有限，通常更偏好 reverse KL（MiniLLM）；
但极端 reverse KL 会牺牲多样性，所以 GKD 用 JSD 在两者之间插值。

运行：python code/compression/kd_loss.py
"""

import torch
import torch.nn.functional as F


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def kd_loss(student_logits, teacher_logits, T=2.0, alpha=1.0,
            labels=None, mode="forward", beta=0.5):
    """
    mode: "forward" | "reverse" | "jsd"
    alpha: 软标签项权重（labels 为 None 时忽略）
    """
    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    p_s = log_p_s.exp()
    p_t = F.softmax(teacher_logits / T, dim=-1)

    if mode == "forward":
        # KL(p_t || p_s) = sum p_t * (log p_t - log p_s)
        kl = (p_t * (p_t.clamp_min(1e-12).log() - log_p_s)).sum(dim=-1).mean()
    elif mode == "reverse":
        # KL(p_s || p_t) = sum p_s * (log p_s - log p_t)
        kl = (p_s * (log_p_s - p_t.clamp_min(1e-12).log())).sum(dim=-1).mean()
    elif mode == "jsd":
        # JSD(beta) = beta*KL(p_t||p_m) + (1-beta)*KL(p_s||p_m)
        p_m = (beta * p_t + (1 - beta) * p_s).clamp_min(1e-12)
        kl = beta * (p_t * (p_t.clamp_min(1e-12).log() - p_m.log())).sum(dim=-1)
        kl += (1 - beta) * (p_s * (log_p_s - p_m.log())).sum(dim=-1)
        kl = kl.mean()
    else:
        raise ValueError(f"未知 mode: {mode}")

    loss = T * T * kl
    if labels is not None:
        loss = (1 - alpha) * F.cross_entropy(student_logits, labels) + alpha * loss
    return loss


def main():
    torch.manual_seed(0)

    # ---------- 1. 完全匹配时 KL = 0 ----------
    z = torch.randn(4, 10)
    _check(kd_loss(z, z, T=2.0, mode="forward").abs() < 1e-6, "forward KL 应为 0")
    _check(kd_loss(z, z, T=2.0, mode="reverse").abs() < 1e-6, "reverse KL 应为 0")
    print("[1] student == teacher 时 forward/reverse KL 均为 0  OK")

    # ---------- 2. 梯度公式：d(T^2 KL)/dz = T*(p_s - p_t) ----------
    zs = torch.randn(3, 6, requires_grad=True)
    zt = torch.randn(3, 6)
    loss = kd_loss(zs, zt, T=2.0, mode="forward")
    loss.backward()

    expected = 2.0 * (torch.softmax(zs / 2.0, -1) - torch.softmax(zt / 2.0, -1))
    # loss 对 batch 取均值 → 梯度也除以 batch
    expected = expected / zs.shape[0]

    _check(torch.allclose(zs.grad, expected, atol=1e-6),
           f"梯度不符：max diff={(zs.grad - expected).abs().max().item():.2e}")
    print("[2] 自动微分结果 == T*(p_s - p_t)/batch，T^2 补偿正确  OK")

    # ---------- 3. forward KL 是 zero-avoiding ----------
    # teacher 均匀，student 把概率几乎全给一个 token
    teacher = torch.tensor([[0.0, 0.0]])
    student = torch.tensor([[0.0, 10.0]])
    fwd = kd_loss(student, teacher, T=1.0, mode="forward").item()
    rev = kd_loss(student, teacher, T=1.0, mode="reverse").item()
    _check(fwd > rev, "student 漏掉 teacher 的一个模式时，forward KL 应更大")
    print(f"[3] student 漏模式：forward={fwd:.3f} > reverse={rev:.3f}（zero-avoiding）  OK")

    # ---------- 4. reverse KL 惩罚「在 teacher 零概率处放质量」 ----------
    # teacher 几乎只给 token0，student 均匀
    teacher = torch.tensor([[10.0, 0.0]])
    student = torch.tensor([[0.0, 0.0]])
    fwd = kd_loss(student, teacher, T=1.0, mode="forward").item()
    rev = kd_loss(student, teacher, T=1.0, mode="reverse").item()
    _check(rev > fwd, "student 在 teacher 的低概率区放质量时，reverse KL 应更大")
    print(f"[4] 在 teacher 零概率区放质量：forward={fwd:.3f} < reverse={rev:.3f}（mode-seeking）  OK")

    # ---------- 5. JSD 是有界的 ----------
    a = torch.randn(2, 5)
    b = torch.randn(2, 5) * 3
    jsd_val = kd_loss(a, b, T=1.0, mode="jsd", beta=0.5).item()
    _check(0 <= jsd_val <= torch.log(torch.tensor(2.0)).item() + 1e-5,
           f"JSD 应落在 [0, log2]，实际 {jsd_val}")
    print(f"[5] JSD = {jsd_val:.3f}，落在 [0, log2] 内  OK")

    # ---------- 6. T^2 让梯度量级对温度不敏感 ----------
    zs = torch.randn(5, 8) * 0.1
    zt = torch.randn(5, 8) * 0.1
    grads = []
    for T in (1.0, 4.0):
        z = zs.clone().requires_grad_(True)
        kd_loss(z, zt, T=T, mode="forward").backward()
        grads.append(z.grad.norm().item())
    rel = abs(grads[0] - grads[1]) / grads[0]
    _check(rel < 0.5, f"T^2 补偿后梯度量级应接近，实际 {grads}")
    print(f"[6] T=1 vs T=4 的梯度范数：{grads[0]:.4f} vs {grads[1]:.4f}（相对差 {rel:.1%}）  OK")

    # ---------- 7. 带硬标签时 alpha 起作用 ----------
    labels = torch.tensor([0, 1, 2])
    zs = torch.randn(3, 5)
    zt = torch.randn(3, 5)
    l_soft = kd_loss(zs, zt, T=2.0, alpha=0.0, labels=labels)
    l_hard = kd_loss(zs, zt, T=2.0, alpha=1.0, labels=labels)
    l_mix = kd_loss(zs, zt, T=2.0, alpha=0.5, labels=labels)
    _check(l_mix < max(l_soft.item(), l_hard.item()) + 1e-6, "混合损失应位于两者之间")
    print(f"[7] alpha=0/0.5/1 的 loss：{l_soft:.3f} / {l_mix:.3f} / {l_hard:.3f}  OK")

    print("\n全部通过。")


if __name__ == "__main__":
    main()
