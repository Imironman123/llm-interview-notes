"""
LLM-as-Judge 的校准：Cohen's kappa 与位置偏差检测
==================================================

把 Judge 当成一个有偏差的分类器来管理：
  先测它与人工标注的一致性，达标后再用它做规模化评估。

为什么不能只看「一致率」：
  若 90% 样本标签都是「好」，Judge 全说「好」也能拿 90% 一致率。
  Cohen's kappa 扣除了随机猜测的期望一致率，才是真实可靠性。

运行：python code/agent/judge_calibration.py
"""

from collections import Counter


def agreement(judge, human):
    """朴素一致率。有缺陷，仅供参照。"""
    assert len(judge) == len(human) and judge
    return sum(j == h for j, h in zip(judge, human)) / len(judge)


def cohens_kappa(judge, human):
    """
    kappa = (p_o - p_e) / (1 - p_e)

    p_o: 观察到的一致率
    p_e: 两个标注者独立按各自边际分布随机标注时的期望一致率
    """
    assert len(judge) == len(human) and judge
    n = len(judge)
    p_o = sum(j == h for j, h in zip(judge, human)) / n

    cj, ch = Counter(judge), Counter(human)
    p_e = sum((cj[k] / n) * (ch[k] / n) for k in set(cj) | set(ch))

    if p_e >= 1.0:
        # 双方都只给同一类：kappa 无定义，视为完全一致
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def kappa_level(k):
    """Landis & Koch 的经验分级。"""
    if k < 0.0:
        return "差于随机"
    if k < 0.21:
        return "轻微"
    if k < 0.41:
        return "一般"
    if k < 0.61:
        return "中等"
    if k < 0.81:
        return "较好"
    return "几乎完全一致"


def detect_position_bias(judge_fn, pairs):
    """
    检测 Judge 是否偏好 A/B 中的某一边。

    pairs: [(answer_a, answer_b, 人类判定的胜者 "A"|"B"|"tie"), ...]
    judge_fn(a, b) -> "A" | "B" | "tie"

    做法：对同一对样本交换顺序各评一次。
      - 两次结论一致  → 无位置偏差
      - 交换后结论翻转 → 说明它在「看位置」而不是「看内容」
    """
    flips = 0
    preference = Counter()

    for a, b, _ in pairs:
        first = judge_fn(a, b)
        second = judge_fn(b, a)
        # 交换后应当翻转，才是自洽的
        expected_flip = {"A": "B", "B": "A", "tie": "tie"}[first]
        if second == expected_flip:
            preference["一致(自洽)"] += 1
        else:
            flips += 1
            # 不论内容如何都选第一个 → 位置偏好
            if first == "A" and second == "A":
                preference["偏好第一个"] += 1
            elif first == "B" and second == "B":
                preference["偏好第二个"] += 1
            else:
                preference["其他不一致"] += 1

    return {
        "n": len(pairs),
        "flips": flips,
        "flip_rate": flips / len(pairs) if pairs else 0.0,
        "breakdown": dict(preference),
    }


# ---------------------------------------------------------------- tests

def _test():
    # --- 完美一致的 kappa = 1 ---
    k = cohens_kappa(["好", "差", "好", "好"], ["好", "差", "好", "好"])
    assert abs(k - 1.0) < 1e-9, k
    print(f"完全一致：kappa = {k:.3f}")

    # --- 完全相反：kappa 应为负 ---
    k = cohens_kappa(["好", "好", "差", "差"], ["差", "差", "好", "好"])
    assert k < 0, k
    print(f"完全相反：kappa = {k:.3f}（{kappa_level(k)}）")

    # --- 关键案例：类别不平衡时一致率骗人 ---
    # 90% 是「好」，judge 全说「好」
    human = ["好"] * 9 + ["差"]
    judge = ["好"] * 10
    acc = agreement(judge, human)
    kap = cohens_kappa(judge, human)
    print(f"\n不平衡案例：一致率 {acc:.0%}，但 kappa = {kap:.3f}（{kappa_level(kap)}）")
    assert acc == 0.9
    assert kap < 0.5, "类别不平衡时 kappa 应显著低于一致率"

    # --- 中等一致 ---
    human = ["好", "好", "差", "差", "好", "差", "好", "好"]
    judge = ["好", "差", "差", "差", "好", "好", "好", "好"]
    k = cohens_kappa(judge, human)
    print(f"部分一致：一致率 {agreement(judge, human):.0%}，"
          f"kappa = {k:.3f}（{kappa_level(k)}）")

    # --- 位置偏差检测：一个「永远选第一个」的坏 Judge ---
    pairs = [("回答甲", "回答乙", "A"), ("回答丙", "回答丁", "B"),
             ("回答戊", "回答己", "A"), ("回答庚", "回答辛", "tie")]

    def first_biased_judge(a, b):
        return "A"          # 不论内容，永远选第一个

    report = detect_position_bias(first_biased_judge, pairs)
    print(f"\n位置偏差检测：翻转率 {report['flip_rate']:.0%}，"
          f"明细 {report['breakdown']}")
    assert report["flip_rate"] == 1.0, "永远选第一个的 Judge 应 100% 被识别"
    assert report["breakdown"].get("偏好第一个", 0) == 4

    # --- 自洽的 Judge 应当零翻转 ---
    ranking = {"回答甲": 2, "回答乙": 1, "回答丙": 1, "回答丁": 2,
               "回答戊": 2, "回答己": 1, "回答庚": 1, "回答辛": 1}

    def consistent_judge(a, b):
        if ranking[a] > ranking[b]:
            return "A"
        if ranking[b] > ranking[a]:
            return "B"
        return "tie"

    report = detect_position_bias(consistent_judge, pairs)
    print(f"自洽 Judge：翻转率 {report['flip_rate']:.0%} ✓")
    assert report["flip_rate"] == 0.0

    print("\njudge_calibration.py 全部通过")


if __name__ == "__main__":
    _test()
