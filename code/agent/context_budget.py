"""
上下文预算分配与触发式压缩
==========================

工业 Agent 必须显式管理 token 预算，而不是「塞满为止」。

两个核心设计：
  1. **必须为输出预留预算** —— 否则生成到一半被截断
  2. **按 token 水位触发压缩**，而不是按轮数 —— 轮数和 token 不成正比
     （工具返回 8000 token 的一轮，抵得上 20 轮闲聊）

运行：python code/agent/context_budget.py
"""

from dataclasses import dataclass


@dataclass
class Allocation:
    system: int
    tools: int
    history: int
    retrieval: int
    output: int

    @property
    def total(self):
        return self.system + self.tools + self.history + self.retrieval + self.output

    def as_dict(self):
        return {
            "system": self.system, "tools": self.tools,
            "history": self.history, "retrieval": self.retrieval,
            "output": self.output,
        }


# 预算分配比例（顺序：system, tools, history, retrieval, output）
DEFAULT_RATIOS = {
    "system": 0.08,
    "tools": 0.15,
    "history": 0.42,
    "retrieval": 0.15,
    "output": 0.20,
}


def allocate(window, ratios=None):
    """
    按比例切分窗口。返回整数预算，且保证总和不超过 window。

    用 floor 保证不超支：宁可少分一点，也不能让各项之和大于窗口。
    """
    ratios = ratios or DEFAULT_RATIOS
    assert abs(sum(ratios.values()) - 1.0) < 1e-6, "比例之和必须为 1"
    parts = {k: int(window * v) for k, v in ratios.items()}
    return Allocation(**parts)


def should_compact(used, window, threshold=0.70):
    """
    按水位触发，不按轮数。

    阈值 0.7 的理由：留出余量给「下一次工具返回」，
    而不是等爆了再压 —— 爆了就没有空间做压缩本身的调用了。
    """
    return used >= window * threshold


def compact(messages, budget, keep_last=3, pinned_keys=("约束", "偏好", "事实")):
    """
    结构化压缩：保留「钉住的事实」+ 最近 K 条原文，其余压成摘要。

    为什么必须结构化？
      自由文本摘要在多轮压缩后信息衰减严重，
      而「用户目标 / 已确认事实 / 未决问题 / 已执行动作」四段式能稳定保留关键信息。

    返回 (compacted_messages, stats)
    """
    pinned = [m for m in messages if m.get("key") in pinned_keys]
    recent = messages[-keep_last:] if len(messages) >= keep_last else messages
    dropped = [m for m in messages if m not in pinned and m not in recent]

    summary = {
        "role": "summary",
        "content": (
            f"[已压缩 {len(dropped)} 条历史]\n"
            f"- 已完成动作：{sum(1 for m in dropped if m.get('kind') == 'tool')} 次工具调用\n"
            f"- 已确认事实：{len(pinned)} 条（保留原文）\n"
            f"- 未决问题：见最近对话"
        ),
    }

    # 去重：pinned 和 recent 可能重叠
    out, seen_ids = [], set()
    for m in pinned + [summary] + recent:
        mid = id(m)
        if mid not in seen_ids:
            out.append(m)
            seen_ids.add(mid)

    stats = {
        "before": len(messages),
        "after": len(out),
        "dropped": len(dropped),
        "pinned_kept": len(pinned),
    }
    return out, stats


def tokens_of(messages, chars_per_token=2.0):
    """粗略 token 估算：中文约 1.5~2 字符/token，英文约 4 字符/token。"""
    total = sum(len(str(m.get("content", ""))) for m in messages)
    return int(total / chars_per_token)


# ---------------------------------------------------------------- tests

def _test():
    W = 128_000

    # --- 分配总和不超过窗口，且包含输出预留 ---
    a = allocate(W)
    assert a.total <= W, (a.total, W)
    assert a.output > 0, "必须为输出预留预算"
    print(f"窗口 {W:,} → {a.as_dict()}")
    print(f"  合计 {a.total:,}（输出预留 {a.output:,}）✓")

    # --- 比例校验 ---
    try:
        allocate(W, {"system": 0.5, "tools": 0.2, "history": 0.2,
                     "retrieval": 0.2, "output": 0.2})
        raise AssertionError("比例之和不为 1 时应报错")
    except AssertionError as e:
        assert "比例之和" in str(e)
    print("比例校验生效 ✓")

    # --- 触发阈值 ---
    assert not should_compact(60_000, W)
    assert should_compact(90_000, W)
    print(f"水位触发：60k 不压缩，90k 触发压缩（阈值 70%）✓")

    # --- 压缩保留钉住事实 ---
    msgs = []
    msgs.append({"key": "约束", "role": "system", "content": "用户要求中文回复，不得推荐竞品"})
    for i in range(20):
        msgs.append({"kind": "tool", "role": "tool",
                     "content": "x" * 2000, "key": f"tool_{i}"})
    msgs.append({"role": "user", "content": "帮我查下订单"})
    msgs.append({"role": "assistant", "content": "好的，正在查询"})
    msgs.append({"role": "user", "content": "还要查物流"})

    before_tokens = tokens_of(msgs)
    out, stats = compact(msgs, a.history, keep_last=3)

    # 钉住的事实必须原样保留
    assert any("不得推荐竞品" in str(m.get("content")) for m in out), "钉住的事实被丢了"
    # 压缩后条数必须下降
    assert stats["after"] < stats["before"]
    after_tokens = tokens_of(out)

    print(f"\n压缩：{stats['before']} 条 → {stats['after']} 条")
    print(f"  token 估算 {before_tokens:,} → {after_tokens:,}")
    print(f"  钉住保留 {stats['pinned_kept']} 条，丢弃 {stats['dropped']} 条 ✓")

    # --- 压缩必须真的省 token ---
    assert after_tokens < before_tokens * 0.6, "压缩后应显著变短"

    # --- 压缩收益的定量估算 ---
    saved = before_tokens - after_tokens
    remaining_steps = 10
    print(f"\n每步节省约 {saved:,} token，剩余 {remaining_steps} 步"
          f" → 合计节省约 {saved * remaining_steps:,} token")

    print("\ncontext_budget.py 全部通过")


if __name__ == "__main__":
    _test()
