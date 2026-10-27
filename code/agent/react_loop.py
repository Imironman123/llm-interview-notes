"""
最小 ReAct 循环：工具注册 + 循环检测 + 预算守卫 + 优雅降级
============================================================

ReAct 的骨架只有三步：Thought → Action → Observation。难的是**终止**。

本实现重点演示三件事：
  1. 预算守卫是唯一有严格终止保证的机制（不是模型自己说停）
  2. 循环检测：同一 (tool, args) 重复时注入纠偏提示，而不是盲目重跑
  3. 预算耗尽时**优雅收敛**，返回部分结果而不是抛错

运行：python code/agent/react_loop.py
"""

from dataclasses import dataclass, field


@dataclass
class Budget:
    """硬预算。这是保证 Agent 一定停下的唯一可靠手段。"""
    max_steps: int = 10
    max_tokens: int = 50_000
    steps_used: int = 0
    tokens_used: int = 0

    def charge(self, tokens=0):
        self.steps_used += 1
        self.tokens_used += tokens

    def exhausted(self):
        return self.steps_used >= self.max_steps or self.tokens_used >= self.max_tokens

    def reason(self):
        if self.steps_used >= self.max_steps:
            return f"达到最大步数 {self.max_steps}"
        if self.tokens_used >= self.max_tokens:
            return f"达到 token 预算 {self.max_tokens}"
        return ""


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, name, fn, readonly=True):
        self._tools[name] = {"fn": fn, "readonly": readonly}

    def call(self, name, args):
        if name not in self._tools:
            raise KeyError(f"未注册的工具: {name}")
        return self._tools[name]["fn"](**args)

    def __contains__(self, name):
        return name in self._tools


@dataclass
class Step:
    thought: str
    action: str
    args: dict
    observation: str


@dataclass
class LoopResult:
    status: str                     # finished / budget_exhausted / max_steps
    answer: str = ""
    steps: list = field(default_factory=list)
    reason: str = ""


def react_loop(policy, tools: ToolRegistry, budget: Budget, stuck_after=2):
    """
    Args:
        policy: 可调用对象 policy(history) -> {"thought","action","args"}
                —— 真实场景是 LLM 调用；测试里用脚本化策略
        tools: 工具注册表
        budget: 硬预算
        stuck_after: 连续多少次无效动作判定为原地打转
    """
    history = []
    seen = {}                 # (action, args) -> observation
    no_progress = 0

    while not budget.exhausted():
        decision = policy(history)
        budget.charge(tokens=decision.get("tokens", 0))

        action = decision["action"]
        args = decision.get("args", {})
        thought = decision.get("thought", "")

        # ---- 终止条件 1：显式 finish ----
        if action == "finish":
            return LoopResult("finished", answer=args.get("answer", ""), steps=history)

        # ---- 循环检测：同一调用重复出现 ----
        key = (action, repr(sorted(args.items())))
        if key in seen:
            no_progress += 1
            observation = (f"你已经用相同参数调用过 {action}，"
                           f"结果是：{seen[key]}。请换一个思路或直接 finish。")
            if no_progress >= stuck_after:
                return LoopResult(
                    "max_steps",
                    answer=seen[key],
                    steps=history,
                    reason="检测到原地打转，提前终止",
                )
        else:
            no_progress = 0
            try:
                observation = str(tools.call(action, args))
            except KeyError as e:
                observation = f"工具不存在：{e}"
            seen[key] = observation

        history.append(Step(thought, action, args, observation))

    # ---- 终止条件 2：预算耗尽 → 优雅降级，不抛错 ----
    partial = seen.get(list(seen)[-1], "") if seen else ""
    return LoopResult("budget_exhausted", answer=partial, steps=history,
                      reason=budget.reason())


# ---------------------------------------------------------------- tests

def _test():
    # 场景：查两个城市的天气再求和
    db = {"北京": 25, "上海": 30}

    def get_weather(city):
        return f"{city}: {db[city]}℃"

    tools = ToolRegistry()
    tools.register("get_weather", get_weather)

    # --- 场景 1：正常完成 ---
    script = iter([
        {"thought": "查北京", "action": "get_weather", "args": {"city": "北京"}},
        {"thought": "查上海", "action": "get_weather", "args": {"city": "上海"}},
        {"thought": "可以回答了", "action": "finish", "args": {"answer": "北京25 上海30"}},
    ])

    def policy(history):
        return next(script)

    r = react_loop(policy, tools, Budget(max_steps=10))
    assert r.status == "finished", r.status
    assert len(r.steps) == 2
    assert "北京: 25℃" in r.steps[0].observation
    print(f"正常完成：{r.answer}，用了 {len(r.steps)} 步 ✓")

    # --- 场景 2：死循环 → 循坏检测提前终止 ---
    def looping_policy(history):
        return {"thought": "再查一次", "action": "get_weather", "args": {"city": "北京"}}

    r = react_loop(looping_policy, tools, Budget(max_steps=50), stuck_after=2)
    assert r.status == "max_steps", r.status
    assert "原地打转" in r.reason
    assert len(r.steps) <= 3, f"应在检测到打转后尽快停止，实际 {len(r.steps)} 步"
    print(f"死循环被检测：{r.reason}（{len(r.steps)} 步后停止）✓")

    # --- 场景 3：预算耗尽 → 优雅降级 ---
    cities = iter(["北京", "上海"] * 100)

    def varied_policy(history):
        return {"thought": "继续查", "action": "get_weather",
                "args": {"city": next(cities)}}

    r = react_loop(varied_policy, tools, Budget(max_steps=3))
    assert r.status == "budget_exhausted", r.status
    assert r.answer, "预算耗尽时也应返回可用的部分结果"
    print(f"预算耗尽 → 优雅降级，返回部分结果：{r.answer} ✓")

    # --- 场景 4：不存在的工具也要变成 Observation 而不是崩溃 ---
    script2 = iter([
        {"action": "no_such_tool", "args": {}},
        {"action": "finish", "args": {"answer": "ok"}},
    ])
    r = react_loop(lambda h: next(script2), tools, Budget(max_steps=5))
    assert r.status == "finished"
    assert "工具不存在" in r.steps[0].observation
    print("未知工具 → 转为 Observation，不中断循环 ✓")

    # --- 场景 5：token 预算守卫 ---
    b = Budget(max_steps=100, max_tokens=100)
    b.charge(tokens=60); b.charge(tokens=60)
    assert b.exhausted() and "token" in b.reason()
    print("token 预算守卫生效 ✓")

    print("\nreact_loop.py 全部通过")


if __name__ == "__main__":
    _test()
