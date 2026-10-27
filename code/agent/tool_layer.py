"""
工具层的三件套：错误分类重试 / 幂等键 / 熔断器
================================================

这是 Agent 从 Demo 走向生产最容易被低估的一层。

核心认知：**代码层盲目重试是错误的**。
  参数错误重试 100 次也不会成功，应该把错误回灌给模型让它改参数 ——
  这是 Agent 相比传统服务最独特的错误处理方式。

面试考点：
  1. 错误分类 → 不同动作（重试 / 回灌 / 熔断 / 升级）
  2. 幂等键必须由**确定性输入**推导，否则重试会生成不同 key
  3. 熔断三态：Closed / Open / Half-Open
  4. 「意图日志」：先落盘再执行，才能应对「执行后崩溃」

运行：python code/agent/tool_layer.py
"""

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from enum import Enum


class ErrorKind(Enum):
    TRANSIENT = "transient"      # 网络抖动、429 —— 退避重试
    PARAM = "param"              # schema 校验失败 —— 回灌给模型
    BUSINESS = "business"        # 余额不足、权限不足 —— 不重试
    DEPENDENCY = "dependency"    # 下游宕机 —— 熔断
    UNKNOWN = "unknown"          # 其他 —— 交给模型决策


class ToolError(Exception):
    def __init__(self, kind: ErrorKind, message: str):
        self.kind = kind
        self.message = message
        super().__init__(message)


@dataclass
class RetryDecision:
    should_retry: bool
    backoff_seconds: float
    feedback: str            # 回灌给模型的文本


def exponential_backoff(attempt, base=0.5, cap=30.0, rng=None):
    """
    指数退避 + jitter。

    jitter 不是可有可无的：大量并发请求在同一时刻失败，
    如果退避时间相同，它们会同时重试形成**重试风暴**，把下游彻底打垮。
    """
    rng = rng or random
    raw = min(cap, base * (2 ** attempt))
    return raw * (0.5 + rng.random())      # 落在 [0.5x, 1.5x]


def decide(err: ToolError, attempt: int, max_attempts=3, rng=None) -> RetryDecision:
    """错误分类 → 决定动作。这是整个工具层的核心分派逻辑。"""
    if err.kind is ErrorKind.PARAM:
        # 关键：不重试，把校验错误回灌给模型让它自己改
        return RetryDecision(False, 0.0,
                             f"参数校验失败，请修正后重试：{err.message}")

    if err.kind is ErrorKind.BUSINESS:
        return RetryDecision(False, 0.0,
                             f"业务规则拒绝，重试无效，请改变策略：{err.message}")

    if err.kind is ErrorKind.DEPENDENCY:
        return RetryDecision(False, 0.0,
                             f"依赖服务不可用，建议降级或稍后再试：{err.message}")

    if attempt + 1 >= max_attempts:
        return RetryDecision(False, 0.0,
                             f"重试次数耗尽（{max_attempts} 次）：{err.message}")

    if err.kind is ErrorKind.TRANSIENT:
        return RetryDecision(True, exponential_backoff(attempt, rng=rng),
                             f"瞬时错误，将自动重试：{err.message}")

    return RetryDecision(True, exponential_backoff(attempt, rng=rng),
                         f"未知错误，重试一次：{err.message}")


def idempotency_key(session_id, step_index, tool_name, args) -> str:
    """
    确定性推导幂等键。

    ⚠️ args 里不能有 time.time() / uuid4() 这类非确定值，
    否则每次重试都生成新 key，幂等直接失效。
    """
    payload = "|".join([
        str(session_id), str(step_index), str(tool_name),
        json.dumps(args, sort_keys=True, ensure_ascii=False),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CircuitBreaker:
    """
    Closed --(失败数达阈值)--> Open --(冷却结束)--> Half-Open
      ^                                              |
      +-----------(试探成功)--------------------------+

    Half-Open 时只放行少量试探请求，成功则恢复，失败则重新 Open。
    """

    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, failure_threshold=3, reset_timeout=5.0, clock=time.monotonic):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.clock = clock
        self.state = self.CLOSED
        self.failures = 0
        self.opened_at = 0.0

    def allow(self) -> bool:
        if self.state is self.CLOSED:
            return True
        if self.state is self.OPEN:
            if self.clock() - self.opened_at >= self.reset_timeout:
                self.state = self.HALF_OPEN
                return True
            return False
        return True        # HALF_OPEN：放行试探

    def on_success(self):
        self.failures = 0
        self.state = self.CLOSED

    def on_failure(self):
        if self.state is self.HALF_OPEN:
            self.state = self.OPEN
            self.opened_at = self.clock()
            return
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.state = self.OPEN
            self.opened_at = self.clock()


# ---------------------------------------------------------------- tests

def _test():
    rng = random.Random(0)

    # --- 参数错误：不重试，且回灌文本包含原始错误 ---
    d = decide(ToolError(ErrorKind.PARAM, "date 格式应为 YYYY-MM-DD"), 0, rng=rng)
    assert not d.should_retry
    assert "YYYY-MM-DD" in d.feedback
    print("参数错误 → 不重试，回灌给模型 ✓")

    # --- 业务错误：不重试 ---
    d = decide(ToolError(ErrorKind.BUSINESS, "余额不足"), 0, rng=rng)
    assert not d.should_retry
    print("业务错误 → 不重试 ✓")

    # --- 瞬时错误：退避重试，且退避时间随次数增长 ---
    delays = []
    for attempt in range(4):
        d = decide(ToolError(ErrorKind.TRANSIENT, "429"), attempt, max_attempts=10, rng=rng)
        assert d.should_retry
        delays.append(d.backoff_seconds)
    assert delays[0] < delays[3], delays
    print(f"瞬时错误 → 退避时间递增: {[round(x, 2) for x in delays]}")

    # --- 重试次数耗尽 ---
    d = decide(ToolError(ErrorKind.TRANSIENT, "timeout"), 2, max_attempts=3, rng=rng)
    assert not d.should_retry and "耗尽" in d.feedback
    print("超过最大重试次数 → 放弃 ✓")

    # --- 幂等键：同输入同 key，参数不同则 key 不同 ---
    k1 = idempotency_key("s1", 3, "issue_refund", {"order": 123, "amount": 50})
    k2 = idempotency_key("s1", 3, "issue_refund", {"amount": 50, "order": 123})
    k3 = idempotency_key("s1", 4, "issue_refund", {"order": 123, "amount": 50})
    assert k1 == k2, "键顺序不同不应影响幂等键"
    assert k1 != k3, "不同步骤应产生不同键"
    print(f"幂等键确定性 ✓  {k1[:16]}...")

    # --- 熔断器状态机 ---
    now = [0.0]
    cb = CircuitBreaker(failure_threshold=3, reset_timeout=5.0, clock=lambda: now[0])

    assert cb.state == CircuitBreaker.CLOSED and cb.allow()
    cb.on_failure(); cb.on_failure()
    assert cb.state == CircuitBreaker.CLOSED, "未达阈值不应熔断"
    cb.on_failure()
    assert cb.state == CircuitBreaker.OPEN, "达到阈值应熔断"
    assert not cb.allow(), "Open 状态应拒绝请求"

    now[0] = 5.1                      # 冷却结束
    assert cb.allow(), "冷却后应放行试探"
    assert cb.state == CircuitBreaker.HALF_OPEN
    cb.on_success()
    assert cb.state == CircuitBreaker.CLOSED and cb.failures == 0
    print("熔断器：Closed → Open → Half-Open → Closed ✓")

    # --- Half-Open 试探失败 → 重新 Open ---
    for _ in range(3):
        cb.on_failure()
    now[0] += 10
    cb.allow()                         # 进入 Half-Open
    cb.on_failure()
    assert cb.state == CircuitBreaker.OPEN
    print("Half-Open 试探失败 → 重新熔断 ✓")

    print("\ntool_layer.py 全部通过")


if __name__ == "__main__":
    _test()
