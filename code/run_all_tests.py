"""
跑通 code/ 下所有文件的自测。

每个子文件都能单独运行（python code/xxx/yyy.py），这里只是批量执行并汇总。
用子进程而不是 import，是为了避免目录间相互 import 的路径问题 ——
每个文件运行时，它所在目录会自动进入 sys.path，可以直接 import 同级模块。

运行：python code/run_all_tests.py
"""

import subprocess
import sys
from pathlib import Path

SKIP = {"run_all_tests.py"}


def main():
    root = Path(__file__).parent
    files = sorted(f for f in root.rglob("*.py") if f.name not in SKIP)

    if not files:
        print("没有找到可执行的测试文件")
        return 1

    passed, failed = [], []
    for f in files:
        rel = f.relative_to(root)
        print(f"\n{'=' * 60}")
        print(f"▶ {rel}")
        print("=" * 60)

        result = subprocess.run(
            [sys.executable, str(f)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.stdout:
            print(result.stdout.rstrip())

        if result.returncode == 0:
            passed.append(str(rel))
        else:
            failed.append(str(rel))
            if result.stderr:
                print(result.stderr.rstrip())

    # ---- 汇总 ----
    print(f"\n{'=' * 60}")
    print(f"通过 {len(passed)} / {len(files)}")
    for name in failed:
        print(f"  ✗ {name}")
    print("=" * 60)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
