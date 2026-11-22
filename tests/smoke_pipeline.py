"""兼容原命令：运行与 pytest 默认收集相同的离线全流程测试。"""

from pathlib import Path
import sys

import pytest


def main() -> int:
    target = Path(__file__).resolve().with_name("test_smoke_pipeline.py")
    sys.path.insert(0, str(target.parents[1]))
    return int(pytest.main([str(target), "-q"]))


if __name__ == "__main__":
    raise SystemExit(main())
