"""启动本地看板。

    python scripts/serve.py            # 默认 127.0.0.1:8710
    python scripts/serve.py --port 9000 --open
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8710)
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = ap.parse_args()

    import uvicorn

    url = f"http://{args.host}:{args.port}"
    if args.open:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"信号看板：{url}")
    uvicorn.run("web.app:app", host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
