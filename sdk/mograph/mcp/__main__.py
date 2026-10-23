"""python3 -m mograph.mcp [--list | --call TOOL 'JSON' | selftest]   (no args = stdio MCP server)"""
from __future__ import annotations

import json
import sys

from .server import TOOLS, call, selftest, serve


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "selftest":
        ok, msg = selftest()
        print(("  PASS  " if ok else "  FAIL  ") + msg)
        print(f"mcp selftest: {int(ok)}/1 pass")
        return 0 if ok else 1
    if argv and argv[0] == "--list":
        for n, (_, d, props, req) in TOOLS.items():
            print(f"  {n:<22} {d}\n  {'':<22} args: {', '.join((k + ('*' if k in req else '')) for k in props) or '-'}")
        return 0
    if argv and argv[0] == "--call":
        res = call(argv[1], json.loads(argv[2]) if len(argv) > 2 else {})
        print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
        return 0
    serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
