#!/usr/bin/env python3
"""
Regenerate sdk/TOOLS_INDEX.md — the machine-built catalogue of EVERY script / module / CLI / asset in the SDK.

  python3 sdk/tools/build_tools_index.py          (run after adding or changing any tool — it is cheap)

How it works: reads the module docstring of every .py under sdk/mograph, sdk/mograph/v14 and sdk/tools,
lists the public functions (with their first doc line) and the CLI entry points, plus models / binaries / fonts.
The hand-written part (what to use when, gaps, how to extend) lives in sdk/SKILL_TOOLBOX.md; this file is the
always-correct inventory underneath it.  If a module has no docstring, it is flagged "UNDOCUMENTED" → fix it.
"""
from __future__ import annotations

import ast
import datetime
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # sdk/


def _mod_info(path):
    src = open(path, encoding="utf-8").read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return {"doc": f"⚠ SYNTAX ERROR: {e}", "funcs": [], "cli": False}
    doc = ast.get_docstring(tree) or ""
    funcs = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith("_"):
            d = (ast.get_docstring(n) or "").strip().split("\n")[0]
            args = ""
            if isinstance(n, ast.FunctionDef):
                a = [x.arg for x in n.args.args]
                args = "(" + ", ".join(a[:6]) + (", …" if len(a) > 6 else "") + ")"
            funcs.append((n.name + args, d))
    cli = "__main__" in src and ("argparse" in src or "sys.argv" in src)
    return {"doc": doc, "funcs": funcs, "cli": cli}


def _section(title, folder, mod_prefix, out, skip=()):
    out.append(f"\n## {title}\n")
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".py") or fn in skip:
            continue
        p = os.path.join(folder, fn)
        info = _mod_info(p)
        name = fn[:-3]
        first = info["doc"].strip().split("\n")[0] if info["doc"] else "**UNDOCUMENTED — add a module docstring**"
        rel = os.path.relpath(p, os.path.dirname(ROOT))
        cli = f"  · CLI: `python3 -m {mod_prefix}{name} …`" if info["cli"] and mod_prefix else ""
        out.append(f"### `{mod_prefix}{name}`  ({rel}){cli}\n{first}\n")
        body = "\n".join(info["doc"].strip().split("\n")[1:12]).strip()
        if body:
            out.append("<details><summary>details</summary>\n\n```\n" + body + "\n```\n</details>\n")
        if info["funcs"]:
            out.append("| function / class | what |\n|---|---|")
            for f, d in info["funcs"][:25]:
                out.append(f"| `{f}` | {d.replace('|', '/')[:140]} |")
            out.append("")


def main():
    out = ["# TOOLS_INDEX — auto-generated inventory of the SDK",
           f"_generated {datetime.date.today()} by `python3 sdk/tools/build_tools_index.py` — do not edit by hand; "
           "edit docstrings instead. The curated guide is `SKILL_TOOLBOX.md`._\n"]
    _section("v14 engine — talking-head footage (USE THIS FOR REAL VIDEO)", os.path.join(ROOT, "mograph", "v14"),
             "mograph.v14.", out, skip=("__init__.py",))
    _section("legacy engine v5–v13 — pure vector / infographic videos", os.path.join(ROOT, "mograph"), "mograph.",
             out, skip=("__init__.py",))
    _section("tools/ — shell & maintenance scripts", os.path.join(ROOT, "tools"), "", out)
    out.append("\n## shell scripts\n")
    for fn in sorted(os.listdir(os.path.join(ROOT, "tools"))):
        if fn.endswith(".sh"):
            lines = open(os.path.join(ROOT, "tools", fn)).read().split("\n")
            desc = " ".join(l.lstrip("# ") for l in lines[1:4] if l.startswith("#"))
            out.append(f"- `tools/{fn}` — {desc}")
    out.append("\n## models / binaries / assets\n")
    for sub in ("models", "models/rnnoise", "bin", "assets_3d"):
        d = os.path.join(ROOT, "mograph", sub)
        if os.path.isdir(d):
            fs = [f for f in sorted(os.listdir(d)) if not os.path.isdir(os.path.join(d, f))]
            sz = sum(os.path.getsize(os.path.join(d, f)) for f in fs) / 1e6
            out.append(f"- `mograph/{sub}/` ({len(fs)} files, {sz:.1f} MB): " + ", ".join(fs[:20]) +
                       (" …" if len(fs) > 20 else ""))
    fd = os.path.join(ROOT, "fonts")
    if os.path.isdir(fd):
        fams = sorted({f.split("-")[0].split("_")[0] for f in os.listdir(fd) if f.lower().endswith((".ttf", ".otf"))})
        out.append(f"- `fonts/` ({len(os.listdir(fd))} files) families: " + ", ".join(fams))
    with open(os.path.join(ROOT, "TOOLS_INDEX.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(os.path.join(ROOT, "TOOLS_INDEX.md"))


if __name__ == "__main__":
    main()
