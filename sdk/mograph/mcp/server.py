"""
mograph.mcp.server — v22 MODEL CONTEXT PROTOCOL server for VE (idea: Concat src/crates/concat-mcp).

    python3 -m mograph.mcp                 # stdio MCP server (JSON-RPC 2.0, one message per line)
    python3 -m mograph.mcp --list          # print the tool catalogue
    python3 -m mograph.mcp --call ve_lint '{"spec": "projects/maghale/edit_v14.json"}'   # one-shot, for scripts

WARM SESSION: the server keeps one Renderer per spec in memory (fonts, tracking cache, palette, 3D meshes, layer
prep) and only rebuilds it when the spec file changes on disk (mtime) → `ve_preview_frame` after the first call
costs one frame (~2 s) instead of a full Python start + Renderer build each time.  RAM: one Renderer ≈ 250–350 MB,
so at most MAX_WARM (1) renderer is kept (the sandbox has 1 GB).  Pure stdlib: no `mcp` pip package needed.

Register in Claude Code / Cursor:
    {"mcpServers": {"ve": {"command": "python3", "args": ["-m", "mograph.mcp"],
                           "cwd": "/home/user/webapp/sdk"}}}

Tools (all paths relative to the repo root /home/user/webapp or absolute):
    ve_doctor            environment + caches status (no tests by default)
    ve_resolve_glossary  Persian/English sentence → engine keys (motion glossary)
    ve_lint_spec         spec linter (errors / warnings)
    ve_qa_audit          on-screen QA (face cover, subtitles, collisions …)
    ve_check_originality v21 anti-copy / anti-repetition audit
    ve_timeline          text timeline panel of the edit
    ve_words             transcript words with edited times (for "w:" anchors)
    ve_preview_frame     render one frame → JPEG path (optionally half size)
    ve_render_sheet      contact sheet of N frames → JPEG path
    ve_spec_read / ve_spec_patch   read the spec / JSON-merge-patch it (layers by id, auto backup, lint after)
    ve_models            v22 model cache status / fetch one model
    ve_vad               speech regions of an audio/video file (+ optional words.json snap)
    ve_cutout            product photo → RGBA PNG
    ve_enhance           low-res image → restored PNG (tiled Real-ESRGAN)
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import shutil
import sys
import time
import traceback

SDK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(SDK)
PROTOCOL = "2024-11-05"
MAX_WARM = 1
VERSION = "22.0"


def _p(path):
    if not path:
        raise ValueError("path required")
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


@contextlib.contextmanager
def _quiet():
    """engine code prints progress to stdout — stdout is the MCP channel, so capture it."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = old


# ============================================================================================== warm session
class Session:
    def __init__(self):
        self.R = {}            # spec path → (mtime, Renderer)
        self.stats = {"builds": 0, "hits": 0}

    def renderer(self, spec):
        spec = _p(spec)
        mt = os.path.getmtime(spec)
        hit = self.R.get(spec)
        if hit and hit[0] == mt:
            self.stats["hits"] += 1
            return hit[1]
        while len(self.R) >= MAX_WARM:
            self.R.pop(next(iter(self.R)))
        import gc
        gc.collect()
        from ..v14.compositor import Renderer
        R = Renderer(spec)
        self.R[spec] = (mt, R)
        self.stats["builds"] += 1
        return R


S = Session()


def _out_dir(spec):
    d = os.path.join(os.path.dirname(_p(spec)), "out")
    os.makedirs(d, exist_ok=True)
    return d


# ============================================================================================== tools
def t_doctor(a):
    from ..pro.doctor import doctor
    lines = []
    bad = doctor(fix=bool(a.get("fix")), tests=bool(a.get("tests")), log=lines.append)
    from ..models_hub import status
    lines.append("v22 models: " + "  ".join(f"{n}:{'✓' if ok else '·'}" for n, ok, *_ in status()))
    return {"ok": not bad, "report": "\n".join(lines)}


def t_glossary(a):
    from ..motion.glossary import resolve
    return {"concepts": resolve(a["text"])}


def t_lint(a):
    from ..pro.lint import lint
    E, W = lint(_p(a["spec"]))
    return {"clean": not (E or W), "errors": E, "warnings": W}


def t_qa(a):
    from ..v14.qa import qa_warnings
    R = S.renderer(a["spec"])
    with _quiet():
        w = qa_warnings(R)
    return {"clean": not w, "warnings": w}


def t_originality(a):
    from ..style_bank import originality_report
    sp = _p(a["spec"])
    E, W, st = originality_report(json.load(open(sp, encoding="utf-8")), sp)
    return {"ok": not E, "errors": E, "warnings": W, "stats": st}


def t_timeline(a):
    from ..pro.export import timeline
    return {"timeline": timeline(S.renderer(a["spec"]))}


def t_words(a):
    R = S.renderer(a["spec"])
    return {"duration": round(R.duration, 3), "cuts": [round(c, 3) for c in R.tl.cuts],
            "words": [{"w": w["w"], "t": round(w["t"], 3), "end": round(w["end"], 3)} for w in R.tl.words]}


def t_frame(a):
    import cv2
    R = S.renderer(a["spec"])
    t = float(a.get("t", R.duration / 2))
    t0 = time.time()
    with _quiet():
        img, _ = R.render(t)
    if a.get("half", True):
        img = cv2.resize(img, (R.W // 2, R.H // 2), interpolation=cv2.INTER_AREA)
    name = os.path.splitext(os.path.basename(a["spec"]))[0]
    p = a.get("out") or os.path.join(_out_dir(a["spec"]), f"{name}_mcp_t{t:05.2f}.jpg")
    cv2.imwrite(_p(p), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {"path": _p(p), "t": t, "secs": round(time.time() - t0, 2), "warm": S.stats}


def t_sheet(a):
    import cv2
    import numpy as np
    R = S.renderer(a["spec"])
    n = int(a.get("n", 12))
    ts = a.get("t") or list(np.linspace(0.3, R.duration - 0.3, n))
    ims = []
    with _quiet():
        for t in ts:
            img, _ = R.render(float(t))
            im = cv2.resize(img, (270, 480), interpolation=cv2.INTER_AREA)
            cv2.putText(im, f"{float(t):.2f}", (8, 24), 0, 0.7, (0, 255, 255), 2)
            ims.append(im)
    cols = 6
    while len(ims) % cols:
        ims.append(np.zeros_like(ims[0]))
    sheet = np.vstack([np.hstack(ims[i:i + cols]) for i in range(0, len(ims), cols)])
    name = os.path.splitext(os.path.basename(a["spec"]))[0]
    p = _p(a.get("out") or os.path.join(_out_dir(a["spec"]), f"{name}_mcp_sheet.jpg"))
    cv2.imwrite(p, sheet, [cv2.IMWRITE_JPEG_QUALITY, 86])
    return {"path": p, "times": [round(float(t), 2) for t in ts]}


def t_spec_read(a):
    sp = json.load(open(_p(a["spec"]), encoding="utf-8"))
    if a.get("summary", True):
        return {"meta": sp.get("meta"), "source": sp.get("source"),
                "layers": [{"id": L.get("id"), "type": L.get("type"), "t": L.get("t"), "out": L.get("out"),
                            "text": L.get("text")} for L in sp.get("layers", [])],
                "subtitles": sp.get("subtitles") if isinstance(sp.get("subtitles"), dict) else None,
                "grade": sp.get("grade")}
    return sp


def merge_patch(target, patch):
    """RFC 7386 JSON merge patch; plus: "layers" given as a list of objects WITH "id" → merged by id
    (a {"id": x, "_delete": true} entry removes the layer, unknown ids are appended)."""
    if not isinstance(patch, dict):
        return copy.deepcopy(patch)
    out = dict(target) if isinstance(target, dict) else {}
    for k, v in patch.items():
        if k == "layers" and isinstance(v, list) and all(isinstance(x, dict) and "id" in x for x in v) \
                and isinstance(out.get("layers"), list):
            layers = [dict(L) for L in out["layers"]]
            for L in v:
                idx = next((i for i, x in enumerate(layers) if x.get("id") == L["id"]), None)
                if L.get("_delete"):
                    if idx is not None:
                        layers.pop(idx)
                elif idx is None:
                    layers.append({kk: vv for kk, vv in L.items() if vv is not None})
                else:
                    layers[idx] = merge_patch(layers[idx], L)
            out["layers"] = layers
        elif v is None:
            out.pop(k, None)
        else:
            out[k] = merge_patch(out.get(k), v) if isinstance(v, dict) else copy.deepcopy(v)
    return out


def t_spec_patch(a):
    sp = _p(a["spec"])
    cur = json.load(open(sp, encoding="utf-8"))
    new = merge_patch(cur, a["patch"])
    bk = sp + ".bak"
    shutil.copy2(sp, bk)
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(new, f, ensure_ascii=False, indent=1)
    from ..pro.lint import lint
    E, W = lint(sp)
    if E and a.get("revert_on_error", True):
        shutil.copy2(bk, sp)
        return {"applied": False, "reverted": True, "errors": E, "warnings": W}
    return {"applied": True, "backup": bk, "errors": E, "warnings": W, "layers": len(new.get("layers", []))}


def t_models(a):
    from ..models_hub import status, try_ensure
    if a.get("fetch"):
        lines = []
        p = try_ensure(a["fetch"], log=lines.append)
        return {"fetched": bool(p), "path": p, "log": lines}
    return {"models": [{"name": n, "cached": ok, "mb": mb, "licence": lic, "opt_in": oi}
                       for n, ok, mb, lic, oi in status()]}


def t_vad(a):
    from ..vad import load_audio, snap_words, speech_regions
    pcm = load_audio(_p(a["audio"]))
    regs = speech_regions(pcm)
    res = {"speech": [[round(x, 3), round(y, 3)] for x, y in regs]}
    if a.get("words"):
        d = json.load(open(_p(a["words"]), encoding="utf-8"))
        ws = d["words"] if isinstance(d, dict) else d
        fx = snap_words(ws, regs)
        res["moved"] = sum(1 for x, y in zip(ws, fx) if abs(float(x.get("t", x.get("start"))) - y["t"]) > 0.05)
        if a.get("write"):
            out = _p(a["write"])
            json.dump({"words": fx} if isinstance(d, dict) else fx, open(out, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            res["written"] = out
    return res


def t_cutout(a):
    from ..pro.cutout import cutout_file
    lines = []
    out = cutout_file(_p(a["image"]), _p(a["out"]) if a.get("out") else None, a.get("quality", "fast"),
                      float(a.get("feather", 1.0)), log=lines.append)
    return {"path": out, "log": lines}


def t_enhance(a):
    import cv2
    from ..pro.enhance import enhance
    src = _p(a["image"])
    res, info = enhance(cv2.imread(src), int(a.get("factor", 2)), float(a.get("mix", 0.6)))
    out = _p(a.get("out") or os.path.splitext(src)[0] + f"_enh{int(a.get('factor', 2))}x.png")
    cv2.imwrite(out, res)
    return {"path": out, "size": [res.shape[1], res.shape[0]], **info}


_S = {"type": "string"}
_SPEC = {"spec": {"type": "string", "description": "edit.json path (repo-relative or absolute)"}}
TOOLS = {
    "ve_doctor": (t_doctor, "Environment, caches, binaries and v22 model status.",
                  {"fix": {"type": "boolean"}, "tests": {"type": "boolean"}}, []),
    "ve_resolve_glossary": (t_glossary, "Map a Persian/English request sentence to VE engine keys.",
                            {"text": _S}, ["text"]),
    "ve_lint_spec": (t_lint, "Lint an edit.json (unknown types/keys, anchors, files, overlaps).", _SPEC, ["spec"]),
    "ve_qa_audit": (t_qa, "On-screen QA of an edit (face cover, subtitle length/collisions, edges).", _SPEC, ["spec"]),
    "ve_check_originality": (t_originality, "v21 anti-copy / anti-repetition audit (must be ok before render).",
                             _SPEC, ["spec"]),
    "ve_timeline": (t_timeline, "Text timeline panel: cuts, retime, layers, subtitles, camera, grade.", _SPEC,
                    ["spec"]),
    "ve_words": (t_words, "Transcript words with EDITED times (use them as \"w:word\" anchors).", _SPEC, ["spec"]),
    "ve_preview_frame": (t_frame, "Render one composited frame to JPEG (warm renderer).",
                         dict(_SPEC, t={"type": "number"}, half={"type": "boolean"}, out=_S), ["spec"]),
    "ve_render_sheet": (t_sheet, "Contact sheet across the edit (LOOK before rendering).",
                        dict(_SPEC, n={"type": "integer"}, t={"type": "array", "items": {"type": "number"}}, out=_S),
                        ["spec"]),
    "ve_spec_read": (t_spec_read, "Read the spec (summary by default).", dict(_SPEC, summary={"type": "boolean"}),
                     ["spec"]),
    "ve_spec_patch": (t_spec_patch, "JSON-merge-patch the spec; layers with \"id\" merge by id, {\"id\",\"_delete\":true} "
                                    "removes. Backup + lint; reverts on lint errors.",
                      dict(_SPEC, patch={"type": "object"}, revert_on_error={"type": "boolean"}), ["spec", "patch"]),
    "ve_models": (t_models, "v22 optional model cache (outside git): list, or fetch one by name.",
                  {"fetch": _S}, []),
    "ve_vad": (t_vad, "Speech regions (Silero VAD) of an audio/video; optionally snap a words.json to onsets.",
               {"audio": _S, "words": _S, "write": _S}, ["audio"]),
    "ve_cutout": (t_cutout, "Remove the background of a product/object photo → RGBA PNG.",
                  {"image": _S, "out": _S, "quality": {"type": "string", "enum": ["fast", "high"]},
                   "feather": {"type": "number"}}, ["image"]),
    "ve_enhance": (t_enhance, "Restore/upscale a low-res image (tiled Real-ESRGAN with natural mix).",
                   {"image": _S, "out": _S, "factor": {"type": "integer", "enum": [1, 2, 4]}, "mix": {"type": "number"}},
                   ["image"]),
}


def tool_list():
    return [{"name": n, "description": d, "inputSchema": {"type": "object", "properties": props, "required": req}}
            for n, (_, d, props, req) in TOOLS.items()]


def call(name, args):
    if name not in TOOLS:
        raise KeyError(f"unknown tool '{name}'")
    fn = TOOLS[name][0]
    with _quiet() as buf:
        res = fn(args or {})
    log = buf.getvalue().strip()
    if log:
        res = dict(res, log=log[-4000:])
    return res


# ============================================================================================== JSON-RPC
def handle(msg):
    mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": params.get("protocolVersion", PROTOCOL),
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "ve-mograph", "version": VERSION},
            "instructions": "VE video engine. Loop: ve_lint_spec → ve_qa_audit → ve_check_originality → "
                            "ve_render_sheet (LOOK) → edit with ve_spec_patch → repeat. Never copy references."}}
    if method in ("notifications/initialized", "initialized") or (mid is None and method.startswith("notifications/")):
        return None
    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": tool_list()}}
    if method == "tools/call":
        try:
            res = call(params.get("name"), params.get("arguments"))
            return {"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text", "text": json.dumps(res, ensure_ascii=False, indent=1, default=str)}],
                "isError": False}}
        except Exception as e:  # noqa: BLE001  — tool errors are results, not protocol errors
            return {"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text", "text": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"}],
                "isError": True}}
    if mid is None:
        return None
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}


def serve(stdin=None, stdout=None):
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            stdout.write(json.dumps({"jsonrpc": "2.0", "id": None,
                                     "error": {"code": -32700, "message": "parse error"}}) + "\n")
            stdout.flush()
            continue
        batch = msg if isinstance(msg, list) else [msg]
        outs = [r for r in (handle(m) for m in batch) if r is not None]
        if outs:
            stdout.write(json.dumps(outs if isinstance(msg, list) else outs[0], ensure_ascii=False) + "\n")
            stdout.flush()


def selftest():
    """protocol round trip in-process (initialize → tools/list → tools/call) + merge-patch semantics."""
    import tempfile
    r = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": PROTOCOL}})
    ok1 = r["result"]["serverInfo"]["name"] == "ve-mograph"
    tl = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
    ok2 = len(tl) == len(TOOLS) and all("inputSchema" in t for t in tl)
    g = handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": "ve_resolve_glossary", "arguments": {"text": "شمشیر نوری قرمز"}}})
    ok3 = not g["result"]["isError"] and "saber" in g["result"]["content"][0]["text"]
    bad = handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "nope", "arguments": {}}})
    ok4 = bad["result"]["isError"] and handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    base = {"meta": {"w": 1080}, "layers": [{"id": "a", "type": "kinetic_title", "text": "x"},
                                            {"id": "b", "type": "broll", "src": "y"}]}
    m = merge_patch(base, {"meta": {"fps": 30}, "layers": [{"id": "a", "text": "z"}, {"id": "b", "_delete": True},
                                                           {"id": "c", "type": "disc"}]})
    ok5 = (m["meta"] == {"w": 1080, "fps": 30} and [L["id"] for L in m["layers"]] == ["a", "c"]
           and m["layers"][0]["text"] == "z" and base["layers"][0]["text"] == "x")
    sp = os.path.join(tempfile.gettempdir(), "_mg_mcp.json")
    json.dump({"source": {"src": "nope.mp4"}, "layers": []}, open(sp, "w"))
    pr = t_spec_patch({"spec": sp, "patch": {"layers": [{"id": "q", "type": "kinetc_title"}]}})
    ok6 = pr["reverted"] and json.load(open(sp))["layers"] == []
    ok = ok1 and ok2 and ok3 and ok4 and ok5 and ok6
    return ok, (f"initialize/list/call ok, {len(TOOLS)} tools, tool errors as results, merge-patch by layer id, "
                f"bad patch reverted by lint {ok6}")
