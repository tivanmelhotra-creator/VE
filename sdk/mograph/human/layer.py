"""
mograph.human.layer — `human_lineart` layer for the v14 renderer (registered in pro/bridge.py).

    {"type": "human_lineart", "t": "w:قدم", "out": "we:خانه+0.4",
     "character": "office_man",                       # preset name … or {"body": {...}, "outfit": {...}, "face": ...}
     "motion": "walk_cycle",                          # see mograph.human.clip.motion_fn (keys / gestures / poses)
     "x": 760, "y": 1700, "height": 900,               # screen px: feet position (x, y) and figure height
     "yaw": 70, "yaw_end": null, "pitch": 4,
     "color": [1, 1, 1], "width": 3.2,                # stroke colour (0..1) and outer stroke width at 1000 px height
     "draw_on": 0.6,                                  # seconds: strokes draw on in sequence at the entrance
     "enter": "draw" | "fade" | "none",  "exit": "undraw" | "fade" | "none", "exit_dur": 0.4,
     "pass": "front"}

Stable framing: the figure keeps a constant scale (height px ↔ body height) and the feet stay planted on (x, y);
walking in place is the default — use "travel_px": 300 to slide the figure across the screen over its life.
"""
from __future__ import annotations

import math

import numpy as np

from .character import Character, preset
from .lineart import Camera, Style, render_lines


def _character(L):
    key = "_human_ch"
    if key not in L:
        spec = L.get("character", "casual_man")
        st = Style(color=tuple(L.get("color", (0.07, 0.07, 0.08))), width=float(L.get("width", 3.2)))
        args = preset(spec) if isinstance(spec, str) else dict(spec)
        L[key] = Character(**args, style=st)
    return L[key]


def _strokes_at(L, lt, W, H):
    from .clip import motion_fn, smoother
    from . import poses as PZ
    ch = _character(L)
    cache = L.setdefault("_human_cache", {})
    k = round(lt * 48) / 48.0                         # quantise to 48 fps (renderer may sub-sample for blur)
    if k in cache:
        return cache[k]
    f = L.get("_human_f")
    if f is None:
        f = L["_human_f"] = motion_fn(L.get("motion", "idle"), L.get("cycle_s"))
    p = ch.pose(f(k))
    G, X = PZ.pose_body(ch.body, p)
    # fixed camera in BODY space: framed on the rest pose, then mapped so feet = (x, y), height = height px
    if "_human_cam" not in L:
        hpx = float(L.get("height", 900))
        cam = Camera(yaw=float(L.get("yaw", 30)), pitch=float(L.get("pitch", 4)), fov=float(L.get("fov", 18)),
                     W=W, H=H)
        top = ch.body.rest[:13380, 1].max()
        cam.target = (0.0, top / 2, 0.0)
        # distance so that body height ↔ hpx
        cam.dist = top / (hpx / H) / 2 / math.tan(math.radians(cam.fov) / 2)
        for _ in range(3):                            # exact: measure the projected rest height and rescale
            xy, _ = cam.project(ch.body.rest[:13380:7])
            cam.dist *= float(np.ptp(xy[:, 1])) / hpx
        xy, _ = cam.project(np.array([[0, 0.0, 0]]))
        L["_human_cam"] = cam
        L["_human_off"] = (float(L.get("x", W / 2)) - xy[0, 0], float(L.get("y", H * 0.92)) - xy[0, 1])
        L["_human_fig"] = hpx
    cam = L["_human_cam"]
    if L.get("yaw_end") is not None:
        life = max(0.1, float(L.get("_life", 3.0)))
        cam.yaw = float(L.get("yaw", 30)) + (float(L["yaw_end"]) - float(L.get("yaw", 30))) * smoother(lt / life)
    parts = ch.parts(G, X)
    lf = render_lines(parts, cam, ch.style, extra=ch.face_strokes(G, X), ss=1.25, fig_px=L["_human_fig"],
                      seed=int(L.get("seed", 7)))
    if len(cache) > 400:
        cache.clear()
    cache[k] = lf.strokes
    return lf.strokes


def draw_v14(ctx, L, F):
    if getattr(F, "prepass", False) and getattr(F, "ectx", None) is not None:
        return
    lt = F.t - L["_t0"]
    if lt < 0:
        return
    out = L.get("_out_t")
    olt = None if out is None else F.t - out
    exit_dur = float(L.get("exit_dur", 0.4))
    if olt is not None and olt > exit_dur:
        return
    strokes = _strokes_at(L, lt, F.W, F.H)
    ox, oy = L["_human_off"]
    tx = float(L.get("travel_px", 0.0))
    if tx:
        life = max(0.1, float(L.get("_life", 3.0)))
        ox += tx * min(1.0, lt / life)
    alpha = 1.0
    prog = 1.0
    don = float(L.get("draw_on", 0.6))
    if L.get("enter", "draw") == "draw" and lt < don:
        prog = lt / don
    elif L.get("enter") == "fade" and lt < 0.3:
        alpha = lt / 0.3
    if olt is not None and olt > 0:
        if L.get("exit", "undraw") == "undraw":
            prog = min(prog, 1 - olt / exit_dur)
        elif L.get("exit") == "fade":
            alpha *= 1 - olt / exit_dur
    col = L.get("color", (0.07, 0.07, 0.08))
    ctx.save()
    ctx.translate(ox, oy)
    ctx.set_source_rgba(col[0], col[1], col[2], alpha)
    n = len(strokes)
    # draw-on: strokes appear in order of their vertical position (head → feet), each one growing along its length
    order = L.get("_order")
    if order is None or len(order) != n:
        ys = [float(np.mean(s.pts[:, 1])) for s in strokes]
        order = L["_order"] = list(np.argsort(ys))
    from .lineart import ribbon
    for rank, i in enumerate(order):
        s = strokes[i]
        if prog < 1.0:
            start = rank / max(1, n) * 0.7
            local = (prog - start) / 0.3
            if local <= 0:
                continue
            m = max(2, int(len(s.pts) * min(1.0, local)))
            pts, w = s.pts[:m], s.w[:m]
        else:
            pts, w = s.pts, s.w
        if len(pts) < 2:
            continue
        Lr, Rr = ribbon(pts, w)
        ctx.new_path()
        ctx.move_to(*Lr[0])
        for q in Lr[1:]:
            ctx.line_to(*q)
        for q in Rr[::-1]:
            ctx.line_to(*q)
        ctx.close_path()
        ctx.fill()
        for j in (0, -1):
            ctx.arc(pts[j][0], pts[j][1], w[j] / 2, 0, 2 * math.pi)
            ctx.fill()
    ctx.restore()


def prepare(L, R):
    L["_out_t"] = L.get("_t1") if L.get("out") is not None else None
    if L["_out_t"] is not None:
        L["_life"] = max(0.1, L["_out_t"] - L.get("_t0", 0.0))
    _character(L)                                   # build the shaped character + garments once
    return L
