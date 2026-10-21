"""
mograph.aefx.bridge — glue between the AE-FX engines and the renderers (v14 footage compositor, v5-13 scenes).

LAYER FX STACK (any v14 layer, and any v5-13 element):
    "fx": [
      {"motion": "scale_01_in"},                       # Premiere-style IN preset at the layer start ("t")
      {"motion": "glitch_03_out", "at": "out"},        # OUT preset ending exactly at the layer "out"
      {"motion": "shake_02", "at": "w:بوم"},           # one-shot FX preset at any time / word anchor
      {"glow": {"radius": 80, "exposure": 0.5, "tint": "#ff4fd8"}},           # deep_glow on the layer alone
      {"path_blur": {"path": [[x,y],...], "end_blur": 60, "bidirectional": false}, "t": 2.0, "dur": 0.6}
    ]
    keys of a motion item: dur (retime), speed, motion_blur (shutter °, default 180 = real motion blur),
                            origin "layer" (default: pivot = layer bbox centre) | "frame"
    path_blur / glow items: "t"/"dur" window (default whole layer life), "ramp" 0.15 s fade of the strength

FRAME (post) FX — v14 spec "post": {"aefx": [...same items...]} or adjustment layer {"type": "aefx_adjust", ...}
    are applied to the whole composited frame (camera transitions: {"motion": "position_l_to_r_01_in", "at": 3.2}).

CUT TRANSITIONS — v14 "source": {"cut_presets": {"out": "blur_02_out", "in": "blur_02_in", "dur": 0.5}} or a list
    per cut index → the outgoing shot plays an OUT preset, the incoming one the IN preset, centred on each cut.
"""
from __future__ import annotations

import math

import cairo
import cv2
import numpy as np

from . import motion as AM
from .deepglow import deep_glow
from .pathblur import path_blur


# ------------------------------------------------------------------------------------- helpers
def _surf_arr(s):
    s.flush()
    return np.ndarray((s.get_height(), s.get_stride() // 4, 4), np.uint8, s.get_data())[:, :s.get_width()]


def resolve_items(items, resolve, t0, t1):
    """resolve times of an fx list once ("at": "in"|"out"|time|"w:word")."""
    out = []
    for it in items or ():
        it = dict(it)
        if "motion" in it:
            p = AM.retimed(AM.get(it["motion"]), it.get("dur"), it.get("speed", 1.0))
            it["_p"] = p
            at = it.get("at", "out" if p["dir"] == "out" else "in")
            if at == "in":
                it["_start"] = t0
            elif at == "out":
                it["_start"] = t1 - p["dur"]
            else:
                it["_start"] = resolve(at)
                if p["dir"] == "out":
                    it["_start"] -= p["dur"]
            it["_end"] = it["_start"] + p["dur"]
        else:
            ws = resolve(it["t"]) if "t" in it else t0
            it["_start"] = ws
            it["_end"] = ws + float(it["dur"]) if "dur" in it else t1
        out.append(it)
    return out


def active(items, t):
    """items that change the image at time t (IN presets before their start keep the first state = hidden)."""
    res = []
    for it in items:
        if "motion" in it:
            p = it["_p"]
            if p["dir"] == "in" and t >= it["_end"]:
                continue
            if p["dir"] == "fx" and not (it["_start"] <= t <= it["_end"]):
                continue
            if p["dir"] == "out" and t < it["_start"]:
                continue
            res.append(it)
        elif it["_start"] - 1e-6 <= t <= it["_end"] + 1e-6:
            res.append(it)
    return res


def _ramp(it, t):
    r = float(it.get("ramp", 0.15))
    if r <= 0:
        return 1.0
    return max(0.0, min(1.0, (t - it["_start"]) / r, (it["_end"] - t) / r))


def apply_items(img, items, t, fps=30.0, pivot=None):
    """Apply active fx items to a BGRA premultiplied (or BGR) image.  pivot = (x, y) px used as the anchor of motion
    presets (layer centre) so a preset scales/rotates the LAYER about itself, not the frame centre."""
    H, W = img.shape[:2]
    out = img
    for it in items:
        if "motion" in it:
            p = it["_p"]
            lt = t - it["_start"]
            fx = p["fx"]
            if pivot is not None and it.get("origin", "layer") == "layer":
                u, v = pivot[0] / W, pivot[1] / H
                fx = [_repivot(f, u, v) for f in fx]
            out = AM.apply_stack(out, fx, max(0.0, lt), fps, it.get("motion_blur", 180))
        elif "glow" in it:
            g = dict(it["glow"]) if isinstance(it["glow"], dict) else {}
            k = _ramp(it, t)
            g["intensity"] = float(g.get("intensity", 1.0)) * k
            if out.shape[-1] == 4:                       # layer: glow from its own pixels, alpha grows with light
                out = deep_glow(out, threshold=float(g.pop("threshold", 0.0)), **g)
            else:
                out = deep_glow(out, **g)
        elif "path_blur" in it:
            pb = dict(it["path_blur"])
            k = _ramp(it, t)
            pb["end_blur"] = float(pb.get("end_blur", 60.0)) * k
            pb["start_blur"] = float(pb.get("start_blur", 0.0)) * k
            if isinstance(pb.get("path"), list):
                pb["path"] = [tuple(map(float, q)) for q in pb["path"]]
            out = path_blur(out, **pb)
    return out


def _repivot(f, u, v):
    """move a Premiere transform/basic3d pivot from the frame centre to (u, v) (normalised): anchor & position shift
    together so the motion is identical but happens around the layer."""
    if f["fx"] != "transform":
        return f
    P = dict(f["p"])
    du, dv = u - 0.5, v - 0.5
    for key in ("anchor", "position"):
        val = P.get(key, [0.5, 0.5])
        if isinstance(val, dict):
            val = {"k": [[r[0], [r[1][0] + du, r[1][1] + dv]] + list(r[2:]) for r in val["k"]]}
        else:
            val = [val[0] + du, val[1] + dv]
        P[key] = val
    return {"fx": "transform", "p": P}


# ------------------------------------------------------------------------------------- layer wrapper (cairo)
def draw_with_fx(ctx, draw_fn, items, t, fps=30.0, bbox_hint=None):
    """Render draw_fn(ctx2) into an offscreen BGRA layer, apply the active fx items, composite onto ctx.
    Returns True when drawn through the stack (False → caller should draw normally)."""
    act = active(items, t)
    if not act:
        return False
    tgt = ctx.get_target()
    W, H = tgt.get_width(), tgt.get_height()
    layer = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    lc = cairo.Context(layer)
    lc.set_matrix(ctx.get_matrix())
    draw_fn(lc)
    a = _surf_arr(layer)
    alpha = a[..., 3]
    ys, xs = np.nonzero(alpha > 3)
    if not len(ys):
        return True
    pivot = ((xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0)
    out = apply_items(np.ascontiguousarray(a), act, t, fps, pivot)
    out = np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))
    out[..., :3] = np.minimum(out[..., :3], out[..., 3:4])        # keep premultiplied invariant
    s = cairo.ImageSurface.create_for_data(memoryview(out), cairo.FORMAT_ARGB32, W, H)
    ctx.save()
    ctx.identity_matrix()
    ctx.set_source_surface(s, 0, 0)
    ctx.paint()
    ctx.restore()
    return True


# ------------------------------------------------------------------------------------- cut transitions
def cut_transition(frame_at, cuts, cfg, t, fps=30.0):
    """frame_at(t) → BGR; cuts = edited-time cut list.  Returns processed frame or None when no cut is near.
    cfg: {"out": preset, "in": preset, "dur": 0.5} or a list (one cfg per cut, None = plain cut)."""
    for i, c in enumerate(cuts):
        cf = cfg[i] if isinstance(cfg, list) else cfg
        if not cf:
            continue
        d = float(cf.get("dur", 0.5))
        if c - d / 2 <= t < c + d / 2:
            img = frame_at(t)
            if t < c and cf.get("out"):
                p = AM.retimed(AM.get(cf["out"]), d / 2)
                return AM.apply_stack(img, p["fx"], t - (c - d / 2), fps, cf.get("motion_blur", 180))
            if t >= c and cf.get("in"):
                p = AM.retimed(AM.get(cf["in"]), d / 2)
                return AM.apply_stack(img, p["fx"], t - c, fps, cf.get("motion_blur", 180))
            return img
    return None


def paired_in(out_name):
    """'blur_02_out' → 'blur_02_in' when it exists (presets come in matching pairs)."""
    lib = AM.library()
    k = out_name[:-4] + "_in" if out_name.endswith("_out") else out_name
    return k if k in lib else None
