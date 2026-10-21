"""
mograph.motion.bridge — v18 motion kit inside BOTH renderers.

v14 footage layers / v5–v13 scene elements (same keys):
    {"type": "saber", "preset": "neon", "core": "text", "text": "مقاله", "t": "w:مقاله", "out": "we:مقاله+1",
     "anchor": "head_top", "offset": [0, -0.6], "size": 150, "reveal": {"dur": 0.5}, "pass": "front"}
        v14: "anchor"/"offset" place text/shape cores on the speaker (x, y computed per frame); path/saber
        coords stay in screen px.  Light is ADDED (screen-like) and also feeds the v14 emission buffer so the glow
        spills onto the speaker.  "reveal"/"out" times are RELATIVE to the layer start / out (seconds).
    {"type": "character", "rig": "bean", "x": 540, "y": 1700, "size": 420, "actions": [...], "lip_sync": {...}}
    {"type": "mouth", "x": 540, "y": 900, "size": 200, "lip_sync": {"words": "words.json"}}   # mouth only
fx stack item (any v14 layer / v5 element, aefx bridge):
    "fx": [{"saber": {"preset": "electric"}, "t": 2.0, "dur": 1.5}]      → the LAYER's own alpha is the core
"""
from __future__ import annotations

import math

import numpy as np

from . import saber as SB


# ------------------------------------------------------------------------------------- saber
def _saber_spec(L, t0, t1, t, x=None, y=None, k=1.0):
    sp = {kk: v for kk, v in L.items() if not kk.startswith("_") and kk not in ("type", "t", "out", "id", "pass", "z")}
    if x is not None and sp.get("core", "saber") in ("text", "shape"):
        sp["x"], sp["y"] = x, y
    if k != 1.0:
        sp["size"] = float(sp.get("size", 160)) * k
        if "w" in sp:
            sp["w"] = float(sp["w"]) * k
        if "h" in sp:
            sp["h"] = float(sp["h"]) * k
    rv = sp.get("reveal")
    if rv is not None:
        rv = dict(rv) if isinstance(rv, dict) else {"dur": float(rv)}
        rv["t"] = t0 + float(rv.get("t", 0.0))
        sp["reveal"] = rv
    else:
        sp["reveal"] = {"t": t0, "dur": float(L.get("in_dur", 0.45)), "ease": "out_expo"}
    if t1 is not None and t1 < 1e8:
        od = dict(sp.get("out_fx") or {}) if isinstance(sp.get("out_fx"), dict) else {}
        sp["out"] = {"t": t1 + float(od.get("t", 0.0)), "dur": float(od.get("dur", 0.35)), "ease": od.get("ease", "in_cubic")}
    else:
        sp.pop("out", None)
    import zlib                                   # stable across processes (hash() is salted → chunked renders differed)
    sp.setdefault("seed", zlib.crc32(str(L.get("id", L.get("text", ""))).encode("utf-8")) % 997)
    return sp


def draw_saber_layer(ctx, L, F):
    """v14 DRAW entry."""
    from ..v14 import layers as LY
    t = F.t
    t0, t1 = L["_t0"], L["_t1"]
    if t < t0 or t > t1 + 0.6:
        return None
    if getattr(F, "prepass", False) and F.ectx is None:
        return None
    x = y = None
    k = 1.0
    if L.get("anchor", "screen") != "screen" or L.get("core") in ("text", "shape"):
        x, y, k = LY.pos(L, F)
    sp = _saber_spec(L, t0, t1, t, x, y, k)
    if "color" in sp and isinstance(sp["color"], str) and sp["color"].startswith("@"):
        sp["color"] = L["_pal"].get(sp["color"][1:], "#4fc3ee")
    if getattr(F, "prepass", False):                     # emission pre-pass → light the speaker
        if F.ectx is not None and L.get("spill", False):            # opt-in: the v14 bloom re-glows it
            fx = _cached(L, F, sp, t)
            import cairo
            h2, w2 = F.H // 2, F.W // 2
            import cv2
            # spill = soft glow only (the sharp core would be re-bloomed by the compositor and smear the letters)
            small = cv2.GaussianBlur(cv2.resize(fx, (w2, h2), interpolation=cv2.INTER_AREA), (0, 0), 14)
            small = np.ascontiguousarray(small)
            s = cairo.ImageSurface.create_for_data(memoryview(small), cairo.FORMAT_ARGB32, w2, h2)
            F.ectx.save()
            F.ectx.identity_matrix()
            F.ectx.set_source_surface(s, 0, 0)
            F.ectx.paint_with_alpha(float(L.get("spill_amount", 0.12)))     # v14 bloom amplifies emission a lot
            F.ectx.restore()
        return None
    fx = _cached(L, F, sp, t)
    SB.blit_cairo(ctx, fx)
    return None


def _cached(L, F, sp, t):
    key = ("_sb", round(t, 4))
    c = L.get("_sbc")
    if c is not None and c[0] == key:
        return c[1]
    fx = SB.render(sp, t, F.W, F.H, k=float(L.get("quality", 0.5)))
    L["_sbc"] = (key, fx)
    return fx


# ------------------------------------------------------------------------------------- character / mouth
def _prep_char(L, resolve, base, words_out=None):
    from . import character as CH
    if "_ch" not in L:
        spec = dict(L)
        ls = spec.get("lip_sync")
        if ls and words_out is not None and isinstance(ls, dict) and not ls.get("visemes") and not ls.get("raw"):
            spec["lip_sync"] = dict(ls, map=words_out)          # v14: anchor to the EDITED / retimed transcript
        L["_ch"] = CH.prepare(spec, resolve, base)
    return L["_ch"]


def draw_character_layer(ctx, L, F):
    from . import character as CH
    if getattr(F, "prepass", False):
        return None
    R = F.R
    words = [(w["t"], w["end"], w["w"]) for w in R.tl.words]
    C = _prep_char(L, R.tl.resolve, R.base, words)
    from ..v14 import layers as LY
    if L.get("anchor", "screen") != "screen":
        x, y, k = LY.pos(L, F)
    else:
        x, y, k = float(L.get("x", F.W / 2)), float(L.get("y", F.H * 0.85)), 1.0
    t = F.t
    if t < L["_t0"] - 0.01:
        return None
    a = 1.0
    if t > L["_t1"]:
        a = max(0.0, 1 - (t - L["_t1"]) / 0.3)
    return CH.draw(ctx, C, t, x, y, float(L.get("size", 420)) * k, a)


def draw_mouth_layer(ctx, L, F):
    from .lipsync import draw_mouth, shape_at, track
    if getattr(F, "prepass", False):
        return None
    if "_vis" not in L:
        ls = L.get("lip_sync") or {}
        L["_vis"] = ls.get("visemes") or track([(w["t"], w["end"], w["w"]) for w in F.R.tl.words], None, F.R.fps)
    from ..v14 import layers as LY
    x, y, k = LY.pos(L, F) if L.get("anchor", "screen") != "screen" else (float(L.get("x", F.W / 2)),
                                                                         float(L.get("y", F.H / 2)), 1.0)
    draw_mouth(ctx, shape_at(L["_vis"], F.t), x, y, float(L.get("size", 200)) * k)
    return None


def register_v14(DRAW):
    DRAW["saber"] = draw_saber_layer
    DRAW["character"] = draw_character_layer
    DRAW["mouth"] = draw_mouth_layer


# ------------------------------------------------------------------------------------- v5–v13 elements
def _register_v5():
    from ..elements import element

    @element("saber", dur=0.0, enter="none", exit="fade")
    def saber_element(ctx, el, s):
        tgt = ctx.get_target()
        W, H = tgt.get_width(), tgt.get_height()
        t0 = float(el.get("t", 0.0))
        t1 = float(el["out"]) if isinstance(el.get("out"), (int, float)) else None
        sp = _saber_spec(el, t0, t1, s.t)
        sp["opacity"] = max(0.0, min(1.0, s.alpha)) * float(el.get("opacity", 1.0))
        fx = SB.render(sp, s.t, W, H, k=float(el.get("quality", 0.5)))
        SB.blit_cairo(ctx, fx)
        return None

    @element("character", dur=0.0, enter="none", exit="fade")
    def character_element(ctx, el, s):
        from . import character as CH
        C = _prep_char(el, None, ".")
        return CH.draw(ctx, C, s.t, float(el.get("x", 540)), float(el.get("y", 1600)), float(el.get("size", 420)),
                       max(0.0, min(1.0, s.alpha)))

    @element("mouth", dur=0.0, enter="none", exit="fade")
    def mouth_element(ctx, el, s):
        from .lipsync import draw_mouth, shape_at, track
        if "_vis" not in el:
            ls = el.get("lip_sync") or {}
            el["_vis"] = ls.get("visemes") or track(ls.get("words", []), ls.get("audio"))
        draw_mouth(ctx, shape_at(el["_vis"], s.t), float(el.get("x", 540)), float(el.get("y", 900)),
                   float(el.get("size", 200)))
        return None


# ------------------------------------------------------------------------------------- fx-stack item
def apply_saber_item(img, it, t, ramp=1.0):
    """aefx bridge hook: glow a BGRA premultiplied layer with saber using its own alpha as the core."""
    sp = dict(it["saber"]) if isinstance(it["saber"], dict) else {"preset": str(it["saber"])}
    sp.setdefault("core", "alpha")
    sp.setdefault("text_mode", "outline" if sp.get("outline") else "fill")
    if img.shape[-1] != 4:
        A = img.max(axis=2).astype(np.float32) / 255.0
    else:
        A = img[..., 3].astype(np.float32) / 255.0
    H, W = img.shape[:2]
    sp["glow_intensity"] = float(SB.params(sp).get("glow_intensity", 1.0)) * ramp
    fx = SB.render(sp, t, W, H, alpha=A)
    if img.shape[-1] == 4:
        import cv2
        out = img.copy()
        keep = 1.0 if sp.get("keep_layer", True) else 0.0
        rgb = cv2.add((out[..., :3] * keep).astype(np.uint8), fx[..., :3])
        out[..., :3] = rgb
        out[..., 3] = np.maximum((out[..., 3] * keep).astype(np.uint8), fx[..., 3])
        return out
    import cv2
    return cv2.add(img, fx[..., :3])


def install():
    """idempotent: register everything (called from mograph.motion __init__)."""
    try:
        from .easing import install_into_core
        install_into_core()
    except Exception:  # noqa: BLE001
        pass
    try:
        _register_v5()
    except Exception:  # noqa: BLE001
        pass
