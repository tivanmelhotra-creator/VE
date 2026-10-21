"""
mograph.fx — v11 "Studio Pantheon": real-footage editing & compositing inside the vector engine.

ELEMENT  footage   — one clip = a complete edit-suite layer (everything is a pure function of t → frame-exact,
                     worker-safe, cached).  All keys optional except src.
  placement   x y w h (box, cover-fit by default)  fit: cover|contain   radius  border (#hex|true)  border_w
              shadow (true | {"blur":40,"alpha":0.45,"dy":24})   anchor_uv [u,v] (source point placed at the box centre)
  reframe     "frame": [[t, [u, v, zoom]], …]   keyframed virtual camera inside the clip (u,v = source point at centre)
              "reframe": "face" | "<tracker>"   auto-follow (keeps that point at "reframe_at" [0.5,0.4]), "reframe_zoom"
              "punch": [t, …] | "cuts" | "beats"  instant editorial punch-in toggling 1.0 ↔ 1+punch_amt (0.12)
              "zooms": [{"t", "dur", "to": 1.25, "ease": "out_expo", "back": t2}]  smooth push-ins (Hormozi zoom)
              "beat_pump": 0.02   scale bump on every music beat (spec.beats)   "shake": [{"t","dur","amp","freq"}]
  time        clip_start  speed  loop
              "ramps": [{"t": 2.0, "dur": 0.35, "speed": 0.35, "hold": 0.8}]  speed ramp: ease down to `speed`
                       over dur, hold, ease back to 1.0 (smooth time-remapping; source time = ∫ speed dt)
              "remap": [[t_local, src_t], …]   explicit time-remap curve (freeze = two keys with the same src_t)
              "beat_ramp": {"slow": 0.45, "fast": 1.6, "every": 1}  slow on each beat, fast between (music-locked)
              "freeze": [[t0, t1], …]  freeze frames
  colour      "grade": preset | {"exposure","contrast","saturation","temp","tint","gamma","lift","fade"}
                       presets: none flat natural vivid warm cool teal_orange mono noir bleach cyan_negative
              "grade_keys": [[t, preset|dict], …]   animated grade (raw → corrected reveal)
              "selective": {"keep": "#e14c2c" | hue, "tol": 14, "t": .., "dur": .., "amount": 1}  everything grey
                           except one colour (ref01: orange scrubs on grey)
              "hue_shift": [{"t", "dur": 0.25, "from": "#e14c2c"|hue, "to": "#1e7bff"|hue, "tol": 14,
                             "back": t2, "within": "person|all"}]   colour keyframing (clothes change colour)
              "halftone": [{"t","dur","cell":14,"color":"#e14c2c","bg":"#f0e6dc","fade":0.15}]   dot-screen look
              "flash": [t | {"t","dur":0.18,"color":"#fff","amount":0.85}]   exposure flash pulse
              "looks": [{"t","dur","look":"invert_cyan|mono|noir|negative"}]   short stylised shots (ref01 VIBE)
  matte       "matte": false|"auto"|"onnx"|"chroma"|"grabcut"   (automatic when anything needs the person cut-out)
              "matte_h": 512   "bg": "keep" | "blur" | "none" | "#hex" | "world"   replace the background behind the person
              "roto": {"mode": "color", "at"|"hsv"|"color", "tol"} | {"mode": "person"}   object mask for glow/hue
  depth       any chapter element with  "behind": "<footage id>"  is drawn BETWEEN the plate and the person
              (text passes behind the speaker — depth masking / roto layering).  "front" = normal elements.
  glow        "edge_glow": {"t","dur"|"out","color":"#2ad1ff","width":10,"core":"#ffffff","pulse":2.0,
                            "target":"person|roto","flicker":0.15}   saber stroke / neon rim around the cut-out
  tracking    "trackers": {"hand": {"kind":"hand","point":"index_tip"}, "box": {"kind":"planar","quad":[...]}}
              any element: "follow": "<footage id>.<tracker>"  (+ follow_offset [dx,dy], follow_rot, follow_scale,
              follow_lag seconds) → attached to the motion (Motion Tracking / Point Track).
  audio       "audio": true (mix the clip's own audio, follows remap)  "audio_gain"  "audio_enhance": true
              (high-pass + FFT denoise + compressor, cached) — ref04 "Enhance audio".

ELEMENT  planar_insert   {"follow": "clip.box", "src": "screen.png"|"video.mp4"|"color", "fit":"cover",
                          "edge": 0, "blend":"normal|screen", "grade"}  — screen replacement / box-art swap
                          (corners of a planar tracker, perspective-warped).
"""
from __future__ import annotations

import math
import os
import subprocess

import cairo
import cv2
import numpy as np

from .core import clamp, color as _color, ease_in_out, ease_out, get_ease, lerp, prog
from .elements import REGISTRY, compute_state, element

# ---------------------------------------------------------------------------
# colour helpers
# ---------------------------------------------------------------------------
GRADES = {
    "none": {},
    "flat": {"contrast": 0.82, "saturation": 0.72, "lift": 0.05, "exposure": -0.05},
    "natural": {"contrast": 1.06, "saturation": 1.08, "exposure": 0.03},
    "vivid": {"contrast": 1.14, "saturation": 1.32, "exposure": 0.04, "temp": 0.04},
    "warm": {"temp": 0.12, "saturation": 1.1, "contrast": 1.05},
    "cool": {"temp": -0.12, "saturation": 1.0, "contrast": 1.05},
    "teal_orange": {"temp": 0.05, "tint": -0.03, "saturation": 1.18, "contrast": 1.12, "split": 0.12},
    "mono": {"saturation": 0.0, "contrast": 1.08},
    "noir": {"saturation": 0.0, "contrast": 1.35, "exposure": -0.08, "gamma": 1.1},
    "bleach": {"saturation": 0.45, "contrast": 1.25, "fade": 0.04},
    "moody": {"saturation": 0.85, "contrast": 1.18, "exposure": -0.1, "temp": -0.03, "fade": 0.03},
    "cyan_negative": {"invert": 1.0, "saturation": 0.0, "tint_rgb": [0.25, 0.95, 1.0]},
}
_NUM = ("exposure", "contrast", "saturation", "temp", "tint", "gamma", "lift", "fade", "split", "invert")


def to_hue(c):
    """'#rrggbb' | [r,g,b] | number(OpenCV hue 0..180) → OpenCV hue."""
    if isinstance(c, (int, float)):
        return float(c)
    r, g, b, _ = _color(c)
    hsv = cv2.cvtColor(np.uint8([[[b * 255, g * 255, r * 255]]]), cv2.COLOR_BGR2HSV)[0, 0]
    return float(hsv[0])


def grade_params(g):
    if g is None:
        return {}
    if isinstance(g, str):
        return dict(GRADES.get(g, {}))
    out = dict(GRADES.get(g.get("preset", "none"), {}))
    out.update({k: v for k, v in g.items() if k != "preset"})
    return out


def lerp_grade(a, b, p):
    out = {}
    for k in set(a) | set(b):
        if k in _NUM:
            d = 1.0 if k in ("contrast", "saturation", "gamma") else 0.0
            out[k] = lerp(float(a.get(k, d)), float(b.get(k, d)), p)
        elif k == "tint_rgb":
            va, vb = a.get(k, [1, 1, 1]), b.get(k, [1, 1, 1])
            out[k] = [lerp(x, y, p) for x, y in zip(va, vb)]
    return out


_LUT_CACHE = {}


def _lut(ch, exposure, contrast, gamma, lift, fade, gain):
    key = (ch, round(exposure, 3), round(contrast, 3), round(gamma, 3), round(lift, 3), round(fade, 3), round(gain, 3))
    lut = _LUT_CACHE.get(key)
    if lut is None:
        x = np.arange(256, dtype=np.float32) / 255.0
        x = x * (2 ** exposure) * gain
        x = (x - 0.5) * contrast + 0.5
        x = np.clip(x, 0, 1) ** (1.0 / max(0.05, gamma))
        x = lift + x * (1 - lift)
        x = fade + x * (1 - 2 * fade) if fade else x
        lut = np.clip(x * 255, 0, 255).astype(np.uint8)
        if len(_LUT_CACHE) > 512:
            _LUT_CACHE.clear()
        _LUT_CACHE[key] = lut
    return lut


def apply_grade(fr, g):
    """BGR uint8 → graded BGR uint8 (LUT per channel + saturation mix: ~4 ms at 720p)."""
    if not g:
        return fr
    if g.get("invert", 0) >= 0.5:
        fr = 255 - fr
    t, ti = float(g.get("temp", 0.0)), float(g.get("tint", 0.0))
    gains = (1 - t * 0.8 + ti * 0.2, 1 - ti * 0.6, 1 + t * 0.8 + ti * 0.2)     # B, G, R
    args = [float(g.get(k, d)) for k, d in (("exposure", 0), ("contrast", 1), ("gamma", 1), ("lift", 0), ("fade", 0))]
    if any(abs(a - d) > 1e-3 for a, d in zip(args, (0, 1, 1, 0, 0))) or abs(t) > 1e-3 or abs(ti) > 1e-3:
        b, gg, r = cv2.split(fr)
        fr = cv2.merge([cv2.LUT(b, _lut(0, *args, gains[0])), cv2.LUT(gg, _lut(1, *args, gains[1])),
                        cv2.LUT(r, _lut(2, *args, gains[2]))])
    s = float(g.get("saturation", 1.0))
    if abs(s - 1.0) > 1e-3:
        gray = cv2.cvtColor(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        fr = cv2.addWeighted(fr, s, gray, 1 - s, 0)
    if g.get("tint_rgb"):
        r_, g_, b_ = g["tint_rgb"]
        fr = cv2.multiply(fr, np.array([b_, g_, r_, 0], np.float64))
    if g.get("split"):
        k = float(g["split"])
        lum = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        sh = (1 - lum)[..., None] * np.array([k * 40, k * 18, -k * 10], np.float32)      # teal shadows
        hi = lum[..., None] * np.array([-k * 25, k * 5, k * 35], np.float32)            # orange highs
        fr = np.clip(fr.astype(np.float32) + sh + hi, 0, 255).astype(np.uint8)
    return fr


def hue_mask(fr_hsv, h0, tol, smin=50, vmin=35, feather=3):
    h = fr_hsv[..., 0].astype(np.int16)
    d = np.abs(h - int(round(h0)))
    d = np.minimum(d, 180 - d).astype(np.float32)
    m = np.clip((tol + 4 - d) / 8.0, 0, 1)
    m *= np.clip((fr_hsv[..., 1].astype(np.float32) - smin * 0.6) / (smin * 0.6 + 1), 0, 1)
    m *= np.clip((fr_hsv[..., 2].astype(np.float32) - vmin * 0.6) / (vmin * 0.6 + 1), 0, 1)
    if feather:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    return m


def selective_color(fr, keep_hue, tol, amount=1.0, smin=70):
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    m = hue_mask(hsv, keep_hue, tol, smin=smin)
    gray = cv2.cvtColor(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    k = (1 - m * 1.0)[..., None] * amount
    return (fr.astype(np.float32) * (1 - k) + gray.astype(np.float32) * k).astype(np.uint8)


def hue_shift(fr, h_from, h_to, tol, p, within=None, sat_boost=1.0, smin=110, vmin=50):
    """Shift pixels of hue h_from toward h_to by progress p (0..1) — mask feathered, optional limit mask.
    smin (default 110/255) keeps skin tones (low saturation orange) out of the selection."""
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    m = hue_mask(hsv, h_from, tol, smin=smin, vmin=vmin) * p
    if within is not None:
        m *= within
    if m.max() < 0.01:
        return fr
    d = ((h_to - h_from + 90) % 180) - 90
    h = hsv[..., 0].astype(np.float32)
    hsv2 = hsv.copy()
    hsv2[..., 0] = np.mod(h + d * m, 180).astype(np.uint8)
    if sat_boost != 1.0:
        hsv2[..., 1] = np.clip(hsv[..., 1] * (1 + (sat_boost - 1) * m), 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)


def halftone(fr, cell, ink, paper, amount=1.0):
    """Dot-screen look: dot radius ∝ darkness (cell px)."""
    h, w = fr.shape[:2]
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (max(1, w // cell), max(1, h // cell)), interpolation=cv2.INTER_AREA)
    out = np.empty_like(fr)
    pc, ic = _color(paper), _color(ink)
    out[:] = (int(pc[2] * 255), int(pc[1] * 255), int(pc[0] * 255))
    col = (int(ic[2] * 255), int(ic[1] * 255), int(ic[0] * 255))
    sh, sw = small.shape
    rmax = cell * 0.62
    for j in range(sh):
        cy = int(j * cell + cell / 2)
        row = small[j]
        for i in range(sw):
            r = rmax * (1 - row[i] / 255.0) ** 0.8
            if r >= 0.7:
                cv2.circle(out, (int(i * cell + cell / 2), cy), int(round(r)), col, -1, cv2.LINE_AA)
    if amount < 1:
        out = cv2.addWeighted(out, amount, fr, 1 - amount, 0)
    return out


# ---------------------------------------------------------------------------
# time remapping
# ---------------------------------------------------------------------------
def _speed_at(el, lt, beats_local):
    sp = float(el.get("speed", 1.0))
    for r in el.get("ramps") or []:
        t0 = float(r["t"]); d = float(r.get("dur", 0.35)); hold = float(r.get("hold", 0.6))
        target = float(r.get("speed", 0.35))
        back = float(r.get("back_dur", d))
        if lt < t0 or lt > t0 + d + hold + back:
            continue
        if lt < t0 + d:
            q = ease_in_out(prog(lt, t0, d))
        elif lt < t0 + d + hold:
            q = 1.0
        else:
            q = 1 - ease_in_out(prog(lt, t0 + d + hold, back))
        sp = lerp(sp, target, q)
    br = el.get("beat_ramp")
    if br and beats_local:
        slow, fast = float(br.get("slow", 0.45)), float(br.get("fast", 1.6))
        every = int(br.get("every", 1))
        bs = beats_local[::every]
        i = np.searchsorted(bs, lt)
        if 0 < i < len(bs):
            a, b = bs[i - 1], bs[i]
            ph = (lt - a) / max(1e-3, b - a)
            # slow right on the beat, accelerate mid-interval, decelerate into the next beat (mean ≈ 1)
            w = 0.5 - 0.5 * math.cos(2 * math.pi * ph)
            sp *= lerp(slow, fast, w)
    for f0, f1 in el.get("freeze") or []:
        if float(f0) <= lt < float(f1):
            return 0.0
    return sp


def build_time_map(el, scene, span):
    """Table lt → src_t (60 Hz) for the element's life span (cached on the element)."""
    tm = el.get("_tmap")
    if tm is not None:
        return tm
    t_in = float(el.get("t", 0.0))
    cs = float(el.get("clip_start", 0.0))
    if el.get("remap"):
        pts = np.array([[float(a), float(b)] for a, b in el["remap"]], np.float64)
        tm = ("remap", pts)
    else:
        hz = 60.0
        n = int(max(1.0, span) * hz) + 2
        beats_local = [b - t_in for b in (getattr(scene, "beats", None) or []) if b >= t_in - 1] if scene else []
        ts = np.arange(n) / hz
        sp = np.array([_speed_at(el, t, beats_local) for t in ts])
        src = cs + np.concatenate([[0.0], np.cumsum((sp[1:] + sp[:-1]) / 2 / hz)])
        tm = ("table", (ts, src))
    el["_tmap"] = tm
    return tm


def src_time(el, scene, t):
    """Scene time t → source clip time."""
    if el.get("segments"):                      # v12: EDL from mograph.cut (edited timeline, scene time)
        from .cut import to_src_time
        return to_src_time(el["segments"], t)
    lt = max(0.0, t - float(el.get("t", 0.0)))
    simple = not (el.get("ramps") or el.get("remap") or el.get("beat_ramp") or el.get("freeze"))
    if simple:
        st = lt * float(el.get("speed", 1.0)) + float(el.get("clip_start", 0.0))
    else:
        end = el.get("out")
        span = (float(end) - float(el.get("t", 0.0)) + 1.0) if end is not None else (scene.duration if scene else 60.0)
        kind, data = build_time_map(el, scene, span)
        if kind == "remap":
            st = float(np.interp(lt, data[:, 0], data[:, 1]))
        else:
            st = float(np.interp(lt, data[0], data[1]))
    if scene is not None and getattr(scene, "cut_map", None) is not None and el.get("follow_cuts"):
        st = scene.cut_map(st)
    return st


# ---------------------------------------------------------------------------
# spec normalisation: every time inside a footage element is LOCAL (seconds after its "t");
# strings ("w:word", "b:12", "abs:3.2", "end-1") are resolved by the scene and converted to local.
# ---------------------------------------------------------------------------
_LOCAL_LISTS = ("ramps", "zooms", "flash", "hue_shift", "halftone", "looks", "shake")
_LOCAL_DICTS = ("selective", "edge_glow")


def normalize_footage(el, scene, ch_t, ch_end):
    t0 = float(el.get("t", ch_t))

    def loc(v):
        if isinstance(v, (int, float)):
            return float(v)
        return scene.resolve_time(v, ch_t, ch_end) - t0

    for k in _LOCAL_LISTS:
        out = []
        for it in el.get(k) or []:
            if isinstance(it, (int, float, str)):
                out.append(loc(it))
                continue
            it = dict(it)
            for tk in ("t", "back", "out"):
                if tk in it and it[tk] is not None:
                    it[tk] = loc(it[tk])
            out.append(it)
        if k in el:
            el[k] = out
    for k in _LOCAL_DICTS:
        if isinstance(el.get(k), dict):
            d = dict(el[k])
            for tk in ("t", "out"):
                if tk in d and d[tk] is not None:
                    d[tk] = loc(d[tk])
            el[k] = d
    if isinstance(el.get("punch"), list):
        el["punch"] = [loc(v) for v in el["punch"]]
    for k in ("frame", "grade_keys", "remap"):
        if isinstance(el.get(k), list):
            el[k] = [[loc(kk[0])] + list(kk[1:]) for kk in el[k]]
    if isinstance(el.get("freeze"), list):
        el["freeze"] = [[loc(a), loc(b)] for a, b in el["freeze"]]


# ---------------------------------------------------------------------------
# view (placement + virtual camera) — shared by drawing, follow and planar inserts
# ---------------------------------------------------------------------------
def _env_keys(keys, t, default):
    """[[t, [u,v,z]], …] eased interpolation."""
    if not keys:
        return default
    ks = sorted(((float(k[0]), k[1], k[2] if len(k) > 2 else "in_out") for k in keys), key=lambda k: k[0])
    if t <= ks[0][0]:
        return ks[0][1]
    for (t0, v0, _), (t1, v1, e) in zip(ks, ks[1:]):
        if t < t1:
            p = get_ease(e)((t - t0) / max(1e-6, t1 - t0))
            return [lerp(a, b, p) for a, b in zip(v0, v1)]
    return ks[-1][1]


def view(el, scene, t, fw, fh, src_t=None):
    """→ dict(x, y, W, H, k, cu, cv): source pixel (px, py) maps to local
       (x + (px - cu*fw)*k, y + (py - cv*fh)*k), clipped to the W×H box centred at (x, y)."""
    from .elements import xy
    x, y = xy(el, t)
    sc = scene
    W = float(el.get("w", sc.W if sc else 1080))
    H = float(el.get("h", (W * fh / fw) if el.get("fit") == "contain" and not el.get("h") else (sc.H if sc else 1920)))
    if el.get("w") and not el.get("h") and el.get("fit") != "contain":
        H = W * fh / fw
    fit = el.get("fit", "cover")
    k0 = max(W / fw, H / fh) if fit == "cover" else min(W / fw, H / fh)
    lt = t - float(el.get("t", 0.0))
    au = el.get("anchor_uv", [0.5, 0.5])
    u, v, z = _env_keys(el.get("frame"), lt, [float(au[0]), float(au[1]), 1.0])
    rf = el.get("reframe")
    if rf and scene is not None:
        tr = footage_tracker(el, scene, "face" if rf == "face" else rf)
        if tr is not None:
            st = src_time(el, scene, t) if src_t is None else src_t
            s = tr.sample(st)
            at = el.get("reframe_at", [0.5, 0.4])
            z = float(el.get("reframe_zoom", z))
            # place tracked point at `at` of the box
            u = s["x"] - (at[0] - 0.5) * W / (fw * k0 * z)
            v = s["y"] - (at[1] - 0.5) * H / (fh * k0 * z)
    z *= punch_scale(el, scene, t) * zoom_scale(el, lt) * pump_scale(el, scene, t) * auto_push(el, t)
    k = k0 * z
    # never reveal outside the clip in cover mode: clamp the source centre
    if fit == "cover":
        hu = W / (2 * fw * k); hv = H / (2 * fh * k)
        u = clamp(u, min(hu, 0.5), max(1 - hu, 0.5)); v = clamp(v, min(hv, 0.5), max(1 - hv, 0.5))
    dx, dy = shake_offset(el, lt)
    return {"x": x + dx, "y": y + dy, "W": W, "H": H, "k": k, "cu": u, "cv": v, "fw": fw, "fh": fh}


def map_point(vw, u, v):
    """normalised source point → chapter-local coordinates."""
    return (vw["x"] + (u - vw["cu"]) * vw["fw"] * vw["k"], vw["y"] + (v - vw["cv"]) * vw["fh"] * vw["k"])


def auto_push(el, t):
    """v12: "auto_push": 0.05 → every edit segment slowly pushes in by 5 % (resets on each cut)."""
    k = float(el.get("auto_push", 0.0) or 0.0)
    segs = el.get("segments")
    if not k or not segs:
        return 1.0
    e0, e1 = segs[0][2], segs[0][2] + segs[0][1] - segs[0][0]
    for a, b, e in segs:
        if t >= e:
            e0, e1 = e, e + (b - a)
    return 1 + k * ease_in_out(clamp((t - e0) / max(0.5, e1 - e0)))


def punch_scale(el, scene, t):
    pv = el.get("punch")
    if not pv:
        return 1.0
    amt = float(el.get("punch_amt", 0.12))
    if pv == "cuts":
        ts = list(getattr(scene, "cuts", None) or [])
    elif pv == "beats":
        ts = list(getattr(scene, "beats", None) or [])[::int(el.get("punch_every", 4))]
    else:
        ts = [float(x) + float(el.get("t", 0.0)) for x in pv]      # local → scene time
    past = [c for c in ts if c <= t]
    if not past:
        return 1.0
    kk = len(past)
    target = 1 + (amt if kk % 2 else 0)
    prev = 1 + (amt if (kk - 1) % 2 else 0)
    return lerp(prev, target, ease_in_out(clamp((t - past[-1]) / 0.06)))


def zoom_scale(el, t):
    z = 1.0
    for zz in el.get("zooms") or []:
        t0 = float(zz["t"]); d = float(zz.get("dur", 0.5)); to = float(zz.get("to", 1.2))
        p = get_ease(zz.get("ease", "out_expo"))(prog(t, t0, d))
        if zz.get("back") is not None:
            p *= 1 - ease_in_out(prog(t, float(zz["back"]), float(zz.get("back_dur", d))))
        z *= lerp(1.0, to, p)
    return z


def pump_scale(el, scene, t):
    a = float(el.get("beat_pump", 0.0) or 0.0)
    if not a or scene is None or not scene.beats:
        return 1.0
    b = scene.beats
    i = int(np.searchsorted(b, t)) - 1
    if i < 0:
        return 1.0
    d = t - b[i]
    return 1 + a * math.exp(-d * 9.0)


def shake_offset(el, t):
    dx = dy = 0.0
    for sh in el.get("shake") or []:
        t0 = float(sh["t"]); d = float(sh.get("dur", 0.3))
        if t0 <= t < t0 + d:
            env = (1 - (t - t0) / d) ** 2
            f = float(sh.get("freq", 22)); a = float(sh.get("amp", 14))
            dx += a * env * math.sin(2 * math.pi * f * t + 1.3)
            dy += a * env * math.sin(2 * math.pi * f * 1.37 * t + 0.4)
    return dx, dy


# ---------------------------------------------------------------------------
# trackers bound to a footage element
# ---------------------------------------------------------------------------
def footage_src(el):
    p = el.get("_src_enh")
    if p:
        return p
    p = el.get("_src_abs") or el["src"]
    if el.get("enhance") and os.path.exists(p):   # v12: cached upscale + denoise + sharpen proxy
        from .cut import enhance_video
        try:
            p = enhance_video(p, el["enhance"])
        except Exception as e:  # noqa: BLE001
            print(f"[enhance] skipped: {e}")
        el["_src_enh"] = p
    return p


def footage_audio_src(el):
    return el.get("_src_abs") or el["src"]


def footage_tracker(el, scene, name):
    trs = el.get("trackers") or {}
    spec = trs.get(name)
    if spec is None and name == "face":
        spec = {"kind": "face"}
    if spec is None:
        return None
    cache = el.setdefault("_trk", {})
    if name not in cache:
        from .track import get_track
        cache[name] = get_track(footage_src(el), spec, log=print)
    return cache[name]


def resolve_follow(scene, ref, t):
    """'clip.hand' → (x, y, scale, rot_deg, quad|None) in the clip chapter's local coords, or None."""
    if not ref or "." not in ref or scene is None:
        return None
    fid, name = ref.split(".", 1)
    el = scene.by_id.get(fid)
    if el is None or el.get("type") != "footage":
        return None
    tr = footage_tracker(el, scene, name)
    if tr is None:
        return None
    from .media import reader
    r = reader(footage_src(el))
    st = src_time(el, scene, t)
    s = tr.sample(st)
    vw = view(el, scene, t, r.w, r.h, src_t=st)
    x, y = map_point(vw, s["x"], s["y"])
    quad = [map_point(vw, a, b) for a, b in s["quad"]] if s.get("quad") else None
    return x, y, s["s"], math.degrees(s["rot"]), quad, s.get("conf", 1.0)


def follow_xform(scene, el, t):
    """For render.draw_world: translation (+rot/scale) that glues an element to a tracker.
    The element is authored at its own x/y (or 0,0); `follow_offset` shifts it from the tracked point."""
    ref = el.get("follow")
    if not ref:
        return None
    lag = float(el.get("follow_lag", 0.0))
    if lag > 0:
        acc = None
        n = 5
        for i in range(n):
            r = resolve_follow(scene, ref, t - lag * i / (n - 1))
            if r is None:
                continue
            acc = r if acc is None else tuple((a + b) if isinstance(a, float) else a for a, b in zip(acc, r))
        if acc is None:
            return None
        P = (acc[0] / n, acc[1] / n, acc[2] / n, acc[3] / n)
    else:
        r = resolve_follow(scene, ref, t)
        if r is None:
            return None
        P = r[:4]
    from .elements import xy
    ex, ey = xy(el, t)
    off = el.get("follow_offset", [0, 0])
    if el.get("follow_axis") == "x":
        return (P[0] + off[0] - ex, off[1], 1.0, 0.0, ex, ey)
    if el.get("follow_axis") == "y":
        return (off[0], P[1] + off[1] - ey, 1.0, 0.0, ex, ey)
    rot = (P[3] - float(el.get("follow_rot0", 0.0))) if el.get("follow_rot") else 0.0
    scl = P[2] ** float(el.get("follow_scale", 0.0)) if el.get("follow_scale") else 1.0
    return (P[0] + off[0] - ex, P[1] + off[1] - ey, scl, rot, ex, ey)


# ---------------------------------------------------------------------------
# frame pipeline
# ---------------------------------------------------------------------------
def _env(t, t0, dur, fade=0.12, out=None):
    if out is not None:
        dur = float(out) - t0
    if t < t0 or t > t0 + dur + fade:
        return 0.0
    return min(clamp((t - t0) / max(1e-3, fade)), clamp((t0 + dur + fade - t) / max(1e-3, fade)))


def _grade_at(el, lt):
    gk = el.get("grade_keys")
    if gk:
        ks = sorted(((float(k[0]), grade_params(k[1]), k[2] if len(k) > 2 else 0.35) for k in gk), key=lambda k: k[0])
        cur = grade_params(el.get("grade"))
        for t0, g, d in ks:
            if lt >= t0:
                cur = lerp_grade(cur, g, ease_in_out(prog(lt, t0, d)))
        return cur
    return grade_params(el.get("grade"))


_AUTO_CACHE = {}


def auto_levels_params(path, src_t, shots, opts):
    """v12: per-shot black/white point + neutral balance measured on 5 frames of that shot (cached)."""
    bounds = [0.0] + list(shots or []) + [1e9]
    k = max(i for i, b in enumerate(bounds[:-1]) if src_t >= b - 1e-6)
    key = (path, k)
    p = _AUTO_CACHE.get(key)
    if p is None:
        from .media import reader
        r = reader(path)
        a0, a1 = bounds[k], min(bounds[k + 1], r.duration or bounds[k] + 1)
        L, BGR = [], []
        for q in np.linspace(0.1, 0.9, 5):
            f = r.frame(a0 + (a1 - a0) * q).copy()
            L.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).ravel()[::7]); BGR.append(f.reshape(-1, 3)[::7])
        L = np.concatenate(L); BGR = np.concatenate(BGR).astype(np.float32)
        lo, hi = np.percentile(L, float(opts.get("black_pct", 0.6))), np.percentile(L, float(opts.get("white_pct", 99.6)))
        mids = BGR[(L > np.percentile(L, 35)) & (L < np.percentile(L, 75))].mean(0)
        bal = mids.mean() / np.maximum(mids, 1)
        p = {"lo": float(lo), "hi": float(max(hi, lo + 40)), "bal": bal.tolist()}
        _AUTO_CACHE[key] = p
    return p


def auto_grade(fr, p, opts):
    amt = float(opts.get("balance", 0.6))
    lo, hi = p["lo"], p["hi"]
    x = np.arange(256, dtype=np.float32)
    y = np.clip((x - lo) / (hi - lo), 0, 1)
    c = float(opts.get("curve", 0.35))                       # filmic S-curve
    y = y + c * (y - y * y) * (2 * y - 1) * -1.0 if c else y
    y = np.clip(y, 0, 1) ** float(opts.get("gamma", 0.95))
    luts = []
    for ch in range(3):
        g = 1 + (p["bal"][ch] - 1) * amt
        luts.append(np.clip(y * 255 * g, 0, 255).astype(np.uint8))
    b, g_, r = cv2.split(fr)
    out = cv2.merge([cv2.LUT(b, luts[0]), cv2.LUT(g_, luts[1]), cv2.LUT(r, luts[2])])
    vib = float(opts.get("vibrance", 0.25))
    if vib:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        sat = hsv[..., 1].astype(np.float32)
        hsv[..., 1] = np.clip(sat + vib * (255 - sat) * (sat / 255.0) * 1.2, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return out


def skin_correct(fr, opts):
    """v12: natural skin — pixels in the skin hue band (incl. magenta/orange casts) are pulled toward a target hue
    and their saturation tamed.  opts: {"hue": 12 (OpenCV 0..180), "amount": 0.6, "sat": 0.82, "band": 22}"""
    o = opts if isinstance(opts, dict) else {}
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s_, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    tgt = float(o.get("hue", 12)); band = float(o.get("band", 22))
    d = (h - tgt + 90) % 180 - 90                        # signed hue distance
    m = np.clip(1 - np.abs(d) / band, 0, 1) * np.clip((s_ - 25) / 40, 0, 1) * np.clip((v - 50) / 50, 0, 1)
    m = cv2.GaussianBlur(m, (0, 0), 2)
    amt = float(o.get("amount", 0.6))
    hsv[..., 0] = (h - d * m * amt) % 180
    hsv[..., 1] = s_ * (1 - m * (1 - float(o.get("sat", 0.82))))
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def process_frame(el, scene, fr, t, lt, person=None):
    """All colour operations of the footage element on a BGR frame at scene time t (local lt)."""
    ag = el.get("auto_grade")
    if ag:
        opts = ag if isinstance(ag, dict) else {}
        shots = ((scene.spec.get("_edit") or {}).get("shots") if scene is not None else None) or []
        st = src_time(el, scene, t)
        fr = auto_grade(fr, auto_levels_params(footage_src(el), st, shots, opts), opts)
    if el.get("skin"):
        fr = skin_correct(fr, el["skin"])
    fr = apply_grade(fr, _grade_at(el, lt))
    sel = el.get("selective")
    if sel:
        a = _env(lt, float(sel.get("t", 0)), float(sel.get("dur", 1e9)), float(sel.get("fade", 0.3)), sel.get("out"))
        if a > 0:
            fr = selective_color(fr, to_hue(sel.get("keep", "#e14c2c")), float(sel.get("tol", 14)),
                                 float(sel.get("amount", 1.0)) * a, float(sel.get("smin", 70)))
    for hs in el.get("hue_shift") or []:
        t0 = float(hs["t"]); d = float(hs.get("dur", 0.25))
        p = ease_out(prog(lt, t0, d))
        if hs.get("back") is not None:
            p *= 1 - ease_in_out(prog(lt, float(hs["back"]), d))
        if p > 0.001:
            within = person if (hs.get("within") == "person" and person is not None) else None
            fr = hue_shift(fr, to_hue(hs.get("from", "#e14c2c")), to_hue(hs.get("to", "#1e7bff")),
                           float(hs.get("tol", 10)), p, within, float(hs.get("sat", 1.0)),
                           float(hs.get("smin", 110)), float(hs.get("vmin", 50)))
    for lk in el.get("looks") or []:
        a = _env(lt, float(lk["t"]), float(lk.get("dur", 0.6)), float(lk.get("fade", 0.06)))
        if a > 0:
            look = lk.get("look", "invert_cyan")
            g = {"invert_cyan": "cyan_negative", "negative": {"invert": 1}, "mono": "mono", "noir": "noir"}.get(look, look)
            alt = apply_grade(fr, grade_params(g))
            fr = alt if a >= 0.999 else cv2.addWeighted(alt, a, fr, 1 - a, 0)
    for ht in el.get("halftone") or []:
        a = _env(lt, float(ht["t"]), float(ht.get("dur", 0.8)), float(ht.get("fade", 0.15)))
        if a > 0:
            fr = halftone(fr, int(ht.get("cell", 14)), ht.get("color", "#e14c2c"), ht.get("bg", "#f3ebe3"), a)
    return fr


def flash_amount(el, lt):
    best = 0.0
    col = "#ffffff"
    for f in el.get("flash") or []:
        f = {"t": f} if isinstance(f, (int, float)) else f
        t0 = float(f["t"]); d = float(f.get("dur", 0.18))
        if t0 - 0.03 <= lt <= t0 + d:
            v = clamp((lt - t0 + 0.03) / 0.03) if lt < t0 else (1 - (lt - t0) / d) ** 2
            v *= float(f.get("amount", 0.85))
            if v > best:
                best, col = v, f.get("color", "#ffffff")
    return best, col


def surface_from_bgr(fr, alpha=None):
    from .matte import premultiplied_surface
    from .media import to_surface
    if alpha is None:
        return to_surface(fr)
    return premultiplied_surface(fr, alpha)


def glow_surfaces(alpha, color, width, core=None, flicker=0.0, t=0.0):
    """Saber-style edge glow from a matte: outer coloured haze + thin bright core (both premultiplied)."""
    from .matte import premultiplied_surface
    a8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    edge = cv2.morphologyEx(a8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    w = max(2, int(width))
    if flicker:
        w = max(2, int(w * (1 + flicker * math.sin(t * 37.0) * math.sin(t * 11.3))))
    haze = cv2.GaussianBlur(cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=max(1, w // 3)), (0, 0), w)
    haze = np.clip(haze.astype(np.float32) / max(1, haze.max()) * 1.6, 0, 1)
    c = _color(color)
    h, wd = a8.shape
    bgr = np.empty((h, wd, 3), np.uint8)
    bgr[:] = (int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))
    outs = [premultiplied_surface(bgr, haze)]
    if core:
        cc = _color(core)
        core_a = cv2.GaussianBlur(edge, (0, 0), 1.0).astype(np.float32) / 255.0
        bgr2 = np.empty_like(bgr)
        bgr2[:] = (int(cc[2] * 255), int(cc[1] * 255), int(cc[0] * 255))
        outs.append(premultiplied_surface(bgr2, np.clip(core_a * 1.8, 0, 1)))
    return outs


def _needs_matte(el, scene):
    if el.get("matte") is False:
        return False
    if el.get("matte") or el.get("bg", "keep") != "keep" or (el.get("edge_glow") or {}).get("target", "person") == "person" and el.get("edge_glow"):
        return True
    if any(h.get("within") == "person" for h in el.get("hue_shift") or []):
        return True
    return bool(_behind_elements(el, scene))


def _behind_elements(el, scene):
    if scene is None or el.get("_chapter") is None:
        return []
    cache = el.get("_behind_list")
    if cache is None:
        cache = [e for e in scene.chapters[el["_chapter"]]["elements"] if e.get("behind") == el.get("id")]
        el["_behind_list"] = cache
    return cache


def roto_mask(el, fr, person):
    ro = el.get("roto") or {}
    mode = ro.get("mode", "person")
    if mode == "person":
        return person
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    if ro.get("hsv"):
        h0 = float(ro["hsv"][0])
    elif ro.get("color"):
        h0 = to_hue(ro["color"])
    else:
        u, v = ro.get("at", [0.5, 0.5])
        H, W = fr.shape[:2]
        h0 = float(np.median(hsv[int(v * H) - 2:int(v * H) + 3, int(u * W) - 2:int(u * W) + 3, 0]))
    m = hue_mask(hsv, h0, float(ro.get("tol", 12)), feather=0)
    m = cv2.morphologyEx((m > 0.5).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(np.float32)
    if ro.get("largest", True):
        n, lab, st, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8))
        if n > 1:
            j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            m = (lab == j).astype(np.float32)
    if person is not None and ro.get("within_person"):
        m *= person
    return cv2.GaussianBlur(m, (0, 0), 1.5)


@element("footage", dur=0.3, enter="fade", sfx="none")
def el_footage(ctx, el, s):
    from .media import reader
    from .matte import cached_matte
    sc = s.scene
    r = reader(footage_src(el))
    st = src_time(el, sc, s.t)
    fr = r.frame(st, loop=bool(el.get("loop", False)))
    idx = int(round(st * r.fps))
    lt = s.t - float(el.get("t", 0.0))
    fh, fw = fr.shape[:2]
    vw = view(el, sc, s.t, fw, fh, src_t=st)
    a = s.alpha * (1 - s.out_p) if s.exit == "fade" else s.alpha
    person = None
    has_active_behind = any(compute_state(bel, s.t, sc).ok() for bel in _behind_elements(el, sc))
    if _needs_matte(el, sc) and (el.get("bg", "keep") != "keep" or el.get("matte") or el.get("edge_glow") or has_active_behind):
        mh = int(el.get("matte_h", 512))
        ms = min(1.0, mh / fh)
        small = cv2.resize(fr, (int(fw * ms), int(fh * ms)), interpolation=cv2.INTER_AREA) if ms < 1 else fr
        m = el.get("matte") if isinstance(el.get("matte"), str) else "auto"
        pa = cached_matte(r.path, idx, small, m, el.get("key"))
        person = cv2.resize(pa, (fw, fh), interpolation=cv2.INTER_LINEAR) if ms < 1 else pa
    fr = process_frame(el, sc, fr, s.t, lt, person)
    # v13.9 Studio Headroom Extension: shifts speaker down by `headroom` px while seamlessly extending the top room wall!
    hr_px = int(float(el.get("headroom", 0)) * (fh / 1920.0))
    if hr_px > 4 and hr_px < fh // 2:
        fr_shifted = cv2.copyMakeBorder(fr[:fh - hr_px], hr_px, 0, 0, 0, cv2.BORDER_REFLECT_101)
        band_h = min(fh, hr_px + 64)
        fr_shifted[:band_h] = cv2.GaussianBlur(fr_shifted[:band_h], (31, 31), 11.0)
        fr = fr_shifted
        if person is not None:
            person = np.pad(person[:fh - hr_px], ((hr_px, 0), (0, 0)), mode="constant", constant_values=0.0)
    fa, fcol = flash_amount(el, lt)
    x, y, W, H, k = vw["x"], vw["y"], vw["W"], vw["H"], vw["k"]
    rad = float(el.get("radius", 0))
    sh = el.get("shadow")
    ctx.save()
    if s.scale != 1.0:
        ctx.translate(x, y + s.dy); ctx.scale(s.scale, s.scale); ctx.translate(-x, -(y + s.dy))
    else:
        ctx.translate(0, s.dy)
    if sh:
        sh = sh if isinstance(sh, dict) else {}
        _soft_shadow(ctx, x, y, W, H, rad, float(sh.get("blur", 36)), float(sh.get("alpha", 0.42)) * a,
                     float(sh.get("dy", 22)))
    # clip to the box
    ctx.save()
    _box_path(ctx, x, y, W, H, rad)
    ctx.clip()

    def paint(surf, alpha_mul=1.0, op=cairo.OPERATOR_OVER):
        ctx.save()
        ctx.translate(x, y); ctx.scale(k, k); ctx.translate(-vw["cu"] * fw, -vw["cv"] * fh)
        ctx.set_source_surface(surf, 0, 0)
        ctx.get_source().set_filter(cairo.FILTER_GOOD)
        ctx.set_operator(op)
        ctx.paint_with_alpha(clamp(a * alpha_mul))
        ctx.restore()

    # v13 Visual-DNA: support "dof_blur": [[t0, t1, sigma, dim_alpha], ...] to softly blur/matte the speaker himself for full-screen hero titles!
    dof_k = 0.0
    dof_sigma = 18.0
    dof_dim = 0.58
    for db in el.get("dof_blur") or []:
        t0_d, t1_d = float(db[0]), float(db[1])
        sig_d = float(db[2]) if len(db) > 2 else 18.0
        dim_d = float(db[3]) if len(db) > 3 else 0.58
        fade_d = 0.22
        if t0_d - fade_d <= s.t <= t1_d + fade_d:
            kd = min(clamp((s.t - (t0_d - fade_d)) / fade_d), clamp(((t1_d + fade_d) - s.t) / fade_d))
            if kd > dof_k:
                dof_k, dof_sigma, dof_dim = kd, sig_d, dim_d
    if dof_k > 0.01:
        rad_b = max(1, int(dof_sigma * dof_k))
        blurred = cv2.GaussianBlur(fr, (0, 0), rad_b)
        fr = cv2.addWeighted(blurred, dof_k, fr, 1.0 - dof_k, 0)
        # Apply cinematic dark-warm studio matte tint so full-screen typography pops with 100% contrast
        fr = (fr.astype(np.float32) * (1.0 - dof_dim * dof_k)).astype(np.uint8)

    bgm = el.get("bg", "keep")
    bg_dim = float(el.get("bg_dim", 0.0))
    if (bgm == "keep" or person is None) and bg_dim <= 0.01:
        surf, _k1 = surface_from_bgr(fr)
        paint(surf)
    elif bgm == "keep" and person is not None and bg_dim > 0.01:
        # Darken & warm-grade the background wall behind the segmented speaker (like @novadent.ir!)
        tint_hex = el.get("bg_tint", "#1C140C")
        tr, tg, tb = _color(tint_hex)[:3]
        tint_bgr = np.array([tb * 255.0, tg * 255.0, tr * 255.0], dtype=np.float32)
        bg_fr = cv2.GaussianBlur(fr, (21, 21), 6.0).astype(np.float32)
        bg_fr = np.clip(bg_fr * (1.0 - bg_dim) + tint_bgr * bg_dim * 0.65, 0, 255).astype(np.uint8)
        surf, _k1 = surface_from_bgr(bg_fr)
        paint(surf)
    elif bgm == "blur":
        surf, _k1 = surface_from_bgr(cv2.GaussianBlur(fr, (0, 0), max(4, fw / 60)))
        paint(surf)
    elif isinstance(bgm, str) and bgm.startswith("#"):
        ctx.set_source_rgba(*_color(bgm, a)); ctx.paint()
    # (b) depth layer: elements authored with "behind": "<this id>"
    any_behind_active = False
    for bel in _behind_elements(el, sc):
        bs = compute_state(bel, s.t, sc)
        if bs.ok():
            any_behind_active = True
            bs.alpha *= a
            ctx.save()
            ctx.translate(0, -s.dy)
            fx = follow_xform(sc, bel, s.t)
            if fx:
                _apply_follow(ctx, fx)
            REGISTRY[bel["type"]]["fn"](ctx, bel, bs)
            ctx.restore()
    # (c) person on top (only needed when something is behind it or the bg was replaced)
    if person is not None and (bgm != "keep" or _behind_elements(el, sc)):
        p_mat = np.clip((person - 0.24) / 0.72, 0.0, 1.0).astype(np.float32) if any_behind_active else person
        # Subtle edge dark-multiply defringe so bright wall halo disappears on dark studio behind-layers
        fr_fg = fr
        if any_behind_active:
            edge_band = np.clip(p_mat * (1.0 - p_mat) * 4.0, 0.0, 1.0)[:, :, None]
            # Harmonize speaker contrast & warm studio key-light to match the gridded backdrop + clean edge defringe
            f32 = fr.astype(np.float32)
            f32 = np.clip((f32 - 118.0) * 1.08 + 116.0, 0, 255)
            fr_fg = np.clip(f32 * (1.0 - 0.52 * edge_band), 0, 255).astype(np.uint8)
        surf_p, _k2 = surface_from_bgr(fr_fg, p_mat)
        paint(surf_p)
    # (d) edge glow / saber
    eg = el.get("edge_glow")
    if eg:
        ga = _env(lt, float(eg.get("t", 0)), float(eg.get("dur", 1e9)), float(eg.get("fade", 0.15)), eg.get("out"))
        if ga > 0:
            target = person if eg.get("target", "person") == "person" else roto_mask(el, fr, person)
            if target is None:
                from .matte import cached_matte as _cm
                target = _cm(r.path, idx, fr, "auto")
            sm = 0.5
            tsm = cv2.resize(target, (int(fw * sm), int(fh * sm)), interpolation=cv2.INTER_AREA)
            pul = float(eg.get("pulse", 0.0))
            gp = 1.0 if not pul else 0.7 + 0.3 * math.sin(2 * math.pi * pul * lt)
            for j, gs in enumerate(glow_surfaces(tsm, eg.get("color", "#2ad1ff"), float(eg.get("width", 10)) * sm,
                                                 eg.get("core", "#ffffff"), float(eg.get("flicker", 0.0)), lt)):
                ctx.save()
                ctx.translate(x, y); ctx.scale(k / sm, k / sm); ctx.translate(-vw["cu"] * fw * sm, -vw["cv"] * fh * sm)
                ctx.set_source_surface(gs[0], 0, 0)
                ctx.set_operator(cairo.OPERATOR_ADD)
                ctx.paint_with_alpha(clamp(a * ga * (gp if j == 0 else 1.0) * float(eg.get("alpha", 1.0))))
                ctx.restore()
    if fa > 0.002:
        ctx.set_source_rgba(*_color(fcol, a * fa)); ctx.set_operator(cairo.OPERATOR_ADD); ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)
    ctx.restore()     # box clip
    b = el.get("border")
    if b:
        ctx.save()
        _box_path(ctx, x, y, W, H, rad)
        ctx.set_source_rgba(*_color(b if isinstance(b, str) else "#ffffff", a))
        ctx.set_line_width(float(el.get("border_w", 4)))
        ctx.stroke()
        ctx.restore()
    ctx.restore()
    return (x - W / 2, y - H / 2, x + W / 2, y + H / 2)


def _box_path(ctx, x, y, W, H, rad):
    ctx.new_path()
    if rad > 0:
        from .geometry import compile_path, rounded_rect_d
        ctx.append_path(compile_path(rounded_rect_d(x - W / 2, y - H / 2, W, H, rad)).whole)
    else:
        ctx.rectangle(x - W / 2, y - H / 2, W, H)


_SHADOW = {}


def _soft_shadow(ctx, x, y, W, H, rad, blur, alpha, dy):
    """Pre-blurred rounded-rect shadow (cached per size)."""
    key = (int(W), int(H), int(rad), int(blur))
    if key not in _SHADOW:
        pad = int(blur * 2.2)
        w, h = int(W) + 2 * pad, int(H) + 2 * pad
        m = np.zeros((h, w), np.uint8)
        cv2.rectangle(m, (pad, pad), (pad + int(W), pad + int(H)), 255, -1)
        if rad:
            m[:] = 0
            r = int(rad)
            cv2.rectangle(m, (pad + r, pad), (pad + int(W) - r, pad + int(H)), 255, -1)
            cv2.rectangle(m, (pad, pad + r), (pad + int(W), pad + int(H) - r), 255, -1)
            for cx, cy in ((pad + r, pad + r), (pad + int(W) - r, pad + r), (pad + r, pad + int(H) - r),
                           (pad + int(W) - r, pad + int(H) - r)):
                cv2.circle(m, (cx, cy), r, 255, -1)
        m = cv2.GaussianBlur(m, (0, 0), blur / 2)
        from .matte import premultiplied_surface
        _SHADOW.clear()
        _SHADOW[key] = (premultiplied_surface(np.zeros((h, w, 3), np.uint8), m.astype(np.float32) / 255.0), pad)
    (surf, keep), pad = _SHADOW[key]
    ctx.save()
    ctx.set_source_surface(surf, x - W / 2 - pad, y - H / 2 - pad + dy)
    ctx.paint_with_alpha(clamp(alpha))
    ctx.restore()


def _apply_follow(ctx, fx):
    dx, dy, scl, rot, ex, ey = fx
    ctx.translate(dx, dy)
    if rot or scl != 1.0:
        ctx.translate(ex, ey)
        if rot:
            ctx.rotate(math.radians(rot))
        if scl != 1.0:
            ctx.scale(scl, scl)
        ctx.translate(-ex, -ey)


# ---------------------------------------------------------------------------
# planar insert (screen replacement / box-art swap)
# ---------------------------------------------------------------------------
_IMG = {}


def _load_img(path):
    if path not in _IMG:
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise FileNotFoundError(path)
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
        elif im.shape[2] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
        _IMG[path] = im
    return _IMG[path]


@element("planar_insert", dur=0.3, enter="fade", sfx="none")
def el_planar_insert(ctx, el, s):
    sc = s.scene
    r = resolve_follow(sc, el.get("follow"), s.t)
    if r is None or not r[4]:
        return None
    quad = np.float32(r[4])
    src = el.get("src")
    base = sc.base_dir if sc else "."
    if src and src.lower().endswith((".mp4", ".mov", ".webm", ".mkv")):
        from .media import reader
        rd = reader(src if os.path.isabs(src) else os.path.join(base, src))
        im = cv2.cvtColor(rd.frame(max(0.0, s.t - float(el["t"])) * float(el.get("speed", 1.0)),
                                   loop=True), cv2.COLOR_BGR2BGRA)
    elif src and not str(src).startswith("#"):
        im = _load_img(src if os.path.isabs(src) else os.path.join(base, src))
    else:
        c = _color(src or "#ff3355")
        im = np.zeros((64, 64, 4), np.uint8); im[:] = (c[2] * 255, c[1] * 255, c[0] * 255, 255)
    if el.get("grade"):
        im = np.dstack([apply_grade(np.ascontiguousarray(im[..., :3]), grade_params(el["grade"])), im[..., 3]])
    ih, iw = im.shape[:2]
    x0, y0 = np.floor(quad.min(axis=0)).astype(int) - 2
    x1, y1 = np.ceil(quad.max(axis=0)).astype(int) + 2
    w, h = max(2, x1 - x0), max(2, y1 - y0)
    e = float(el.get("edge", 0.0))
    c = quad.mean(axis=0)
    q = (quad - c) * (1 + e) + c - np.float32([x0, y0])
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [iw, 0], [iw, ih], [0, ih]]), q)
    warped = cv2.warpPerspective(im, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    al = warped[..., 3].astype(np.float32) / 255.0
    from .matte import premultiplied_surface
    surf, _keep = premultiplied_surface(np.ascontiguousarray(warped[..., :3]), al)
    ctx.save()
    ctx.set_source_surface(surf, x0, y0)
    ctx.set_operator({"screen": cairo.OPERATOR_SCREEN, "multiply": cairo.OPERATOR_MULTIPLY,
                      "add": cairo.OPERATOR_ADD}.get(el.get("blend"), cairo.OPERATOR_OVER))
    ctx.paint_with_alpha(clamp(s.alpha * (1 - s.out_p)))
    ctx.restore()
    return (float(x0), float(y0), float(x1), float(y1))


# ---------------------------------------------------------------------------
# audio: clip sound following the time map  (+ optional enhancement)
# ---------------------------------------------------------------------------
VOICE_CHAIN = ("highpass=f=85,lowpass=f=15000,afftdn=nf=-28:tn=1,equalizer=f=220:t=q:w=1.0:g=-2,"
               "equalizer=f=3200:t=q:w=1.2:g=3,equalizer=f=9000:t=q:w=1.5:g=1.5,deesser=i=0.35,"
               "acompressor=threshold=-22dB:ratio=3.2:attack=4:release=90:makeup=4,"
               "alimiter=limit=0.89:level=false")


def enhanced_audio_path(path):
    """v12: broadcast voice chain — rumble cut, FFT denoise, mud cut, presence + air, de-ess, comp, limiter."""
    out = path + ".enh2.wav"
    if not os.path.exists(out):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-vn", "-af", VOICE_CHAIN,
                        "-ac", "1", "-ar", "48000", out], check=True)
    return out


def splice_segments(src, tab, n, sr, xf=0.012):
    """Source audio → edited timeline: each [src0, src1, edit0] copied with raised-cosine edges."""
    out = np.zeros(n, np.float32)
    f = max(1, int(xf * sr))
    for a, b, e0 in tab:
        i0, i1 = int(a * sr), min(len(src), int(b * sr))
        o0 = int(e0 * sr)
        seg = src[max(0, i0 - f):i1 + f].astype(np.float32).copy()
        o0 -= (i0 - max(0, i0 - f))
        if len(seg) < 2 * f + 2:
            continue
        w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, 2 * f))
        seg[:2 * f] *= w; seg[-2 * f:] *= w[::-1]
        lo = max(0, o0); hi = min(n, o0 + len(seg))
        if hi > lo:
            out[lo:hi] += seg[lo - o0:hi - o0]
    return out


def footage_audio(scene, n, sr):
    """Sum of every footage element with "audio": true, time-remapped, as a mono float32 track."""
    from .audio import read_audio
    out = np.zeros(n, np.float32)
    for ch in scene.chapters:
        for el in ch["elements"]:
            if el.get("type") != "footage" or not el.get("audio"):
                continue
            p = footage_audio_src(el)
            if el.get("audio_enhance"):
                try:
                    p = enhanced_audio_path(p)
                except Exception:  # noqa: BLE001
                    pass
            try:
                src = read_audio(p)
            except Exception:  # noqa: BLE001
                continue
            if el.get("segments"):                 # v12: EDL splice with equal-power micro crossfades
                out += splice_segments(src, el["segments"], n, sr, float(el.get("xfade_ms", 30)) / 1000) * \
                    float(el.get("audio_gain", 1.0))
                continue
            t0 = float(el["t"])
            t1 = float(el["out"]) + float(el.get("out_dur", 0.3)) if el.get("out") is not None else scene.duration
            t1 = min(t1, ch["end"] if ch.get("end") else scene.duration, scene.duration)
            i0, i1 = int(t0 * sr), min(n, int(t1 * sr))
            if i1 <= i0:
                continue
            ts = np.arange(i0, i1) / sr
            # source time per output sample (evaluate the map on a 1 kHz grid, interpolate)
            grid = np.arange(t0, t1 + 0.002, 0.001)
            sg = np.array([src_time(el, scene, g) for g in grid]) if (el.get("ramps") or el.get("remap") or
                                                                       el.get("beat_ramp") or el.get("freeze")) \
                else (grid - t0) * float(el.get("speed", 1.0)) + float(el.get("clip_start", 0.0))
            st = np.interp(ts, grid, sg)
            idx = st * sr
            ok = (idx >= 0) & (idx < len(src) - 1)
            seg = np.zeros(len(ts), np.float32)
            ii = idx[ok].astype(np.int64)
            fr = (idx[ok] - ii).astype(np.float32)
            seg[ok] = src[ii] * (1 - fr) + src[ii + 1] * fr
            fade = min(len(seg), int(0.02 * sr))
            if fade:
                seg[:fade] *= np.linspace(0, 1, fade); seg[-fade:] *= np.linspace(1, 0, fade)
            out[i0:i1] += seg * float(el.get("audio_gain", 1.0))
    return out
