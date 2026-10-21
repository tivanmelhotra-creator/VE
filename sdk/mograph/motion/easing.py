"""
mograph.motion.easing — ONE resolver for every way a user / agent can say "ease".

    from mograph.motion.easing import ease
    f = ease("out_back")                      # Penner name (also "easeOutBack", "ease-out-back", "back.out")
    f = ease("cubic-bezier(.2,.8,.2,1)")      # CSS / After Effects graph editor (x1,y1,x2,y2), overshoot y allowed
    f = ease([0.68, -0.35, 0.27, 1.35])       # same as a list
    f = ease({"type": "elastic", "dir": "out", "amp": 1.2, "period": 0.3})
    f = ease({"type": "back", "dir": "in_out", "overshoot": 2.2})
    f = ease({"type": "bounce", "dir": "out", "bounces": 4, "decay": 0.5})
    f = ease({"type": "spring", "stiffness": 260, "damping": 11, "dur": 1.0})   # physical (normalised to dur)
    f = ease({"type": "ae", "out_speed": 0, "out_influence": 0.75, "in_speed": 0, "in_influence": 0.9})
    f = ease({"type": "steps", "n": 6})       # stepped / limited animation (on twos for a 12-frame move)
    f = ease("ease_out")  ==  ease("out")  ==  ease("ease out")  ==  ease("کاهش سرعت")   (glossary)
    y = f(p)      p ∈ [0, 1] → y (may leave [0, 1] for back / elastic)

Types of the glossary: ease_in, ease_out, ease_in_out, cubic, quart, expo, back, elastic, bounce, linear
(+ sine, quad, quint, circ, spring, steps, ae, bezier) × direction in / out / in_out / out_in.

Measurements (graph editor numbers, so the agent can CHOOSE instead of guessing):
    measure(f) → {"overshoot": %, "undershoot": %, "peak_velocity": ×linear, "t_peak_velocity": p,
                  "settle_90": p where y stays within ±10 % of the end, "start_velocity", "end_velocity"}
Graph editor image:  graph_image(["out_back", "cubic-bezier(.2,.8,.2,1)"], "g.png")  — value AND speed graph
(AE "Value Graph" / "Speed Graph"), handles drawn for bezier curves.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

# =============================================================================================== primitives


def _c(p):
    return 0.0 if p <= 0 else 1.0 if p >= 1 else p


def _poly(n):
    return lambda p: _c(p) ** n


def _sine(p):
    return 1 - math.cos(_c(p) * math.pi / 2)


def _expo(p):
    p = _c(p)
    return 0.0 if p <= 0 else 2 ** (10 * (p - 1)) if p < 1 else 1.0


def _circ(p):
    p = _c(p)
    return 1 - math.sqrt(max(0.0, 1 - p * p))


def _back(s=1.70158):
    return lambda p: (s + 1) * _c(p) ** 3 - s * _c(p) ** 2


def _elastic_in(amp=1.0, period=0.3):
    a = max(1.0, amp)
    s = period / (2 * math.pi) * math.asin(1 / a)

    def f(p):
        p = _c(p)
        if p in (0.0, 1.0):
            return p
        q = p - 1
        return -(a * 2 ** (10 * q) * math.sin((q - s) * (2 * math.pi) / period))
    return f


def _bounce_out(bounces=4, decay=0.25):
    """generalised bounce-out: `bounces` hits, each bounce height × decay (default 4 hits × 0.25 = Penner bounce exactly)."""
    bounces = max(1, int(bounces))
    r = math.sqrt(max(1e-4, min(0.95, decay)))          # time ratio between successive arcs (height ∝ time²)
    durs = [1.0] + [2 * r ** i for i in range(1, bounces)]
    tot = sum(durs)
    durs = [d / tot for d in durs]

    def f(p):
        p = _c(p)
        t0 = 0.0
        for i, d in enumerate(durs):
            if p <= t0 + d or i == len(durs) - 1:
                if i == 0:                              # first fall: 0 → 1 on a parabola
                    u = (p - t0) / d
                    return u * u
                u = (p - t0) / d                        # arc i: 1 → 1-h → 1
                h = r ** (2 * i)
                return 1 - h * (1 - (2 * u - 1) ** 2)
            t0 += d
        return 1.0
    return f


def _directional(fin, d):
    """direction wrapper from an IN curve."""
    if d == "in":
        return fin
    if d == "out":
        return lambda p: 1 - fin(1 - _c(p))
    if d == "in_out":
        return lambda p: fin(2 * _c(p)) / 2 if p < 0.5 else 1 - fin(2 - 2 * _c(p)) / 2
    if d == "out_in":
        fo = _directional(fin, "out")
        return lambda p: fo(2 * _c(p)) / 2 if p < 0.5 else 0.5 + fin(2 * _c(p) - 1) / 2
    raise ValueError(d)


def cubic_bezier(x1, y1, x2, y2):
    """CSS / AE graph-editor cubic-bezier: Newton + bisection solve of x(u) = p (x1, x2 clamped to [0, 1])."""
    x1, x2 = min(1.0, max(0.0, float(x1))), min(1.0, max(0.0, float(x2)))
    y1, y2 = float(y1), float(y2)
    cx, bx = 3 * x1, 3 * (x2 - x1) - 3 * x1
    ax = 1 - cx - bx
    cy, by = 3 * y1, 3 * (y2 - y1) - 3 * y1
    ay = 1 - cy - by

    def sx(u): return ((ax * u + bx) * u + cx) * u
    def dx(u): return (3 * ax * u + 2 * bx) * u + cx
    def sy(u): return ((ay * u + by) * u + cy) * u

    def f(p):
        if p <= 0:
            return 0.0
        if p >= 1:
            return 1.0
        u = p
        for _ in range(8):
            e = sx(u) - p
            if abs(e) < 1e-7:
                return sy(u)
            d = dx(u)
            if abs(d) < 1e-6:
                break
            u -= e / d
        lo, hi, u = 0.0, 1.0, p
        for _ in range(40):
            if sx(u) < p:
                lo = u
            else:
                hi = u
            u = (lo + hi) / 2
        return sy(u)
    f.bezier = (x1, y1, x2, y2)
    return f


def ae_ease(out_speed=0.0, out_influence=1 / 3, in_speed=0.0, in_influence=1 / 3):
    """After Effects keyframe velocity dialog for a normalised 0→1 move over 1 s.
    speed in 'units per second' of the normalised move (1.0 = linear speed), influence 0.1 %–100 %."""
    io = min(1.0, max(1e-3, float(out_influence)))
    ii = min(1.0, max(1e-3, float(in_influence)))
    return cubic_bezier(io, float(out_speed) * io, 1 - ii, 1 - float(in_speed) * ii)


def spring_curve(stiffness=180.0, damping=12.0, mass=1.0, v0=0.0, dur=1.0):
    """physical damped spring 0→1, time normalised so p=1 ↔ dur seconds (ends wherever the spring is)."""
    w0 = math.sqrt(stiffness / mass)
    z = damping / (2 * math.sqrt(stiffness * mass))

    def f(p):
        t = _c(p) * dur
        if t <= 0:
            return 0.0
        if z < 1:
            wd = w0 * math.sqrt(1 - z * z)
            x = math.exp(-z * w0 * t) * (-math.cos(wd * t) + ((v0 - z * w0) / wd) * math.sin(wd * t))
        else:
            x = math.exp(-w0 * t) * (-1 + (v0 - w0) * t)
        return 1 + x
    return f


def steps(n=6, mode="end"):
    """stepped timing (CSS steps()): limited animation / on twos when n = frames / 2."""
    n = max(1, int(n))
    if mode == "start":
        return lambda p: min(1.0, math.ceil(_c(p) * n) / n)
    return lambda p: 1.0 if p >= 1 else math.floor(_c(p) * n) / n


# =============================================================================================== registry
FAMILIES = {
    "linear": None, "quad": _poly(2), "cubic": _poly(3), "quart": _poly(4), "quint": _poly(5),
    "sine": _sine, "expo": _expo, "circ": _circ, "back": _back(), "elastic": _elastic_in(), "bounce": None,
}
# the glossary types → (family, direction) ;  plain "ease_in/out/in_out" = cubic (AE Easy Ease ≈ cubic in/out)
GLOSSARY_TYPES = {
    "ease_in": ("cubic", "in"), "ease_out": ("cubic", "out"), "ease_in_out": ("cubic", "in_out"),
    "cubic": ("cubic", "in_out"), "quart": ("quart", "out"), "expo": ("expo", "out"), "back": ("back", "out"),
    "elastic": ("elastic", "out"), "bounce": ("bounce", "out"), "linear": ("linear", "in"),
}
# named bezier looks (CSS + motion-design houses), every one is a cubic-bezier → editable in a graph editor
NAMED_BEZIER = {
    "css_ease": (0.25, 0.1, 0.25, 1.0), "css_ease_in": (0.42, 0, 1, 1), "css_ease_out": (0, 0, 0.58, 1),
    "css_ease_in_out": (0.42, 0, 0.58, 1),
    "smooth": (0.22, 1.0, 0.36, 1.0), "snappy": (0.16, 1.0, 0.3, 1.0), "anticipate": (0.68, -0.35, 0.27, 1.35),
    "material_standard": (0.2, 0.0, 0.0, 1.0), "material_decelerate": (0.0, 0.0, 0.2, 1.0),
    "material_accelerate": (0.3, 0.0, 1.0, 1.0), "material_emphasized": (0.05, 0.7, 0.1, 1.0),
    "apple_spring": (0.32, 0.72, 0.0, 1.0), "whip": (0.85, 0.0, 0.15, 1.0), "swift_out": (0.55, 0.0, 0.1, 1.0),
    "ae_easy_ease": (1 / 3, 0, 2 / 3, 1), "ae_easy_ease_out": (1 / 3, 0, 2 / 3, 2 / 3),
    "ae_easy_ease_in": (1 / 3, 1 / 3, 2 / 3, 1), "ae_expo_75": (0.75, 0, 0.25, 1), "ae_hard_out": (0.0, 0.0, 0.1, 1.0),
    "punch": (0.7, -0.4, 0.4, 1.4), "soft_land": (0.15, 0.85, 0.35, 1.0),
}
_DIRS = {"in": "in", "out": "out", "inout": "in_out", "in_out": "in_out", "outin": "out_in", "out_in": "out_in"}
_FA = {"کاهش سرعت": "ease_out", "شتاب": "ease_in", "نرم": "ease_in_out", "اینرسی": "ease_out", "خطی": "linear",
       "فنری": "elastic", "الاستیک": "elastic", "بانس": "bounce", "اورشوت": "back", "پرشی": "bounce"}


def _norm_name(s):
    s = s.strip()
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", s).lower()             # easeOutBack → ease_out_back
    s = re.sub(r"[\s\-\.]+", "_", s)
    s = s.replace("easeinout", "ease_in_out").replace("inout", "in_out")
    return s


@lru_cache(maxsize=512)
def _from_name(name):
    raw = name.strip()
    if raw in _FA:
        raw = _FA[raw]
    m = re.match(r"^\s*cubic[-_ ]?bezier\s*\(\s*([^)]*)\)\s*$", raw, re.I)
    if m:
        return cubic_bezier(*[float(v) for v in re.split(r"[,\s]+", m.group(1).strip()) if v])
    m = re.match(r"^\s*steps\s*\(\s*(\d+)\s*(?:,\s*(start|end|jump-start|jump-end))?\s*\)\s*$", raw, re.I)
    if m:
        return steps(int(m.group(1)), "start" if (m.group(2) or "").endswith("start") else "end")
    s = _norm_name(raw)
    if s in NAMED_BEZIER:
        return cubic_bezier(*NAMED_BEZIER[s])
    if s in ("linear", "none"):
        return _c
    if s in GLOSSARY_TYPES:
        fam, d = GLOSSARY_TYPES[s]
        return _family(fam, d)
    if s in ("in", "out", "in_out", "out_in"):                       # legacy mograph.core names: cubic
        return _family("cubic", s)
    s2 = s[5:] if s.startswith("ease_") else s
    # forms: out_back | back_out | back.out (GSAP) | in_out_expo | expo_in_out
    for fam in FAMILIES:
        for d in sorted(_DIRS, key=len, reverse=True):
            if s2 in (f"{d}_{fam}", f"{fam}_{d}", f"{d}{fam}", f"{fam}{d}"):
                return _family(fam, _DIRS[d])
        if s2 == fam:
            return _family(fam, GLOSSARY_TYPES.get(fam, (fam, "out"))[1])
    # legacy mograph.core / v14 physics names
    try:
        from ..core import EASINGS
        if s in EASINGS:
            return EASINGS[s]
    except Exception:  # noqa: BLE001
        pass
    raise KeyError(f"unknown easing '{name}'  (python3 -m mograph.motion ease list)")


def _family(fam, d, **kw):
    if fam == "linear":
        return _c
    if fam == "bounce":
        bo = _bounce_out(kw.get("bounces", 4), kw.get("decay", 0.25))
        fin = lambda p: 1 - bo(1 - _c(p))  # noqa: E731
    elif fam == "back":
        fin = _back(float(kw.get("overshoot", 1.70158)) * (1.525 if d == "in_out" and "overshoot" not in kw else 1))
    elif fam == "elastic":
        fin = _elastic_in(float(kw.get("amp", 1.0)), float(kw.get("period", 0.3 if d != "in_out" else 0.45)))
    elif fam in ("quad", "cubic", "quart", "quint") and "power" in kw:
        fin = _poly(float(kw["power"]))
    else:
        fin = FAMILIES[fam]
    return _directional(fin, d)


def ease(spec=None, default="ease_in_out"):
    """any easing spec → callable p → y.   None → default.  Callables pass through."""
    if spec is None:
        spec = default
    if callable(spec):
        return spec
    if isinstance(spec, str):
        return _from_name(spec)
    if isinstance(spec, (list, tuple)) and len(spec) == 4:
        return cubic_bezier(*spec)
    if isinstance(spec, dict):
        ty = _norm_name(str(spec.get("type", spec.get("family", "cubic"))))
        d = _DIRS.get(_norm_name(str(spec.get("dir", spec.get("direction", ""))))) if spec.get(
            "dir", spec.get("direction")) else None
        if ty in ("bezier", "cubic_bezier", "css"):
            return cubic_bezier(*spec.get("p", spec.get("points", (0.25, 0.1, 0.25, 1))))
        if ty in ("ae", "after_effects", "speed_influence", "velocity"):
            return ae_ease(spec.get("out_speed", 0.0), spec.get("out_influence", 1 / 3),
                           spec.get("in_speed", 0.0), spec.get("in_influence", 1 / 3))
        if ty == "spring":
            return spring_curve(float(spec.get("stiffness", 180)), float(spec.get("damping", 12)),
                                float(spec.get("mass", 1)), float(spec.get("v0", 0)), float(spec.get("dur", 1.0)))
        if ty == "steps":
            return steps(int(spec.get("n", 6)), spec.get("mode", "end"))
        if ty in GLOSSARY_TYPES and d is None:
            fam, d = GLOSSARY_TYPES[ty]
        else:
            fam = ty if ty in FAMILIES else GLOSSARY_TYPES.get(ty, ("cubic", "out"))[0]
        kw = {k: spec[k] for k in ("amp", "period", "overshoot", "bounces", "decay", "power") if k in spec}
        f = _family(fam, d or "out", **kw)
        if spec.get("mirror"):                                  # go there and come back (yo-yo)
            g = f
            f = lambda p: g(2 * _c(p)) if p < 0.5 else g(2 - 2 * _c(p))  # noqa: E731
        return f
    raise TypeError(f"easing spec {spec!r}")


def names():
    """every NAME accepted by ease() (for list / suggest)."""
    out = list(GLOSSARY_TYPES) + list(NAMED_BEZIER)
    for fam in FAMILIES:
        if fam == "linear":
            continue
        for d in ("in", "out", "in_out", "out_in"):
            out.append(f"{d}_{fam}")
    return out


# =============================================================================================== measurements
def measure(spec, n=400):
    """numbers of a curve (graph-editor readout).  velocities in × the linear speed."""
    f = ease(spec)
    ps = [i / n for i in range(n + 1)]
    ys = [f(p) for p in ps]
    v = [(ys[i + 1] - ys[i]) * n for i in range(n)]
    vmax = max(range(n), key=lambda i: abs(v[i]))
    settle = 0.0
    for i in range(n, -1, -1):
        if abs(ys[i] - 1.0) > 0.1:
            settle = ps[min(n, i + 1)]
            break
    return {"overshoot": round(max(0.0, max(ys) - 1.0) * 100, 1), "undershoot": round(max(0.0, -min(ys)) * 100, 1),
            "peak_velocity": round(abs(v[vmax]), 2), "t_peak_velocity": round(ps[vmax], 3),
            "start_velocity": round(v[0], 2), "end_velocity": round(v[-1], 2), "settle_90": round(settle, 3),
            "y_mid": round(f(0.5), 3)}


def character(spec):
    """one-line human description from the measurements (agent-facing)."""
    m = measure(spec)
    bits = []
    if m["start_velocity"] < 0.3 and m["end_velocity"] < 0.3:
        bits.append("slow-in & slow-out")
    elif m["start_velocity"] < 0.3:
        bits.append("starts slow (ease in), hits at full speed")
    elif m["end_velocity"] < 0.3:
        bits.append("fast start, soft landing (ease out)")
    else:
        bits.append("constant-ish speed")
    if m["undershoot"] > 2:
        bits.append(f"anticipation {m['undershoot']} %")
    if m["overshoot"] > 2:
        bits.append(f"overshoot {m['overshoot']} %")
    bits.append(f"peak speed ×{m['peak_velocity']} at {int(m['t_peak_velocity'] * 100)} %")
    return ", ".join(bits)


# =============================================================================================== graph editor
def graph_image(specs, out=None, size=360, speed=True, labels=None):
    """AE-style graph editor: per curve a VALUE graph (with bezier handles) and a SPEED graph, side by side.
    Returns the BGR image (np.ndarray); writes `out` when given."""
    import cv2
    import numpy as np
    if not isinstance(specs, (list, tuple)) or (len(specs) == 4 and all(isinstance(v, (int, float)) for v in specs)):
        specs = [specs]
    rows = []
    for k, sp in enumerate(specs):
        f = ease(sp)
        lab = (labels[k] if labels else (sp if isinstance(sp, str) else str(sp)))[:46]
        S, pad = size, 34
        cell = np.full((S, S, 3), 24, np.uint8)
        top, bot = 1.45, -0.45
        mapy = lambda y: int(pad + (top - y) / (top - bot) * (S - 2 * pad))  # noqa: E731
        mapx = lambda x: int(pad + x * (S - 2 * pad))  # noqa: E731
        for gy in (0.0, 0.5, 1.0):
            cv2.line(cell, (pad, mapy(gy)), (S - pad, mapy(gy)), (60, 60, 60) if gy != 0.5 else (42, 42, 42), 1)
        for gx in (0.0, 0.25, 0.5, 0.75, 1.0):
            cv2.line(cell, (mapx(gx), pad), (mapx(gx), S - pad), (40, 40, 40), 1)
        pts = np.array([[mapx(i / 200), mapy(f(i / 200))] for i in range(201)], np.int32)
        bz = getattr(f, "bezier", None)
        if bz:
            x1, y1, x2, y2 = bz
            for (ax, ay), (bx, by) in (((0, 0), (x1, y1)), ((1, 1), (x2, y2))):
                cv2.line(cell, (mapx(ax), mapy(ay)), (mapx(bx), mapy(by)), (200, 160, 60), 1, cv2.LINE_AA)
                cv2.circle(cell, (mapx(bx), mapy(by)), 5, (230, 190, 80), -1, cv2.LINE_AA)
        cv2.polylines(cell, [pts], False, (90, 230, 255), 2, cv2.LINE_AA)
        m = measure(sp)
        cv2.putText(cell, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(cell, f"value  ov {m['overshoot']}%  un {m['undershoot']}%", (6, S - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
        cells = [cell]
        if speed:
            sc = np.full((S, S, 3), 24, np.uint8)
            vs = [(f(min(1, (i + 1) / 200)) - f(i / 200)) * 200 for i in range(200)]
            vm = max(2.0, max(abs(v) for v in vs) * 1.1)
            my = lambda v: int(S / 2 - v / vm * (S / 2 - pad))  # noqa: E731
            cv2.line(sc, (pad, my(0)), (S - pad, my(0)), (70, 70, 70), 1)
            cv2.line(sc, (pad, my(1)), (S - pad, my(1)), (45, 70, 45), 1)
            pts2 = np.array([[mapx(i / 200), my(v)] for i, v in enumerate(vs)], np.int32)
            cv2.polylines(sc, [pts2], False, (120, 255, 140), 2, cv2.LINE_AA)
            cv2.putText(sc, f"speed  peak x{m['peak_velocity']} @ {int(m['t_peak_velocity'] * 100)}%", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(sc, "green line = linear speed", (6, S - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (150, 150, 150), 1, cv2.LINE_AA)
            cells.append(sc)
        rows.append(np.hstack(cells))
    cols = 2 if len(rows) > 3 else 1
    while len(rows) % cols:
        rows.append(np.zeros_like(rows[0]))
    img = np.vstack([np.hstack(rows[i:i + cols]) for i in range(0, len(rows), cols)])
    if out:
        cv2.imwrite(out, img)
    return img


def install_into_core():
    """make every legacy `get_ease(name)` (v5–13 elements, v14 layers) understand the new specs too:
    cubic-bezier strings, lists, dicts, Penner camelCase … — unknown names still fall back to in_out."""
    from .. import core
    if getattr(core.get_ease, "_motion", False):
        return
    old = core.get_ease

    def get_ease(e):
        if callable(e) or e is None or (isinstance(e, str) and e in core.EASINGS):
            return old(e)
        try:
            return ease(e)
        except Exception:  # noqa: BLE001
            return old(e if isinstance(e, str) else None)
    get_ease._motion = True
    core.get_ease = get_ease
    try:
        from .. import elements
        elements.get_ease = get_ease
    except Exception:  # noqa: BLE001
        pass
