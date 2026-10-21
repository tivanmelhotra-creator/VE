"""
mograph.three_anim — v20.1 PROCEDURAL 3D ANIMATION for `three3d` models (state changes *inside* the object, not only
turning it):  hourglass that pours and flips, laptop lid on a real hinge, parts that assemble / fly apart.

Why a module (and not "just keyframe yaw"):  the v14 `object_3d` layer can only spin / bob the WHOLE object.  An
hourglass needs its SAND to move: the top pile drains through the neck (crater forms), a stream falls, the bottom
pile grows (mound forms) — and when the top is empty the glass turns over and it starts again.  That is geometry that
changes every frame, so it must be rebuilt from a state, with a LOCKED camera framing (otherwise the bounding box
change makes the object breathe on screen) and with the sand volume conserved (otherwise the sand visibly grows or
shrinks — an instant "fake" tell).

    {"type": "object_3d", "three_model": "hourglass_3d", "anim": "hourglass_loop",
     "pour_dur": 3.0,          # seconds for a full pour (ignored between word-synced flips: fitted to the gap)
     "flip_dur": 0.7, "hold": 0.25, "settle": 0.3,   # flip turn time · pause when empty · sand slides to the neck
     "start_fill": 0.0,        # 0 = starts full on top … 0.9 = almost poured at entrance
     "flips_at": ["w:دوباره", "w:زمان"],   # optional: flip EXACTLY on these words (sand empties just before)
     "flip_axis": "z",         # z = turn in the screen plane (classic) · x = tumble towards the camera
     "flips": 99}              # max number of automatic flips (no flips_at)

    {"type": "object_3d", "three_model": "laptop_3d", "anim": "laptop_open", "open_dur": 0.9, "close_at": "w:…"}
    {"type": "object_3d", "three_model": "star_trophy_3d", "anim": "assemble", "assemble_dur": 1.1}
    {"type": "object_3d", "three_model": "emerald_tree_3d", "explode_at": "w:…", "explode_to": 0.6}   # any model

Every state is a pure function of the layer-local time → deterministic in parallel chunk rendering.
CLI:  python3 -m mograph.three_anim sheet hourglass_loop --out s.png   ·   python3 -m mograph.three_anim selftest
"""
from __future__ import annotations

import math
import os
import sys
import time
from functools import lru_cache

import numpy as np

ANIMS = ("hourglass_loop", "laptop_open", "assemble")
DEFAULTS = {"pour_dur": 3.0, "flip_dur": 0.7, "hold": 0.25, "settle": 0.3, "start_fill": 0.0, "flip_axis": "z",
            "flips": 99, "open_dur": 0.9, "open_delay": 0.15, "assemble_dur": 1.1, "explode_from": 1.0,
            "explode_dur": 0.6, "explode_to": 0.6}


def _ease_io(p):                        # cubic in-out
    p = min(1.0, max(0.0, p))
    return 4 * p * p * p if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


def _ease_out_back(p, s=1.4):
    p = min(1.0, max(0.0, p)) - 1.0
    return 1 + (s + 1) * p ** 3 + s * p ** 2


def _smooth(p):
    p = min(1.0, max(0.0, p))
    return p * p * (3 - 2 * p)


# ============================================================================================ hourglass physics
_Y = None
_CUM = None


def _table():
    """cumulative sand-volume table V(y) of the glass interior (π r² dy), shared by both bulbs (symmetric)."""
    global _Y, _CUM
    if _Y is None:
        from .three3d import hg_inner_r
        _Y = np.linspace(-0.82, 0.82, 3281)
        a = math.pi * np.array([hg_inner_r(float(y)) for y in _Y]) ** 2
        _CUM = np.concatenate([[0.0], np.cumsum((a[1:] + a[:-1]) * 0.5 * np.diff(_Y))])
    return _Y, _CUM


def _vol(lo, hi):
    Y, C = _table()
    return float(np.interp(hi, Y, C) - np.interp(lo, Y, C))


def _cone(y, k):
    from .three3d import hg_inner_r
    return math.pi * hg_inner_r(y) ** 2 * k / 3.0          # volume of the surface cone (+ mound / − crater)


def _solve_hi(lo, target, k_hi_fn, hi_max):
    """hi such that slab(lo, hi) + cone(hi, k_hi(hi)) == target (bisection, monotonic)."""
    a, b = lo, hi_max
    for _ in range(48):
        m = 0.5 * (a + b)
        if _vol(lo, m) + _cone(m, k_hi_fn(m)) < target:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def _solve_lo(hi, target, k_lo, lo_min):
    a, b = lo_min, hi
    for _ in range(48):
        m = 0.5 * (a + b)
        if _vol(m, hi) - _cone(m, k_lo) > target:     # k_lo < 0 (cone pointing down) ADDS volume below lo
            a = m
        else:
            b = m
    return 0.5 * (a + b)


NECK = 0.03                 # top pile rests on the funnel just above the neck
SAND_FRAC = 0.40            # sand = 40 % of one bulb (looks like a real 1-minute glass)
MOUND = 0.085               # height of the bottom cone at the end of a pour
CRATER = 0.11               # depth of the top funnel crater mid-pour


@lru_cache(maxsize=1)
def sand_total():
    from .three3d import HG_END
    return SAND_FRAC * _vol(NECK, HG_END)


def _top_slab(vol, p):
    """top bulb, pile resting on the neck funnel; crater deepens as it drains."""
    from .three3d import HG_END
    if vol < 1e-5:
        return None
    lo = NECK
    crater = -CRATER * _smooth(p / 0.25) * (1 - _smooth((p - 0.85) / 0.15) * 0.6)
    k_hi = lambda h: max(crater, -(h - lo) * 0.85)                       # noqa: E731
    hi = _solve_hi(lo, vol, k_hi, HG_END)
    return (lo, hi, -0.025, k_hi(hi))


def _bot_slab(vol, p):
    """bottom bulb, pile on the cap, cone mound grows with the pour."""
    from .three3d import HG_END
    if vol < 1e-5:
        return None
    lo = -HG_END
    mound = MOUND * math.sqrt(max(0.0, min(1.0, p)))
    k_hi = lambda h: min(mound, max(0.0, -0.05 - h))                     # noqa: E731
    hi = _solve_hi(lo, vol, k_hi, -NECK)
    return (lo, hi, 0.0, k_hi(hi))


def _cap_slab(vol, mound):
    """mirror of a full bottom pile, now in the TOP bulb against the cap (right after a flip)."""
    from .three3d import HG_END
    hi = HG_END
    lo = _solve_lo(hi, vol, -mound, NECK)
    return (lo, hi, -mound, 0.0)


def hourglass_sand(p, settle=1.0, stream_front=1.0, stream_tail=0.0):
    """sand state for pour fraction p (0 = all on top … 1 = all at the bottom).
    settle 0→1 = right after a flip: the pile slides from the cap down to the neck (volume kept exactly).
    stream_front 0→1 = the falling stream reaches the bottom pile; stream_tail 0→1 = the last grains fall."""
    V = sand_total()
    p = min(1.0, max(0.0, p))
    vt, vb = V * (1 - p), V * p
    if settle < 1.0 and vt > 1e-5:
        a = _cap_slab(vt, MOUND)
        b = _top_slab(vt, 0.0)
        s = _smooth(settle)
        lo = a[0] + (b[0] - a[0]) * s
        k_lo = a[2] + (b[2] - a[2]) * s
        from .three3d import HG_END
        hi = _solve_hi(lo, vt + _cone(lo, k_lo), lambda h: 0.0, HG_END)
        top = (lo, hi, k_lo, 0.0)
    else:
        top = _top_slab(vt, p)
    bot = _bot_slab(vb, p)
    stream = None
    if 0.0 < p < 1.0 or stream_tail < 1.0 and p >= 1.0 and stream_tail > 0.0:
        y_pile = (bot[1] + bot[3]) if bot is not None else -0.80
        y0 = 0.0 + (y_pile - 0.0) * _smooth(stream_front)           # falling front
        y1 = 0.0 + (y_pile - 0.0) * _smooth(stream_tail)            # tail after the top empties
        if y1 - y0 > 0.01:
            stream = (y0, y1)
    r = lambda x: None if x is None else tuple(round(float(v), 4) for v in x)   # noqa: E731
    return (r(top), r(bot), r(stream))


def _flip_times(cfg, t_end):
    """local flip start times: word-synced (cfg['_flips']) or an automatic cycle."""
    if cfg.get("_flips") is not None:
        return sorted(float(f) for f in cfg["_flips"] if f >= 0)
    pour, hold, flip, settle = (float(cfg[k]) for k in ("pour_dur", "hold", "flip_dur", "settle"))
    out, t = [], (1 - float(cfg["start_fill"])) * pour + hold
    while len(out) < int(cfg["flips"]) and t < t_end:
        out.append(t)
        t += flip + settle + pour + hold
    return out


def hourglass_state(cfg, lt, t_end=60.0):
    """→ (sand, root_rot (rx, ry, rz) rad, lift 0..1)  at layer-local time lt."""
    flips = _flip_times(cfg, t_end)
    pour, hold, fd, st = (float(cfg[k]) for k in ("pour_dur", "hold", "flip_dur", "settle"))
    axis = 0 if str(cfg.get("flip_axis", "z")) == "x" else 2
    i = sum(1 for f in flips if f <= lt)
    rot = [0.0, 0.0, 0.0]
    if i > 0 and lt < flips[i - 1] + fd:                                # ---- turning over
        q = (lt - flips[i - 1]) / fd
        rot[axis] = math.pi * _ease_io(q)
        return hourglass_sand(1.0, stream_tail=1.0), tuple(rot), math.sin(math.pi * q)
    if i == 0:                                                          # ---- first pour (may start mid-way)
        p0 = float(cfg["start_fill"])
        s0, sett = 0.0, 1.0
        end = (flips[0] - hold) if flips else None
    else:
        s0 = flips[i - 1] + fd
        p0, sett = 0.0, min(1.0, (lt - s0) / max(1e-3, st))
        s0 += st
        end = (flips[i] - hold) if i < len(flips) else None
    dur = (end - s0) / max(1e-3, 1.0 - p0) if end is not None else pour
    dur = max(0.3, dur)
    if lt < s0:                                                         # settling after the flip
        return hourglass_sand(0.0, settle=sett, stream_front=0.0), tuple(rot), 0.0
    tau = lt - s0
    p = p0 + tau / dur
    fall = 0.12                                                         # seconds for grains to reach the pile
    front = 1.0 if p0 > 0 else min(1.0, tau / fall)
    if p >= 1.0:
        t_empty = s0 + (1.0 - p0) * dur
        tail = min(1.0, (lt - t_empty) / fall)
        return hourglass_sand(1.0, stream_front=1.0, stream_tail=tail), tuple(rot), 0.0
    return hourglass_sand(p, settle=sett, stream_front=front), tuple(rot), 0.0


# ============================================================================================ generic state
def config(L):
    """normalise a layer / element dict → cfg (anim name + parameters) or None (static model)."""
    a = L.get("anim")
    if isinstance(a, dict):
        cfg = dict(DEFAULTS, **{k: v for k, v in a.items() if k != "type"})
        name = a.get("type")
    else:
        cfg = dict(DEFAULTS)
        name = a
        for k in DEFAULTS:
            if k in L:
                cfg[k] = L[k]
    for k in ("_flips", "_close", "_explode_at"):
        cfg[k] = None
    if not name and L.get("_explode_at") is None and L.get("explode_at") is None:
        return None
    cfg["name"] = name
    return cfg


def state(cfg, model, lt, t_end=60.0):
    """→ (builder kwargs tuple, root rotation, explode override | None, lift)."""
    name = cfg.get("name")
    ex = None
    if cfg.get("_explode_at") is not None:
        q = (lt - cfg["_explode_at"]) / max(1e-3, float(cfg["explode_dur"]))
        ex = float(cfg["explode_to"]) * _ease_out_back(q, 1.2) if q > 0 else 0.0
    if name == "hourglass_loop":
        sand, rot, lift = hourglass_state(cfg, lt, t_end)
        return (("sand", sand),), rot, ex, lift
    if name == "laptop_open":
        q = (lt - float(cfg["open_delay"])) / max(1e-3, float(cfg["open_dur"]))
        ang = 102.6 * _ease_out_back(q, 1.25) if q > 0 else 0.0
        if cfg.get("_close") is not None and lt > cfg["_close"]:
            ang *= 1.0 - _ease_io((lt - cfg["_close"]) / max(1e-3, float(cfg["open_dur"]) * 0.8))
        return (("lid_open", round(ang, 1)),), (0.0, 0.0, 0.0), ex, 0.0
    if name == "assemble":
        q = lt / max(1e-3, float(cfg["assemble_dur"]))
        e = float(cfg["explode_from"]) * (1.0 - _ease_out_back(q, 1.1)) if q < 1 else 0.0
        return (), (0.0, 0.0, 0.0), (e if ex is None else ex + e), 0.0
    return (), (0.0, 0.0, 0.0), ex, 0.0


_BUILDERS = {"hourglass_3d": "build_hourglass_3d", "laptop_3d": "build_laptop_3d"}


@lru_cache(maxsize=16)
def _frame(model):
    """locked framing = the model's REST bounding sphere (identical to the static three3d render)."""
    from . import three3d as T
    sc = T.get_compiled_scene(model, 0)
    return tuple(float(v) for v in sc.center), float(sc.radius)


@lru_cache(maxsize=48)
def scene(model, kwargs, rot, explode_q):
    from . import three3d as T
    key = T.resolve_model_name(model) or model
    fn_name = _BUILDERS.get(key)
    fn = getattr(T, fn_name) if fn_name else T.MODELS[key]
    root = T.THREE.Group()
    inner = fn(**dict(kwargs)) if kwargs else fn()
    root.add(inner)
    root.rotation.set(*rot)
    c, r = _frame(key)
    return T.CompiledThreeScene(root, explode=explode_q / 100.0, frame=(c, r))


@lru_cache(maxsize=96)
def _render(model, kwargs, rot, explode_q, q_sz, yaw, pitch, roll, ssaa):
    from . import three3d as T
    from .real3d import to_bgra_premul
    rgba, D = T.render_three(scene(model, kwargs, rot, explode_q), q_sz, yaw, pitch, roll, ssaa=ssaa)
    return np.clip(to_bgra_premul(rgba) + 0.5, 0, 255).astype(np.uint8), D


def render_state(model, cfg, lt, size, yaw=0.0, pitch=15.0, roll=0.0, explode=0.0, ssaa=2, t_end=60.0):
    """→ (premultiplied uint8 BGRA D×D, D, lift)"""
    kwargs, rot, ex, lift = state(cfg, model, lt, t_end)
    e = explode if ex is None else ex
    rot = tuple(round(v, 3) for v in rot)
    pm, D = _render(str(model), kwargs, rot, int(round(e * 100)), max(24, int(round(size / 8.0) * 8)),
                    round(yaw * 2) / 2, round(pitch * 2) / 2, round(roll * 2) / 2, int(ssaa))
    return pm, D, lift


def paint(ctx, model, cfg, lt, cx, cy, size, yaw=0.0, pitch=15.0, roll=0.0, explode=0.0, alpha=1.0, shadow=True,
          ssaa=2, t_end=60.0):
    import cv2
    from .obj3d import contact_shadow, paint_premul
    pm, D, lift = render_state(model, cfg, lt, size, yaw, pitch, roll, explode, ssaa, t_end)
    q_sz = max(24, int(round(size / 8.0) * 8))
    if abs(size - q_sz) > 1.0:
        D2 = max(16, int(round(D * size / q_sz)))
        pm = cv2.resize(pm, (D2, D2), interpolation=cv2.INTER_LINEAR)
        D = D2
    hop = size * 0.10 * lift                         # the glass is lifted a little while it turns over
    if shadow:
        contact_shadow(ctx, cx, cy + size * 0.46, size * 0.82 * (1 - 0.25 * lift), alpha * (1 - 0.45 * lift),
                       strength=0.52)
    paint_premul(ctx, pm, cx - D * 0.5, cy - hop - D * 0.5, alpha)
    return (cx - size * 0.5, cy - hop - size * 0.5, cx + size * 0.5, cy - hop + size * 0.5)


# ============================================================================================ audit sheet
def sheet(name="hourglass_loop", out_path=None, cell=240, n=10, **kw):
    """10 frames across one full cycle (pour → empty → flip → settle → pour) to LOOK at the animation."""
    import cv2
    model = {"hourglass_loop": "hourglass_3d", "laptop_open": "laptop_3d", "assemble": "star_trophy_3d"}[name]
    cfg = config(dict({"anim": name}, **kw))
    if name == "hourglass_loop":
        T = float(cfg["pour_dur"]) + float(cfg["hold"]) + float(cfg["flip_dur"]) + float(cfg["settle"]) + 0.8
        ts = [0.0, 0.05 * T, 0.3 * T, 0.55 * T] + list(np.linspace(cfg["pour_dur"] + cfg["hold"] + 0.05,
                                                                   cfg["pour_dur"] + cfg["hold"] + cfg["flip_dur"], 4)) + \
            [cfg["pour_dur"] + cfg["hold"] + cfg["flip_dur"] + 0.15, T]
    else:
        ts = list(np.linspace(0, 1.6, n))
    rows, cols = 2, 5
    out = np.full((rows * cell, cols * cell, 3), 18, np.uint8)
    for i, t in enumerate(ts[:n]):
        pm, D, lift = render_state(model, cfg, float(t), int(cell * 0.62), yaw=20, pitch=12, ssaa=2)
        tile = np.full((cell, cell, 3), 18, np.float32)
        y0 = (cell - D) // 2 - int(cell * 0.62 * 0.10 * lift)
        x0 = (cell - D) // 2
        ys, xs = slice(max(0, y0), min(cell, y0 + D)), slice(max(0, x0), min(cell, x0 + D))
        sub = pm[max(0, -y0):max(0, -y0) + (ys.stop - ys.start), max(0, -x0):max(0, -x0) + (xs.stop - xs.start)]
        a = sub[:, :, 3:4].astype(np.float32) / 255.0
        tile[ys, xs] = sub[:, :, :3].astype(np.float32) + tile[ys, xs] * (1 - a)
        tile = np.clip(tile, 0, 255).astype(np.uint8)
        cv2.putText(tile, f"{name} t={t:.2f}s", (8, cell - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 230, 245), 1,
                    cv2.LINE_AA)
        r, c = divmod(i, cols)
        out[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = tile
    out_path = out_path or f"{name}_anim_sheet.png"
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, out)
    return out_path


# ============================================================================================ selftests
def _t_volume():
    """sand volume (top + bottom + cones) is conserved through the pour and the post-flip settle."""
    V = sand_total()
    errs = []
    for p in np.linspace(0, 1, 21):
        for s in ((0.0, 0.5, 1.0) if p == 0 else (1.0,)):
            top, bot, _ = hourglass_sand(float(p), settle=s)
            v = 0.0
            if top is not None:
                v += _vol(top[0], top[1]) + _cone(top[1], top[3]) - _cone(top[0], top[2])
            if bot is not None:
                v += _vol(bot[0], bot[1]) + _cone(bot[1], bot[3]) - _cone(bot[0], bot[2])
            errs.append(abs(v - V) / V)
    e = max(errs)
    return e < 0.03, f"sand volume conserved within {e * 100:.2f} % over pour + settle"


def _t_inside():
    """every sand vertex stays inside the glass (the glass z-test can never be beaten by sand)."""
    from . import three3d as T
    worst = 0.0
    for p in np.linspace(0, 1, 11):
        top, bot, _ = hourglass_sand(float(p))
        for slab in (top, bot):
            if slab is None:
                continue
            for r, y in T._hg_sand_profile(*slab):
                worst = max(worst, r - T.hg_glass_r(y) * 0.97)
    return worst <= 1e-6, f"sand inside glass (max overshoot {worst:+.4f})"


def _t_flip_seam():
    """the frame at the END of a flip and the frame at the START of the settle are the same picture."""
    cfg = config({"anim": "hourglass_loop", "pour_dur": 1.0, "hold": 0.2, "flip_dur": 0.5, "settle": 0.3})
    f0 = 1.0 + 0.2
    a, _, _ = render_state("hourglass_3d", cfg, f0 + 0.5 - 1e-4, 140, yaw=20, pitch=12, ssaa=1)
    b, _, _ = render_state("hourglass_3d", cfg, f0 + 0.5 + 1e-4, 140, yaw=20, pitch=12, ssaa=1)
    d = float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())
    return d < 1.5, f"flip seam Δ = {d:.2f} / 255 (end of turn == start of next pour)"


def _t_word_sync():
    """with flips_at the top bulb is empty exactly `hold` s before each flip, and full again after it."""
    cfg = config({"anim": "hourglass_loop", "hold": 0.25})
    cfg["_flips"] = [2.0, 4.5]
    pre, _, _ = hourglass_state(cfg, 2.0 - 0.25 + 0.02)
    mid, _, _ = hourglass_state(cfg, 1.0)
    post, _, _ = hourglass_state(cfg, 2.0 + 0.7 + 0.3 + 0.01)
    ok = pre[0] is None and mid[0] is not None and mid[1] is not None and post[1] is None and post[0] is not None
    return ok, "word-synced flips: empty before the word, full on top after the turn"


def _t_frame_lock():
    """locked framing: the static frame and the animated hourglass have the same scale (no breathing)."""
    from . import three3d as T
    cfg = config({"anim": "hourglass_loop"})
    ws = []
    for t in (0.0, 1.5, 2.9):
        pm, D, _ = render_state("hourglass_3d", cfg, t, 140, ssaa=1)
        ws.append(int((pm[:, :, 3] > 25).any(axis=1).sum()))
    st, _ = T.render_three("hourglass_3d", 136, ssaa=1)
    ws.append(int((st[:, :, 3] > 0.1).any(axis=1).sum()))
    return max(ws) - min(ws) <= 2, f"silhouette height {ws} px (locked camera framing)"


def _t_laptop():
    cfg = config({"anim": "laptop_open"})
    c, _, _ = render_state("laptop_3d", cfg, 0.0, 140, pitch=10, ssaa=1)
    o, _, _ = render_state("laptop_3d", cfg, 2.0, 140, pitch=10, ssaa=1)
    hc = int((c[:, :, 3] > 25).any(axis=1).sum())
    ho = int((o[:, :, 3] > 25).any(axis=1).sum())
    return ho > hc * 1.8, f"laptop lid on its hinge: closed {hc} px → open {ho} px tall"


def _t_determinism():
    cfg = config({"anim": "hourglass_loop", "start_fill": 0.3})
    a, _, _ = render_state("hourglass_3d", cfg, 1.234, 120, yaw=33, ssaa=1)
    _render.cache_clear()
    scene.cache_clear()
    b, _, _ = render_state("hourglass_3d", cfg, 1.234, 120, yaw=33, ssaa=1)
    return bool(np.array_equal(a, b)), "pure function of time (identical after cache flush)"


TESTS = [("volume", _t_volume), ("inside", _t_inside), ("flip_seam", _t_flip_seam), ("word_sync", _t_word_sync),
         ("frame_lock", _t_frame_lock), ("laptop", _t_laptop), ("determinism", _t_determinism)]


def selftest():
    t0 = time.time()
    passed = 0
    for name, fn in TESTS:
        t1 = time.time()
        try:
            ok, msg = fn()
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"{type(e).__name__}: {e}"
        passed += bool(ok)
        print(f"  {'✓' if ok else '✗'} {name:<11} {msg}  ({(time.time() - t1) * 1000:.0f} ms)")
    print(f"three_anim selftest: {passed}/{len(TESTS)} pass  ({time.time() - t0:.1f}s)")
    return 0 if passed == len(TESTS) else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        raise SystemExit(selftest())
    if cmd == "sheet":
        nm = sys.argv[2] if len(sys.argv) > 2 else "hourglass_loop"
        out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
        print(sheet(nm, out))
        raise SystemExit(0)
    print(__doc__)
