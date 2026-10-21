"""
mograph.v14.layers — the v14 motion-design component library (every layer = physics + 3D + light).

Each layer is a JSON dict.  Common keys:
  type      layer type (below)               id      name (for QA)
  t, out    in / out time ("w:word", "we:word+0.1", seconds)
  pass      "back" (behind the speaker) | "front" (default) | "both" (3D layers split themselves by depth)
  anchor    tracked point the layer is glued to: head_top face nose chin neck chest belly shoulder_l/r hand_l/r hands
            or "screen" (fixed).  "offset": [dx, dy] in units of HEAD WIDTH (so it scales with the speaker/zoom)
            "x","y" = screen px when anchor == "screen".   "follow": 0..1 how much it follows (default 1)
  size      base font px (scaled by "head_scale" × head width when anchored, default true)
  text      MARKUP (see glyphs.py):  "آمپول‌های|*لاغری*"
  styles    per-layer style overrides       palette  {"accent": "#FF2E42", ...}
  sfx       override the automatic sound    z       draw order inside its pass

LAYER TYPES
  kinetic_title   multi-font title with per-glyph PHYSICS entrances:
                  enter: "jelly_pop" | "drop_bounce" | "flip_3d" | "swing_3d" | "rise_blur" | "stretch_in" | "type"
                  exit:  "pop_out" | "fall" | "flip_out" | "sink" | "fade"
                  idle:  "float" | "breathe" | "jelly" | "none";   "plane": {"rx","ry","rz"}  "extrude": {...}
                  "hero_hit": time → extra jelly impact + flash on hero words
  orbit_ring      text wrapped on a TRUE 3D cylinder around the head; back half is drawn behind the speaker,
                  front half in front; "spin" deg/s, "tilt", "radius" (head widths), "land": time → unwraps flat
  disc            dark vinyl/CD disc with curved arc texts (top/bottom), centre label, half-rim neon that ROTATES
                  (ease-in-out) at "flip_at" and swaps which half is lit → its light spills on the speaker
  glass_shape     real frosted glass hex / circle / pill / dome / diamond / square with gold stroke, specular bloom,
                  jelly-spring open, and a kinetic text inside.  "shape": "hex"|"circle"|"pill"|"dome"|"diamond"|"rsquare"
  halo            concentric 2-tone neon halo behind the head (pass back) with text on top half
  frame_box       tilted wireframe rounded square whose hero letters BREAK OUT of the border + diamond badge
  strike          animated hand-drawn strike-through / circle-scribble / underline on a target layer word
  counter_badge   "۱ / ۲" step badge with spring flip
  sweep_light     light streak across a layer (shine)
  flash           full-frame light flash (cut accent)
  progress        thin progress line at the top
  ambient_bokeh   v14.8: depth-sandwich out-of-focus octagonal dust (12 behind her, 6 large in front, 1.3x parallax)
                  for a time window — {"type": "ambient_bokeh", "t": 0, "out": 6.0, "back": 12, "front": 6} (bokeh.py)
  object_3d       v14.4: a REAL turning 3D object (any RGBA image → solid extruded/cushion/multiview object, or a
                  procedural sculpt3d model) glued to a tracked point; spring-pop + spin-in entrance (object3d.py)
"""
from __future__ import annotations

import math

import cairo
import numpy as np

from . import glyphs as G
from . import physics as ph
from .compositor import glass

# =========================================================================================== prepare / helpers
DEFAULT_SFX = {"kinetic_title": "whoosh_pop", "orbit_ring": "whoosh", "disc": "whoosh", "glass_shape": "jelly",
               "halo": "boom", "frame_box": "pop", "strike": "scribble", "counter_badge": "tick",
               "flash": "hit", "sweep_light": "shine", "object_3d": "whoosh_pop"}


def prepare(L, R):
    L = dict(L)
    L["_t0"] = R.tl.resolve(L.get("t", 0.0))
    L["_t1"] = R.tl.resolve(L.get("out"), R.duration + 1)
    if L.get("type") == "halo":              # anatomy defaults (above the head, 2 discs, VFX) — see HALO_DEFAULT
        for kk, v in HALO_DEFAULT.items():
            L.setdefault(kk, v)
    for k in ("land", "flip_at", "hero_hit", "pulse_at", "turn_on"):
        if k in L and L[k] is not None:
            v = L[k]
            L["_" + k] = [R.tl.resolve(x) for x in v] if isinstance(v, list) else R.tl.resolve(v)
    if "words_at" in L:                      # per-word reveal times
        L["_words_at"] = [R.tl.resolve(x) for x in L["words_at"]]
    if "lines_at" in L:
        L["_lines_at"] = [R.tl.resolve(x) for x in L["lines_at"]]
    pal = G.PALETTE               # the designed (auto) palette is installed globally by the Renderer
    if L.get("palette"):
        pal = dict(G.PALETTE)
        pal.update(L["palette"])
    L["_pal"] = pal
    L["_theme"] = R.spec.get("meta", {}).get("theme", "doran")
    return L


def in_pass(L, p):
    lp = L.get("pass", "front")
    if L["type"] in ("orbit_ring",) or lp == "both":
        return True
    if L["type"] == "halo" and L.get("peek", True) and L.get("text"):
        return True                       # halo body + text behind her; the "peek" glyphs are re-drawn in front
    return lp == p


def pos(L, F):
    a = L.get("anchor", "screen")
    if a == "screen":
        x, y = float(L.get("x", F.W / 2)), float(L.get("y", F.H / 2))
        return x, y, 1.0
    ax, ay = F.anchor(a)
    hw = F.head_px()
    off = L.get("offset", [0, 0])
    fl = float(L.get("follow", 1.0))
    if fl < 1.0:                              # partial follow: blend with the clip-average anchor
        mx, my = F.R.tracks.A[a].mean(axis=0)
        bx, by = F.uv(mx, my)
        ax, ay = bx + (ax - bx) * fl, by + (ay - by) * fl
    # depth parallax: "depth" < 1 → follows the head less (sits further back); > 1 → more (closer to camera)
    dpt = float(L.get("depth", 1.0 if L.get("pass", "front") == "front" else 0.85))
    if dpt != 1.0:
        mx, my = F.R.tracks.A[a].mean(axis=0)
        bx, by = F.uv(mx, my)
        ax, ay = bx + (ax - bx) * dpt, by + (ay - by) * dpt
    # rotate the offset with the head tilt (layer orbits with the head instead of floating rigidly)
    roll = F.R.tracks.head_roll(F.src_t) if a in ("head_top", "head_c", "face", "nose", "chin") else \
        F.R.tracks.shoulder_roll(F.src_t)
    tilt = float(L.get("tilt_follow", 0.8)) * roll
    L["_roll"] = tilt
    ca, sa = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
    ox, oy = off[0] * hw, off[1] * hw
    x = ax + ox * ca - oy * sa
    y = ay + ox * sa + oy * ca
    if "clamp_x" in L:
        x = min(max(x, L["clamp_x"][0]), L["clamp_x"][1])
    if "clamp_y" in L:
        y = min(max(y, L["clamp_y"][0]), L["clamp_y"][1])
    # LAYOUT GUARD: never over the face/mouth, always inside the safe frame (stage.py)
    r_est = element_radius(L, F, hw)
    if r_est:
        from .stage import guard
        x, y = guard(L, F, x, y, r_est)
    # scale with the speaker, but damped (sqrt) so titles never shrink to unreadable or explode on push-ins
    k = (hw / float(L.get("head_ref", 300.0))) ** 0.5 if L.get("head_scale", True) else 1.0
    return x, y, k


def element_radius(L, F, hw):
    """Approximate bounding radius (px) of a layer — used by the layout guard and the stage light."""
    t = L["type"]
    if t == "disc":
        return float(L.get("radius", 1.0)) * hw
    if t == "glass_shape":
        return 0.5 * max(float(L.get("w", 1.6)), float(L.get("h", 1.6))) * hw
    if t == "frame_box":
        return 0.5 * max(float(L.get("w", 2.0)), float(L.get("h", 1.3))) * hw
    if t == "object_3d":
        return 0.5 * (float(L.get("size", 360)) if L.get("anchor", "screen") == "screen"
                      else float(L.get("size", 1.4)) * hw)
    if t in ("kinetic_title", "counter_badge", "tracked_tag"):
        if t == "counter_badge":
            return float(L.get("r", 44)) * 1.2
        if t == "tracked_tag":
            return 0
        b = blk(L)
        k = (hw / float(L.get("head_ref", 300.0))) ** 0.5
        return 0.5 * max(b.w, b.h * 1.6) * k * float(L.get("scale", 1.0))
    return 0


def blk(L, size_mul=1.0, text_key="text", size_key="size"):
    key = f"_blk_{text_key}_{size_mul:.3f}"
    if key not in L:
        L[key] = G.typeset(L.get(text_key, ""), float(L.get(size_key, 60)) * size_mul, L.get("styles"),
                           float(L.get("leading", 0.10)), L.get("align", "center"), float(L.get("word_gap", 0.28)),
                           float(L.get("tracking", 0.0)), L.get("lines"), lockup=L.get("lockup", True),
                           tuck=float(L.get("tuck", 0.04)), theme=L.get("theme", L.get("_theme", "doran")))
    return L[key]


def draw(ctx, L, F):
    fn = DRAW.get(L["type"])
    if fn is None:
        raise KeyError(f"unknown v14 layer type '{L['type']}'")
    return fn(ctx, L, F)


def _sfx(F, L, kind, t, gain=1.0, pan=0.0):
    pass                                       # sound is scheduled by audio.py from the same timing maths


# =========================================================================================== motion recipes
def _glyph_motion(L, F, b, k, lt, out_lt):
    """Return motion(g) for kinetic entrances / exits / idle.  lt = local time since t0, out_lt = time since out."""
    enter = L.get("enter", "jelly_pop")
    exit_ = L.get("exit", "pop_out")
    idle = L.get("idle", "float")
    stag = float(L.get("stagger", 0.035))
    unit = L.get("unit", "word")
    n = max(1, b.n)
    words_at = L.get("_words_at")
    lines_at = L.get("_lines_at")
    hero_hit = L.get("_hero_hit")
    t = F.t
    pal = L["_pal"]

    def m(g):
        s = G.GS()
        # ---- per-glyph local time (RTL order = reading order)
        if words_at and g.word < len(words_at):
            gt = t - words_at[g.word]
        elif lines_at and g.line < len(lines_at):
            gt = t - lines_at[g.line] - g.word * stag
        elif unit == "glyph":
            gt = lt - g.order * stag
        else:                                   # word unit: whole word moves as one cursive piece
            gt = lt - g.line * stag * 2.5 - g.word * stag * 1.6
        hero = g.sname in ("hero", "callig", "alert", "hero_white")
        if gt <= 0:
            s.visible = False
            return s
        h = g.size
        if enter == "jelly_pop":
            p = ph.sp(gt, "jelly" if hero else "bouncy")
            sc = max(0.0, p)
            s.sx, s.sy = sc, sc
            jx, jy, jsh = ph.jelly(gt, 1.3 if hero else 0.7, 3.4, 4.5)
            s.sx *= jx; s.sy *= jy; s.shear = jsh
            s.ty = (1 - min(1.0, p)) * h * 0.35
            s.alpha = min(1.0, gt / 0.06)
        elif enter == "drop_bounce":
            p = ph.sp(gt, "bouncy")
            s.ty = -(1 - p) * h * 2.2
            v = ph.sp_vel(gt, "bouncy")
            st = 1 + min(0.45, abs(v) * 0.05)
            s.sy, s.sx = st, 1 / st
            if gt > 0.18:
                jx, jy, _ = ph.jelly(gt - 0.18, 0.9, 4, 6)
                s.sx *= jy; s.sy *= jx
            s.alpha = min(1.0, gt / 0.05)
        elif enter == "flip_3d":
            p = ph.sp(gt, "bouncy")
            s.rx = (1 - p) * -100
            s.tz = (1 - min(1, p)) * 300
            s.alpha = min(1.0, gt / 0.08)
        elif enter == "swing_3d":
            p = ph.sp(gt, "jelly")
            s.ry = (1 - p) * 95
            s.tx = (1 - p) * -h * 0.8
            s.alpha = min(1.0, gt / 0.08)
        elif enter == "rise_blur":
            p = ph.EXPO_OUT(min(1, gt / 0.55))
            s.ty = (1 - p) * h * 0.9
            s.sy = 1 + (1 - p) * 0.6
            s.sx = 1 / s.sy
            s.alpha = p
        elif enter == "stretch_in":
            p = ph.sp(gt, "jelly")
            s.sx = 0.15 + 0.85 * p
            s.sy = 1 / max(0.35, s.sx) ** 0.5
            s.alpha = min(1.0, gt / 0.05)
        elif enter == "type":
            s.alpha = 1.0 if gt > 0 else 0.0
            p = ph.sp(gt, "snappy")
            s.sx = s.sy = 0.7 + 0.3 * p
        else:
            s.alpha = min(1.0, gt / 0.25)
        # ---- hero impact (second beat on the key word)
        if hero_hit is not None and hero:
            hl = t - hero_hit
            if hl > 0:
                jx, jy, jsh = ph.jelly(hl, 1.6, 3.0, 4.0)
                s.sx *= jx; s.sy *= jy; s.shear += jsh
                if hl < 0.35:
                    s.glow = float(g.style.get("glow", 0.6)) + 1.4 * (1 - hl / 0.35)
        # ---- idle life
        if idle == "float":
            s.ty += ph.osc(t, 0.35, h * 0.035, g.word * 1.3 + g.line)
            s.rz += ph.osc(t, 0.27, 1.2, g.word + 7)
        elif idle == "breathe":
            b_ = 1 + 0.025 * math.sin(t * 2.4 + g.word)
            s.sx *= b_; s.sy *= b_
        elif idle == "jelly":
            s.shear += 0.04 * math.sin(t * 5.1 + g.word * 1.7)
            s.sy *= 1 + 0.03 * math.sin(t * 4.3 + g.word * 2.3)
        # ---- exit
        if out_lt is not None and out_lt > 0:
            ot = out_lt - ((n - 1 - g.order) * stag * 0.5 if unit == "glyph" else (b.nwords - 1 - g.word) * stag * 1.5)
            if ot > 0:
                if exit_ == "pop_out":
                    p = ph.ease_in(min(1, ot / 0.22))
                    s.sx *= 1 + 0.25 * math.sin(min(1, ot / 0.22) * math.pi) - p
                    s.sy *= 1 - p
                    s.alpha *= 1 - p
                elif exit_ == "fall":
                    s.ty += ot * ot * 3200
                    s.rz += ot * 140 * (1 if g.word % 2 else -1)
                    s.alpha *= max(0, 1 - ot / 0.45)
                elif exit_ == "flip_out":
                    p = ph.ease_in(min(1, ot / 0.3))
                    s.rx += p * 95
                    s.alpha *= 1 - p
                elif exit_ == "sink":
                    p = ph.ease_in(min(1, ot / 0.3))
                    s.ty += p * h
                    s.sy *= 1 - 0.5 * p
                    s.alpha *= 1 - p
                else:
                    s.alpha *= max(0, 1 - ot / 0.25)
        return s
    return m


def _out_lt(L, F):
    return (F.t - L["_t1"]) if F.t > L["_t1"] - 1e-6 else None


# =========================================================================================== kinetic_title
class _ShiftT:
    """Frame proxy at an earlier instant (motion blur: the block one shutter interval ago).  Only `t` differs —
    tracked positions are NOT re-evaluated (the layout guard keeps per-frame state; the speaker moves slowly)."""
    __slots__ = ("t", "_F")

    def __init__(self, F, t):
        self.t, self._F = t, F

    def __getattr__(self, k):
        return getattr(self._F, k)


def _mblur_for(L, F, b, k, lt, olt, plane_fn=None, x=None, y=None):
    """v14.8 motion-blur descriptor for G.draw_block (None when off / not moving)."""
    if F.prepass:
        return None
    from . import motionblur as MB
    cfg = MB.config(L, F)
    if cfg is None:
        return None
    if cfg["auto"]:                          # automatic: only during fast physics entrances / exits
        fast_in = (L.get("enter") in MB.AUTO_ENTERS or L.get("plane_in")) and lt < 1.1 + b.nwords * 0.06
        fast_out = olt is not None and olt > 0 and L.get("exit") in MB.AUTO_EXITS
        if not (fast_in or fast_out):
            return None
    dt = MB.shutter_dt(cfg, F.R.fps)
    Fp = _ShiftT(F, F.t - dt)
    olt_p = (olt - dt) if olt is not None else None
    if olt_p is not None and olt_p <= 0:
        olt_p = None
    return {"motion": _glyph_motion(L, Fp, b, k, lt - dt, olt_p), "plane": plane_fn(Fp.t, lt - dt) if plane_fn else None,
            "x": x, "y": y, "cfg": cfg}


def _title_plane(L, t, lt):
    plane = dict(L.get("plane", {}))
    sway = L.get("sway", 1.0)
    if sway:
        plane["ry"] = plane.get("ry", 0) + ph.osc(t, 0.18, 7 * sway, 3)
        plane["rx"] = plane.get("rx", 0) + ph.osc(t, 0.14, 4 * sway, 9)
    plane["rz"] = plane.get("rz", 0) + L.get("_roll", 0.0)
    pin = L.get("plane_in")              # {"ry": 70} → block swings in from that angle with a jelly spring
    if pin:
        p = ph.sp(lt, L.get("plane_spring", "jelly"))
        for kk, vv in pin.items():
            plane[kk] = plane.get(kk, 0) + vv * (1 - p)
    return plane


def draw_kinetic_title(ctx, L, F):
    x, y, k = pos(L, F)
    b = blk(L)
    lt = F.t - L["_t0"]
    olt = _out_lt(L, F)
    if olt is not None and olt > 0.9:
        return
    sc = k * float(L.get("scale", 1.0))
    # whole-block 3D sway (real perspective, not a 2D rotate) + plane_in swing
    plane = _title_plane(L, F.t, lt)
    field = None
    fd = L.get("field")
    if fd:
        kw = dict(fd)
        kind = kw.pop("kind")
        if kind in ("wave", "jelly"):
            kw["phase"] = F.t * float(kw.pop("speed", 4.0))
        field = G.Field(kind, **kw)
    mb = _mblur_for(L, F, b, k, lt, olt, lambda t_, lt_: _title_plane(L, t_, lt_), x, y)
    G.draw_block(ctx, b, x, y, motion=_glyph_motion(L, F, b, k, lt, olt), field=field, plane=plane, pal=L["_pal"],
                 ectx=F.ectx, extrude=L.get("extrude"), scale=sc, shadow=L.get("shadow", {"alpha": 0.35, "dy": 4}),
                 stroke=L.get("stroke"), mblur=mb)


# =========================================================================================== orbit_ring
def ring_fit(text_w, R, sep, min_scale=0.55, max_span=0.86, max_R=None, fill=True, coverage=0.40):
    """GEOMETRY SOLVER for text on a 3D ring — the agent never has to think about it.
    On a cylinder seen from the front, the front and back halves CONVERGE at the two silhouettes, so text that
    wraps the whole ring always collides there (the "last letter runs into the first" bug).  Rule:
       total ink (all copies) ≤ coverage × circumference   (default 40 % ≈ the readable front arc) → a phrase
       never wraps round to meet itself; ≥ 60 % of the ring stays empty,
       split evenly between copies, so every silhouette crossing happens in a gap.
    Order of fixes when one copy is too long: shrink text (≥ min_scale) → grow the ring (≤ max_R) → shrink more.
    → (n_copies, scale, R)."""
    cov = coverage if fill else min(coverage, max_span)
    C = 2 * math.pi * R
    s = 1.0
    if text_w > cov * C:
        s = max(min_scale, cov * C / max(1.0, text_w))
        if text_w * s > cov * C:
            need = text_w * s / (2 * math.pi * cov)
            R = min(max_R or need, need)
            C = 2 * math.pi * R
            if text_w * s > cov * C:
                s = max(0.35, cov * C / max(1.0, text_w))
    n = max(1, int((cov * C) // max(1.0, text_w * s))) if fill else 1
    return n, s, R


def draw_orbit_ring(ctx, L, F):
    """Text on a TRUE 3D cylinder around the head (back half drawn behind her, front half in front).
    "text" = ONE phrase (no manual repetition!). The ring solver repeats it evenly ("fill": true, default) and
    auto-fits size/radius so the end of the text can NEVER run into its start.
    "max_copies" caps the repetitions (default 3), "sep" = gap between copies in em (default 1.2)."""
    lt = F.t - L["_t0"]
    olt = _out_lt(L, F)
    if olt is not None and olt > 0.6:
        return
    x, y, k = pos(L, F)
    hw = F.head_px()
    b = blk(L)
    R = float(L.get("radius", 0.95)) * hw
    sep = float(L.get("sep", 1.2)) * float(L.get("size", 60)) * k
    max_R = min(F.W * 0.47, float(L.get("max_radius", 1.45)) * hw)
    n, fs, R = ring_fit(b.w * k, R, sep, fill=L.get("fill", True), max_R=max_R)
    n = min(n, int(L.get("max_copies", 3)))
    if L.get("_fitlog") is None:
        L["_fitlog"] = (n, fs, R)
    p_in = ph.sp(lt, "bouncy")
    spin = float(L.get("spin", 55.0))
    # MONOTONIC spin-in (bug fix v14.3): the old  spin·t − (1−spring)·140  REVERSED direction whenever the
    # spring overshot (>1) → "moves, goes back, moves again".  Angle must never change direction: the ring
    # enters with extra angular velocity that decays exponentially to the cruise speed (always same sign).
    # ∫ (spin + boost·e^(−t/τ)) dt  = spin·t + boost·τ·(1 − e^(−t/τ))   → strictly increasing.
    boost, tau = float(L.get("spin_in", 260.0)), 0.32
    lt0 = max(0.0, lt)
    ang = float(L.get("phase", 180.0)) + spin * F.t - boost * tau + boost * tau * (1 - math.exp(-lt0 / tau))
    tilt = float(L.get("tilt", -16.0)) + ph.osc(F.t, 0.2, 3, 2)
    land = L.get("_land")
    # grow-in with ONE soft settle (critically-damped-ish), not a 30 % bounce that reads as a stutter
    sc = k * fs * float(L.get("scale", 1.0)) * (0.6 + 0.4 * min(1.04, ph.sp(lt, "smooth")))
    alpha = min(1.0, lt / 0.15)
    R_eff = R * (0.85 + 0.15 * ph.EXPO_OUT(min(1.0, lt0 / 0.5)))
    if olt is not None and olt > 0:
        alpha *= max(0, 1 - olt / 0.4)
        R_eff *= 1 + olt * 0.6
    if land is not None and F.t > land:
        q = ph.SILK(min(1, (F.t - land) / 0.7))
        R_eff = R_eff * (1 + 9 * q * q)                    # unwrap: radius → ∞ = flat
        ang = ang * (1 - q)
        tilt *= 1 - q
        n = 1
        y = y + (float(L.get("land_dy", 2.2)) * hw) * q
        if q >= 1 and F.pass_ == "back":
            return
    want = "back" if F.pass_ == "back" else "front"
    jel = L.get("jelly", 1.0)

    cur = {"ang": ang}

    def m(g):
        s = G.GS()
        gl = lt - g.word * 0.05
        if gl <= 0:
            s.visible = False
            return s
        jx, jy, jsh = ph.jelly(gl, 0.35 * jel, 2.6, 4.5)      # gentle, fast-decaying jelly (no wobble-back)
        s.sx, s.sy, s.shear = jx, jy, jsh
        s.ty = ph.osc(F.t + g.word * 0.7, 0.6, g.size * 0.03, 1)        # ribbon ripple travelling along the ring
        # FACING FADE: letters near the cylinder silhouette are foreshortened and pile onto each other —
        # a pro ring fades them by how much they face the camera (|cos|), so only readable letters show.
        if R_eff < 5000:
            a_ = (g.cx * sc) / R_eff + math.radians(cur["ang"])
            f_ = abs(math.cos(a_))
            s.alpha = min(1.0, max(0.0, (f_ - 0.30) / 0.35)) ** 1.6
        return s
    for i in range(n):
        cur["ang"] = ang + i * 360.0 / n
        field = G.Field("cylinder", radius=R_eff, angle=cur["ang"], tilt=tilt)
        G.draw_block(ctx, b, x, y, motion=m, field=field, f=1400.0, pal=L["_pal"], ectx=F.ectx, depth=want,
                     extrude=L.get("extrude", {"depth": 6, "light": 0.5}), scale=sc,
                     alpha=alpha * (0.55 if want == "back" else 1.0))


# =========================================================================================== band text
def band_fit(blk_w, blk_h, r_in, r_out, span_deg=150.0, want=None):
    """Fit a text block INTO a circular band (r_in..r_out). Returns (scale, radius_of_text_centre).
    Height ≤ 62 % of the band, arc length ≤ span_deg of the text circle → text can never leave the band."""
    band = max(1.0, r_out - r_in)
    rc = (r_in + r_out) / 2
    s_h = band * 0.62 / max(1.0, blk_h)
    s_w = math.radians(span_deg) * rc / max(1.0, blk_w)
    s = min(s_h, s_w, want if want else 9e9)
    return s, rc


def draw_band_text(ctx, F, L, b, x, y, r_in, r_out, angle, col, alpha, lt, pal, span=150.0, glow=0.0,
                   emit=True, size_mul=1.0):
    s, rc = band_fit(b.w, b.h, r_in, r_out, span)
    s *= size_mul
    G.draw_block(ctx, b, x, y, motion=_arc_motion(lt, col, alpha, glow), pal=pal,
                 field=G.Field("arc_top", radius=rc, angle=angle), scale=s, ectx=F.ectx if emit else None)
    return s


# =========================================================================================== disc (vinyl / CD)
def draw_disc(ctx, L, F):
    """@novadent vinyl: dark translucent glass body, thin rim, world-locked NEON half-rim on top that spills red
    light on the speaker, two band texts that live INSIDE the band (auto-fitted), white centre label lockup.
    "flip_at" spins the disc 180° → the second text (bottom_text) rotates up into the lit top half.
    Placement is guarded (never over the mouth/face) by pos()."""
    lt = F.t - L["_t0"]
    olt = _out_lt(L, F)
    if olt is not None and olt > 0.5:
        return
    x, y, k = pos(L, F)
    pal = L["_pal"]
    r = float(L.get("radius", 1.0)) * F.head_px()
    p = ph.sp(lt, "jelly")
    jx, jy, jsh = ph.jelly(lt, 0.6, 2.6, 3.4)
    a_out = min(1.0, lt / 0.12)
    if olt is not None and olt > 0:
        q = ph.ease_in(min(1, olt / 0.35))
        p *= 1 - q * 0.6
        a_out *= 1 - q
    if p <= 0.01:
        return
    flip = L.get("_flip_at")
    fq = ph.SILK(min(1, max(0, (F.t - flip) / 0.8))) if flip is not None else 0.0
    rot = 180 * fq + float(L.get("spin", 4.0)) * lt
    rr = r * p
    neon = G.rgba(L.get("neon", "@red"), pal)
    # --- body: TRUE glass, dark (reference vinyl is ~80 % black, the shirt still reads through)
    if F.pass_ == "front" and not F.prepass:
        glass(ctx, F, lambda c: c.arc(x, y, rr, 0, 2 * math.pi), (x - rr, y - rr, x + rr, y + rr), blur=10,
              tint=(0.02, 0.02, 0.025, float(L.get("body_dark", 0.80))), lift=0.0, sat=0.5, alpha=a_out)
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(jx, jy)
    # warm red inner glow from the lit rim (inside the body, top-heavy)
    rg = cairo.RadialGradient(0, -rr * 0.95, rr * 0.05, 0, -rr * 0.6, rr * 1.1)
    rg.add_color_stop_rgba(0, neon[0], neon[1], neon[2], 0.30 * a_out)
    rg.add_color_stop_rgba(1, neon[0], neon[1], neon[2], 0.0)
    ctx.arc(0, 0, rr, 0, 2 * math.pi); ctx.set_source(rg); ctx.fill()
    # grooves (very subtle)
    for i in range(7):
        ctx.arc(0, 0, rr * (0.46 + i * 0.07), 0, 2 * math.pi)
        ctx.set_source_rgba(1, 1, 1, 0.03 * a_out); ctx.set_line_width(1.0); ctx.stroke()
    # specular sweep (rotates with the disc)
    sa = math.radians(rot)
    lg = cairo.LinearGradient(-rr * math.cos(sa), -rr * math.sin(sa), rr * math.cos(sa), rr * math.sin(sa))
    lg.add_color_stop_rgba(0.0, 1, 1, 1, 0.0)
    lg.add_color_stop_rgba(0.5, 1, 1, 1, 0.07 * a_out)
    lg.add_color_stop_rgba(1.0, 1, 1, 1, 0.0)
    ctx.arc(0, 0, rr, 0, 2 * math.pi); ctx.set_source(lg); ctx.fill()
    # thin full rim
    ctx.arc(0, 0, rr, 0, 2 * math.pi)
    ctx.set_source_rgba(1, 1, 1, 0.22 * a_out); ctx.set_line_width(1.6); ctx.stroke()
    # NEON top half-rim (world-locked; draws on from the centre outwards)
    dr = ph.SILK(min(1.0, max(0.0, lt - 0.1) / 0.45))
    a0, a1 = 1.5 * math.pi - 0.5 * math.pi * dr * 1.0, 1.5 * math.pi + 0.5 * math.pi * dr * 1.0
    for lw, aa in ((16, 0.16), (7, 0.45), (2.6, 1.0)):
        ctx.arc(0, 0, rr - 1, a0, a1)
        ctx.set_source_rgba(neon[0], neon[1], neon[2], aa * a_out)
        ctx.set_line_width(lw * k); ctx.set_line_cap(cairo.LINE_CAP_ROUND); ctx.stroke()
    ctx.restore()
    if F.ectx is not None and dr > 0:
        e = F.ectx
        e.save(); e.translate(x, y)
        e.arc(0, 0, rr - 1, a0, a1)
        e.set_source_rgba(neon[0], neon[1], neon[2], a_out * float(L.get("spill", 1.0)))
        e.set_line_width(30 * k); e.stroke()
        e.restore()
    # --- band texts (INSIDE the band between label and rim)
    cr = rr * float(L.get("label_ratio", 0.40))
    r_in, r_out = cr * 1.08, rr * 0.93
    for key, off in (("top_text", 0.0), ("bottom_text", 180.0)):
        t = L.get(key)
        if not t:
            continue
        bt = G.typeset(t, 60, L.get("band_styles", {"thin": {"color": "#FFFFFF"}}), lockup=False)
        ang = rot + off                                    # 0 = text centred at the top
        up = 0.5 + 0.5 * math.cos(math.radians(ang))       # 1 at top (lit by the neon) … 0 at bottom
        c = (0.42 + 0.58 * up,) * 3 + (1.0,)
        draw_band_text(ctx, F, L, bt, x, y, r_in, r_out, ang, c, a_out * (0.35 + 0.65 * up), lt, pal,
                       span=float(L.get("band_span", 150)), glow=0.25 * up, emit=up > 0.6, size_mul=p)
    # small neon dot ornament just under the band text (reference detail)
    ctx.save(); ctx.translate(x, y)
    ctx.arc(0, -(r_in - cr * 0.02), 4.5 * k, 0, 2 * math.pi)
    ctx.set_source_rgba(neon[0], neon[1], neon[2], a_out); ctx.set_line_width(1.6 * k); ctx.stroke()
    # --- centre label
    ctx.arc(0, 0, cr, 0, 2 * math.pi)
    rg = cairo.RadialGradient(-cr * .3, -cr * .3, cr * .1, 0, 0, cr)
    rg.add_color_stop_rgba(0, 1, 1, 1, a_out)
    rg.add_color_stop_rgba(1, .90, .90, .92, a_out)
    ctx.set_source(rg); ctx.fill()
    ctx.restore()
    ct = L.get("center_text")
    if ct:
        bc = G.typeset(ct, 60, L.get("center_styles", {
            "thin": {"color": "#6A6D74", "font": "Doran-Light.ttf"},
            "hero": {"color": "#111216", "gradient": None, "scale": 1.45, "glow": 0}}), leading=0.0)
        s = min(cr * 1.45 / max(1.0, bc.w), cr * 1.2 / max(1.0, bc.h))      # fit inside the label
        G.draw_block(ctx, bc, x, y, pal=dict(pal), scale=s * p, alpha=a_out,
                     motion=lambda g: _pop_gs(lt - 0.15 - g.word * 0.08))


def _arc_motion(lt, col, a_out, glow=0.0):
    def m(g):
        s = G.GS()
        gl = lt - 0.12 - g.word * 0.06
        if gl <= 0:
            s.visible = False
            return s
        p = ph.sp(gl, "bouncy")
        s.sx = s.sy = max(0, p)
        s.color = col
        s.alpha = a_out
        s.glow = glow
        return s
    return m


def _pop_gs(gl, preset="jelly"):
    s = G.GS()
    if gl <= 0:
        s.visible = False
        return s
    p = ph.sp(gl, preset)
    s.sx = s.sy = max(0, p)
    return s


# =========================================================================================== glass_shape
def _shape_path(shape, x, y, w, h, rad=0.18, rot=0.0):
    def fn(c):
        c.save()
        c.translate(x, y)
        c.rotate(math.radians(rot))
        if shape == "circle":
            c.arc(0, 0, w / 2, 0, 2 * math.pi)
        elif shape == "hex":
            pts = [(w / 2 * math.cos(math.radians(a)), h / 2 * math.sin(math.radians(a))) for a in range(-90, 270, 60)]
            _rounded(c, pts, min(w, h) * rad * 0.6)
        elif shape == "diamond":
            pts = [(0, -h / 2), (w / 2, 0), (0, h / 2), (-w / 2, 0)]
            _rounded(c, pts, min(w, h) * rad)
        elif shape == "dome":
            c.move_to(-w / 2, h / 2)
            c.arc(0, h / 2, w / 2, math.pi, 2 * math.pi)
            c.close_path()
        elif shape == "pill":
            r = h / 2
            c.arc(-w / 2 + r, 0, r, math.pi / 2, 3 * math.pi / 2)
            c.arc(w / 2 - r, 0, r, -math.pi / 2, math.pi / 2)
            c.close_path()
        else:   # rsquare
            pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
            _rounded(c, pts, min(w, h) * rad)
        c.restore()
    return fn


def _rounded(c, pts, r):
    n = len(pts)
    for i in range(n):
        p0, p1, p2 = pts[i - 1], pts[i], pts[(i + 1) % n]
        v1 = (p0[0] - p1[0], p0[1] - p1[1]); v2 = (p2[0] - p1[0], p2[1] - p1[1])
        l1, l2 = math.hypot(*v1), math.hypot(*v2)
        rr = min(r, l1 / 2, l2 / 2)
        a = (p1[0] + v1[0] / l1 * rr, p1[1] + v1[1] / l1 * rr)
        b = (p1[0] + v2[0] / l2 * rr, p1[1] + v2[1] / l2 * rr)
        if i == 0:
            c.move_to(*a)
        else:
            c.line_to(*a)
        c.curve_to(p1[0], p1[1], p1[0], p1[1], *b)
    c.close_path()


def draw_glass_shape(ctx, L, F):
    if L.get("style", "nested") == "nested":
        from .glassbadge import draw_glass_badge
        return draw_glass_badge(ctx, L, F, pos, _glyph_motion, blk, _out_lt)
    lt = F.t - L["_t0"]
    olt = _out_lt(L, F)
    if olt is not None and olt > 0.5:
        return
    x, y, k = pos(L, F)
    pal = L["_pal"]
    hw = F.head_px()
    shape = L.get("shape", "hex")
    w = float(L.get("w", 1.6)) * hw
    h = float(L.get("h", 1.45 if shape == "hex" else 1.6)) * hw
    if shape == "circle":
        h = w
    p = ph.sp(lt, L.get("spring", "jelly"))
    jx, jy, jsh = ph.jelly(lt, 1.0, 2.8, 3.6)
    jx2, jy2, _ = (1, 1, 0)
    bump = L.get("_pulse_at")
    if bump is not None:
        for bt in (bump if isinstance(bump, list) else [bump]):
            if F.t > bt:
                a1, b1, _ = ph.jelly(F.t - bt, 0.9, 3.4, 4.5)
                jx2 *= a1; jy2 *= b1
    a = min(1.0, lt / 0.08)
    if olt is not None and olt > 0:
        q = ph.ease_in(min(1, olt / 0.3))
        p *= 1 - q * 1.05
        a *= 1 - q
    if p <= 0.02:
        return
    rot = float(L.get("rot", 0.0)) + ph.osc(F.t, 0.2, 2.0, 4) + (1 - min(1, p)) * float(L.get("spin_in", -25))
    ww, hh = w * p * jx * jx2, h * p * jy * jy2
    path = _shape_path(shape, x, y, ww, hh, float(L.get("round", 0.18)), rot)
    if F.pass_ == "front":
        glass(ctx, F, path, (x - ww, y - hh, x + ww, y + hh), blur=float(L.get("blur", 18)),
              tint=tuple(L.get("tint", (0.05, 0.05, 0.06, 0.38))), lift=float(L.get("lift", 0.04)), alpha=a,
              refract=L.get("refract"), chroma=L.get("chroma"), caustic=L.get("caustic"))
    else:
        ctx.save(); path(ctx); ctx.set_source_rgba(0.05, 0.05, 0.06, 0.55 * a); ctx.fill(); ctx.restore()
    # stroke (gold) + inner highlight
    sc = G.rgba(L.get("stroke", "@gold"), pal)
    ctx.save()
    path(ctx)
    ctx.set_source_rgba(sc[0], sc[1], sc[2], a)
    ctx.set_line_width(float(L.get("stroke_w", 3.0)) * k)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.stroke()
    ctx.restore()
    # glow of stroke into emission
    if F.ectx is not None:
        F.ectx.save(); path(F.ectx)
        F.ectx.set_source_rgba(sc[0], sc[1], sc[2], a * float(L.get("glow", 0.55)))
        F.ectx.set_line_width(8 * k); F.ectx.stroke(); F.ectx.restore()
    # specular bloom on top vertex
    if L.get("specular", True):
        vx = x + math.sin(math.radians(rot)) * hh / 2
        vy = y - math.cos(math.radians(rot)) * hh / 2
        tw = 0.6 + 0.4 * math.sin(F.t * 3.0)
        rg = cairo.RadialGradient(vx, vy, 0, vx, vy, 34 * k)
        rg.add_color_stop_rgba(0, 1, 0.97, 0.85, 0.95 * a * tw)
        rg.add_color_stop_rgba(1, 1, 0.85, 0.4, 0)
        ctx.set_source(rg)
        ctx.arc(vx, vy, 34 * k, 0, 2 * math.pi)
        ctx.fill()
        if F.ectx is not None:
            F.ectx.set_source_rgba(1, .9, .6, 0.8 * a * tw)
            F.ectx.arc(vx, vy, 10 * k, 0, 2 * math.pi)
            F.ectx.fill()
    # text inside
    if L.get("text"):
        b = blk(L)
        L2 = dict(L)
        L2.setdefault("enter", "jelly_pop")
        L2.setdefault("idle", "none")
        L2["_t0"] = L["_t0"] + float(L.get("text_delay", 0.12))
        mot = _glyph_motion(L2, F, b, k, F.t - L2["_t0"], olt)
        G.draw_block(ctx, b, x, y + float(L.get("text_dy", 0)) * hw, motion=mot, pal=pal, ectx=F.ectx,
                     scale=k * min(1.0, p * 1.0) * float(L.get("text_scale", 1.0)), alpha=a,
                     plane={"rz": rot * 0.5}, shadow={"alpha": .4, "dy": 3})


# =========================================================================================== halo
# HALO ANATOMY (v14.3, from the user's reference):
#   outer  = bigger TRANSLUCENT tinted disc + thin neon rim (radius rr)
#   inner  = small BRIGHT disc (radius rr*inner_ratio, radial gradient, hot centre)
#   centre = slightly ABOVE the head  (anchor head_top, default offset [0, -0.30] head-widths)
#   text   = INSIDE the bright disc.  text_mode "center" → stacked lockup centred (like the ref);
#            "arc" → text bent along the inner disc's upper edge (half-circle);  "auto" → arc when the
#            text is one wide line that would get tiny if centred, else center.
#            optional "arc_text" = a small line bent along the top edge ABOVE a centred hero.
#   VFX    = entrance shockwave ring + neon flicker, breathing glow, light sweep across the bright disc,
#            rim bloom + light spill into the emission buffer. The head occludes the lower part (back pass);
#            "peek" re-draws the hero's first glyph(s) in front.
HALO_DEFAULT = {"anchor": "head_top", "offset": [0, -0.22], "radius": 1.18, "inner_ratio": 0.74,
                "outer_alpha": 0.30, "text_mode": "auto", "sweep": True, "pulse": 0.04, "shock": True}


def _halo_geo(L, F):
    for kk, v in HALO_DEFAULT.items():
        L.setdefault(kk, v)
    lt = F.t - L["_t0"]
    olt = _out_lt(L, F)
    x, y, k = pos(L, F)
    hw = F.head_px()
    p = ph.sp(lt, "jelly")
    a = min(1.0, lt / 0.1)
    if olt is not None and olt > 0:
        q = ph.ease_in(min(1, olt / 0.35)); p *= 1 + q * 0.3; a *= 1 - q
    beat = 1 + float(L["pulse"]) * math.sin(F.t * 5.2) * (0.35 + 0.65 * math.exp(-max(0, lt - 0.3)))
    rr = float(L["radius"]) * hw * p * beat
    # SAFE FRAME: the outer disc never leaves the top of the frame — first push the centre down (it may
    # overlap the head more; the matte hides it), then shrink if still needed
    top = float(L.get("safe_top", 24))
    if y - rr < top:
        y = min(y + (top - (y - rr)), F.anchor("head_top")[1] + 0.05 * hw)
        rr = min(rr, max(0.5 * rr, y - top))
    return lt, olt, x, y, k, hw, rr, a


def draw_halo(ctx, L, F):
    olt = _out_lt(L, F)
    if olt is not None and olt > 0.5:
        return
    lt, olt, x, y, k, hw, rr, a = _halo_geo(L, F)
    pal = L["_pal"]
    base = G.rgba(L.get("color", "@alert"), pal)            # ONE colour → derived tones (works for any palette)
    c1 = G.rgba(L["outer"], pal) if "outer" in L else base
    c2 = G.rgba(L["inner"], pal) if "inner" in L else base
    neon = G.rgba(L.get("neon", L.get("color", "@alert")), pal)
    ri = rr * float(L["inner_ratio"])
    if F.pass_ == "front":                # front pass only re-draws the peeking glyph(s) — skip the body
        return _halo_text(ctx, L, F, x, y, k, hw, ri, a, olt, pal)
    # neon flicker on entrance (two quick dips), then steady
    flick = 1.0
    if lt < 0.45:
        flick = 0.55 + 0.45 * (0.5 + 0.5 * math.cos(lt * 47.0)) if lt > 0.08 else lt / 0.08
    glow = 0.85 + 0.15 * math.sin(F.t * 3.1)                  # breathing glow
    ctx.save()
    # ---- outer translucent disc (tinted glass) ------------------------------------------------
    og = cairo.RadialGradient(x, y, ri * 0.9, x, y, rr)
    oa = float(L["outer_alpha"]) * a
    og.add_color_stop_rgba(0, c1[0] * .55, c1[1] * .55, c1[2] * .55, oa * 1.2)
    og.add_color_stop_rgba(1, c1[0] * .35, c1[1] * .35, c1[2] * .35, oa * 0.75)
    ctx.arc(x, y, rr, 0, 2 * math.pi); ctx.set_source(og); ctx.fill()
    # ---- inner bright disc (hot centre, saturated edge) --------------------------------------
    rg = cairo.RadialGradient(x, y - ri * .25, ri * .05, x, y, ri)
    rg.add_color_stop_rgba(0, min(1, c2[0] * 1.15 + .12), min(1, c2[1] * 1.15 + .10), min(1, c2[2] * 1.15 + .10), a)
    rg.add_color_stop_rgba(0.7, c2[0], c2[1], c2[2], a)
    rg.add_color_stop_rgba(1, c2[0] * .72, c2[1] * .72, c2[2] * .72, a)
    ctx.arc(x, y, ri, 0, 2 * math.pi); ctx.set_source(rg); ctx.fill()
    # ---- VFX: light sweep across the bright disc (soft diagonal band, every ~2.4 s) ----------
    if L.get("sweep", True):
        ph_s = ((lt - 0.25) % 2.4) / 0.9
        if 0 <= ph_s <= 1:
            sx = x - ri * 1.4 + ph_s * ri * 2.8
            ctx.save(); ctx.arc(x, y, ri, 0, 2 * math.pi); ctx.clip()
            lg = cairo.LinearGradient(sx - ri * .35, y - ri, sx + ri * .35, y + ri)
            for st_, al in ((0, 0), (0.5, 0.22 * a), (1, 0)):
                lg.add_color_stop_rgba(st_, 1, .92, .88, al)
            ctx.set_source(lg); ctx.paint(); ctx.restore()
    # ---- neon rims: outer rim + inner rim ---------------------------------------------------
    for rad, lws in ((rr, ((12, 0.18), (3.0, 0.95))), (ri, ((6, 0.20), (1.6, 0.7)))):
        for lw, aa in lws:
            ctx.arc(x, y, rad, 0, 2 * math.pi)
            ctx.set_source_rgba(min(1, neon[0] + .15), min(1, neon[1] + .1), min(1, neon[2] + .1), aa * a * flick * glow)
            ctx.set_line_width(lw * k); ctx.stroke()
    # ---- VFX: entrance shockwave ring --------------------------------------------------------
    if L.get("shock", True) and 0.02 < lt < 0.6:
        q = (lt - 0.02) / 0.58
        ctx.arc(x, y, rr * (1.0 + 0.55 * ph.EXPO_OUT(q)), 0, 2 * math.pi)
        ctx.set_source_rgba(neon[0], neon[1], neon[2], 0.55 * (1 - q) ** 1.5)
        ctx.set_line_width(5 * k * (1 - q) + 1); ctx.stroke()
    ctx.restore()
    if F.ectx is not None:                # rim bloom + light spill (env light / light-wrap on her hair)
        e = F.ectx
        e.arc(x, y, rr, 0, 2 * math.pi)
        e.set_source_rgba(neon[0], neon[1], neon[2], 0.85 * a * flick * glow)
        e.set_line_width(26 * k); e.stroke()
        e.arc(x, y, ri, 0, 2 * math.pi)
        e.set_source_rgba(c2[0], c2[1], c2[2], 0.45 * a * glow * float(L.get("spill", 1.0)))
        e.fill()
    _halo_text(ctx, L, F, x, y, k, hw, ri, a, olt, pal)


def _halo_mode(L, b, ri):
    m = L.get("text_mode", "auto")
    if m != "auto":
        return m
    nlines = len(getattr(b, "lines", []) or [1])
    ts_center = min(1.6 * ri / max(1.0, b.w), 1.15 * ri / max(1.0, b.h))
    ts_arc = min(math.pi * ri * 0.62 * 0.85 / max(1.0, b.w), 0.34 * ri / max(1.0, b.h))
    return "arc" if (nlines == 1 and ts_arc > ts_center * 1.25) else "center"


def _halo_text(ctx, L, F, x, y, k, hw, ri, a, olt, pal):
    """Text INSIDE the bright inner disc (radius ri). center = lockup, arc = bent along the upper edge."""
    mode_pass = "back" if F.pass_ == "back" else "front"
    htop = F.anchor("head_top")[1]
    if L.get("arc_text") and mode_pass == "back":       # small line bent along the top edge
        ba = blk(L, text_key="arc_text", size_key="arc_size") if "arc_size" in L else \
            blk(L, 0.42, text_key="arc_text")
        Ra = ri * float(L.get("arc_r", 0.84))
        tsa = min(k, math.pi * Ra * 0.55 / max(1.0, ba.w), 0.20 * ri / max(1.0, ba.h))
        La = dict(L); La["_t0"] = L["_t0"] + 0.30; La.setdefault("enter", "drop_bounce")
        mota = _glyph_motion(La, F, ba, k, F.t - La["_t0"], olt)
        G.draw_block(ctx, ba, x, y, motion=mota, pal=pal, ectx=F.ectx, scale=tsa, alpha=a,
                     field=G.Field("arc_top", radius=Ra), shadow={"alpha": .35, "dy": 2})
    if not L.get("text"):
        return
    b = blk(L)
    L2 = dict(L); L2["_t0"] = L["_t0"] + 0.15; L2.setdefault("enter", "drop_bounce")
    mot = _glyph_motion(L2, F, b, k, F.t - L2["_t0"], olt)
    mode = _halo_mode(L, b, ri)
    if mode == "arc":
        R = ri * float(L.get("arc_r", 0.72))
        tsc = min(k, math.pi * R * 0.85 / max(1.0, b.w), 0.34 * ri / max(1.0, b.h))
        if mode_pass == "back":
            G.draw_block(ctx, b, x, y, motion=mot, pal=pal, ectx=F.ectx, scale=tsc, alpha=a,
                         field=G.Field("arc_top", radius=R), shadow={"alpha": .4, "dy": 3})
        return                                        # arc text sits high in the disc → no peek needed
    # centered lockup: fit into the disc; centre slightly above the disc centre (ref look)
    ty = y + float(L.get("text_dy_r", -0.05 if L.get("arc_text") else -0.12)) * ri
    if L.get("arc_text"):
        ty += 0.10 * ri
    tsc = min(k, 1.55 * ri / max(1.0, b.w), (1.05 if L.get("arc_text") else 1.25) * ri / max(1.0, b.h))
    # OCCLUSION RULE: at most max_hidden of the block may sit below her hairline (head_top − hair margin).
    # If the disc has no room above the head (head high in frame), the text SHRINKS to fit between the top
    # of the bright disc and that limit, instead of disappearing behind her.
    hair = htop - float(L.get("hair_margin", 0.06)) * hw
    mh = float(L.get("max_hidden", 0.22))
    top_lim = y - ri * (0.86 if not L.get("arc_text") else 0.50)   # arc_text owns the top band of the disc
    avail = (hair - top_lim) / max(0.2, 1 - mh)
    tsc = max(0.45 * tsc, min(tsc, avail / max(1.0, b.h)))
    bh = b.h * tsc
    bottom_allowed = hair + mh * bh
    if ty + bh / 2 > bottom_allowed:
        ty = max(top_lim + bh / 2, bottom_allowed - bh / 2)
    if mode_pass == "back":
        G.draw_block(ctx, b, x, ty, motion=mot, pal=pal, ectx=F.ectx, scale=tsc, alpha=a,
                     shadow={"alpha": .45, "dy": 4})
    elif L.get("peek", True):
        # PEEK: the hero's first letter (RTL start = right-most glyph) is re-drawn in FRONT of her head
        hero = [g for g in b.glyphs if g.sname in ("hero", "hero_white", "alert")]
        if hero:
            hw_i = hero[0].word
            peek = sorted([g for g in hero if g.word == hw_i], key=lambda g: -g.x1)[:int(L.get("peek_glyphs", 1))]
            ids = {id(g) for g in peek}

            def mot2(g, _m=mot):
                st = _m(g)
                if id(g) not in ids:
                    st.visible = False
                return st
            G.draw_block(ctx, b, x, ty, motion=mot2, pal=pal, ectx=None, scale=tsc, alpha=a)


# =========================================================================================== frame_box
def draw_frame_box(ctx, L, F):
    lt = F.t - L["_t0"]
    olt = _out_lt(L, F)
    if olt is not None and olt > 0.6:
        return
    x, y, k = pos(L, F)
    hw = F.head_px()
    pal = L["_pal"]
    w = float(L.get("w", 2.0)) * hw
    h = float(L.get("h", 1.3)) * hw
    rot = float(L.get("rot", -8.0)) + ph.osc(F.t, 0.22, 1.5, 3)
    # stroke DRAWS ON (trim path) with ease, then jelly
    dp = ph.SILK(min(1, lt / 0.55))
    a = 1.0
    if olt is not None and olt > 0:
        q = ph.ease_in(min(1, olt / 0.35)); a = 1 - q; dp = 1 - q
    jx, jy, jsh = ph.jelly(lt - 0.4, 0.6, 2.6, 3.5)
    path = _shape_path("rsquare", x, y, w * jx, h * jy, 0.22, rot)
    ctx.save()
    path(ctx)
    flat = ctx.copy_path_flat()
    ctx.new_path()
    pts = [p[1] for p in flat if p[0] in (cairo.PATH_MOVE_TO, cairo.PATH_LINE_TO)]
    if pts:
        tot = sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) for i in range(1, len(pts)))
        lim = tot * dp
        acc = 0.0
        ctx.move_to(*pts[0])
        for i in range(1, len(pts)):
            d = math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            if acc + d > lim:
                f = (lim - acc) / max(d, 1e-6)
                ctx.line_to(pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * f, pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * f)
                break
            ctx.line_to(*pts[i])
            acc += d
        col = G.rgba(L.get("stroke", "@ink"), pal)
        ctx.set_source_rgba(col[0], col[1], col[2], 0.9 * a)
        ctx.set_line_width(float(L.get("stroke_w", 3.0)) * k)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.stroke()
    ctx.restore()
    # diamond badge at top-left corner (with optional icon text/emoji-free glyph)
    if L.get("badge", True):
        bp = ph.sp(lt - 0.25, "jelly")
        if bp > 0.02:
            cx = x - w / 2 * math.cos(math.radians(rot)) + h / 2 * math.sin(math.radians(rot))
            cy = y - w / 2 * math.sin(math.radians(rot)) - h / 2 * math.cos(math.radians(rot))
            bs = 0.55 * hw * bp
            dpath = _shape_path("diamond", cx, cy, bs * 1.25, bs * 1.25, 0.18, ph.osc(F.t, .3, 4, 1))
            if F.pass_ == "front":
                glass(ctx, F, dpath, (cx - bs, cy - bs, cx + bs, cy + bs), blur=10, tint=(0.95, 0.75, 0.2, 0.35), alpha=a)
            ctx.save(); dpath(ctx)
            gc = G.rgba(L.get("badge_color", "@gold"), pal)
            ctx.set_source_rgba(gc[0], gc[1], gc[2], a); ctx.set_line_width(3 * k); ctx.stroke(); ctx.restore()
            if L.get("badge_text"):
                bb = G.typeset(L["badge_text"], 0.30 * hw, {"thin": {"font": "Doran-ExtraBold.ttf", "color": "@ink"}})
                G.draw_block(ctx, bb, cx, cy, scale=bp, alpha=a)
    # text — hero letters deliberately overflow the frame
    if L.get("text"):
        b = blk(L)
        L2 = dict(L)
        L2["_t0"] = L["_t0"] + 0.2
        L2.setdefault("enter", "jelly_pop")
        mot = _glyph_motion(L2, F, b, k, F.t - L2["_t0"], olt)
        G.draw_block(ctx, b, x, y, motion=mot, pal=pal, ectx=F.ectx, scale=k, alpha=a,
                     plane={"rz": rot * 0.6, "ry": ph.osc(F.t, .2, 8, 5)}, extrude=L.get("extrude"),
                     shadow={"alpha": .5, "dy": 5})


# =========================================================================================== strike / underline
def draw_strike(ctx, L, F):
    """v15 Variable-Width Tapered Brush Strike / Underline / Circle (using `mograph.varstroke`)."""
    from ..varstroke import draw_tapered_stroke, Taper, Wave
    lt = F.t - L["_t0"]
    olt = _out_lt(L, F)
    if olt is not None and olt > 0.4:
        return
    x, y, k = pos(L, F)
    hw = F.head_px()
    w = float(L.get("w", 1.6)) * hw
    pal = L["_pal"]
    col = G.rgba(L.get("color", "@red"), pal)
    p = ph.SILK(min(1, lt / float(L.get("dur", 0.35))))
    a = 1.0 if olt is None else max(0, 1 - olt / 0.3)
    kind = L.get("kind", "strike")
    base_w = float(L.get("width", 10)) * k
    if kind == "circle":
        rx, ry = w / 2, float(L.get("h", 0.6)) * hw / 2
        n = 64
        pts = []
        for i in range(n + 1):
            th = -math.pi * 0.6 + i / n * 2 * math.pi * 1.12
            wob = 1 + 0.04 * math.sin(i * 0.6)
            pts.append([x + rx * wob * math.cos(th), y + ry * wob * math.sin(th)])
    else:
        x0, x1 = x + w / 2, x - w / 2
        yy = y + (0 if kind == "strike" else float(L.get("dy", 0.35)) * hw)
        u = np.linspace(0.0, 1.0, 32)
        px_ = x0 + (x1 - x0) * u
        py_ = yy + 4.0 * (1.0 - u) - 5.0 * np.sin(u * math.pi)
        pts = np.column_stack([px_, py_])
    tap = L.get("taper", Taper(start_len=0.22, end_len=0.38, start_width=0.12, end_width=0.02))
    wav = Wave(amount=0.12, cycles=2.5) if L.get("wave", True) else None
    rgba_c = (col[0], col[1], col[2], a)
    draw_tapered_stroke(ctx, pts, base_w, rgba_c, taper=tap, wave=wav, progress=p)
    if F.ectx is not None:
        draw_tapered_stroke(F.ectx, pts, base_w * 1.6, (col[0], col[1], col[2], 0.65 * a), taper=tap, progress=p)


# =========================================================================================== small layers
def draw_flash(ctx, L, F):
    lt = F.t - L["_t0"]
    d = float(L.get("dur", 0.18))
    if lt < 0 or lt > d:
        return
    a = (1 - lt / d) ** 2 * float(L.get("amount", 0.5))
    c = G.rgba(L.get("color", "#FFF4E0"))
    ctx.set_source_rgba(c[0], c[1], c[2], a)
    ctx.set_operator(cairo.OPERATOR_ADD)
    ctx.paint()


def draw_progress(ctx, L, F):
    p = min(1, F.t / max(1e-3, F.R.duration))
    c = G.rgba(L.get("color", "@gold"), L["_pal"])
    ctx.rectangle(F.W - F.W * p, 0, F.W * p, float(L.get("h", 6)))     # RTL: grows right → left
    ctx.set_source_rgba(c[0], c[1], c[2], 0.9)
    ctx.fill()


def draw_counter_badge(ctx, L, F):
    lt = F.t - L["_t0"]
    olt = _out_lt(L, F)
    if olt is not None and olt > 0.4:
        return
    x, y, k = pos(L, F)
    p = ph.sp(lt, "jelly")
    a = 1.0 if olt is None else max(0, 1 - olt / 0.3)
    r = float(L.get("r", 46)) * k * max(0, p)
    if r < 1:
        return
    pal = L["_pal"]
    c = G.rgba(L.get("color", "@gold"), pal)
    path = _shape_path("circle", x, y, 2 * r, 2 * r)
    if F.pass_ == "front":
        glass(ctx, F, path, (x - r, y - r, x + r, y + r), blur=10, tint=(0.08, 0.06, 0.02, 0.5), alpha=a)
    ctx.save(); path(ctx); ctx.set_source_rgba(c[0], c[1], c[2], a); ctx.set_line_width(3 * k); ctx.stroke(); ctx.restore()
    b = G.typeset(L.get("text", "۱"), float(L.get("size", 54)), {"thin": {"font": "Doran-ExtraBold.ttf", "color": "@gold"}})
    G.draw_block(ctx, b, x, y, scale=k * max(0, p), alpha=a, plane={"ry": (1 - min(1, p)) * 160}, ectx=F.ectx,
                 motion=lambda g: _glow_gs(0.5))


def _glow_gs(gl):
    s = G.GS()
    s.glow = gl
    return s


def draw_sweep_light(ctx, L, F):
    lt = F.t - L["_t0"]
    d = float(L.get("dur", 0.6))
    if lt < 0 or lt > d:
        return
    p = ph.SILK(lt / d)
    x = F.W * (1.2 - 1.4 * p)
    lg = cairo.LinearGradient(x - 160, 0, x + 160, 0)
    lg.add_color_stop_rgba(0, 1, 1, 1, 0)
    lg.add_color_stop_rgba(.5, 1, .95, .85, .14)
    lg.add_color_stop_rgba(1, 1, 1, 1, 0)
    ctx.set_source(lg)
    ctx.set_operator(cairo.OPERATOR_ADD)
    ctx.paint()


DRAW = {"kinetic_title": draw_kinetic_title, "orbit_ring": draw_orbit_ring, "disc": draw_disc,
        "glass_shape": draw_glass_shape, "halo": draw_halo, "frame_box": draw_frame_box, "strike": draw_strike,
        "flash": draw_flash, "progress": draw_progress, "counter_badge": draw_counter_badge,
        "sweep_light": draw_sweep_light}

# subtitles are defined in subs.py (jelly capsule) and registered here
from .subs import prepare_subs, draw_subs  # noqa: E402

DRAW["subtitles"] = draw_subs


# =========================================================================================== tracked_tag
def draw_tracked_tag(ctx, L, F):
    """MOTION-TRACKING SHOWCASE: a small glass chip glued to a tracked point (hand / shoulder) with a curved
    leader line, a pulsing target ring ON the point, and a short motion trail — so the viewer SEES the tracking.
    "anchor": "hand_r", "offset": [0.9, -0.6] (chip position relative to the point, in head widths), "text"."""
    lt = F.t - L["_t0"]
    olt = _out_lt(L, F)
    if olt is not None and olt > 0.4:
        return
    a = min(1.0, lt / 0.15) * (1.0 if olt is None else max(0.0, 1 - olt / 0.35))
    hw = F.head_px()
    an = L.get("anchor", "hand_r")
    px_, py_ = F.anchor(an)
    k = (hw / 300.0) ** 0.5
    pal = L["_pal"]
    col = G.rgba(L.get("color", "@accent"), pal)
    # trail: previous positions of the point
    for i in range(1, 9):
        st = F.src_t - i * 0.035
        tx, ty = F.anchor(an, st)
        ctx.arc(tx, ty, 5 * k * (1 - i / 10), 0, 2 * math.pi)
        ctx.set_source_rgba(col[0], col[1], col[2], 0.35 * a * (1 - i / 9))
        ctx.fill()
    # target ring on the point (spring in + pulse)
    pr = ph.sp(lt, "jelly")
    rr = 26 * k * max(0, pr) * (1 + 0.12 * math.sin(F.t * 9))
    for lw, aa in ((6, .25), (2, 1)):
        ctx.arc(px_, py_, rr, 0, 2 * math.pi)
        ctx.set_source_rgba(col[0], col[1], col[2], aa * a)
        ctx.set_line_width(lw * k)
        ctx.stroke()
    ctx.arc(px_, py_, 4 * k, 0, 2 * math.pi)
    ctx.set_source_rgba(1, 1, 1, a)
    ctx.fill()
    if F.ectx is not None:
        F.ectx.arc(px_, py_, rr, 0, 2 * math.pi)
        F.ectx.set_source_rgba(col[0], col[1], col[2], 0.8 * a)
        F.ectx.set_line_width(8 * k)
        F.ectx.stroke()
    # chip follows with a slight lag (secondary motion) — spring towards the target
    off = L.get("offset", [0.9, -0.6])
    lag = float(L.get("lag", 0.08))
    lx, ly = F.anchor(an, F.src_t - lag)
    cx_, cy_ = lx + off[0] * hw, ly + off[1] * hw
    if L.get("text"):     # keep the chip inside the safe frame; the leader still points at the true tracked point
        b0 = blk(L)
        pw0, ph0 = b0.w * k + 44 * k, b0.h * k + 30 * k
        cx_ = min(max(cx_, pw0 / 2 + 40), F.R.W - pw0 / 2 - 40)
        cy_ = min(max(cy_, ph0 / 2 + 140), F.R.H - ph0 / 2 - 260)
        from .stage import face_zone          # the chip may never sit on her face: slide it out sideways
        zx, zy, rx, ry = face_zone(F, 0.10)
        rr0 = 0.5 * max(pw0, ph0)
        if ((cx_ - zx) / (rx + rr0)) ** 2 + ((cy_ - zy) / (ry + rr0)) ** 2 < 1.0:
            sgn = 1 if cx_ >= zx else -1
            cand = zx + sgn * (rx + pw0 / 2 + 8)
            if cand - pw0 / 2 < 40 or cand + pw0 / 2 > F.R.W - 40:      # no room on that side → go below the chin
                cy_ = zy + ry + ph0 / 2 + 8
            else:
                cx_ = cand
    # leader line (quadratic curve) drawn on
    dp = ph.SILK(min(1, max(0, lt - 0.1) / 0.3))
    if dp > 0.01:
        mx, my = (px_ + cx_) / 2, min(py_, cy_) - 0.25 * hw
        ctx.save()
        ctx.move_to(px_, py_)
        n = 24
        for i in range(1, int(n * dp) + 1):
            u = i / n
            qx = (1 - u) ** 2 * px_ + 2 * (1 - u) * u * mx + u * u * cx_
            qy = (1 - u) ** 2 * py_ + 2 * (1 - u) * u * my + u * u * cy_
            ctx.line_to(qx, qy)
        ctx.set_source_rgba(1, 1, 1, 0.75 * a)
        ctx.set_line_width(1.6 * k)
        ctx.set_dash([5 * k, 4 * k])
        ctx.stroke()
        ctx.restore()
    if L.get("text") and lt > 0.3:
        b = blk(L)
        pw, ph_ = b.w * k + 44 * k, b.h * k + 30 * k
        sp_ = ph.sp(lt - 0.3, "jelly")
        from .glassbadge import shape_path
        path = shape_path("pill", cx_, cy_, pw * min(1.0, sp_), ph_, 0.5, 0)
        if F.pass_ == "front" and not F.prepass:
            glass(ctx, F, path, (cx_ - pw, cy_ - ph_, cx_ + pw, cy_ + ph_), blur=12,
                  tint=(0.04, 0.04, 0.05, 0.55), alpha=a)
        ctx.save(); path(ctx)
        ctx.set_source_rgba(col[0], col[1], col[2], 0.9 * a); ctx.set_line_width(1.6 * k); ctx.stroke(); ctx.restore()
        L2 = dict(L)
        L2["_t0"] = L["_t0"] + 0.38
        L2.setdefault("enter", "jelly_pop")
        L2.setdefault("idle", "none")
        G.draw_block(ctx, b, cx_, cy_, motion=_glyph_motion(L2, F, b, k, F.t - L2["_t0"], olt), pal=pal,
                     ectx=F.ectx, scale=k, alpha=a)


DRAW["tracked_tag"] = draw_tracked_tag


# =========================================================================================== object_3d (v14.4)
def draw_object_3d(ctx, L, F):
    from .object3d import draw_object_3d as _d
    return _d(ctx, L, F, pos, _out_lt)


DRAW["object_3d"] = draw_object_3d


# =========================================================================================== ambient_bokeh (v14.8)
def draw_ambient_bokeh(ctx, L, F):
    """Rendered by the compositor (bokeh.py) in the two depth slots; nothing to draw on the layer canvases."""
    return None


DRAW["ambient_bokeh"] = draw_ambient_bokeh


# =========================================================================================== v15 Craft Layers
def draw_vegas_comet(ctx, L, F):
    """Ported from EffectCraft `crates/effects/src/generate3.rs` (`vegas`):
    Runs N glowing comet-tail pulses with tapered widths chasing around a shape contour."""
    from ..varstroke import draw_tapered_stroke, Taper
    lt = F.t - L["_t0"]
    olt = _out_lt(L, F)
    if lt < 0 or (olt is not None and olt > 0.4):
        return
    x, y, k = pos(L, F)
    hw = F.head_px()
    w = float(L.get("w", 1.8)) * hw
    h = float(L.get("h", 1.2)) * hw
    shape = L.get("shape", "hept")
    path = _shape_path(shape, x, y, w, h, float(L.get("round", 0.18)), float(L.get("rot", 0.0)))
    ctx.save()
    path(ctx)
    flat = ctx.copy_path_flat()
    ctx.new_path()
    ctx.restore()
    pts = np.array([p[1] for p in flat if p[0] in (cairo.PATH_MOVE_TO, cairo.PATH_LINE_TO)], dtype=np.float32)
    if len(pts) < 4:
        return
    col = G.rgba(L.get("color", "@accent"), L["_pal"])
    a = min(1.0, lt / 0.2) * (1.0 if olt is None else max(0.0, 1.0 - olt / 0.35))
    n_comets = int(L.get("comets", 2))
    speed = float(L.get("speed", 0.55))
    span = int(max(4, len(pts) * float(L.get("length", 0.28))))
    for ci in range(n_comets):
        head_idx = int(((lt * speed + ci / n_comets) % 1.0) * len(pts))
        seg_pts = np.array([pts[(head_idx - span + j) % len(pts)] for j in range(span)], dtype=np.float32)
        tap = Taper(start_len=0.85, end_len=0.05, start_width=0.0, end_width=0.2)
        draw_tapered_stroke(ctx, seg_pts, float(L.get("width", 5.5)) * k, (col[0], col[1], col[2], a), taper=tap)
        if F.ectx is not None:
            draw_tapered_stroke(F.ectx, seg_pts, float(L.get("width", 11.0)) * k, (col[0], col[1], col[2], 0.75 * a), taper=tap)


def draw_god_rays(ctx, L, F):
    """Ported from EffectCraft `crates/effects/src/generate2.rs` (`light_rays` / `light_burst`):
    Volumetric radial god-rays casting outward behind hero typography."""
    lt = F.t - L["_t0"]
    olt = _out_lt(L, F)
    if lt < 0 or (olt is not None and olt > 0.5):
        return
    x, y, k = pos(L, F)
    hw = F.head_px()
    R = float(L.get("radius", 1.6)) * hw
    col = G.rgba(L.get("color", "@accent"), L["_pal"])
    a = min(1.0, lt / 0.25) * (1.0 if olt is None else max(0.0, 1.0 - olt / 0.4)) * float(L.get("alpha", 0.38))
    n_rays = int(L.get("rays", 12))
    rot = F.t * float(L.get("spin", 14.0))
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_SCREEN)
    for i in range(n_rays):
        ang = math.radians(rot + i * (360.0 / n_rays))
        half_w = math.radians(float(L.get("beam_deg", 7.5)) * (0.7 + 0.3 * math.sin(F.t * 3.0 + i)))
        rg = cairo.RadialGradient(x, y, R * 0.05, x, y, R)
        rg.add_color_stop_rgba(0.0, col[0], col[1], col[2], a)
        rg.add_color_stop_rgba(0.55, col[0], col[1], col[2], a * 0.35)
        rg.add_color_stop_rgba(1.0, col[0], col[1], col[2], 0.0)
        ctx.new_path()
        ctx.move_to(x, y)
        ctx.arc(x, y, R, ang - half_w, ang + half_w)
        ctx.close_path()
        ctx.set_source(rg)
        ctx.fill()
    ctx.restore()


DRAW["vegas_comet"] = draw_vegas_comet
DRAW["god_rays"] = draw_god_rays
