"""
mograph.v14.stage — two automatic "editor brains" every layer goes through (a weak agent gets them for free).

A) LAYOUT GUARD  (never cover the face / mouth, never leave the safe frame)
   protected zone = an ellipse around the tracked face, from forehead to below the chin, + a margin, computed
   per frame from the pose.  Before a layer is drawn its bounding circle/box is tested; if it overlaps the zone
   it is MOVED along the layer's preferred escape direction (down for chest elements, up for head titles,
   sideways for side tags) — smoothly (the shift itself is spring-smoothed over time so nothing jumps) — and
   also clamped to the 9:16 safe frame (top 140 px UI bar, bottom 260 px caption zone, 40 px sides).
   Layers opt out with "guard": false (e.g. halo, orbit ring and full-screen flashes are designed around her).

B) STAGE LIGHT  (the user's taste: a translucent engineering grid, low opacity, that DARKENS the plate locally,
   with a centre glow in the element's colour that fades with distance)
   For every "stage"-enabled element (default: disc, glass_shape, halo, frame_box, kinetic_title heroes) the
   background around it — behind the speaker, never on her (the speaker pass composites over it) — gets:
     1 scrim   radial darkening (strength from the plate brightness under the element, so bright skies get
               more, already-dark plates almost none)  → contrast for the element without a box
     2 grid    0.6–1 px coordinate grid + '+' ticks at intersections, world-locked to the background
               (moves with the camera), alpha falling off with distance from the element
     3 light   a coloured pool in the element's colour at the centre, fading out (this is what makes the
               element look like it EMITS light) — the same colour also feeds the emission buffer so it spills
               onto the speaker.
   Strength per layer: "stage": {"scrim": 0.55, "grid": 0.22, "light": 0.5, "radius": 1.7}  or  "stage": false.
   v21: the GRID IS OFF BY DEFAULT (grid 0.0) — only scrim + coloured light pool are drawn.  Turn the grid on
   explicitly per layer ("stage": {"grid": 0.25}) and at most once per video (originality guard warns otherwise).
"""
from __future__ import annotations

import math

import cairo
import numpy as np

from . import glyphs as G
from . import physics as ph

STAGE_DEFAULT = {"disc": True, "glass_shape": True, "halo": True, "frame_box": True, "kinetic_title": True,
                 "orbit_ring": False, "tracked_tag": False, "strike": False, "flash": False, "progress": False,
                 "counter_badge": False, "sweep_light": False}
GUARD_DEFAULT = {"disc": "down", "glass_shape": "down", "frame_box": "up", "kinetic_title": "up",
                 "counter_badge": "side", "tracked_tag": "side", "strike": None, "halo": None, "orbit_ring": None,
                 "flash": None, "progress": None, "sweep_light": None,
                 "object_3d": "side"}


# ------------------------------------------------------------------------------------------------ A) guard
def face_zone(F, margin=0.18, crown=True):
    """Protected ellipse (cx, cy, rx, ry) in screen px: HAIR CROWN … below the chin.
    v21 fix: the top used to stop at the eyebrows (fy − 0.55·hw) → titles were pushed onto the forehead.  Now the
    whole forehead + hair crown (head_top − 0.18·hw) is inside the ellipse."""
    hw = F.head_px()
    fx, fy = F.anchor("face")
    cx_, cy_ = F.anchor("chin")
    if not crown:                                  # QA "covers the face" test: eyes … chin only
        top = fy - 0.55 * hw
        bot = cy_ + 0.20 * hw
        return fx, (top + bot) / 2, hw * (0.5 + margin), (bot - top) / 2 + hw * margin
    try:
        _, htop_y = F.anchor("head_top")
    except Exception:  # noqa: BLE001
        htop_y = fy - 0.85 * hw
    top = min(fy - 0.85 * hw, htop_y - 0.18 * hw)
    bot = cy_ + 0.20 * hw
    return fx, (top + bot) / 2, hw * (0.5 + margin), (bot - top) / 2 + hw * margin


def _overlap(zone, x, y, r):
    zx, zy, rx, ry = zone
    # distance from element circle to the ellipse (approx: scale to unit circle)
    dx, dy = (x - zx) / (rx + r), (y - zy) / (ry + r)
    return dx * dx + dy * dy < 1.0


# v22: the default push direction follows the ANCHOR, not only the type.  A title on the chest / shoulders / neck
# that grazes the chin must move DOWN a few px — the old type default "up" threw it over the whole face to the top.
ANCHOR_GUARD = {"chest": "down", "belly": "down", "neck": "down", "shoulders": "down", "shoulder_l": "down",
                "shoulder_r": "down", "hands": "down", "hand_l": "side", "hand_r": "side", "chin": "down",
                "head_top": "up", "head_c": "up", "face": "up", "nose": "up"}


def guard_mode(L, kind=None):
    if kind is not None:
        return kind
    if "guard" in L:
        return L["guard"]
    base = GUARD_DEFAULT.get(L["type"], None)
    if base in ("up", "down"):
        return ANCHOR_GUARD.get(L.get("anchor", "screen"), base)
    return base


def guard(L, F, x, y, r, kind=None):
    """Return a safe (x, y) for an element of radius r.  Smooth over time (no jumps)."""
    mode = guard_mode(L, kind)
    if not mode:
        return x, y
    zone = face_zone(F, float(L.get("guard_margin", 0.12)))
    zx, zy, rx, ry = zone
    tx, ty = x, y
    if _overlap(zone, x, y, r):
        if mode == "down":
            ty = zy + ry + r + 6
        elif mode == "up":
            ty = zy - ry - r - 6
        else:                                     # side: push away horizontally
            s = 1 if x >= zx else -1
            tx = zx + s * (rx + r + 6)
    # Also prevent "side" elements (object_3d, tracked_tag) from overlapping active chest hero widgets (disc, glass_shape)
    if mode == "side" and hasattr(F, "R") and hasattr(F.R, "layers"):
        for other in F.R.layers:
            if other is L or other.get("type") not in ("disc", "glass_shape"):
                continue
            if other.get("_t0", 9e9) - 0.05 <= F.t <= other.get("_t1", -9e9) + 0.05:
                cx_c, cy_c = F.anchor(other.get("anchor", "chest"))
                hw_c = F.head_px()
                off_c = other.get("offset", [0, 0.35])
                cy_c += off_c[1] * hw_c
                rc_c = hw_c * 0.85
                if abs(ty - cy_c) < (rc_c + r * 0.85) and abs(tx - cx_c) < (rc_c + r * 0.75):
                    s = 1 if tx >= cx_c else -1
                    tx = cx_c + s * (rc_c + r * 0.75 + 12)
    # safe frame
    W, H = F.W, F.H
    top_lim = r * 0.90 + 145
    # wide text blocks: r ≈ half the block width → the whole width must stay inside (v21: side titles were clipped)
    sx_k = 1.0 if L.get("type") in ("kinetic_title", "shatter_text") else 0.60
    tx = min(max(tx, r * sx_k + 40), W - r * sx_k - 40)
    ty = min(max(ty, top_lim), H - r * 1.0 - 80)              # element must stay fully on screen
    # v21 HIGH-HEAD FIX: an "up" element (title above the head) that the top safe boundary pushed back DOWN onto
    # the head must never land on the forehead → slide it sideways into the open upper-left / upper-right
    # shoulder space; if the frame is too narrow for that, shrink it (L["_gscale"]) so it stays above the hair.
    gscale = 1.0
    if mode == "up" and _overlap(zone, tx, ty, r):
        hw = F.head_px()
        try:
            htop_y = F.anchor("head_top")[1]
        except Exception:  # noqa: BLE001
            htop_y = zy - ry
        room_l, room_r = zx - rx, W - (zx + rx)
        s = -1 if room_l > room_r else 1
        if abs(x - zx) > 0.25 * rx:
            s = 1 if x >= zx else -1
        tx = zx + s * (rx + r * 0.80 + 10)
        ty = max(top_lim, min(ty, zy - 0.25 * ry))
        tx_c = min(max(tx, r * sx_k + 40), W - r * sx_k - 40)
        if abs(tx_c - tx) > 2 or _overlap(zone, tx_c, ty, r):
            # no side room either → scale down so the element ends above the hair crown
            avail = max(20.0, (htop_y - 0.08 * hw) - 145)
            gscale = max(0.5, min(1.0, avail / max(1.0, 1.9 * r)))
            r2 = r * gscale
            ty = max(r2 * 0.90 + 145, min(ty, htop_y - 0.08 * hw - r2))
            side_x = zx + s * (rx + r2 * 0.80 + 10)
            rc = max(r2, r * 0.95) * sx_k * 1.08              # drawn ink (glow / extrude) is wider than r2
            tx_c = min(max(side_x, rc + 40), W - rc - 40)
        tx = tx_c
    gs_prev = L.get("_gscale", 1.0)
    L["_gscale"] = gscale if abs(gscale - gs_prev) < 0.02 else gs_prev + (gscale - gs_prev) * 0.35
    # temporal smoothing of the CORRECTION (state on the layer, keyed by frame time)
    dx, dy = tx - x, ty - y
    st = L.setdefault("_guard", {"t": None, "dx": 0.0, "dy": 0.0})
    if st["t"] == F.t:                              # same frame (stage pass + draw) → same answer
        return x + st["dx"], y + st["dy"]
    if st["t"] is not None and 0 < F.t - st["t"] < 0.2:
        a = 1 - math.exp(-(F.t - st["t"]) / 0.12)
        st["dx"] += (dx - st["dx"]) * a
        st["dy"] += (dy - st["dy"]) * a
    else:
        st["dx"], st["dy"] = dx, dy
    st["t"] = F.t
    return x + st["dx"], y + st["dy"]


# ------------------------------------------------------------------------------------------------ B) stage
def _grid(ctx, F, cx, cy, R, col, alpha, k):
    """World-locked grid (anchored to the background via the camera matrix) with radial fade."""
    if alpha <= 0.004:
        return
    M = F.M
    step = 46.0 * float(M[0, 0])                    # grid cell scales with the camera zoom
    ox, oy = float(M[0, 2]) % step, float(M[1, 2]) % step
    rg = cairo.RadialGradient(cx, cy, R * 0.15, cx, cy, R)
    rg.add_color_stop_rgba(0, col[0], col[1], col[2], alpha)
    rg.add_color_stop_rgba(0.6, col[0], col[1], col[2], alpha * 0.45)
    rg.add_color_stop_rgba(1, col[0], col[1], col[2], 0)
    ctx.save()
    ctx.arc(cx, cy, R, 0, 2 * math.pi)
    ctx.clip()
    x0 = cx - R - ((cx - R - ox) % step)
    y0 = cy - R - ((cy - R - oy) % step)
    ctx.set_line_width(max(0.6, 0.8 * k))
    xs = np.arange(x0, cx + R + 1, step)
    ys = np.arange(y0, cy + R + 1, step)
    for gx in xs:
        ctx.move_to(gx, cy - R); ctx.line_to(gx, cy + R)
    for gy in ys:
        ctx.move_to(cx - R, gy); ctx.line_to(cx + R, gy)
    ctx.set_source(rg)
    ctx.stroke()
    # '+' ticks at intersections (brighter, small)
    tk = 4.0 * k
    for gx in xs:
        for gy in ys:
            ctx.move_to(gx - tk, gy); ctx.line_to(gx + tk, gy)
            ctx.move_to(gx, gy - tk); ctx.line_to(gx, gy + tk)
    ctx.set_line_width(max(0.8, 1.1 * k))
    rg2 = cairo.RadialGradient(cx, cy, R * 0.1, cx, cy, R * 0.85)
    rg2.add_color_stop_rgba(0, col[0], col[1], col[2], min(1, alpha * 2.2))
    rg2.add_color_stop_rgba(1, col[0], col[1], col[2], 0)
    ctx.set_source(rg2)
    ctx.stroke()
    ctx.restore()


def plate_brightness(F, cx, cy, R):
    try:
        from .compositor import surface_array
        arr = surface_array(F.surf)
        x0, y0 = int(max(0, cx - R)), int(max(0, cy - R))
        x1, y1 = int(min(F.W, cx + R)), int(min(F.H, cy + R))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return 0.4
        return float(arr[y0:y1:6, x0:x1:6, :3].mean()) / 255.0
    except Exception:  # noqa: BLE001
        return 0.4


def draw_stage(F, L, cx, cy, size, color="@accent", alpha=1.0):
    """Drawn on the BACK canvas (behind the speaker) — called by the renderer's stage pass."""
    cfg = L.get("stage", STAGE_DEFAULT.get(L["type"], False))
    if not cfg or alpha <= 0.01 or F.pass_ != "back" or F.prepass:
        return
    cfg = cfg if isinstance(cfg, dict) else {}
    ctx = cairo.Context(F.surf)
    R = size * float(cfg.get("radius", 1.8))
    k = (F.head_px() / 300.0) ** 0.5
    lum = plate_brightness(F, cx, cy, R * 0.8)
    stage = G.rgba("@stage")
    col = G.rgba(cfg.get("color", color))
    # 1 scrim: stronger on bright plates
    sa = float(cfg.get("scrim", 0.62)) * min(1.0, 0.35 + lum * 1.3) * alpha
    rg = cairo.RadialGradient(cx, cy, size * 0.2, cx, cy, R)
    rg.add_color_stop_rgba(0, stage[0], stage[1], stage[2], sa)
    rg.add_color_stop_rgba(0.55, stage[0], stage[1], stage[2], sa * 0.7)
    rg.add_color_stop_rgba(1, stage[0], stage[1], stage[2], 0)
    ctx.arc(cx, cy, R, 0, 2 * math.pi); ctx.set_source(rg); ctx.fill()
    # 2 grid
    gcol = G.rgba(cfg.get("grid_color", "@grid"))
    _grid(ctx, F, cx, cy, R * 0.95, gcol, float(cfg.get("grid", 0.0)) * alpha, k)   # v21: grid OFF by default
    # 3 coloured light pool (centre → fades)
    la = float(cfg.get("light", 0.42)) * alpha
    rg = cairo.RadialGradient(cx, cy, 0, cx, cy, R * 0.9)
    rg.add_color_stop_rgba(0, col[0], col[1], col[2], la)
    rg.add_color_stop_rgba(0.4, col[0], col[1], col[2], la * 0.35)
    rg.add_color_stop_rgba(1, col[0], col[1], col[2], 0)
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_SCREEN)
    ctx.arc(cx, cy, R * 0.9, 0, 2 * math.pi); ctx.set_source(rg); ctx.fill()
    ctx.restore()
    if F.ectx is not None:                        # also a soft emission → spills onto the speaker
        e = F.ectx
        rg = cairo.RadialGradient(cx, cy, 0, cx, cy, R * 0.7)
        rg.add_color_stop_rgba(0, col[0], col[1], col[2], la * 0.35)
        rg.add_color_stop_rgba(1, col[0], col[1], col[2], 0)
        e.arc(cx, cy, R * 0.7, 0, 2 * math.pi); e.set_source(rg); e.fill()


def element_alpha(L, F):
    """Envelope (0..1) of a layer's presence, for fading its stage with it."""
    lt = F.t - L["_t0"]
    a = min(1.0, max(0.0, lt / 0.25))
    t1 = L.get("_t1")
    if t1 is not None and F.t > t1:
        a *= max(0.0, 1 - (F.t - t1) / 0.35)
    return ph.SILK(a) if 0 < a < 1 else a
