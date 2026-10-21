"""
mograph.v14.object3d — `object_3d` layer: a REAL turning 3D object inside the v14 talking-head engine.

    {"type": "object_3d", "t": "w:لپ‌تاپ", "out": "we:قیمت+0.3",
     "three_model": "hourglass_3d",          # v20 PROCEDURAL Three.js 3D mesh (360° true PBR + glass) — or —
     "three_script": "custom_model.js",      # v20 custom JavaScript Three.js procedural generator script — or —
     "model": "croissant",                   # v14.6 PHOTOREAL scanned PBR model (pbr3d) — or —
     "preset": "laptop_pro_3d",              # procedural sculpt3d model (laptop_pro_3d | ozempic_pen_3d | trophy_3d) — or —
     "src": "laptop_3d.png",                 # auto-upgraded to three3d when a matching 3D model exists!
     "geometry": "auto",                     # auto | three3d | slab | badge | cushion | multiview
     "anchor": "hand_r", "offset": [0.2, -0.6],   # or "anchor": "screen", "x": 760, "y": 640
     "size": 1.4,                            # diameter in HEAD WIDTHS when anchored, px when anchor == screen
     "motion": "hover",                      # turntable_osc | spin_360 | spring_entry | hover | none
     "yaw": -20, "pitch": 12,                # base angles;  "turn_on": "w:…" → one 360° spin at that word
     "explode": 0.0,                         # v20: sub-part explosion factor (0.0 .. 1.0)
     "anim": "hourglass_loop",               # v20.1 procedural animation INSIDE the model (mograph.three_anim):
                                             #   hourglass_loop (pour → empty → flip → pour, "flips_at": ["w:…"])
                                             #   laptop_open ("close_at") · assemble · any model: "explode_at"
     "enter": "spring_pop" | "drop" | "fade",  "exit": "pop_out" | "fall" | "fade",
     "pass": "front" | "back"}

v20 upgrade:
  * Supports `"three_model"` and `"three_script"` powered by `mograph.three3d` (true 360° procedural Three.js geometry
    with two-pass metallic/glass PBR shading).
  * Automatically upgrades known `"src"` sprites (`hourglass_3d.png`, `pen_3d.png`, `tree_3d.png`, `star_3d.png`,
    `shield_3d.png`, `cash_3d.png`, `warn_3d.png`, `laptop_3d.png`) to true 3D procedural meshes unless
    `"allow_slab": true` is explicitly set.
  * Skips heavy 3D rasterisation during the 1×1 dummy emission prepass (`F.prepass`), cutting render time by ~2×.
"""
from __future__ import annotations

import math
import os

from . import physics as ph


def _angles(L, lt):
    from ..obj3d import motion_angles
    yaw0 = float(L.get("yaw", -18.0))
    pitch0 = float(L.get("pitch", 12.0))
    el = {"motion": L.get("motion", "hover"), "yaw_speed": L.get("yaw_speed", 75.0),
          "yaw_amp": L.get("yaw_amp"), "pitch_amp": L.get("pitch_amp"), "speed": L.get("speed", 2.0)}
    el = {k: v for k, v in el.items() if v is not None}
    yaw, pitch = motion_angles(el, lt, yaw0, pitch0)
    # spin-in: the object arrives already turning and settles (monotonic, eased) — the v14 "never fade" rule
    spin_in = float(L.get("spin_in", 200.0))
    d = float(L.get("spin_dur", 0.9))
    if lt < d:
        p = ph.ease_out(max(0.0, lt) / d)
        yaw += (1.0 - p) * -spin_in
    if L.get("_turn_on") is not None:            # one full turn on a word (emphasis)
        lt2 = L["_t_now"] - L["_turn_on"]
        if 0 <= lt2 <= 0.8:
            yaw += 360.0 * ph.ease_in_out(lt2 / 0.8)
    return yaw, pitch


def _motion_state(L, F, lt, olt):
    """(scale, alpha, dy) of the entrance / exit at local time lt (pure function → also used for motion blur)."""
    ent = L.get("enter", "spring_pop")
    dy = 0.0
    if ent == "fade":
        sc, a = 1.0, min(1.0, max(0.0, lt) / 0.3)
    elif ent == "drop":
        p = ph.sp(lt, "bouncy")
        sc, a = 1.0, min(1.0, max(0.0, lt) / 0.15)
        dy -= (1.0 - p) * F.H * 0.35
    else:
        sc, a = max(0.0, ph.sp(lt, "jelly")), min(1.0, max(0.0, lt) / 0.12)
    if olt is not None and olt > 0:
        ex = L.get("exit", "pop_out")
        q = min(1.0, olt / 0.4)
        if ex == "fall":
            dy += ph.ease_in(q) * F.H * 0.4
            a *= 1 - q
        elif ex == "fade":
            a *= 1 - q
        else:
            sc *= max(0.0, 1 + 0.15 * math.sin(q * math.pi) - q * 1.15)
            a *= 1 - ph.ease_in(q)
    return sc, a, dy


def _blur_vector(L, F, lt, olt, size, yaw, sc, dy):
    """v14.8: screen displacement over one 180° shutter interval: spin-in rim speed (horizontal, from the yaw
    derivative) + drop / fall translation (vertical) + spring scale growth (radial, folded into the dominant axis)."""
    from . import motionblur as MB
    cfg = MB.config(L, F)
    if cfg is None or F.prepass:
        return None
    dt = MB.shutter_dt(cfg, F.R.fps)
    olt_p = (olt - dt) if (olt is not None and olt - dt > 0) else None
    sc_p, _, dy_p = _motion_state(L, F, lt - dt, olt_p)
    keep = L.get("_t_now")
    L["_t_now"] = F.t - dt
    yaw_p, _ = _angles(L, lt - dt)
    L["_t_now"] = keep
    dyaw = math.radians(((yaw - yaw_p + 180.0) % 360.0) - 180.0)
    vx = abs(dyaw) * size * 0.5 * 0.85                 # rim speed of the turning object (≈ visible silhouette)
    vy = dy - dy_p
    vs = abs(sc - sc_p) * size * 0.5 / max(sc, 0.2)
    vx = vx if dyaw >= 0 else -vx
    if abs(vy) >= abs(vx):
        vy += math.copysign(vs, vy if vy else 1.0)
    else:
        vx += math.copysign(vs, vx)
    if math.hypot(vx, vy) < float(cfg["threshold"]):
        return None
    return vx, vy, float(cfg["max"])


def draw_object_3d(ctx, L, F, pos, out_lt):
    # Skip ONLY the compositor's 1×1 emission pre-pass (F.ectx set).  v14/qa.py also sets F.prepass=True but renders
    # on a full-size A8 surface with F.ectx=None to measure ink (frame-edge / face checks) — it must still see us.
    if getattr(F, "prepass", False) and getattr(F, "ectx", None) is not None:
        return
    from .. import obj3d as O
    from .. import three3d as T3
    lt = F.t - L["_t0"]
    olt = out_lt(L, F)
    if lt < 0 or (olt is not None and olt > 0.5):
        return
    L["_t_now"] = F.t
    x, y, k = pos(L, F)
    if L.get("anchor", "screen") == "screen":
        size = float(L.get("size", 360))
    else:
        size = float(L.get("size", 1.4)) * F.head_px()
    # entrance / exit
    sc, a, dy = _motion_state(L, F, lt, olt)
    y += dy
    size *= sc
    if size < 6 or a <= 0.01:
        return
    yaw, pitch = _angles(L, lt)
    roll = float(L.get("roll", 0.0)) + float(L.get("_roll", 0.0)) * float(L.get("roll_follow", 0.5))
    base = getattr(F.R, "base", None) or getattr(F.R, "base_dir", None)
    vec = _blur_vector(L, F, lt, olt, size, yaw, sc, dy)

    def _paint(ctx):
        glow = float(L.get("glow", 0.35))
        if glow > 0 and F.pass_ == L.get("pass", "front"):
            import cairo
            from . import glyphs as G
            c = G.rgba(L.get("glow_color", "@accent"), L["_pal"])
            g = cairo.RadialGradient(x, y, 0, x, y, size * 0.75)
            g.add_color_stop_rgba(0, c[0], c[1], c[2], 0.28 * glow * a)
            g.add_color_stop_rgba(1, c[0], c[1], c[2], 0)
            ctx.save()
            ctx.set_source(g)
            ctx.arc(x, y, size * 0.75, 0, 2 * math.pi)
            ctx.fill()
            ctx.restore()
        shadow = bool(L.get("shadow", True))

        # 1. Explicit v20 Three.js procedural model or .js script
        t_script = L.get("three_script")
        if t_script:
            sp = T3.resolve_script_ref(t_script, base)
            T3.paint(ctx, sp, x, y, size, yaw=yaw, pitch=pitch, roll=roll,
                     explode=float(L.get("explode", 0.0)), alpha=a, shadow=shadow, ssaa=int(L.get("ssaa", 2)))
            return
        t_model = L.get("three_model")
        if not t_model and not L.get("allow_slab", False) and L.get("geometry", "auto") in ("auto", "three3d"):
            t_model = T3.resolve_model_name(L.get("src"))
        if t_model:
            acfg = L.get("_anim_cfg")
            if acfg is None and (L.get("anim") or L.get("explode_at")):
                from .. import three_anim as TA
                acfg = L["_anim_cfg"] = TA.config(L) or False
                if acfg:
                    for k in ("_flips", "_close", "_explode_at"):          # word times → layer-local seconds
                        src = {"_flips": "_flips_at", "_close": "_close_at", "_explode_at": "_explode_at"}[k]
                        v = L.get(src)
                        if v is not None:
                            acfg[k] = [float(u) - L["_t0"] for u in v] if isinstance(v, list) else float(v) - L["_t0"]
            if acfg:
                from .. import three_anim as TA
                TA.paint(ctx, t_model, acfg, lt, x, y, size, yaw=yaw, pitch=pitch, roll=roll,
                         explode=float(L.get("explode", 0.0)), alpha=a, shadow=shadow, ssaa=int(L.get("ssaa", 2)),
                         t_end=L["_t1"] - L["_t0"] + 1.0)
                return
            T3.paint(ctx, t_model, x, y, size, yaw=yaw, pitch=pitch, roll=roll,
                     explode=float(L.get("explode", 0.0)), alpha=a, shadow=shadow, ssaa=int(L.get("ssaa", 2)))
            return

        # 2. v14.6 Photoreal scanned PBR model (EffectCraft)
        if L.get("model"):
            _draw_pbr(ctx, L, F, x, y, size, sc, a, shadow)
            return

        # 3. v14.4 Sculpt3D component tree
        if L.get("preset") or L.get("tree"):
            from .. import sculpt3d as S
            tree = L.get("_tree")
            if tree is None:
                tree = L["_tree"] = S.get_tree(L.get("tree"), L.get("preset", "laptop_pro_3d"), base_dir=base,
                                               open_dur=float(L.get("open_dur", 1.1)))
            S.render_sculpt3d_scene(ctx, tree, x, y, size, t=lt, yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
                                    alpha=a, shadow=shadow, ssaa=int(L.get("ssaa", 2)))
            return

        # 4. Multiview or fallback 2D sprite slab extrusion
        src = L.get("src", "laptop_3d.png")
        geom = L.get("geometry", "auto")
        views = L.get("views") or (O.find_views(src, base) if geom in ("auto", "multiview") else None)
        if views:
            O.render_multiview(ctx, views, x, y, size, yaw=yaw, pitch=pitch, roll=roll, alpha=a, shadow=shadow,
                               base_dir=base)
            return
        if geom == "auto":
            low = str(src).lower()
            geom = "cushion" if any(s in low for s in ("pin", "heart", "warn", "cash", "coin")) else "slab"
        prof = {"slab": (0.12, 0.30, 0.0), "badge": (0.06, 0.6, 0.0), "cushion": (0.22, 1.0, 0.45)}.get(geom, (0.12, 0.3, 0))
        O.render_extruded_sprite(ctx, src, x, y, size, yaw=yaw, pitch=pitch, roll=roll,
                                 depth=float(L.get("depth", prof[0])), round_=float(L.get("round", prof[1])),
                                 bulge=float(L.get("bulge", prof[2])), alpha=a, specular=float(L.get("specular", 0.7)),
                                 shadow=shadow, back=L.get("back", "solid"), base_dir=base)

    if vec is not None:                            # v14.8: 180° shutter streak during spin-in / drop / fall
        from .motionblur import paint_blurred
        r = size * 0.85
        paint_blurred(ctx, _paint, (x - r, y - r, x + r, y + r * 1.25), vec[0], vec[1], vec[2])
    else:
        _paint(ctx)


def _draw_pbr(ctx, L, F, x, y, size, sc, a, shadow):
    """`"model": "croissant" | "my.glb"` → pbr3d.  The whole on-screen interval is rendered in ONE EffectCraft call
    at the BASE size (angles incl. spin-in / turn_on baked per frame); springs only scale the sprite."""
    from .. import pbr3d
    if "_pbr" not in L:
        t0 = L["_t0"]
        t1 = min(F.R.duration, L["_t1"] + 0.5)
        if L.get("anchor", "screen") != "screen":
            base_size = float(L.get("size", 1.4)) * F.head_px()
        else:
            base_size = float(L.get("size", 360))
        L["_pbr_size"] = base_size = int(round(min(base_size, 1400)))

        def ang(lt):
            L["_t_now"] = t0 + lt
            yaw, pitch = _angles(L, lt)
            return yaw, pitch, float(L.get("roll", 0.0))
        base = getattr(F.R, "base", None) or getattr(F.R, "base_dir", None)
        L["_pbr"] = pbr3d.track(L["model"], base_size, ang, t0, t1, float(F.R.fps), hdri=L.get("hdri"),
                                light=L.get("light"), shadow=shadow, env_rot=float(L.get("env_rot", 0.0)),
                                base_dir=base)
    pbr3d.paint_scaled(ctx, L["_pbr"](F.t), x, y, size / L["_pbr_size"], a, float(L.get("shadow_strength", 1.0)))
