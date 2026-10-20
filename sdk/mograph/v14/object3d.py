"""
mograph.v14.object3d — `object_3d` layer: a REAL turning 3D object inside the v14 talking-head engine.

    {"type": "object_3d", "t": "w:لپ‌تاپ", "out": "we:قیمت+0.3",
     "model": "croissant",                   # v14.6 PHOTOREAL scanned PBR model (pbr3d) — preferred — or —
     "src": "laptop_3d.png",                 # any RGBA image (assets_3d / project path)  — or —
     "preset": "laptop_pro_3d",              # procedural sculpt3d model (laptop_pro_3d | ozempic_pen_3d | trophy_3d)
     "geometry": "auto",                     # auto | slab | badge | cushion | multiview (image sources)
     "anchor": "hand_r", "offset": [0.2, -0.6],   # or "anchor": "screen", "x": 760, "y": 640
     "size": 1.4,                            # diameter in HEAD WIDTHS when anchored, px when anchor == screen
     "motion": "hover",                      # turntable_osc | spin_360 | spring_entry | hover | none
     "yaw": -20, "pitch": 12,                # base angles;  "turn_on": "w:…" → one 360° spin at that word
     "enter": "spring_pop" | "drop" | "fade",  "exit": "pop_out" | "fall" | "fade",
     "pass": "front" | "back"}

Entrance = jelly spring scale + a quick spin-in (the object TURNS into view — never a flat fade).
Contact shadow + a soft accent glow keep it sitting in the scene; it respects the layout guard (never on the face).
"""
from __future__ import annotations

import math

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


def draw_object_3d(ctx, L, F, pos, out_lt):
    from .. import obj3d as O
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
    ent = L.get("enter", "spring_pop")
    if ent == "fade":
        sc, a = 1.0, min(1.0, lt / 0.3)
    elif ent == "drop":
        p = ph.sp(lt, "bouncy")
        sc, a = 1.0, min(1.0, lt / 0.15)
        y -= (1.0 - p) * F.H * 0.35
    else:
        sc, a = max(0.0, ph.sp(lt, "jelly")), min(1.0, lt / 0.12)
    if olt is not None:
        ex = L.get("exit", "pop_out")
        q = min(1.0, olt / 0.4)
        if ex == "fall":
            y += ph.ease_in(q) * F.H * 0.4
            a *= 1 - q
        elif ex == "fade":
            a *= 1 - q
        else:
            sc *= max(0.0, 1 + 0.15 * math.sin(q * math.pi) - q * 1.15)
            a *= 1 - ph.ease_in(q)
    size *= sc
    if size < 6 or a <= 0.01:
        return
    yaw, pitch = _angles(L, lt)
    roll = float(L.get("roll", 0.0)) + float(L.get("_roll", 0.0)) * float(L.get("roll_follow", 0.5))
    base = getattr(F.R, "base", None) or getattr(F.R, "base_dir", None)
    # soft accent glow behind (sits the object in the light of the scene) — emission pass when available
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
    if L.get("model"):                         # v14.6 photoreal: scanned PBR model rendered by EffectCraft
        _draw_pbr(ctx, L, F, x, y, size, sc, a, shadow)
        return
    if L.get("preset") or L.get("tree"):
        from .. import sculpt3d as S
        tree = L.get("_tree")
        if tree is None:                       # resolve once (preset name | tree dict | tree .json path)
            tree = L["_tree"] = S.get_tree(L.get("tree"), L.get("preset", "laptop_pro_3d"), base_dir=base,
                                           open_dur=float(L.get("open_dur", 1.1)))
        S.render_sculpt3d_scene(ctx, tree, x, y, size, t=lt, yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
                                alpha=a, shadow=shadow, ssaa=int(L.get("ssaa", 2)))
        return
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
