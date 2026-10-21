"""
mograph.motion.character — rubber-hose / cutout cartoon characters driven by the animation principles.

A character = a RIG (bones + parts, cutout style) + ACTIONS on a timeline + optional LIP SYNC.
    {"type": "character", "rig": "bean", "x": 540, "y": 1500, "size": 520, "color": "#ffcc33",
     "actions": [{"do": "enter", "t": 0.0, "from": "left"}, {"do": "hop", "t": 1.2}, {"do": "wave", "t": 2.0, "dur": 1.4},
                 {"do": "talk", "t": 3.0, "dur": 2.5}, {"do": "point", "t": 5.6, "side": "r"}, {"do": "exit", "t": 8.0}],
     "lip_sync": {"words": "words.json", "audio": "media/raw.mp4", "offset": 0.0},     # or "visemes": [...]
     "timing": "twos",   # ones | twos | threes  (limited animation: pose held 2/3 frames — the cartoon feel)
     "style": "rubber_hose", "exaggeration": 1.0, "face": "happy", "blink": true}

PRINCIPLES (all on by default, each is a measurable number — see `principles_report()`):
    squash_and_stretch   body scale sy = 1/sx (volume kept) from vertical velocity + landing impact
    anticipation         every hop / jump / point starts with a counter-move (crouch / wind-up) ≈ 25 % of the action
    follow_through       antenna / hair / arms lag the body with a damped spring (overlapping_action)
    overlapping_action   limbs start/stop at different times (arm delay 2 frames, hand 4)
    secondary_action     blink, breathing, idle sway while the main action plays
    arcs                 hands / jump paths follow parabolas / circular arcs, never straight lines
    exaggeration         "exaggeration": 0.5 … 2.0 multiplies squash, anticipation depth, overshoot
    timing               spacing from motion.easing curves; "timing": twos/threes = stepped exposure
    staging              the character turns 3/4 toward its action target and is kept inside the safe frame

Rigs: "bean" (jelly-bean body, rubber-hose arms & legs, big eyes) · "blob" (no legs, bouncy slime) ·
      "star" (5-point mascot) · custom: {"rig": {"body": "circle|bean|rect", "w": .., "h": .., "arms": true, ...}}
Actions: idle enter exit hop jump walk wave point talk shrug nod shake surprise celebrate look spin squash
"""
from __future__ import annotations

import math

import numpy as np

from .easing import ease as _ease

FPS = 30.0
ACTIONS = ("idle", "enter", "exit", "hop", "jump", "walk", "wave", "point", "talk", "shrug", "nod", "shake",
           "surprise", "celebrate", "look", "spin", "squash")
ACTION_DUR = {"hop": 0.7, "jump": 1.1, "wave": 1.4, "point": 1.0, "talk": 2.0, "shrug": 0.9, "nod": 0.8,
              "shake": 0.9, "surprise": 0.9, "celebrate": 1.6, "look": 1.0, "spin": 1.0, "squash": 0.6,
              "walk": 2.0, "enter": 0.8, "exit": 0.7, "idle": 1.0}
RIGS = {
    "bean": dict(body="bean", w=0.62, h=0.80, arms=True, legs=True, eyes=2, antenna=True, ink="#1a1014"),
    "blob": dict(body="blob", w=0.86, h=0.62, arms=True, legs=False, eyes=2, antenna=False, ink="#10141a"),
    "star": dict(body="star", w=0.86, h=0.86, arms=False, legs=True, eyes=2, antenna=False, ink="#1a1408"),
}


def _hexf(c):
    c = str(c).lstrip("#")
    return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def damped(t, freq=2.2, zeta=0.18):
    """impulse response of an under-damped spring (follow-through wobble), 0 at t=0, decays."""
    if t <= 0:
        return 0.0
    w = 2 * math.pi * freq
    return math.exp(-zeta * w * t) * math.sin(w * math.sqrt(1 - zeta * zeta) * t)


# =============================================================================================== pose solver
class Pose:
    __slots__ = ("x", "y", "sx", "sy", "rot", "lean", "arm_l", "arm_r", "leg_l", "leg_r", "eye", "brow", "look",
                 "mouth", "antenna", "alpha", "face", "flip")

    def __init__(self):
        self.x = self.y = 0.0            # offset in body heights
        self.sx = self.sy = 1.0
        self.rot = self.lean = 0.0       # deg
        self.arm_l = [200.0, 0.0]        # [shoulder angle deg (0 = right, 90 = down), elbow bend deg]
        self.arm_r = [-20.0, 0.0]
        self.leg_l = [0.0, 0.0]          # [hip swing deg, knee]
        self.leg_r = [0.0, 0.0]
        self.eye = 1.0                   # openness 0..1.3
        self.brow = 0.0                  # -1 angry … +1 surprised
        self.look = [0.0, 0.0]
        self.mouth = None                # lip-sync geometry or None (face expression)
        self.antenna = 0.0               # follow-through angle
        self.alpha = 1.0
        self.face = "happy"
        self.flip = 1.0


def _win(t, a):
    """local progress of action a at time t (None if outside)."""
    t0 = a["_t"]
    d = a["_d"]
    if t < t0 or t > t0 + d + a.get("_tail", 0.0):
        return None
    return (t - t0) / d


def _anticip(p, share=0.25, depth=1.0):
    """anticipation curve: dips to -depth·0.3 in the first `share`, then overshoots to 1 (out_back), settles."""
    if p < share:
        q = p / share
        return -0.3 * depth * math.sin(q * math.pi / 2) ** 2
    q = (p - share) / (1 - share)
    return -0.3 * depth + (1 + 0.3 * depth) * _ease({"type": "back", "dir": "out", "overshoot": 1.4 * depth})(q)


def solve(C, t):
    """pose at time t (pure function: frames render in any order)."""
    P = Pose()
    ex = float(C.get("exaggeration", 1.0))
    acts = C["_acts"]
    # ---- secondary: breathing + idle sway, blink
    P.sy += 0.018 * math.sin(t * 2 * math.pi * 0.35)
    P.sx -= 0.018 * math.sin(t * 2 * math.pi * 0.35)
    P.rot += 1.6 * math.sin(t * 2 * math.pi * 0.21 + 0.6)
    P.arm_l = [200.0 + 4 * math.sin(t * 1.7), 18.0]
    P.arm_r = [-20.0 - 4 * math.sin(t * 1.7 + 0.4), -18.0]
    P.face = C.get("face", "happy")
    if C.get("blink", True):
        bt = (t + float(C.get("seed", 0)) * 0.73) % 3.3
        if bt < 0.12:
            P.eye = abs(bt - 0.06) / 0.06
    vy = 0.0                    # vertical velocity proxy (body heights / s) for squash & stretch
    impact = 0.0
    for a in acts:
        p = _win(t, a)
        if p is None:
            continue
        do = a["do"]
        pc = min(1.0, max(0.0, p))
        lt = t - a["_t"]
        if do == "enter":
            side = {"left": -1, "right": 1, "bottom": 0, "top": 0}.get(a.get("from", "bottom"), 0)
            g = _ease("out_back")(pc)
            if side:
                P.x += side * 2.2 * (1 - g)
                P.lean += -side * 14 * (1 - pc) * ex
            else:
                P.y += 1.6 * (1 - _ease("out_cubic")(pc))
                if pc > 0.55:
                    impact = max(impact, damped(lt - 0.55 * a["_d"], 3.0, 0.25) * 0.9)
            P.alpha *= min(1.0, pc * 4)
        elif do == "exit":
            side = {"left": -1, "right": 1}.get(a.get("to", "right"), 1)
            q = _anticip(pc, 0.3, ex)
            P.x += side * 2.4 * max(0.0, q)
            P.lean += side * 18 * max(0.0, q) * ex - side * 8 * min(0.0, q) * 3
            P.sx *= 1 + 0.25 * max(0.0, min(1.0, q)) * ex
            P.sy /= 1 + 0.25 * max(0.0, min(1.0, q)) * ex
            P.alpha *= 1.0 - max(0.0, (pc - 0.75) * 4)
        elif do in ("hop", "jump"):
            hgt = (0.45 if do == "hop" else 1.0) * float(a.get("height", 1.0)) * ex
            sh = 0.28                                      # anticipation share
            land = 0.82
            if pc < sh:                                    # crouch
                q = math.sin(pc / sh * math.pi / 2)
                P.sy *= 1 - 0.22 * q * ex
                P.sx *= 1 + 0.22 * q * ex
                P.arm_l[0], P.arm_r[0] = 200 - 30 * q, -20 + 30 * q
            elif pc < land:                                # air: parabola (arc), stretch on the way up
                q = (pc - sh) / (land - sh)
                P.y -= hgt * 4 * q * (1 - q)
                vy = hgt * 4 * (1 - 2 * q) / ((land - sh) * a["_d"])
                P.arm_l[0], P.arm_r[0] = 150 - 40 * (1 - q), 30 + 40 * (1 - q)
                if do == "jump" and a.get("flip", False):
                    P.rot += 360 * _ease("in_out_cubic")(q)
                P.x += float(a.get("dx", 0.0)) * q
            else:                                          # land: squash impact then settle
                P.x += float(a.get("dx", 0.0))
                impact = max(impact, damped(lt - land * a["_d"], 3.2, 0.22) * (0.8 + 0.4 * hgt))
            P.leg_l[1] = P.leg_r[1] = 30 * (1 - min(1, abs(pc - 0.55) * 4)) if sh < pc < land else 0
        elif do == "walk":
            spd = float(a.get("speed", 1.0))
            ph = lt * 2 * math.pi * 1.6 * spd
            P.x += float(a.get("dx", 1.5)) * pc
            P.y -= 0.05 * abs(math.sin(ph))
            P.leg_l[0], P.leg_r[0] = 28 * math.sin(ph), -28 * math.sin(ph)
            P.arm_l[0] = 200 - 25 * math.sin(ph)
            P.arm_r[0] = -20 - 25 * math.sin(ph)
            P.lean += 6 * spd * (1 if a.get("dx", 1.5) > 0 else -1)
            vy = -0.4 * math.cos(ph) * 2
        elif do == "wave":
            side = a.get("side", "r")
            up = _ease("out_back")(min(1.0, pc * 4)) * (1 - _ease("in_cubic")(max(0.0, pc - 0.85) / 0.15))
            sw = math.sin(lt * 2 * math.pi * 2.6) * 22 * ex * up
            if side == "r":
                P.arm_r = [-20 - 45 * up, -55 * up + sw]
            else:
                P.arm_l = [200 + 45 * up, 55 * up - sw]
            P.lean += (4 if side == "r" else -4) * up
            P.face = "happy"
        elif do == "point":
            side = a.get("side", "r")
            q = _anticip(pc if pc < 0.6 else 0.6, 0.35, ex) if pc < 0.6 else 1.0
            back = 1 - _ease("in_out_cubic")(max(0.0, pc - 0.8) / 0.2)
            ang = (40 - 45 * q) * back + (-20) * (1 - back)        # wind-up down → snap out horizontal
            if side == "r":
                P.arm_r = [ang, 0.0]
            else:
                P.arm_l = [180 - ang, 0.0]
            P.lean += (10 if side == "r" else -10) * q * back
            P.look = [0.6 * (1 if side == "r" else -1) * back, -0.1 * back]
        elif do == "talk":
            P.rot += 3 * math.sin(lt * 2 * math.pi * 1.3)
            P.y -= 0.015 * abs(math.sin(lt * 2 * math.pi * 3.1))
            P.arm_r = [-20 - 35 * (0.5 + 0.5 * math.sin(lt * 4.1)), -30]
            if "_vis" not in C:                            # no lip sync → procedural chatter (U-AI-etc cycle)
                cyc = ["AI", "etc", "E", "O", "MBP", "AI", "U", "etc"]
                from .lipsync import MOUTHS
                P.mouth = dict(MOUTHS[cyc[int(lt * 9) % len(cyc)]], v="auto")
        elif do == "shrug":
            q = math.sin(pc * math.pi)
            P.arm_l = [200 - 55 * q, 70 * q]
            P.arm_r = [-20 + 55 * q, -70 * q]
            P.sy *= 1 - 0.06 * q
            P.y += 0.04 * q
            P.brow = 0.6 * q
            P.face = "meh"
        elif do == "nod":
            P.rot += 0
            P.y += 0.05 * math.sin(pc * 2 * math.pi * 2) * (1 - pc)
            P.sy *= 1 - 0.05 * abs(math.sin(pc * 2 * math.pi * 2)) * (1 - pc)
            P.look[1] = 0.25 * math.sin(pc * 2 * math.pi * 2) * (1 - pc)
        elif do == "shake":
            P.rot += 9 * math.sin(pc * 2 * math.pi * 3) * (1 - pc) * ex
            P.look[0] = 0.5 * math.sin(pc * 2 * math.pi * 3) * (1 - pc)
            P.face = "sad"
        elif do == "surprise":
            q = _anticip(pc, 0.2, ex)
            P.sy *= 1 + 0.18 * max(0.0, q) * ex
            P.sx /= 1 + 0.18 * max(0.0, q) * ex
            P.y -= 0.12 * max(0.0, q)
            P.eye = 1.0 + 0.3 * max(0.0, q)
            P.brow = 1.0
            P.arm_l = [200 + 70 * max(0.0, q), 30]
            P.arm_r = [-20 - 70 * max(0.0, q), -30]
            P.face = "o"
            impact = max(impact, damped(lt - 0.2 * a["_d"], 2.6, 0.2) * 0.4)
        elif do == "celebrate":
            ph = lt * 2 * math.pi * 2.2
            P.arm_l = [200 + 120 + 15 * math.sin(ph), 20]
            P.arm_r = [-20 - 120 - 15 * math.sin(ph + 1), -20]
            hop = abs(math.sin(ph / 2)) * 0.3 * ex
            P.y -= hop
            vy = -math.cos(ph / 2) * 0.3 * 2.2 * math.pi * np.sign(math.sin(ph / 2))
            P.face = "joy"
        elif do == "look":
            d = a.get("dir", [1, 0])
            q = _ease("out_cubic")(min(1.0, pc * 3)) * (1 - _ease("in_out_cubic")(max(0.0, pc - 0.8) / 0.2))
            P.look = [float(d[0]) * 0.7 * q, float(d[1]) * 0.5 * q]
            P.lean += float(d[0]) * 6 * q
        elif do == "spin":
            P.rot += 360 * _ease("in_out_back")(pc) * (1 if a.get("cw", True) else -1)
            P.sx *= 1 - 0.15 * math.sin(pc * math.pi)
        elif do == "squash":
            impact = max(impact, damped(lt, 2.8, 0.2) * float(a.get("amount", 1.0)))
    # ---- squash & stretch from velocity + impact (volume preserved)
    st = max(-0.32, min(0.32, 0.06 * vy * ex))
    imp = max(-0.38, min(0.38, -0.34 * impact * ex))
    k = (1 + st) * (1 + imp)
    P.sy *= k
    P.sx /= k
    # ---- follow-through: antenna lags rotation / vertical motion
    P.antenna = -P.lean * 1.4 + (-st * 120) + impact * 30 + 6 * math.sin(t * 3.1)
    return P


# =============================================================================================== drawing
def _bean_path(ctx, w, h, kind):
    if kind == "blob":
        ctx.move_to(-w / 2, 0)
        ctx.curve_to(-w / 2, -h * 0.75, w / 2, -h * 0.75, w / 2, 0)
        ctx.curve_to(w / 2, h * 0.32, -w / 2, h * 0.32, -w / 2, 0)
    elif kind == "star":
        for i in range(10):
            a = -math.pi / 2 + i * math.pi / 5
            r = (0.5 if i % 2 == 0 else 0.24) * w
            (ctx.move_to if i == 0 else ctx.line_to)(math.cos(a) * r, math.sin(a) * r * h / w)
        ctx.close_path()
        return
    else:                                                   # bean: wider bottom, rounded top
        ctx.move_to(0, -h / 2)
        ctx.curve_to(w * 0.52, -h / 2, w * 0.58, h * 0.1, w * 0.48, h * 0.32)
        ctx.curve_to(w * 0.38, h * 0.52, -w * 0.38, h * 0.52, -w * 0.48, h * 0.32)
        ctx.curve_to(-w * 0.58, h * 0.1, -w * 0.52, -h / 2, 0, -h / 2)
    ctx.close_path()


def _hose(ctx, x0, y0, ang, bend, L, lw, ink, glove=True):
    """rubber-hose limb: a single smooth bezier noodle (no elbow) ending in a cartoon glove / shoe."""
    a = math.radians(ang)
    b = math.radians(ang + bend)
    x1, y1 = x0 + math.cos(a) * L * 0.55, y0 + math.sin(a) * L * 0.55
    x2, y2 = x1 + math.cos(b) * L * 0.5, y1 + math.sin(b) * L * 0.5
    ctx.move_to(x0, y0)
    ctx.curve_to(x0 + math.cos(a) * L * 0.3, y0 + math.sin(a) * L * 0.3, x1, y1, x2, y2)
    ctx.set_source_rgb(*ink)
    ctx.set_line_width(lw)
    ctx.set_line_cap(1)
    ctx.stroke()
    if glove:
        ctx.arc(x2, y2, lw * 1.45, 0, 2 * math.pi)
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        ctx.set_source_rgb(*ink)
        ctx.set_line_width(lw * 0.45)
        ctx.stroke()
    return x2, y2


def draw(ctx, C, t, x, y, size, alpha=1.0):
    """draw the character at screen (x, y) = feet centre, `size` = body height px."""
    from .lipsync import draw_mouth, shape_at
    tq = quantise(t, C.get("timing", "ones"), C.get("fps", FPS))
    P = solve(C, tq)
    rig = C["_rig"]
    H = size
    W = H * rig["w"] / rig["h"]
    ink = _hexf(rig.get("ink", "#1a1014"))
    body = _hexf(C.get("color", "#ffcc33"))
    belly = tuple(min(1, c * 0.75 + 0.25) for c in body)
    lw = max(2.0, H * 0.035)
    a = alpha * P.alpha
    if a <= 0.003:
        return None
    if P.mouth is None and C.get("_vis"):
        P.mouth = shape_at(C["_vis"], tq)
    ctx.save()
    ctx.push_group()
    ctx.translate(x + P.x * H, y + P.y * H)
    # contact shadow (shrinks when airborne)
    air = max(0.0, -P.y)
    ctx.save()
    ctx.scale(1, 0.18)
    ctx.arc(0, 0, W * 0.55 * (1 - 0.5 * min(1, air)), 0, 2 * math.pi)
    ctx.restore()
    ctx.set_source_rgba(0, 0, 0, 0.25 * (1 - 0.6 * min(1, air)))
    ctx.fill()
    leg_h = H * 0.22 if rig.get("legs") else 0.0
    # squash/stretch pivot at the feet
    ctx.rotate(math.radians(P.lean * 0.6))
    ctx.scale(P.sx, P.sy)
    # legs
    if rig.get("legs"):
        for side, lg in ((-1, P.leg_l), (1, P.leg_r)):
            hx = side * W * 0.18
            a0 = 90 + lg[0]
            ex_, ey_ = _hose(ctx, hx, -leg_h, a0, lg[1] * side, leg_h * 1.05, lw * 1.1, ink, glove=False)
            ctx.save()
            ctx.translate(ex_ + side * lw * 0.9, ey_)
            ctx.scale(1, 0.55)
            ctx.arc(0, 0, lw * 2.1, 0, 2 * math.pi)
            ctx.restore()
            ctx.set_source_rgb(*ink)
            ctx.fill()
    # body
    ctx.translate(0, -leg_h - H * 0.40)
    ctx.rotate(math.radians(P.rot + P.lean * 0.4))
    sh_y = -H * 0.02
    if rig.get("arms"):                                     # back arm behind the body
        _hose(ctx, -W * 0.40, sh_y, P.arm_l[0], P.arm_l[1], H * 0.42, lw * 1.05, ink)
    _bean_path(ctx, W, H * 0.8, rig["body"])
    path = ctx.copy_path()
    ctx.set_source_rgb(*body)
    ctx.fill_preserve()
    ctx.save()
    ctx.clip()
    ctx.arc(0, H * 0.28, W * 0.36, 0, 2 * math.pi)          # belly patch
    ctx.set_source_rgb(*belly)
    ctx.fill()
    ctx.arc(-W * 0.18, -H * 0.24, W * 0.12, 0, 2 * math.pi)  # specular
    ctx.set_source_rgba(1, 1, 1, 0.35)
    ctx.fill()
    ctx.restore()
    ctx.new_path()
    ctx.append_path(path)
    ctx.set_source_rgb(*ink)
    ctx.set_line_width(lw)
    ctx.stroke()
    if rig.get("arms"):
        _hose(ctx, W * 0.40, sh_y, P.arm_r[0], P.arm_r[1], H * 0.42, lw * 1.05, ink)
    # antenna (follow-through)
    if rig.get("antenna"):
        ang = math.radians(-90 + P.antenna)
        bx, by = 0, -H * 0.40
        L = H * 0.22
        tx, ty = bx + math.cos(ang) * L, by + math.sin(ang) * L
        ctx.move_to(bx, by)
        ctx.curve_to(bx, by - L * 0.5, tx - math.cos(ang) * L * 0.3, ty, tx, ty)
        ctx.set_source_rgb(*ink)
        ctx.set_line_width(lw * 0.6)
        ctx.stroke()
        ctx.arc(tx, ty, lw * 1.3, 0, 2 * math.pi)
        ctx.set_source_rgb(*_hexf(C.get("accent", "#ff4f6d")))
        ctx.fill_preserve()
        ctx.set_source_rgb(*ink)
        ctx.set_line_width(lw * 0.4)
        ctx.stroke()
    # face
    fy = -H * 0.12
    lx, ly = P.look[0] * W * 0.08, P.look[1] * H * 0.05
    er = W * 0.13
    for side in (-1, 1):
        ex0 = side * W * 0.17 + lx * 0.5
        ctx.save()
        ctx.translate(ex0, fy + ly * 0.5)
        ctx.scale(1, max(0.06, 1.15 * P.eye))
        ctx.arc(0, 0, er, 0, 2 * math.pi)
        ctx.restore()
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        ctx.set_source_rgb(*ink)
        ctx.set_line_width(lw * 0.6)
        ctx.stroke()
        if P.eye > 0.25:
            ctx.arc(ex0 + lx + side * 0.0, fy + ly + er * 0.1, er * 0.48, 0, 2 * math.pi)
            ctx.set_source_rgb(*ink)
            ctx.fill()
            ctx.arc(ex0 + lx - er * 0.18, fy + ly - er * 0.12, er * 0.15, 0, 2 * math.pi)
            ctx.set_source_rgb(1, 1, 1)
            ctx.fill()
        # brow
        by_ = fy - er * 1.5 - P.brow * er * 0.5
        ctx.move_to(ex0 - er * 0.8, by_ + side * P.brow * -er * 0.0 + (er * 0.25 if P.face == "sad" else 0) * -side)
        ctx.line_to(ex0 + er * 0.8, by_ + (er * 0.25 if P.face == "sad" else 0) * side)
        ctx.set_source_rgb(*ink)
        ctx.set_line_width(lw * 0.7)
        ctx.stroke()
    my = fy + H * 0.17
    if P.mouth is not None:
        draw_mouth(ctx, P.mouth, lx * 0.3, my, W * 0.36, ink=ink, lip=(0.85, 0.3, 0.35))
    else:
        mw = W * 0.30
        ctx.set_source_rgb(*ink)
        ctx.set_line_width(lw * 0.8)
        ctx.set_line_cap(1)
        if P.face in ("happy", "joy"):
            ctx.move_to(-mw / 2, my - H * 0.01)
            ctx.curve_to(-mw / 4, my + H * 0.07, mw / 4, my + H * 0.07, mw / 2, my - H * 0.01)
            if P.face == "joy":
                ctx.close_path()
                ctx.set_source_rgb(0.35, 0.05, 0.1)
                ctx.fill_preserve()
                ctx.set_source_rgb(*ink)
            ctx.stroke()
        elif P.face == "sad":
            ctx.move_to(-mw / 2, my + H * 0.04)
            ctx.curve_to(-mw / 4, my - H * 0.03, mw / 4, my - H * 0.03, mw / 2, my + H * 0.04)
            ctx.stroke()
        elif P.face == "o":
            ctx.save()
            ctx.translate(0, my + H * 0.02)
            ctx.scale(1, 1.3)
            ctx.arc(0, 0, W * 0.07, 0, 2 * math.pi)
            ctx.restore()
            ctx.set_source_rgb(0.3, 0.04, 0.08)
            ctx.fill_preserve()
            ctx.set_source_rgb(*ink)
            ctx.stroke()
        else:
            ctx.move_to(-mw / 2.5, my + H * 0.02)
            ctx.line_to(mw / 2.5, my + H * 0.02)
            ctx.stroke()
        # cheeks
        for side in (-1, 1):
            ctx.arc(side * W * 0.32, my - H * 0.01, W * 0.06, 0, 2 * math.pi)
            ctx.set_source_rgba(1, 0.35, 0.45, 0.35)
            ctx.fill()
    ctx.pop_group_to_source()
    ctx.paint_with_alpha(a)
    ctx.restore()
    return (x - W, y - H * 1.4, x + W, y + H * 0.1)


# =============================================================================================== setup
def quantise(t, timing="ones", fps=FPS):
    """limited animation: hold each drawing for 1/2/3 frames (on ones / twos / threes)."""
    n = {"ones": 1, "twos": 2, "threes": 3, 1: 1, 2: 2, 3: 3}.get(timing, 1)
    if n == 1:
        return t
    step = n / float(fps)
    return math.floor(t / step + 1e-6) * step


def prepare(spec, resolve=None, base_dir="."):
    """spec → prepared character dict (actions resolved & sorted, rig, visemes)."""
    import os
    C = dict(spec)
    res = resolve or (lambda v, d=None: float(v) if v is not None else d)
    rig = C.get("rig", "bean")
    C["_rig"] = dict(RIGS["bean"], **rig) if isinstance(rig, dict) else dict(RIGS.get(rig, RIGS["bean"]))
    acts = []
    for a in C.get("actions", []):
        a = dict(a)
        if a.get("do") not in ACTIONS:
            raise ValueError(f"character action '{a.get('do')}' unknown — one of {ACTIONS}")
        a["_t"] = float(res(a.get("t", 0.0)))
        a["_d"] = max(0.1, float(res(a["dur"]) if isinstance(a.get("dur"), str) else a.get("dur", ACTION_DUR[a["do"]])))
        a["_tail"] = 0.9 if a["do"] in ("hop", "jump", "enter", "surprise", "squash") else 0.0
        acts.append(a)
    C["_acts"] = sorted(acts, key=lambda a: a["_t"])
    ls = C.get("lip_sync")
    if ls:
        from .lipsync import track
        if isinstance(ls, dict) and ls.get("visemes"):
            C["_vis"] = ls["visemes"]
        else:
            ls = ls if isinstance(ls, dict) else {"words": ls}
            wp = ls.get("words")
            if isinstance(wp, str) and not os.path.isabs(wp):
                wp = os.path.join(base_dir, wp)
            ap = ls.get("audio")
            if isinstance(ap, str) and not os.path.isabs(ap):
                ap = os.path.join(base_dir, ap)
            words = wp
            if ls.get("map") is not None:                     # v14: words already in edited/out time
                words = ls["map"]
            C["_vis"] = track(words, ap, float(C.get("fps", FPS)), offset=float(ls.get("offset", 0.0)))
    return C


def principles_report(C, t0=0.0, t1=None, fps=FPS):
    """numbers per principle over [t0, t1] — QA that the motion actually has them."""
    t1 = t1 if t1 is not None else max([a["_t"] + a["_d"] + a["_tail"] for a in C["_acts"]] + [t0 + 2])
    ts = np.arange(t0, t1, 1 / fps)
    Ps = [solve(C, float(t)) for t in ts]
    sy = np.array([p.sy for p in Ps])
    vol = np.array([p.sx * p.sy for p in Ps])
    ys = np.array([p.y for p in Ps])
    ant = np.array([p.antenna for p in Ps])
    held = sum(1 for i in range(1, len(ts)) if quantise(ts[i], C.get("timing", "ones"), fps)
               == quantise(ts[i - 1], C.get("timing", "ones"), fps))
    anticip = 0
    for a in C["_acts"]:
        if a["do"] in ("hop", "jump"):
            i0 = int((a["_t"] - t0) * fps)
            i1 = int((a["_t"] + a["_d"] * 0.28 - t0) * fps)
            if 0 <= i0 < i1 <= len(sy) and sy[i0:i1].min() < 0.95:
                anticip += 1
    return {"squash_range": [round(float(sy.min()), 3), round(float(sy.max()), 3)],
            "volume_error_max": round(float(np.abs(vol - (Ps[0].sx * Ps[0].sy)).max()), 3) if len(vol) else 0,
            "jump_height_max": round(float(-ys.min()), 3), "follow_through_deg": round(float(np.ptp(ant)), 1),
            "anticipated_jumps": anticip, "held_frames": held, "frames": len(ts)}
