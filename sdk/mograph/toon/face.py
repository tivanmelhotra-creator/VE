"""
mograph.toon.face — expressive cartoon faces drawn ON the round toon head (style of the reference sheets:
white oval eyes with a black outline + small dot pupils, thin simple brows, no nose, a small mouth that opens into
a big dark mouth with tongue / teeth on strong emotions; extras = sweat drops, tears, blush, star eyes, Zzz).

Features are 2D shapes in head-angle space (x = yaw, y = pitch, radians ≈ head radii) mapped onto the posed ball
and projected, then masked by the head's own pixels in the id buffer (a hand / another actor in front cuts them).

    face.draw(ctx, G, cam, expr="shock", t=t, look=[x,y,z], talk=0..1, width=px)
"""
from __future__ import annotations

import math

import numpy as np

INK = (0.1, 0.095, 0.09)
MOUTH = (0.2, 0.09, 0.09)
TONGUE = (0.88, 0.45, 0.45)
WHITE = (1.0, 1.0, 1.0)
TEAR = (0.55, 0.78, 0.95)

# eye: size factor | "arc" (^ happy) | "closed" (v sleepy) | "heart" | "star" | "black" (sad filled) | "squeeze" (><)
# lid: 0 = none, 0..1 = upper-lid cover (tired / smug); pupil = pupil scale; brow = (raise, tilt + = inner up)
EXPR = {
    "neutral":   dict(eye=1.0, pupil=1.0, brow=(0.0, 0.0), mouth="flat", nobrow=True, ms=0.6),
    "smile":     dict(eye=1.0, pupil=1.0, brow=(0.03, 0.05), mouth="smile"),
    "happy":     dict(eye=1.0, pupil=1.15, hl=True, brow=(0.06, 0.08), mouth="open_smile"),
    "joy":       dict(eye="arc", brow=(0.08, 0.1), mouth="open_smile", ms=1.2),
    "excited":   dict(eye="star", brow=(0.1, 0.1), mouth="open_smile", ms=1.25),
    "laugh":     dict(eye="squeeze", brow=(0.08, 0.15), mouth="open_smile", ms=1.3),
    "sneaky":    dict(eye=0.95, pupil=0.85, lid=0.5, side=0.6, brow=(-0.02, -0.3), mouth="smirk"),
    "smug":      dict(eye=1.0, pupil=0.9, lid=0.45, brow=(0.04, -0.12), brow_asym=0.06, mouth="smirk"),
    "playful":   dict(eye=1.0, pupil=0.9, lid=0.4, brow=(0.03, -0.1), mouth="tongue"),
    "tired":     dict(eye=1.0, pupil=0.9, lid=0.55, low_lid=0.15, brow=(-0.02, 0.15), mouth="flat_down"),
    "shock":     dict(eye=1.35, pupil=0.42, brow=(0.14, 0.1), mouth="shout", ms=1.25),
    "surprised": dict(eye=1.25, pupil=0.55, brow=(0.12, 0.05), mouth="o"),
    "scared":    dict(eye=1.4, pupil=0.38, brow=(0.12, 0.38), mouth="scream", ms=1.35, sweat=True),
    "nervous":   dict(eye=1.25, pupil=0.45, brow=(0.08, 0.35), mouth="wobble", sweat=True),
    "angry":     dict(eye=0.9, pupil=0.75, brow=(-0.07, -0.6), mouth="grit"),
    "furious":   dict(eye=0.85, pupil=0.55, brow=(-0.09, -0.7), mouth="shout", ms=1.4),
    "sad":       dict(eye="black", brow=(0.03, 0.5), mouth="frown"),
    "cry":       dict(eye="squeeze", brow=(0.04, 0.5), mouth="cry", tears=True),
    "worried":   dict(eye=1.05, pupil=0.75, brow=(0.05, 0.42), mouth="wobble"),
    "thinking":  dict(eye=1.0, pupil=0.8, side=0.7, up=0.6, brow=(0.04, 0.0), brow_asym=0.1, mouth="side"),
    "curious":   dict(eye=1.15, pupil=0.7, brow=(0.08, 0.05), brow_asym=0.08, mouth="small_o"),
    "love":      dict(eye="heart", brow=(0.08, 0.1), mouth="open_smile", blush=True),
    "blush":     dict(eye=1.0, pupil=0.9, side=-0.5, brow=(0.04, 0.3), mouth="wobble", blush=True),
    "eat":       dict(eye="arc", brow=(0.05, 0.06), mouth="chew", blush=True),
    "hungry":    dict(eye=1.0, pupil=0.9, lid=0.3, up=-0.4, brow=(0.02, 0.35), mouth="drool"),
    "sleepy":    dict(eye="closed", brow=(0.0, 0.08), mouth="small_o", ms=0.7),
    "calm":      dict(eye="closed", brow=(0.03, 0.05), mouth="smile"),
    "bored":     dict(eye=1.0, pupil=0.9, lid=0.5, brow=(-0.02, 0.0), mouth="flat"),
}


def _bez(p0, p1, p2, p3, n=18):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, float) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def _ellipse(cx, cy, rx, ry, n=44, a0=0.0, a1=2 * math.pi):
    th = np.linspace(a0, a1, n)
    return np.c_[cx + rx * np.cos(th), cy + ry * np.sin(th)]


def _heart(cx, cy, s, n=48):
    t = np.linspace(0, 2 * math.pi, n)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return np.c_[cx + x / 17 * s, cy + y / 17 * s]


def _star(cx, cy, r, k=4, inner=0.32, n=8):
    pts = []
    for i in range(2 * k):
        a = math.pi / 2 + i * math.pi / k
        rr = r if i % 2 == 0 else r * inner
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return np.asarray(pts)


def blink_open(t, seed, every=(2.2, 4.6), dur=0.1):
    rng = np.random.default_rng(seed)
    times = np.cumsum(rng.uniform(*every, 160)) + rng.uniform(0.3, 1.5)
    k = int(np.searchsorted(times, t))
    d = min(abs(t - times[max(0, k - 1)]), abs(times[min(len(times) - 1, k)] - t))
    return d >= dur / 2


def cam_right(cam):
    """world-space unit vector pointing to screen-right."""
    V = cam.view()
    return np.asarray(V[0, :3], float) / (np.linalg.norm(V[0, :3]) + 1e-12)


class ToonFace:
    def __init__(self, head, kind="adult", female=False, eye_sep=0.3, eye_y=0.1, eye_r=0.25, mouth_y=-0.36,
                 lashes=None, seed=0):
        self.h = head
        self.female = female
        self.lashes = female if lashes is None else lashes
        self.kind = kind
        k = 1.1 if kind == "child" else 1.0
        self.ex, self.ey, self.er = eye_sep, eye_y, eye_r * k
        self.my = mouth_y
        self.seed = seed
        self._frames = None

    # ------------------------------------------------------------------------------------------- mapping
    def _proj(self, G, cam, Q, lift=1.01):
        P, _ = self.h.face_points(G, Q, lift)
        return cam.project(P)[0]

    @staticmethod
    def _path(ctx, Q, close=True):
        ctx.new_path()
        ctx.move_to(*Q[0])
        for q in Q[1:]:
            ctx.line_to(*q)
        if close:
            ctx.close_path()

    def _fill(self, ctx, G, cam, Q, rgb, a=1.0):
        self._path(ctx, self._proj(G, cam, Q))
        ctx.set_source_rgba(*rgb, a)
        ctx.fill()

    def _outline(self, ctx, G, cam, Q, w, rgb=INK, close=True):
        self._path(ctx, self._proj(G, cam, Q), close)
        ctx.set_source_rgba(*rgb, 1.0)
        ctx.set_line_width(w)
        ctx.stroke()

    def _line(self, ctx, G, cam, Q, w, rgb=INK, a=1.0):
        """tapered ink stroke (thick middle, thin round ends)."""
        S = self._proj(G, cam, Q)
        if len(S) < 2:
            return
        d = np.gradient(S, axis=0)
        nr = np.c_[-d[:, 1], d[:, 0]]
        nr /= np.linalg.norm(nr, axis=1, keepdims=True) + 1e-9
        s = np.linspace(0, 1, len(S))
        prof = 0.45 + 0.55 * np.sin(np.pi * s) ** 0.6
        ww = (w * prof / 2)[:, None]
        L, R = S + nr * ww, S - nr * ww
        ctx.new_path()
        ctx.move_to(*L[0])
        for q in L[1:]:
            ctx.line_to(*q)
        for q in R[::-1]:
            ctx.line_to(*q)
        ctx.close_path()
        ctx.set_source_rgba(*rgb, a)
        ctx.fill()
        for q in (S[0], S[-1]):
            ctx.arc(q[0], q[1], w * 0.45 / 2, 0, 2 * math.pi)
            ctx.fill()

    # ------------------------------------------------------------------------------------------- eyes
    def _eye(self, ctx, G, cam, side, E, open_, gaze, W):
        sg = 1.0 if side == "L" else -1.0
        cx, cy, r = sg * self.ex, self.ey, self.er
        kind = E.get("eye", 1.0)
        if not isinstance(kind, str) and not open_:
            kind = "blink"
        if kind in ("arc", "closed", "blink", "squeeze"):
            if kind == "arc":            # ^  happy
                Q = _bez((cx - r * 0.95, cy - r * 0.3), (cx - r * 0.5, cy + r * 0.85), (cx + r * 0.5, cy + r * 0.85),
                         (cx + r * 0.95, cy - r * 0.3))
            elif kind == "closed":       # ‿  calm / asleep (lashes down)
                Q = _bez((cx - r * 0.95, cy + r * 0.1), (cx - r * 0.45, cy - r * 0.55), (cx + r * 0.45, cy - r * 0.55),
                         (cx + r * 0.95, cy + r * 0.1))
            elif kind == "squeeze":      # >  <
                Q = np.array([[cx - sg * r * 0.85, cy + r * 0.7], [cx + sg * r * 0.6, cy], [cx - sg * r * 0.85,
                                                                                          cy - r * 0.7]])
                Q = np.vstack([_bez(Q[0], Q[0] * 0.5 + Q[1] * 0.5, Q[1], Q[1], 8),
                               _bez(Q[1], Q[1], Q[1] * 0.5 + Q[2] * 0.5, Q[2], 8)[1:]])
            else:                        # blink: flat lid line
                Q = _bez((cx - r * 0.9, cy - r * 0.05), (cx - r * 0.4, cy - r * 0.3), (cx + r * 0.4, cy - r * 0.3),
                         (cx + r * 0.9, cy - r * 0.05))
            self._line(ctx, G, cam, Q, W * 1.1)
            return
        if kind == "heart":
            H = _heart(cx, cy, r * 1.2)
            self._fill(ctx, G, cam, H, (0.88, 0.24, 0.32))
            self._outline(ctx, G, cam, H, W * 0.8)
            return
        if kind == "black":              # sad: big filled black eyes with a highlight, slightly droopy
            S = _ellipse(cx, cy - r * 0.05, r, r)
            self._fill(ctx, G, cam, S, INK)
            self._fill(ctx, G, cam, _ellipse(cx - r * 0.3, cy + r * 0.35, r * 0.28, r * 0.3), WHITE)
            self._fill(ctx, G, cam, _ellipse(cx + r * 0.35, cy - r * 0.45, r * 0.12, r * 0.12), WHITE)
            return
        if kind == "star":
            S = _ellipse(cx, cy, r * 1.05, r * 1.2)
            self._fill(ctx, G, cam, S, INK)
            self._fill(ctx, G, cam, _star(cx, cy, r * 0.75), WHITE)
            self._outline(ctx, G, cam, S, W * 0.8)
            return
        o = float(kind)
        if E.get("uv_eye"):
            return self._eye_uv(ctx, G, cam, cx, cy, r, o, sg, E, gaze, W)
        # ROUND eyes, built in SCREEN space (the reference draws perfect circles; mapping a circle onto the ball
        # turns it into an oval).  Centre / orientation / scale come from the ball; a turned head only narrows
        # the eye a little across (k), so it never collapses into a thin ellipse.
        c0, ux, uy, R, k = (self._frames or {}).get(side) or self._eye_frame(G, cam, cx, cy, r)
        a, b = R * (0.75 + 0.25 * k) * min(o, 1.3), R * o
        lw = W * 0.85
        ctx.save()
        self._screen_ellipse(ctx, c0, ux, uy, a, b)
        ctx.set_source_rgb(*WHITE)
        ctx.fill_preserve()
        ctx.clip()
        pr = R * 0.36 * E.get("pupil", 1.0)
        gx = float(np.clip(gaze[0] + E.get("side", 0.0), -1, 1)) * max(0.0, a - pr * 1.05) * 0.85
        gy = float(np.clip(gaze[1] + E.get("up", 0.0), -1, 1)) * max(0.0, b - pr * 1.05) * 0.7
        pc = c0 + ux * gx + uy * gy
        self._screen_ellipse(ctx, pc, ux, uy, pr * (0.88 + 0.12 * k), pr)
        ctx.set_source_rgb(*INK)
        ctx.fill()
        if E.get("hl") or E.get("pupil", 1.0) > 1.1:
            self._screen_ellipse(ctx, pc + ux * (-0.3 * pr) + uy * (0.38 * pr), ux, uy, pr * 0.32, pr * 0.32)
            ctx.set_source_rgb(*WHITE)
            ctx.fill()
        lid, low = E.get("lid", 0.0), E.get("low_lid", 0.0)
        tilt = (-0.25 * b * sg if E.get("brow", (0, 0))[1] < -0.2 else 0.0)
        if lid > 0:                       # heavy upper lid (tired / smug / sneaky): skin-white cover
            top = b - 2 * b * lid
            Q = [(-2 * a, top - tilt), (2 * a, top + tilt), (2 * a, 3 * b), (-2 * a, 3 * b)]
            self._screen_poly(ctx, c0, ux, uy, Q)
            ctx.set_source_rgb(0.985, 0.98, 0.965)
            ctx.fill()
        if low > 0:
            bot = -b + 2 * b * low
            self._screen_poly(ctx, c0, ux, uy, [(-2 * a, bot), (2 * a, bot), (2 * a, -3 * b), (-2 * a, -3 * b)])
            ctx.set_source_rgb(0.985, 0.98, 0.965)
            ctx.fill()
        ctx.restore()
        self._screen_ellipse(ctx, c0, ux, uy, a, b)
        ctx.set_source_rgb(*INK)
        ctx.set_line_width(lw)
        ctx.stroke()
        if lid > 0:
            top = b - 2 * b * lid
            half = math.sqrt(max(0.0, 1 - (top / b) ** 2)) * a * 1.08
            xs = np.linspace(-1, 1, 12)
            S = np.array([c0 + ux * (x * half) + uy * (top + x * tilt) for x in xs])
            self._screen_line(ctx, S, W * 1.1)
        if self.lashes:
            for ang, L in ((0.28, 0.32), (0.08, 0.26)):
                th = math.pi * ang if sg > 0 else math.pi * (1 - ang)
                d = np.array([math.cos(th), math.sin(th)])
                p0 = c0 + ux * (a * d[0]) + uy * (b * d[1])
                p1 = p0 + (ux * d[0] + uy * d[1]) * (R * L * 1.15)
                self._screen_line(ctx, np.array([p0, p1]), W * 0.7)

    def _layout_eyes(self, G, cam):
        """both eyes as ROUND screen circles on one line (the head's left/right axis), inside the head outline and
        never overlapping.  3/4 view: the far eye keeps a small gap next to the near one, like the drawn reference;
        profile: the far eye is hidden."""
        fr = {sd: list(self._eye_frame(G, cam, (1.0 if sd == "L" else -1.0) * self.ex, self.ey, self.er))
              for sd in ("L", "R")}
        M = G[self.h.hi].astype(np.float64)
        hc = self.h.center(G)
        P = cam.project(np.array([hc, hc + M[:, :3] @ np.array([1.0, 0, 0]) * self.h.r,
                                  hc + M[:, :3] @ np.array([0, 1.0, 0]) * self.h.r]))[0]
        hp = P[0]
        hr = float(np.linalg.norm(cam.project((hc + cam_right(cam) * self.h.r * self.h.sq[0])[None])[0][0] - hp))
        uy = P[2] - P[0]
        uy /= np.linalg.norm(uy) + 1e-9
        ux = np.array([-uy[1], uy[0]])
        if np.dot(ux, P[1] - P[0]) < 0:
            ux = -ux
        tocam = cam.eye() - hc
        tocam /= np.linalg.norm(tocam)
        vis = {}
        for sd, sgn in (("L", 1.0), ("R", -1.0)):
            dirw = M[:, :3] @ np.array([math.sin(sgn * self.ex), 0, math.cos(sgn * self.ex)])
            vis[sd] = float(np.dot(dirw, tocam))
        self._eye_vis = vis
        rad = {sd: f[3] * (0.75 + 0.25 * f[4]) for sd, f in fr.items()}
        x = {sd: float(np.dot(fr[sd][0] - hp, ux)) for sd in fr}
        y = float(np.mean([np.dot(fr[sd][0] - hp, uy) for sd in fr]))
        need = (rad["L"] + rad["R"]) * 1.1
        near = "L" if vis["L"] >= vis["R"] else "R"
        far = "R" if near == "L" else "L"
        if x["L"] - x["R"] < need:                       # L sits at +ux (the head's +x)
            x[far] = x[near] + (need if far == "L" else -need)
        shown = [sd for sd in fr if vis[sd] >= -0.05]
        # fit inside the head circle at height y: shift the pair along ux, shrink only if it can't fit
        for _ in range(4):
            lo, hi = -1e9, 1e9
            for sd in shown:
                lim = hr * 0.93 - rad[sd]
                half = math.sqrt(max(1.0, lim * lim - y * y))
                lo, hi = max(lo, -half - x[sd]), min(hi, half - x[sd])
            if lo <= hi:
                sh = min(max(0.0, lo), hi) if lo > 0 else (max(min(0.0, hi), lo))
                for sd in fr:
                    x[sd] += sh
                break
            sc = 0.92
            m0 = (x["L"] + x["R"]) / 2
            for sd in fr:
                x[sd] = m0 + (x[sd] - m0) * sc
                fr[sd][3] *= sc
                rad[sd] *= sc
        for sd in fr:
            fr[sd][0] = hp + ux * x[sd] + uy * y
            fr[sd][1], fr[sd][2] = ux, uy
        return {sd: tuple(f) for sd, f in fr.items()}

    # screen-space helpers ------------------------------------------------------------------------------------
    def _eye_frame(self, G, cam, cx, cy, r):
        """→ screen centre, unit 'across' / 'up' axes (from the ball), radius px, foreshortening 0..1"""
        P = self._proj(G, cam, np.array([[cx, cy], [cx + 0.2 * r, cy], [cx - 0.2 * r, cy], [cx, cy + 0.2 * r],
                                         [cx, cy - 0.2 * r]]))
        c0 = P[0]
        ax = (P[1] - P[2]) / (0.4)            # px per r along u
        ay = (P[3] - P[4]) / (0.4)
        la, lb = np.linalg.norm(ax), np.linalg.norm(ay)
        # true (un-foreshortened) radius: the projected size of r at that depth facing the camera
        Pw, _ = self.h.face_points(G, [[cx, cy]], 1.01)
        Pc = cam.project(Pw + cam_right(cam) * (r * self.h.r))[0]
        R = float(np.linalg.norm(Pc[0] - c0))
        uy = ay / (lb + 1e-9)
        # 'across' axis: perpendicular to up in screen (keeps the eye an upright ellipse, never sheared)
        ux = np.array([-uy[1], uy[0]])
        if np.dot(ux, ax) < 0:
            ux = -ux
        k = float(np.clip(la / (R + 1e-9), 0.0, 1.0))
        return c0, ux, uy, R, k

    @staticmethod
    def _screen_ellipse(ctx, c, ux, uy, a, b):
        ctx.new_path()
        n = 48
        for i in range(n + 1):
            t = 2 * math.pi * i / n
            q = c + ux * (a * math.cos(t)) + uy * (b * math.sin(t))
            (ctx.move_to if i == 0 else ctx.line_to)(float(q[0]), float(q[1]))
        ctx.close_path()

    @staticmethod
    def _screen_poly(ctx, c, ux, uy, Q):
        ctx.new_path()
        for i, (x, y) in enumerate(Q):
            q = c + ux * x + uy * y
            (ctx.move_to if i == 0 else ctx.line_to)(float(q[0]), float(q[1]))
        ctx.close_path()

    def _screen_line(self, ctx, S, w, rgb=INK):
        S = np.asarray(S, float)
        d = np.gradient(S, axis=0)
        nr = np.c_[-d[:, 1], d[:, 0]]
        nr /= np.linalg.norm(nr, axis=1, keepdims=True) + 1e-9
        s = np.linspace(0, 1, len(S))
        ww = (w * (0.45 + 0.55 * np.sin(np.pi * s) ** 0.6) / 2)[:, None]
        L, Rr = S + nr * ww, S - nr * ww
        ctx.new_path()
        ctx.move_to(*L[0])
        for q in L[1:]:
            ctx.line_to(*q)
        for q in Rr[::-1]:
            ctx.line_to(*q)
        ctx.close_path()
        ctx.set_source_rgb(*rgb)
        ctx.fill()
        for q in (S[0], S[-1]):
            ctx.arc(q[0], q[1], w * 0.45 / 2, 0, 2 * math.pi)
            ctx.fill()

    def _eye_uv(self, ctx, G, cam, cx, cy, r, o, sg, E, gaze, W):
        """legacy: eye mapped onto the ball (oval when turned)"""
        rx, ry = r * min(o, 1.3), r * o
        S = _ellipse(cx, cy, rx, ry)
        self._fill(ctx, G, cam, S, WHITE)
        ctx.save()
        self._path(ctx, self._proj(G, cam, S))
        ctx.clip()
        pr = r * 0.36 * E.get("pupil", 1.0)
        gx = cx + float(np.clip(gaze[0], -1, 1)) * (rx - pr) * 0.8
        gy = cy + float(np.clip(gaze[1], -1, 1)) * (ry - pr) * 0.7
        self._fill(ctx, G, cam, _ellipse(gx, gy, pr, pr), INK)
        ctx.restore()
        self._outline(ctx, G, cam, S, W * 0.85)

    # ------------------------------------------------------------------------------------------- brows
    def _brow(self, ctx, G, cam, side, E, W):
        sg = 1.0 if side == "L" else -1.0
        h, tilt = E.get("brow", (0.0, 0.0))
        h += E.get("brow_asym", 0.0) * sg
        r = self.er
        eo = E.get("eye", 1.0)
        big = float(eo) if not isinstance(eo, str) else 1.0
        cx, cy = sg * self.ex, self.ey + r * (1.12 * big + 0.55) + h
        inner = np.array([cx - sg * r * 0.8, cy + tilt * r * 0.55])
        outer = np.array([cx + sg * r * 0.8, cy - tilt * r * 0.25])
        mid = (inner + outer) / 2 + [0, r * 0.14 * (1 - abs(tilt))]
        Q = _bez(inner, inner * 0.35 + mid * 0.65, outer * 0.35 + mid * 0.65, outer, 12)
        self._line(ctx, G, cam, Q, W * (1.15 if abs(tilt) > 0.4 else 0.95))

    # ------------------------------------------------------------------------------------------- mouth
    def _dark_mouth(self, ctx, G, cam, Q, W, tongue=True, teeth=False, lower_teeth=False):
        self._fill(ctx, G, cam, Q, MOUTH)
        ctx.save()
        self._path(ctx, self._proj(G, cam, Q))
        ctx.clip()
        x0, x1 = Q[:, 0].min(), Q[:, 0].max()
        y0, y1 = Q[:, 1].min(), Q[:, 1].max()
        wd, ht = x1 - x0, y1 - y0
        if tongue:
            self._fill(ctx, G, cam, _ellipse((x0 + x1) / 2 + wd * 0.08, y0 - ht * 0.05, wd * 0.36, ht * 0.42), TONGUE)
        if teeth:
            band = np.array([[x0 - 0.1, y1 + 0.1], [x1 + 0.1, y1 + 0.1], [x1 + 0.1, y1 - ht * 0.24],
                             [x0 - 0.1, y1 - ht * 0.24]])
            self._fill(ctx, G, cam, band, WHITE)
        if lower_teeth:
            band = np.array([[x0 - 0.1, y0 - 0.1], [x1 + 0.1, y0 - 0.1], [x1 + 0.1, y0 + ht * 0.22],
                             [x0 - 0.1, y0 + ht * 0.22]])
            self._fill(ctx, G, cam, band, WHITE)
        ctx.restore()
        self._outline(ctx, G, cam, Q, W * 0.85)

    def _mouth(self, ctx, G, cam, E, talk, t, W):
        m = E.get("mouth", "flat")
        s = E.get("ms", 1.0)
        y = self.my
        w = 0.22 * s
        if talk > 0.05 and m in ("flat", "smile", "smirk", "frown", "wobble", "flat_down", "side"):
            # talk: the resting mouth opens into a small dark shape that flaps with the voice
            op = 0.025 + 0.085 * talk
            lift = 0.03 if m in ("smile", "smirk") else 0.0
            Q = np.vstack([_bez((-w * 0.8, y + lift), (-w * 0.3, y + 0.012), (w * 0.3, y + 0.012), (w * 0.8, y + lift),
                                10),
                           _bez((w * 0.8, y + lift), (w * 0.45, y - op), (-w * 0.45, y - op), (-w * 0.8, y + lift),
                                12)[1:]])
            self._dark_mouth(ctx, G, cam, Q, W, tongue=talk > 0.4)
            return
        if m == "flat":
            self._line(ctx, G, cam, _bez((-w * 0.7, y), (-w * 0.2, y - 0.01), (w * 0.2, y - 0.01), (w * 0.7, y), 10), W)
        elif m == "flat_down":
            self._line(ctx, G, cam, _bez((-w * 0.7, y - 0.02), (-w * 0.2, y + 0.005), (w * 0.2, y + 0.005),
                                         (w * 0.7, y - 0.02), 10), W)
        elif m == "smile":
            self._line(ctx, G, cam, _bez((-w * 1.0, y + 0.04), (-w * 0.4, y - 0.06), (w * 0.4, y - 0.06),
                                         (w * 1.0, y + 0.04), 14), W)
        elif m == "smirk":
            self._line(ctx, G, cam, _bez((-w * 0.8, y - 0.015), (-w * 0.1, y - 0.04), (w * 0.5, y - 0.03),
                                         (w * 1.0, y + 0.06), 14), W)
        elif m == "side":
            self._line(ctx, G, cam, _bez((-w * 0.2, y - 0.01), (w * 0.2, y - 0.03), (w * 0.6, y - 0.02),
                                         (w * 0.9, y + 0.02), 10), W)
        elif m == "frown":
            self._line(ctx, G, cam, _bez((-w * 0.8, y - 0.04), (-w * 0.3, y + 0.035), (w * 0.3, y + 0.035),
                                         (w * 0.8, y - 0.04), 14), W)
        elif m == "wobble":
            xs = np.linspace(-w * 0.85, w * 0.85, 26)
            self._line(ctx, G, cam, np.c_[xs, y + 0.016 * np.sin(xs / w * 3.2 * math.pi)], W * 0.9)
        elif m in ("o", "small_o"):
            k = 0.55 if m == "o" else 0.32
            Q = _ellipse(0, y - 0.02, w * k, w * k * 1.3)
            self._dark_mouth(ctx, G, cam, Q, W, tongue=False)
        elif m == "open_smile":      # D-shape: flat top, round bottom, tongue
            dep = 0.15 * s
            Q = np.vstack([_bez((-w * 1.2, y + 0.04), (-w * 0.4, y + 0.025), (w * 0.4, y + 0.025), (w * 1.2, y + 0.04),
                                12),
                           _bez((w * 1.2, y + 0.04), (w * 0.95, y - dep), (-w * 0.95, y - dep), (-w * 1.2, y + 0.04),
                                18)[1:]])
            self._dark_mouth(ctx, G, cam, Q, W, tongue=True, teeth=s > 1.1)
        elif m in ("shout", "scream"):   # tall open mouth, upper teeth
            hgt = (0.2 if m == "scream" else 0.17) * s
            Q = np.vstack([_bez((-w * 0.9, y + 0.03), (-w * 0.4, y + 0.06), (w * 0.4, y + 0.06), (w * 0.9, y + 0.03),
                                12),
                           _bez((w * 0.9, y + 0.03), (w * 0.85, y - hgt), (-w * 0.85, y - hgt), (-w * 0.9, y + 0.03),
                                18)[1:]])
            self._dark_mouth(ctx, G, cam, Q, W, tongue=True, teeth=True)
        elif m == "grit":                # angry: wide rectangle of clenched teeth
            Q = np.vstack([_bez((-w * 1.1, y + 0.02), (-w * 0.4, y + 0.045), (w * 0.4, y + 0.045), (w * 1.1, y + 0.02),
                                12),
                           _bez((w * 1.1, y + 0.02), (w * 0.8, y - 0.09), (-w * 0.8, y - 0.09), (-w * 1.1, y + 0.02),
                                14)[1:]])
            self._fill(ctx, G, cam, Q, WHITE)
            self._outline(ctx, G, cam, Q, W * 0.85)
            self._line(ctx, G, cam, np.c_[np.linspace(-w * 1.0, w * 1.0, 10), np.full(10, y - 0.02)], W * 0.6)
            for x in (-w * 0.5, 0.0, w * 0.5):
                self._line(ctx, G, cam, np.array([[x, y + 0.035], [x, y - 0.075]]), W * 0.5)
        elif m == "cry":
            Q = np.vstack([_bez((-w * 1.0, y), (-w * 0.4, y + 0.06), (w * 0.4, y + 0.06), (w * 1.0, y), 12),
                           _bez((w * 1.0, y), (w * 0.6, y - 0.07), (-w * 0.6, y - 0.07), (-w * 1.0, y), 12)[1:]])
            self._dark_mouth(ctx, G, cam, Q, W, tongue=False, teeth=True)
        elif m == "tongue":
            self._line(ctx, G, cam, _bez((-w * 0.9, y + 0.02), (-w * 0.3, y - 0.04), (w * 0.3, y - 0.04),
                                         (w * 0.9, y + 0.02), 12), W)
            T = np.vstack([_bez((w * 0.0, y - 0.03), (w * 0.0, y - 0.13), (w * 0.6, y - 0.13), (w * 0.6, y - 0.025), 12)])
            self._fill(ctx, G, cam, T, TONGUE)
            self._outline(ctx, G, cam, T, W * 0.7, close=False)
        elif m == "chew":
            ph = 0.5 + 0.5 * math.sin(t * 14.0)
            op = 0.015 + 0.05 * ph
            Q = np.vstack([_bez((-w * 0.7, y + 0.02), (-w * 0.2, y + 0.03), (w * 0.2, y + 0.03), (w * 0.7, y + 0.02),
                                10),
                           _bez((w * 0.7, y + 0.02), (w * 0.4, y - op), (-w * 0.4, y - op), (-w * 0.7, y + 0.02),
                                10)[1:]])
            self._dark_mouth(ctx, G, cam, Q, W, tongue=False)
            # full cheek bump
            self._line(ctx, G, cam, _bez((w * 1.6, y + 0.12), (w * 2.1, y + 0.06), (w * 2.1, y - 0.05),
                                         (w * 1.6, y - 0.1), 10), W * 0.7)
        elif m == "drool":
            self._line(ctx, G, cam, _bez((-w * 0.8, y - 0.02), (-w * 0.3, y + 0.02), (w * 0.3, y + 0.02),
                                         (w * 0.8, y - 0.03), 12), W)
            d = _ellipse(w * 0.55, y - 0.08, 0.018, 0.03)
            self._fill(ctx, G, cam, d, TEAR)
            self._outline(ctx, G, cam, d, W * 0.5)

    def _blush(self, ctx, G, cam, W):
        for sg in (-1, 1):
            cx, cy = sg * (self.ex + 0.08), self.ey - self.er * 1.45
            self._fill(ctx, G, cam, _ellipse(cx, cy, 0.09, 0.04), (0.93, 0.55, 0.55), 0.5)
            for k in range(3):
                x = cx - 0.045 + k * 0.045
                self._line(ctx, G, cam, np.array([[x + 0.012, cy + 0.02], [x - 0.012, cy - 0.02]]), W * 0.45,
                           rgb=(0.78, 0.32, 0.32))

    def _tears(self, ctx, G, cam, t, W):
        for sg in (-1, 1):
            x = sg * (self.ex + self.er * 0.3)
            ph = (t * 1.6 + (0.5 if sg > 0 else 0)) % 1.0
            Q = np.vstack([_bez((x - 0.04, self.ey - self.er * 0.6), (x - 0.05, self.ey - 0.3), (x - 0.03, -0.45),
                                (x, -0.5 - 0.05 * ph), 12),
                           _bez((x, -0.5 - 0.05 * ph), (x + 0.03, -0.45), (x + 0.05, self.ey - 0.3),
                                (x + 0.04, self.ey - self.er * 0.6), 12)[1:]])
            self._fill(ctx, G, cam, Q, TEAR, 0.9)
            self._outline(ctx, G, cam, Q, W * 0.5)

    # ------------------------------------------------------------------------------------------- public
    def draw(self, ctx, G, cam, expr="neutral", t=0.0, look=None, talk=0.0, width=4.0, blink=True, **_):
        E = EXPR.get(expr, EXPR["neutral"]) if isinstance(expr, str) else expr
        W = width
        open_ = blink_open(t, self.seed) if blink else True
        gaze = np.zeros(2)
        if look is not None:
            M = G[self.h.hi].astype(np.float64)
            c = M[:, :3] @ self.h.c + M[:, 3]
            d = M[:, :3].T @ (np.asarray(look, float) - c)
            d /= np.linalg.norm(d) + 1e-9
            gaze = np.array([math.atan2(d[0], max(1e-3, d[2])) / 0.7, math.asin(np.clip(d[1], -1, 1)) / 0.6])
        self._frames = self._layout_eyes(G, cam)
        for side in ("L", "R"):
            if getattr(self, "_eye_vis", {}).get(side, 1.0) < -0.05:
                continue
            self._eye(ctx, G, cam, side, E, open_, gaze, W)
            if not E.get("nobrow"):
                self._brow(ctx, G, cam, side, E, W)
        if E.get("blush"):
            self._blush(ctx, G, cam, W)
        self._mouth(ctx, G, cam, E, talk, t, W)
        if E.get("tears"):
            self._tears(ctx, G, cam, t, W)
