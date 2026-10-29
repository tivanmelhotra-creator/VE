"""
mograph.mocap.face — illustrated FACE KIT (webtoon / storyboard style): almond eyes with a heavy upper lash line
and a lifted outer corner, iris partly hidden under the lid, pupil + highlights; thin arched brows; a nose drawn
as a side line + a small rounded tip and nostril (reads 3D, turns with the head); small soft lips with a cupid's
bow and a short lower-lip curve.

All landmarks are MEASURED on the character's own head mesh (eye joints, nose tip = most forward midline point
under the eyes, upper/lower lip = the lip bulges under the nose, chin), so proportions are those of the face
itself — the mouth sits right under the nose, never down on the chin.

Each feature is drawn in a local 2D frame glued to the head bone (anchor + tangent axes projected every frame):
features turn / tilt / foreshorten with the mocap head, the far eye fades out when the head turns away, and
anything in front (a hand, the other character) hides them via the depth buffer.
"""
from __future__ import annotations

import math

import numpy as np

from ..human.body import BODY_N, bone_index, group_vertices


def _sm(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def _bez(p0, p1, p2, p3, n=16):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


class FaceKit:
    def __init__(self, ch, eye_size=1.0, lashes=True, female=None, child=None):
        self.ch = ch
        b = ch.body
        X = b.rest
        self.hi = bone_index("head")
        g = b.params.get("gender", 0.5)
        age = b.params.get("age", 25)
        self.female = (g < 0.5) if female is None else female
        self.child = (age < 13) if child is None else child
        self.lashes = lashes and self.female
        hv = np.nonzero(ch.R.head[:BODY_N] > 0.6)[0]
        P = X[hv]
        eL = X[group_vertices("joint-l-eye")].mean(0)
        ey, ex = float(eL[1]), float(abs(eL[0]))
        mid = P[np.abs(P[:, 0]) < 0.05]

        def zmax(y, x=0.0, r=0.045):
            m = (np.abs(P[:, 0] - x) < r) & (np.abs(P[:, 1] - y) < r)
            return float(P[m, 2].max()) if m.any() else float(P[np.argmin(np.hypot(P[:, 0] - x, P[:, 1] - y)), 2])
        # nose tip: most forward midline point between eye-0.1·s and eye-0.6·s (s = eye separation)
        s = 2 * ex
        ys = np.linspace(ey - 0.12 * s / 0.57, ey - 0.75 * s / 0.57, 40)
        zs = np.array([zmax(y) for y in ys])
        k = int(np.argmax(zs))
        nose_y = float(ys[k])
        # lips: below the nose the profile dips (subnasal) then bulges (upper lip) → first local max below nose
        ys2 = np.linspace(nose_y - 0.05, nose_y - 0.75 * s / 0.57, 40)
        zs2 = np.array([zmax(y) for y in ys2])
        dip = int(np.argmin(zs2[:20]))
        bul = dip + int(np.argmax(zs2[dip:dip + 14]))
        mouth_y = float(ys2[bul]) - 0.035 * s / 0.57
        chin_y = float(mid[:, 1].min()) if len(mid) else mouth_y - 0.5
        self.lm = dict(eye_y=ey, eye_x=ex, nose_y=nose_y, mouth_y=mouth_y, chin_y=chin_y)
        es = eye_size * (1.12 if self.child else 1.0)
        self.u = ex * 0.66 * es                                # eye half-width (dm)
        L = 0.02
        self.pts = {
            "eye.L": np.array([ex * 0.97, ey - 0.01, zmax(ey, ex * 0.97, 0.06) + L]),
            "eye.R": np.array([-ex * 0.97, ey - 0.01, zmax(ey, -ex * 0.97, 0.06) + L]),
            "brow.L": np.array([ex * 1.02, ey + 0.62 * self.u, zmax(ey + 0.62 * self.u, ex, 0.06) + L]),
            "brow.R": np.array([-ex * 1.02, ey + 0.62 * self.u, zmax(ey + 0.62 * self.u, -ex, 0.06) + L]),
            "nose": np.array([0.0, nose_y, zmax(nose_y) + 0.01]),
            "mouth": np.array([0.0, mouth_y, zmax(mouth_y) + 0.015]),
        }
        a = math.radians(24)
        self.nrm = {"eye.L": (math.sin(a), 0.02, math.cos(a)), "eye.R": (-math.sin(a), 0.02, math.cos(a)),
                    "brow.L": (math.sin(a), 0.2, math.cos(a)), "brow.R": (-math.sin(a), 0.2, math.cos(a)),
                    "nose": (0, -0.05, 1), "mouth": (0, -0.1, 1)}
        self.nrm = {k: np.asarray(v, float) / np.linalg.norm(v) for k, v in self.nrm.items()}

    # ------------------------------------------------------------------------------------------- frames
    def _frame(self, G, cam, key):
        M = G[self.hi]
        R, t = M[:, :3].astype(np.float64), M[:, 3].astype(np.float64)
        p = R @ self.pts[key] + t
        n = R @ self.nrm[key]
        right, up = R @ np.array([1.0, 0, 0]), R @ np.array([0, 1.0, 0])
        ax = right - n * np.dot(right, n)
        ay = up - n * np.dot(up, n)
        ax /= np.linalg.norm(ax)
        ay /= np.linalg.norm(ay)
        u = self.u * (0.85 if key == "nose" else 0.8 if key == "mouth" else 1.0)
        xy, z = cam.project(np.array([p, p + ax * u, p + ay * u]))
        v = cam.eye() - p
        facing = float(np.dot(n, v / np.linalg.norm(v)))
        # yaw of the head relative to the camera (−1 left … +1 right) for the 3/4 nose
        fwd = R @ np.array([0, 0, 1.0])
        cr = np.cross(v / np.linalg.norm(v), fwd)
        turn = float(cr[1])
        return xy[0], xy[1] - xy[0], xy[2] - xy[0], z[0], facing, turn

    @staticmethod
    def _vis(depth, ss, xy, z, r=2, tol=0.4):
        if depth is None:
            return 1.0
        H, W = depth.shape
        x, y = int(xy[0] * ss), int(xy[1] * ss)
        if not (0 <= x < W and 0 <= y < H):
            return 0.0
        return 1.0 if depth[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].min() > z - tol else 0.0

    def blink(self, t, seed):
        rng = np.random.default_rng(seed)
        times = np.cumsum(rng.uniform(2.4, 5.0, 80)) + rng.uniform(0.5, 2.0)
        k = int(np.searchsorted(times, t))
        d = min(abs(t - times[max(0, k - 1)]), abs(times[min(len(times) - 1, k)] - t))
        return 1.0 - _sm(1 - d / 0.085) if d < 0.085 else 1.0

    # ------------------------------------------------------------------------------------------- drawing utils
    @staticmethod
    def _map(c, ax, ay, P):
        P = np.asarray(P, float)
        return np.c_[c[0] + ax[0] * P[:, 0] + ay[0] * P[:, 1], c[1] + ax[1] * P[:, 0] + ay[1] * P[:, 1]]

    def _stroke(self, ctx, Q, w0, w1, a, ink, taper=0.35):
        """variable-width ink stroke (thick in the middle, tapered ends) as a filled ribbon."""
        if len(Q) < 2:
            return
        d = np.gradient(Q, axis=0)
        nrm = np.c_[-d[:, 1], d[:, 0]]
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
        s = np.linspace(0, 1, len(Q))
        prof = np.minimum(1, np.minimum(s, 1 - s) / max(1e-3, taper)) ** 0.7
        w = (w0 + (w1 - w0) * np.sin(math.pi * s)) * (0.25 + 0.75 * prof)
        Lp, Rp = Q + nrm * (w / 2)[:, None], Q - nrm * (w / 2)[:, None]
        ctx.set_source_rgba(*ink, a)
        ctx.new_path()
        ctx.move_to(*Lp[0])
        for q in Lp[1:]:
            ctx.line_to(*q)
        for q in Rp[::-1]:
            ctx.line_to(*q)
        ctx.close_path()
        ctx.fill()

    def _poly_path(self, ctx, Q):
        ctx.new_path()
        ctx.move_to(*Q[0])
        for q in Q[1:]:
            ctx.line_to(*q)
        ctx.close_path()

    # ------------------------------------------------------------------------------------------- features
    def _eye(self, ctx, c, ax, ay, side, open_, gaze, a, ink, W, happy):
        sg = 1.0 if side == "L" else -1.0           # +x of the local frame = character's left
        # almond: inner corner lower & rounder, outer corner lifted and pointed
        inner = np.array([-sg * 0.95, -0.04])
        outer = np.array([sg * 1.0, 0.1])
        if happy > 0.5 or open_ < 0.15:              # closed / smiling: a soft downward or upward arc + lashes
            bend = 0.32 if happy > 0.5 else -0.1
            up = _bez(inner, inner + [sg * 0.6, bend], outer + [-sg * 0.5, bend], outer)
            self._stroke(ctx, self._map(c, ax, ay, up), W * 0.7, W * 1.5, a, ink)
            return
        o = open_
        top = _bez(inner, np.array([-sg * 0.6, 0.66 * o]), np.array([sg * 0.45, 0.74 * o + 0.04]), outer, 22)
        bot = _bez(outer, np.array([sg * 0.6, -0.42 * o + 0.02]), np.array([-sg * 0.45, -0.5 * o]), inner, 18)
        lid = np.vstack([top, bot[1:]])
        Qlid = self._map(c, ax, ay, lid)
        # white of the eye (very soft) + iris clipped by the lids
        ctx.save()
        self._poly_path(ctx, Qlid)
        ctx.clip()
        gx = float(np.clip(gaze[0], -1, 1)) * 0.28 * sg * sg
        gy = float(np.clip(gaze[1], -1, 1)) * 0.1
        ir = 0.6
        icx, icy = gx + 0.0 * sg, 0.02 + gy
        th = np.linspace(0, 2 * math.pi, 40)
        iris = self._map(c, ax, ay, np.c_[icx + ir * 0.92 * np.cos(th), icy + ir * np.sin(th)])
        self._poly_path(ctx, iris)
        ctx.set_source_rgba(0.30, 0.27, 0.25, a)
        ctx.fill_preserve()
        ctx.set_source_rgba(*ink, a * 0.9)
        ctx.set_line_width(W * 0.45)
        ctx.stroke()
        # inner iris ring + pupil
        ring = self._map(c, ax, ay, np.c_[icx + ir * 0.62 * np.cos(th), icy + ir * 0.66 * np.sin(th)])
        self._poly_path(ctx, ring)
        ctx.set_source_rgba(0.18, 0.16, 0.15, a)
        ctx.fill()
        pup = self._map(c, ax, ay, np.c_[icx + 0.24 * np.cos(th), icy + 0.27 * np.sin(th)])
        self._poly_path(ctx, pup)
        ctx.set_source_rgba(0.04, 0.04, 0.045, a)
        ctx.fill()
        for hx, hy, hr in ((icx - 0.2 * sg, icy + 0.22, 0.16), (icx + 0.2 * sg, icy - 0.2, 0.07)):
            hl = self._map(c, ax, ay, np.c_[hx + hr * np.cos(th), hy + hr * np.sin(th)])
            self._poly_path(ctx, hl)
            ctx.set_source_rgba(1, 1, 1, a * 0.95)
            ctx.fill()
        ctx.restore()
        # upper lash line: heavy, thickest towards the outer corner, ends in a lifted flick
        Qt = self._map(c, ax, ay, top)
        self._stroke(ctx, Qt, W * 0.7, W * 1.7, a, ink, taper=0.2)
        if self.lashes:
            fl = _bez(outer - [sg * 0.1, 0.0], outer + [sg * 0.06, 0.03], outer + [sg * 0.14, 0.07],
                      outer + [sg * 0.2, 0.14], 8)
            self._stroke(ctx, self._map(c, ax, ay, fl), W * 0.9, W * 0.6, a, ink, taper=0.5)
        # lower lid: short thin line on the outer two-thirds only
        lb = bot[3:13]
        self._stroke(ctx, self._map(c, ax, ay, lb), W * 0.2, W * 0.5, a * 0.6, ink, taper=0.5)
        # double-eyelid crease (small, soft)
        cr = _bez(np.array([-sg * 0.35, 0.82 * o + 0.06]), np.array([-sg * 0.05, 0.98 * o + 0.08]),
                  np.array([sg * 0.4, 0.98 * o + 0.08]), np.array([sg * 0.72, 0.74 * o + 0.1]), 12)
        self._stroke(ctx, self._map(c, ax, ay, cr), W * 0.25, W * 0.5, a * 0.55, ink, taper=0.5)

    def _brow(self, ctx, c, ax, ay, side, a, ink, W, lift):
        sg = 1.0 if side == "L" else -1.0
        b = _bez(np.array([-sg * 0.85, 0.0 + lift]), np.array([-sg * 0.3, 0.32 + lift]),
                 np.array([sg * 0.45, 0.36 + lift]), np.array([sg * 1.05, 0.05 + lift * 0.6]), 16)
        self._stroke(ctx, self._map(c, ax, ay, b), W * 0.7 if not self.female else W * 0.4,
                     W * 1.4 if not self.female else W * 0.95, a * 0.9, ink, taper=0.4)

    def _nose(self, ctx, c, ax, ay, turn, a, ink, W):
        """3/4 nose: the shadow side of the bridge (a soft vertical line on the side turned away), the rounded
        tip and one nostril wing. Front view: just the tip curve + two tiny nostril ticks."""
        s = float(np.clip(turn * 1.6, -1, 1))          # >0 head turned to its left in the image
        k = abs(s)
        side = 1.0 if s >= 0 else -1.0
        # tip: a small rounded "u", shifted toward the turn
        tip = _bez(np.array([-0.22 + 0.15 * s, 0.02]), np.array([-0.12 + 0.15 * s, -0.16]),
                   np.array([0.14 + 0.15 * s, -0.16]), np.array([0.24 + 0.15 * s, 0.0]), 12)
        self._stroke(ctx, self._map(c, ax, ay, tip), W * 0.35, W * 0.8, a * 0.95, ink, taper=0.45)
        if k > 0.15:                                      # bridge side line (only in 3/4)
            br = _bez(np.array([-side * 0.12, 1.15]), np.array([-side * 0.16, 0.8]),
                      np.array([-side * 0.2, 0.42]), np.array([-side * 0.26 + 0.1 * s, 0.12]), 12)
            self._stroke(ctx, self._map(c, ax, ay, br), W * 0.2, W * 0.5, a * 0.55 * min(1, k * 1.5), ink, taper=0.5)
            wing = _bez(np.array([-side * 0.24, 0.1]), np.array([-side * 0.42, 0.06]),
                        np.array([-side * 0.44, -0.12]), np.array([-side * 0.3, -0.16]), 8)
            self._stroke(ctx, self._map(c, ax, ay, wing), W * 0.25, W * 0.55, a * 0.7, ink, taper=0.5)
        else:
            for sg in (-1, 1):
                nt = np.array([[sg * 0.2, -0.08], [sg * 0.3, -0.13]])
                self._stroke(ctx, self._map(c, ax, ay, nt), W * 0.3, W * 0.45, a * 0.6, ink, taper=0.5)

    def _mouth(self, ctx, c, ax, ay, smile, talk, a, ink, W, turn):
        """soft small lips: upper lip line with a gentle cupid's bow and lifted corners, a short lower-lip curve
        underneath; when talking the lips part a little."""
        s = float(np.clip(turn * 1.2, -1, 1)) * 0.12
        wd = 0.58 + 0.1 * smile
        lift = 0.06 + 0.16 * smile
        op = 0.12 * talk
        up = np.array([[-wd, lift], [-wd * 0.55, 0.0], [-0.12, 0.02], [0.0, -0.03], [0.12, 0.02], [wd * 0.55, 0.0],
                       [wd, lift]])
        xs = np.linspace(-wd, wd, 30)
        uy = np.interp(xs, up[:, 0], up[:, 1])
        Q = np.c_[xs + s, uy]
        self._stroke(ctx, self._map(c, ax, ay, Q), W * 0.4, W * 1.25, a, ink, taper=0.3)
        lw = 0.32 + 0.05 * smile
        lo = _bez(np.array([-lw + s, -0.2 - op]), np.array([-lw * 0.4 + s, -0.33 - op]),
                  np.array([lw * 0.4 + s, -0.33 - op]), np.array([lw + s, -0.2 - op]), 12)
        self._stroke(ctx, self._map(c, ax, ay, lo), W * 0.25, W * 0.7, a * (0.55 if self.female else 0.35), ink,
                     taper=0.5)
        if op > 0.02:
            ins = np.c_[np.linspace(-wd * 0.6, wd * 0.6, 12) + s, np.full(12, -0.05 - op * 0.6)]
            self._stroke(ctx, self._map(c, ax, ay, ins), W * 0.3, W * 0.6, a * 0.7, ink, taper=0.5)

    # ------------------------------------------------------------------------------------------- public
    def draw(self, ctx, G, cam, depth=None, ss=1.0, t=0.0, look=None, smile=0.35, talk=0.0, happy=0.0,
             blink_seed=1, alpha=1.0, ink=(0.94, 0.93, 0.9), width=2.0):
        import cairo
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        open_ = self.blink(t, blink_seed)
        W = width
        for side in ("L", "R"):
            c, ax, ay, z, facing, turn = self._frame(G, cam, f"eye.{side}")
            a = _sm((facing - 0.1) / 0.25) * self._vis(depth, ss, c, z) * alpha
            if a > 0.02:
                gaze = np.zeros(2)
                if look is not None:
                    lp, _ = cam.project(np.asarray(look, float)[None])
                    d = lp[0] - c
                    # gaze in the eye's own frame
                    A = np.c_[ax, ay]
                    try:
                        g = np.linalg.solve(A, d)
                        gaze = g / max(1.0, np.linalg.norm(g) / 1.0) * 0.6 / max(0.6, np.linalg.norm(g) / 6)
                        gaze = np.clip(g / 6.0, -1, 1)
                    except np.linalg.LinAlgError:
                        pass
                self._eye(ctx, c, ax, ay, side, open_, gaze, a, ink, W, happy)
            c, ax, ay, z, facing, turn = self._frame(G, cam, f"brow.{side}")
            a = _sm((facing - 0.08) / 0.25) * self._vis(depth, ss, c, z) * alpha
            if a > 0.02:
                self._brow(ctx, c, ax, ay, side, a, ink, W, 0.08 * smile)
        c, ax, ay, z, facing, turn = self._frame(G, cam, "nose")
        a = _sm((facing + 0.25) / 0.3) * self._vis(depth, ss, c, z) * alpha
        if a > 0.02:
            self._nose(ctx, c, ax, ay, turn, a, ink, W)
        c, ax, ay, z, facing, turn = self._frame(G, cam, "mouth")
        a = _sm((facing + 0.15) / 0.3) * self._vis(depth, ss, c, z) * alpha
        if a > 0.02:
            tk = talk * (0.5 + 0.5 * math.sin(t * 17.0)) if talk else 0.0
            self._mouth(ctx, c, ax, ay, smile, tk, a, ink, W, turn)
