"""
mograph.mocap.director — direct REAL mocap actors in one 3D scene; everything they do comes from captured motion,
with only small additive "acting" layers on top (look-at, lean, reach IK) so they interact with each other.

Actor timeline = mocap SEGMENTS (crossfaded) + additive layers:
    seg   {"clip": alias | "ual1:Walk_Loop" | file, "t": 2.0, "in": 0.4, "out": 3.1, "speed": 1, "loop": true,
           "fade": 0.5, "at": [x, z], "heading": deg, "root": "file" | "in_place"}
          heading = facing at the segment start (0 = +Z = towards a camera at yaw 0, 90 = +X).
          Recorded root motion is rotated by the heading → a captured walk keeps every real step and goes where
          aimed. No "at": continues from the previous segment's end (position AND heading, so there is no pop).
    look  {"t", "dur", "target": "girl:head" | [x,y,z], "hold", "out", "w", "tilt"}
    lean  {"t", "dur", "pitch", "side", "twist", "hold", "out"}
    reach {"t", "dur", "arm": "L"|"R", "target": "woman:shoulder.R", "offset": [x,y,z], "hold", "out", "palm"}

All additive layers rotate a bone and its descendants about the joint in WORLD space on the skinning matrices,
so the mocap underneath keeps breathing, swaying and shifting weight.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from ..human.body import bones
from ..human.poses import rmin
from .library import resolve
from .retarget import MocapMotion


def smooth(u):
    u = min(1.0, max(0.0, u))
    return u * u * u * (u * (6 * u - 15) + 10)


def env(t, t0, dur, hold=None, out=None):
    if t < t0:
        return 0.0
    a = smooth((t - t0) / max(1e-6, dur))
    if hold is None:
        return a
    t2 = t0 + dur + hold
    if t < t2:
        return a
    return a * (1 - smooth((t - t2) / max(1e-6, out or dur)))


def roty(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]], np.float64)


def axis_angle(axis, ang):
    a = np.asarray(axis, np.float64)
    n = np.linalg.norm(a)
    if n < 1e-12 or abs(ang) < 1e-9:
        return np.eye(3)
    return Rot.from_rotvec(a / n * ang).as_matrix()


def frac_rot(R, f):
    return Rot.from_rotvec(Rot.from_matrix(R).as_rotvec() * f).as_matrix()


def heading_of(R):
    f = np.asarray(R, np.float64) @ np.array([0, 0, 1.0])
    return math.degrees(math.atan2(f[0], f[2]))


class Segment:
    def __init__(self, actor, spec, base_dir=None):
        self.spec = dict(spec)
        f, c = resolve(spec["clip"], spec.get("name"), base_dir)
        self.mo = MocapMotion(f, c, body=actor.ch.body,
                              root_motion="plant" if spec.get("root") == "plant" else "file",
                              speed=spec.get("speed", 1.0), loop=spec.get("loop", True),
                              mirror=spec.get("mirror", False))
        self.mo.trim = (float(spec.get("in", 0.0)), float(spec.get("out", self.mo.duration)))
        if "in" in spec or "out" in spec or spec.get("speed", 1.0) != 1.0:
            self.mo._precompute(30, True)               # ground over the USED range only (no floating feet)
        self.mo._ground += float(spec.get("lift", 0.0))
        self.t0 = float(spec.get("t", 0.0))
        self.fade = float(spec.get("fade", 0.5))
        self.root = spec.get("root", "file")
        R0, off0 = self.mo.state(0.0)
        self.off0 = off0.copy()
        self.yaw0 = heading_of(R0["root"])
        # "anchor": "world" keeps the capture's own floor coordinates (two people recorded together stay in their
        # recorded relation); "rotate" then turns the whole capture volume, "at" moves its origin.
        self.world = spec.get("anchor") == "world"
        if self.world:
            sc = float(spec.get("pos_scale", 9.5)) / self.mo.scale
            self.off0 = np.zeros(3)
            self._sc = sc
        self.Y = np.eye(3)
        self.P = np.zeros(3)

    def length(self):
        return (self.mo.trim[1] - self.mo.trim[0]) / max(1e-6, self.mo.speed)

    def place(self, prev_end, prev_heading):
        s = self.spec
        heading = s.get("heading", prev_heading)
        self.Y = roty(s["rotate"]) if "rotate" in s else roty(heading - self.yaw0)
        if "at" in s:
            self.P = np.array([s["at"][0], 0, s["at"][1]], np.float64)
        elif prev_end is not None:
            self.P = prev_end.copy()
        self.heading = heading
        if "end_at" in s:
            L = self.length() if not self.mo.loop else float(s.get("dur", self.length()))
            _, off = self.mo.state(min(L, self.length()) - 1e-3)
            o = off - self.off0
            if self.world:
                o = o * self._sc
            d = self.Y @ np.array([o[0], 0, o[2]])
            self.P = np.array([s["end_at"][0], 0, s["end_at"][1]], np.float64) - d

    def _filtered(self, lt):
        """temporal low-pass of the capture (Gaussian over ±2 samples of `smooth_s`): removes marker jitter, keeps
        every real movement (a 60-80 ms window is far shorter than any human gesture)."""
        sw = float(self.spec.get("smooth_s", 0.035))
        if sw <= 0:
            return self.mo.state(lt)
        L = self.length() - 1e-4
        ks = (-2, -1, 0, 1, 2)
        wk = np.exp(-0.5 * np.array(ks, float) ** 2 / 1.1)
        Rs, offs = [], []
        for k in ks:
            u = lt + k * sw
            if not self.mo.loop:
                u = min(max(u, 0.0), L)
            Rs.append(self.mo.state(u))
        names = list(Rs[0][0])
        Q = np.stack([Rot.from_matrix(np.stack([r[0][n] for n in names]).astype(np.float64)).as_quat() for r in Rs])
        Q = np.where((Q * Q[2:3]).sum(-1, keepdims=True) < 0, -Q, Q)
        q = (Q * wk[:, None, None]).sum(0)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        M = Rot.from_quat(q).as_matrix().astype(np.float32)
        off = sum(w * r[1] for w, r in zip(wk, Rs)) / wk.sum()
        return {n: M[i] for i, n in enumerate(names)}, off

    def state(self, t):
        lt = max(0.0, t - self.t0)
        if not self.mo.loop:
            lt = min(lt, self.length() - 1e-4)
        R, off = self._filtered(lt)
        R = dict(R)
        R["root"] = (self.Y @ R["root"]).astype(np.float32)
        o = off - self.off0
        if self.world:
            o = o * self._sc
        h = self.Y @ np.array([o[0], 0, o[2]]) if self.root in ("file", "plant") else np.zeros(3)
        return R, np.array([self.P[0] + h[0], off[1], self.P[2] + h[2]])


class Actor:
    def __init__(self, name, character, segments, look=(), lean=(), reach=(), base_dir=None, breath=1.0,
                 seed=0):
        self.name = name
        self.breath = breath
        self.seed = seed
        self.ch = character
        self.segs = sorted((Segment(self, s, base_dir) for s in segments), key=lambda s: s.t0)
        # chain placement: a segment without "at" starts where the previous one IS at its start time
        pe, ph = None, 0.0
        for k, s in enumerate(self.segs):
            if k > 0 and "at" not in s.spec:
                prev = self.segs[k - 1]
                R, off = prev.state(s.t0)
                pe = np.array([off[0], 0, off[2]])
                ph = heading_of(R["root"])
                # continue from the previous root: match the new clip's own root position offset
            s.place(pe, ph)
            if k > 0 and "at" not in s.spec and "end_at" not in s.spec:
                R1, off1 = s.state(s.t0)
                s.P = s.P + (pe - np.array([off1[0], 0, off1[2]]))
            if "heading" not in s.spec and k > 0 and "at" not in s.spec and "end_at" not in s.spec:
                # match facing exactly (the new clip's start pose may face slightly off its own +Z)
                R1, _ = s.state(s.t0)
                dh = ph - heading_of(R1["root"])
                s.Y = roty(dh) @ s.Y
                R2, off2 = s.state(s.t0)
                s.P = s.P + (pe - np.array([off2[0], 0, off2[2]]))
        self.look, self.lean, self.reach = list(look), list(lean), list(reach)
        B = bones()
        self.lut = {n: i for i, n in enumerate(B)}
        par = self.ch.body.parent
        kids = [[] for _ in B]
        for i, p in enumerate(par):
            if p >= 0:
                kids[p].append(i)
        self._desc = {}

        def desc(i):
            if i not in self._desc:
                out = [i]
                for k2 in kids[i]:
                    out += list(desc(k2))
                self._desc[i] = np.array(out)
            return self._desc[i]
        self.desc = desc
        self._G, self._t, self._Gpre = None, None, None

    def base_state(self, t):
        active = [s for s in self.segs if s.t0 <= t] or [self.segs[0]]
        cur = active[-1]
        R, off = cur.state(t)
        if len(active) >= 2 and t < cur.t0 + cur.fade:
            w = smooth((t - cur.t0) / max(1e-6, cur.fade))
            R0, off0 = active[-2].state(t)
            names = list(R)
            qa = Rot.from_matrix(np.stack([R0[n] for n in names]).astype(np.float64)).as_quat()
            qb = Rot.from_matrix(np.stack([R[n] for n in names]).astype(np.float64)).as_quat()
            qb = np.where((qa * qb).sum(1, keepdims=True) < 0, -qb, qb)
            q = qa * (1 - w) + qb * w
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            M = Rot.from_quat(q).as_matrix().astype(np.float32)
            R = {n: M[k] for k, n in enumerate(names)}
            off = off0 * (1 - w) + off * w
        return R, off

    def _rot_about(self, G, bone, pivot, D):
        ids = self.desc(self.lut[bone])
        G[ids, :, :3] = np.einsum("ij,njk->nik", D, G[ids, :, :3])
        G[ids, :, 3] = np.einsum("ij,nj->ni", D, G[ids, :, 3] - pivot) + pivot

    def joint(self, G, name, tail=False):
        i = self.lut[name]
        b = self.ch.body
        p = b.tails[i] if tail else b.heads[i]
        return G[i, :, :3] @ p + G[i, :, 3]

    def head_point(self, G):
        return self.joint(G, "head") * 0.4 + self.joint(G, "head", True) * 0.6

    def _apply_lean(self, G, t):
        for L in self.lean:
            w = env(t, L["t"], L.get("dur", 0.8), L.get("hold"), L.get("out"))
            if w <= 1e-4:
                continue
            fwd = G[self.lut["root"], :, :3] @ np.array([0, 0, 1.0])
            fwd[1] = 0
            fwd /= np.linalg.norm(fwd) + 1e-9
            left = np.cross(np.array([0, 1.0, 0]), fwd)
            for b, f in (("spine04", 0.2), ("spine03", 0.3), ("spine02", 0.3), ("spine01", 0.2)):
                D = axis_angle(left, math.radians(L.get("pitch", 0)) * w * f) @ \
                    axis_angle(fwd, -math.radians(L.get("side", 0)) * w * f) @ \
                    axis_angle((0, 1, 0), math.radians(L.get("twist", 0)) * w * f)
                self._rot_about(G, b, self.joint(G, b), D)

    def _apply_breath(self, G, t):
        if not self.breath:
            return
        ph = self.seed * 1.7
        b = math.sin(2 * math.pi * t / 3.9 + ph)                       # ~15 breaths / min
        fwd = G[self.lut["root"], :, :3] @ np.array([0, 0, 1.0])
        fwd[1] = 0
        fwd /= np.linalg.norm(fwd) + 1e-9
        left = np.cross(np.array([0, 1.0, 0]), fwd)
        a = math.radians(0.9) * b * self.breath
        for bn in ("spine02", "spine01"):
            self._rot_about(G, bn, self.joint(G, bn), axis_angle(left, -a * 0.5))
        self._rot_about(G, "neck01", self.joint(G, "neck01"), axis_angle(left, a * 0.8))
        # slow head drift (never perfectly still)
        hx = math.radians(0.8) * math.sin(2 * math.pi * t / 6.7 + ph) * self.breath
        hy = math.radians(1.1) * math.sin(2 * math.pi * t / 8.3 + ph * 1.3) * self.breath
        self._rot_about(G, "head", self.joint(G, "head"), axis_angle((0, 1, 0), hy) @ axis_angle(left, hx))

    def _apply_look(self, G, t, scene):
        for L in self.look:
            w = env(t, L["t"], L.get("dur", 0.7), L.get("hold"), L.get("out")) * L.get("w", 1.0)
            if w <= 1e-4:
                continue
            d = scene.point(L["target"], t, exclude=self) - self.head_point(G)
            f = G[self.lut["head"], :, :3] @ np.array([0, 0, 1.0])
            yaw = (math.atan2(d[0], d[2]) - math.atan2(f[0], f[2]) + math.pi) % (2 * math.pi) - math.pi
            yaw = max(-1.2, min(1.2, yaw)) * w
            pitch = math.atan2(d[1], math.hypot(d[0], d[2])) - math.atan2(f[1], math.hypot(f[0], f[2]))
            pitch = max(-0.7, min(0.6, pitch)) * w
            for b, fr in (("spine01", 0.12), ("neck01", 0.2), ("neck02", 0.2), ("neck03", 0.18), ("head", 0.3)):
                right = G[self.lut[b], :, :3] @ np.array([1.0, 0, 0])
                D = axis_angle(right, -pitch * fr) @ axis_angle((0, 1, 0), yaw * fr)
                self._rot_about(G, b, self.joint(G, b), D)
            if L.get("tilt"):
                fw = G[self.lut["head"], :, :3] @ np.array([0, 0, 1.0])
                self._rot_about(G, "head", self.joint(G, "head"), axis_angle(fw, math.radians(L["tilt"]) * w))

    def _apply_reach(self, G, t, scene):
        for L in self.reach:
            w = env(t, L["t"], L.get("dur", 1.0), L.get("hold"), L.get("out"))
            if w <= 1e-4:
                continue
            s = L.get("arm", "L")
            A = self.joint(G, f"upperarm01.{s}")
            B = self.joint(G, f"lowerarm01.{s}")
            C = self.joint(G, f"wrist.{s}")
            T = scene.point(L["target"], t, exclude=self) + np.asarray(L.get("offset", (0, 0, 0)), np.float64)
            # the hand travels on an arc (up and out, never through the body) and lands softly
            u = (t - L["t"]) / max(1e-6, L.get("dur", 1.0))
            lift = math.sin(math.pi * min(1.0, max(0.0, u))) if u < 1 else 0.0
            arc = np.asarray(L.get("arc", (0, 0.8, 0.5)), np.float64) * lift
            Tw = C + (T - C) * w + arc
            a, b = np.linalg.norm(B - A), np.linalg.norm(C - B)
            d = Tw - A
            dist = np.linalg.norm(d)
            dc = min(dist, (a + b) * 0.995)
            dn = d / (dist + 1e-9)
            pole = np.asarray(L.get("pole", (0, -1.0, -0.4)), np.float64)
            pole = pole - np.dot(pole, dn) * dn
            pole /= np.linalg.norm(pole) + 1e-9
            cosA = max(-1.0, min(1.0, (a * a + dc * dc - b * b) / (2 * a * dc)))
            Bn = A + dn * (a * cosA) + pole * (a * math.sqrt(max(0, 1 - cosA * cosA)))
            self._rot_about(G, f"upperarm01.{s}", A, frac_rot(rmin(B - A, Bn - A).astype(np.float64), 1.0))
            C1 = self.joint(G, f"wrist.{s}")
            B1 = self.joint(G, f"lowerarm01.{s}")
            self._rot_about(G, f"lowerarm01.{s}", B1, rmin(C1 - B1, A + dn * dc - B1).astype(np.float64))
            if L.get("palm"):
                Cw = self.joint(G, f"wrist.{s}")
                hd = self.joint(G, f"finger3-1.{s}") - Cw
                fore = Cw - self.joint(G, f"lowerarm01.{s}")
                fore /= np.linalg.norm(fore) + 1e-9
                want = fore * 0.5 + np.array([0, -1.0, 0]) + np.asarray(L.get("palm_dir", (0, 0, 0)), np.float64)
                self._rot_about(G, f"wrist.{s}", Cw, frac_rot(rmin(hd, want).astype(np.float64), w))

    def base_matrices(self, t):
        """pure mocap pose (no additive layers) — what OTHER actors aim at (no mutual recursion)."""
        if getattr(self, "_tb", None) != t:
            R, off = self.base_state(t)
            self._Gb, self._tb = self.segs[0].mo.build(R, off).astype(np.float64), t
        return self._Gb

    def matrices(self, t, scene, pre=False):
        if self._t != t:
            G = self.base_matrices(t).copy()
            self._apply_breath(G, t)
            self._apply_lean(G, t)
            self._apply_look(G, t, scene)
            self._Gpre = G.copy()
            self._apply_reach(G, t, scene)
            self._G, self._t = G, t
        return self._Gpre if pre else self._G

    def posed(self, t, scene):
        G = self.matrices(t, scene).astype(np.float32)
        return G, self.ch.body.skin(G)


class Scene:
    def __init__(self, actors):
        self.actors = {a.name: a for a in actors}

    def point(self, ref, t, exclude=None):
        if callable(ref):
            return np.asarray(ref(t), np.float64)
        if not isinstance(ref, str):
            return np.asarray(ref, np.float64)
        name, j = ref.split(":")
        a = self.actors[name]
        G = a.base_matrices(t)
        if j == "head":
            return a.head_point(G)
        if j.startswith("shoulder."):
            s = j.split(".")[1]
            return a.joint(G, f"shoulder01.{s}") * 0.45 + a.joint(G, f"upperarm01.{s}") * 0.55
        if j == "chest":
            return a.joint(G, "spine01")
        return a.joint(G, j)
