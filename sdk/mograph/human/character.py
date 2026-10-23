"""
mograph.human.character — the practical API: one Character = shape + outfit + face style; pose → line art.

    from mograph.human.character import Character, preset
    ch = Character(**preset("office_man"))
    img = ch.render(pose="walk", phase=0.2, yaw=70, W=720, H=1280)          # RGBA (transparent)
    svg = ch.svg(pose="sit", yaw=40)

Face styles (line-art friendly — never a photoreal face):
    "none"     blank head (pure silhouette, Swiss/editorial style)
    "minimal"  ear + jaw contour only (default — the reference "faceless" style)
    "simple"   + dot eyes, brow strokes, a nose hook and a short mouth line (picked at the right 3D points, so they
               turn with the head)
Body-hiding rule: skin covered by clothing is NOT drawn (clipped by each garment's cover field) and the occluder
geometry still hides lines behind it, so the drawing has no see-through junk.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .body import BODY_N, Body, bone_index, bones, group_vertices, rot_axis
from .garments import Garment, build_outfit
from .lineart import Camera, LineFrame, MeshTopo, Part, Stroke3D, Style, draw_strokes, render_lines, strokes_svg
from . import poses as PZ
from .regions import regions, smoothstep


PRESETS = {
    "office_man":    dict(body=dict(gender=1, age=34, muscle=0.5, weight=0.5, height=0.55),
                          outfit=dict(top="shirt", bottom="trousers", shoes="shoes", hair="side", tuck=True)),
    "office_woman":  dict(body=dict(gender=0, age=30, muscle=0.45, weight=0.45, height=0.5),
                          outfit=dict(top="blouse", bottom="pencil_skirt", length=0.62, shoes="heels", hair="bob")),
    "casual_man":    dict(body=dict(gender=1, age=26, muscle=0.55, weight=0.5),
                          outfit=dict(top="tshirt", bottom="jeans", shoes="sneakers", hair="short")),
    "casual_woman":  dict(body=dict(gender=0, age=25, muscle=0.5, weight=0.45),
                          outfit=dict(top="tshirt", bottom="jeans", shoes="sneakers", hair="ponytail")),
    "business_man":  dict(body=dict(gender=1, age=42, muscle=0.5, weight=0.6, height=0.6),
                          outfit=dict(top="jacket", bottom="trousers", shoes="shoes", hair="short")),
    "dress_woman":   dict(body=dict(gender=0, age=27, muscle=0.45, weight=0.45),
                          outfit=dict(dress="knee", sleeve=0.25, shoes="heels", hair="long")),
    "girl":          dict(body=dict(gender=0, age=7, muscle=0.5, weight=0.5),
                          outfit=dict(dress="knee", sleeve=0.3, shoes="sneakers", hair="ponytail")),
    "boy":           dict(body=dict(gender=1, age=8, muscle=0.5, weight=0.5),
                          outfit=dict(top="tshirt", bottom="shorts", shoes="sneakers", hair="short")),
    "teen":          dict(body=dict(gender=1, age=16, muscle=0.5, weight=0.45, height=0.55),
                          outfit=dict(top="hoodie", bottom="jeans", shoes="sneakers", hair="side")),
    "old_man":       dict(body=dict(gender=1, age=76, muscle=0.35, weight=0.55, height=0.4),
                          outfit=dict(top="sweater", bottom="trousers", shoes="shoes", hair="buzz"),
                          stoop=1.0),
    "old_woman":     dict(body=dict(gender=0, age=74, muscle=0.35, weight=0.6, height=0.35),
                          outfit=dict(top="longsleeve", bottom="skirt", length=0.85, shoes="shoes", hair="bun"),
                          stoop=1.0),
    "heavy_man":     dict(body=dict(gender=1, age=45, muscle=0.35, weight=1.0, fat=1.0, belly=0.8),
                          outfit=dict(top="tshirt", bottom="trousers", shoes="sneakers", hair="short")),
    "heavy_woman":   dict(body=dict(gender=0, age=40, muscle=0.35, weight=1.0, fat=0.9, hips=0.4),
                          outfit=dict(top="longsleeve", bottom="trousers", shoes="shoes", hair="bob")),
    "athlete":       dict(body=dict(gender=1, age=27, muscle=1.0, weight=0.6, height=0.65, shoulders=0.5),
                          outfit=dict(top="tank", bottom="shorts", shoes="sneakers", hair="buzz")),
    "slim_woman":    dict(body=dict(gender=0, age=23, muscle=0.4, weight=0.2, height=0.7, fat=-0.6),
                          outfit=dict(top="tank", bottom="trousers", shoes="sneakers", hair="long")),
    "coat_woman":    dict(body=dict(gender=0, age=35, muscle=0.45, weight=0.5),
                          outfit=dict(top="coat", bottom="trousers", shoes="boots", hair="long")),
}


def preset(name: str, **override) -> dict:
    import copy
    p = copy.deepcopy(PRESETS[name])
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(p.get(k), dict):
            p[k].update(v)
        else:
            p[k] = v
    return p


@dataclass
class _Built:
    body: Body
    garments: list
    skin_clip: np.ndarray


class Character:
    def __init__(self, body: dict | None = None, outfit: dict | None = None, face: str = "minimal",
                 style: Style | None = None, stoop: float = 0.0, line_scale: float = 1.0):
        self.body = Body(**(body or {}))
        self.outfit = outfit or {}
        self.face = face
        self.style = style or Style()
        self.stoop = stoop
        self.line_scale = line_scale
        self.R = regions(self.body)
        self.garments = build_outfit(self.body, self.outfit)
        # skin clip: >0 where some garment covers the body (contours there are dropped)
        clip = np.full(BODY_N, -1.0)
        for g in self.garments:
            if g.cover is not None and g.kind != "hair":
                clip = np.maximum(clip, g.cover[:BODY_N])
        self.skin_clip = np.r_[clip, np.full(len(self.body.rest) - BODY_N, 1.0)]
        self._lm_idx = self._face_landmarks()

    # --------------------------------------------------------------------------------------------- pose
    def pose(self, pose="stand", **kw) -> dict:
        p = PZ.get(pose, **kw) if isinstance(pose, str) else dict(pose)
        if self.stoop:
            sp = list(p.get("spine", (0, 0, 0))) + [0] * 3
            nk = list(p.get("neck", (0, 0, 0))) + [0] * 3
            p["spine"] = (sp[0] + 14 * self.stoop, sp[1], sp[2])
            p["neck"] = (nk[0] - 2 * self.stoop, nk[1], nk[2])
            for s in ("L", "R"):
                lg = p.get(f"leg.{s}")
                if lg and lg.get("shin") is not None and lg.get("thigh") is not None:
                    th, sh = lg["thigh"], lg["shin"]
                    if len(th) == 2:
                        lg["thigh"] = (th[0], th[1] + 6 * self.stoop)
                        lg["shin"] = (sh[0], sh[1] - 2 * self.stoop)
        return p

    def posed(self, pose="stand", **kw):
        p = self.pose(pose, **kw)
        G, X = PZ.pose_body(self.body, p)
        dy = G[bone_index("root"), 1, 3] - self.body.pose_matrices(PZ.solve(self.body, p)[0], (0, 0, 0),
                                                                    PZ.root_rot(p))[bone_index("root"), 1, 3]
        return p, G, X

    # --------------------------------------------------------------------------------------------- face
    def _face_landmarks(self):
        """vertex ids for simple face marks (rest mesh, picked geometrically → robust to shape)."""
        X = self.body.rest
        L = self.R.lm
        hv = np.nonzero(self.R.head[:BODY_N] > 0.6)[0]
        P = X[hv]
        out = {}
        e = X[group_vertices("joint-l-eye")].mean(0)
        out["eye"] = {s: e * np.array([sg, 1, 1]) for s, sg in (("L", 1), ("R", -1))}
        out["hv"] = hv
        return out

    def _nearest_surface(self, X_rest, pts):
        hv = self._lm_idx["hv"]
        P = X_rest[hv]
        ids = []
        for p in pts:
            ids.append(hv[np.argmin(np.linalg.norm(P - p, axis=1))])
        return np.asarray(ids)

    def face_strokes(self, G, X):
        """3D face marks for style "simple": skinned by snapping to surface vertices on the rest mesh."""
        if self.face != "simple":
            return []
        b = self.body
        Xr = b.rest
        L = self.R.lm
        e = self._lm_idx["eye"]
        out = []
        hc = L["head_c"]

        def onskin(rest_pts, lift=0.04):
            ids = self._nearest_surface(Xr, rest_pts)
            n = X[ids] - b.skin(G, Xr[ids] * 0 + hc[None], None)[0:1] if False else None
            P = X[ids]
            # push slightly outward from the head centre so it is not z-fighting the skin
            hcp = b.joint(G, "head") if False else None
            return P
        for s, sg in (("L", 1), ("R", -1)):
            c = e[s] + np.array([0, -0.02, 0.32])
            # eye = tiny closed loop
            loop = [c + np.array([0.09 * math.cos(t), 0.06 * math.sin(t), 0]) for t in np.linspace(0, 2 * math.pi, 9)]
            out.append(Stroke3D(onskin(loop), 0.55, "eye", 0.0))
            brow = [e[s] + np.array([sg * dx, 0.3 + 0.05 * math.cos(dx * 2.5), 0.28]) for dx in
                    np.linspace(-0.05, 0.32, 5)]
            out.append(Stroke3D(onskin(brow), 0.5, "brow", 0.45))
        mouth_y = L["eye"] - 0.95 * (L["eye"] - L["chin"]) * 0.78
        m = [np.array([dx, mouth_y, hc[2] + 1.15]) for dx in np.linspace(-0.22, 0.22, 5)]
        out.append(Stroke3D(onskin(m), 0.45, "mouth", 0.5))
        return out

    # --------------------------------------------------------------------------------------------- parts
    def parts(self, G, X):
        b = self.body
        R = self.R
        sm = np.clip(1 - 0.85 * self.R.hand - 0.3 * self.R.foot, 0.1, 1.0)
        skin_clip = self.skin_clip.copy()
        mj = np.full(len(X), 0.32)
        if self.face in ("none", "minimal", "simple"):
            # inner face contours (nose wings, lips, eye sockets) are dropped; the face SILHOUETTE (profile against
            # the background or the far shoulder) still has a big depth jump, so it stays.
            mj = mj + R.face * 2.2 + R.head * 0.25
        # covered skin is NOT an occluder (it sits inside the clothes; where it pokes through a shell it would
        # punch holes in garment lines). A thin margin keeps the skin right at hems/cuffs/collars.
        occ = ~(self.skin_clip[R.tris] > 0.06).all(1)
        body_part = Part("body", X, R.tris, R.topo, pid=1, smooth_iters=5, smooth_w=sm, min_jump=mj,
                         clip=skin_clip, occ_tris=occ)
        parts = [body_part]
        for k, g in enumerate(self.garments):
            V = b.skin(G, g.rest, g.W)
            parts.append(Part(g.name, V, g.tris, g.topo, pid=2 + k, smooth_iters=g.smooth_iters,
                              min_jump=g.min_jump, weight=g.weight, boundary=g.hem, kind=g.kind))
        return parts

    # --------------------------------------------------------------------------------------------- render
    def camera_for(self, X, yaw, W, H, pitch=4.0, fov=18.0, margin=0.06, frame=None):
        cam = Camera(yaw=yaw, pitch=pitch, fov=fov, W=W, H=H)
        P = X[:BODY_N] if frame is None else frame
        cam.frame(P, margin)
        return cam

    def lines(self, pose="stand", yaw=20.0, W=720, H=1280, cam: Camera | None = None, ss=1.5, seed=0, **kw):
        p, G, X = self.posed(pose, **kw)
        cam = cam or self.camera_for(X, yaw, W, H)
        parts = self.parts(G, X)
        extra = self.face_strokes(G, X)
        fig_px = self._fig_px(cam, X)
        st = self.style
        lf = render_lines(parts, cam, st, extra=extra, ss=ss, fig_px=fig_px * self.line_scale, seed=seed)
        lf.cam = cam
        lf.G, lf.X = G, X
        return lf

    def _fig_px(self, cam, X):
        xy, _ = cam.project(X[:BODY_N][::40])
        return max(200.0, float(np.ptp(xy[:, 1])))

    def render(self, pose="stand", yaw=20.0, W=720, H=1280, cam=None, bg=None, **kw):
        lf = self.lines(pose, yaw, W, H, cam, **kw)
        st = Style(**{**self.style.__dict__, "bg": bg})
        return draw_strokes(lf.strokes, lf.cam.W, lf.cam.H, st)

    def svg(self, pose="stand", yaw=20.0, W=720, H=1280, cam=None, **kw):
        lf = self.lines(pose, yaw, W, H, cam, **kw)
        return strokes_svg(lf.strokes, lf.cam.W, lf.cam.H, self.style)
