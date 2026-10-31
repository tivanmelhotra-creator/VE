"""
toonface.kit.facecomp — v32 professional face-rig compositor for blank-face cartoon video.

    comp = FaceRigCompositor("boy", frames)              # tracks + calibrates
    out  = comp.render(i, face_frame)                    # FaceFrame from expressive.ExpressivePerformer
    comp.qa()                                            # attachment / jitter / level / occlusion metrics

Pipeline
  1. TRACK      facetrack.HeadTracker — multi-region planar similarity (crown/left/right/jaw), One-Euro, limits.
  2. CALIBRATE  the kit's OWN blank plate (front / 3/4 / mirrored 3/4) is chamfer-fitted to the video face on the
                cheek + jaw outline (hairlines differ).  The features land where the character designer put them on
                that face: correct size, spacing and height.  No guessed "eye_span_ratio" (the cause of the tiny,
                close-set "silly" look in v31.4).
  3. LEVEL      LevelPolicy — eye line ≤ 1°, mouth line ≤ 3° relative to the head; calibration roll < 2.5° snaps
                to 0.  The face can only tilt when the TRACKED skull tilts beyond the dead-band.
  4. RENDER     vector features are drawn by cairo DIRECTLY in frame space through  T = M_track ∘ A_cal
                (no raster warp → crisp lines, no resampling shimmer, sub-pixel exact).
  5. OCCLUDE    per frame, skin pieces inside the tracked face are classified face / hand (how far they stick out
                of the face + new ink edges).  Only the covered part is hidden — a hand on the forehead hides a
                brow, not the mouth.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess

import cairo
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize

from .. import ink
from .blankplate import load_blank
from .expressive import FaceFrame
from .facetrack import HeadTracker, TrackerConfig, apply, face_component, invert, segment
from .rig25 import INK, Face, Kit, draw_brow, draw_eye, draw_furrow, draw_mouth, draw_nose, eye_geom
from .turn25 import PLATES


# ================================================================================================= helpers
def compose(M2: np.ndarray, M1: np.ndarray) -> np.ndarray:
    """2×3 affine: apply M1 then M2."""
    return (np.vstack([M2, [0, 0, 1]]) @ np.vstack([M1, [0, 0, 1]]))[:2]


def contour_pts(mask: np.ndarray) -> np.ndarray:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)


def lower_contour(mask: np.ndarray, keep_from: float = 0.38) -> np.ndarray:
    """cheek + jaw part of the face outline (hairline / forelock excluded)."""
    c = contour_pts(mask)
    y0, y1 = c[:, 1].min(), c[:, 1].max()
    return c[c[:, 1] >= y0 + keep_from * (y1 - y0)]


def row_profile(mask: np.ndarray):
    rows = np.where(mask.any(1))[0]
    x0 = np.array([np.argmax(mask[y]) for y in rows])
    x1 = np.array([mask.shape[1] - 1 - np.argmax(mask[y, ::-1]) for y in rows])
    return rows, x0, x1


def chin_offset(mask: np.ndarray) -> float:
    """(chin x − cheek-centre x) / face width: ≈0 front, ±0.1 for a 3/4 face (sign = facing direction)."""
    rows, x0, x1 = row_profile(mask)
    w = x1 - x0
    iy = int(np.argmax(w))
    below = np.where((np.arange(len(rows)) > iy) & (w < 0.33 * w.max()))[0]
    ic = below[0] if len(below) else len(rows) - 1
    band = slice(max(0, iy - len(rows) // 10), iy + len(rows) // 10)
    cheek = np.median(0.5 * (x0[band] + x1[band]))
    return float((0.5 * (x0[ic] + x1[ic]) - cheek) / max(1, w.max()))


def edge(m: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))


# ================================================================================================= level policy
class LevelPolicy:
    """explicit geometric rules that keep the drawn face LEVEL inside a non-rotating head."""

    def __init__(self, max_eye_tilt_deg: float = 1.0, max_mouth_tilt_deg: float = 3.0,
                 cal_roll_deadband: float = 2.5):
        self.max_eye = max_eye_tilt_deg
        self.max_mouth = max_mouth_tilt_deg
        self.cal_roll_deadband = cal_roll_deadband

    @staticmethod
    def _tilt(a, b):
        return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))

    def apply(self, view: dict) -> dict:
        v = {"eyes": {k: dict(e) for k, e in view["eyes"].items()},
             "brows": {k: dict(b) for k, b in view["brows"].items()},
             "nose": view["nose"].copy(), "mouth": {k: m.copy() for k, m in view["mouth"].items()}}
        cL, cR = v["eyes"]["L"]["outline"].mean(0), v["eyes"]["R"]["outline"].mean(0)
        tilt = self._tilt(cL, cR)
        if abs(tilt) > self.max_eye:
            excess = tilt - math.copysign(self.max_eye, tilt)
            half = 0.5 * math.tan(math.radians(excess)) * (cR[0] - cL[0])
            for s, dy in (("L", +half), ("R", -half)):
                e = v["eyes"][s]
                e["outline"] = e["outline"] + [0, dy]
                e["pupil"] = e["pupil"] + [0, dy]
                v["brows"][s]["line"] = v["brows"][s]["line"] + [0, dy]
        m = v["mouth"]
        mt = self._tilt(m["left"], m["right"])
        if abs(mt) > self.max_mouth:
            ex = -math.radians(mt - math.copysign(self.max_mouth, mt))
            c = 0.5 * (m["left"] + m["right"])
            R = np.array([[math.cos(ex), -math.sin(ex)], [math.sin(ex), math.cos(ex)]])
            for k in ("left", "right", "low"):
                m[k] = c + (m[k] - c) @ R.T
        return v


# ================================================================================================= drawing
def draw_face_sided(ctx, kit: Kit, view: dict, t_turn: float, ff: FaceFrame,
                    parts=("brows", "eyes", "nose", "mouth")):
    """rig25 renderer with an independent Face per side (asymmetric brows / lids) + secondary lines."""
    e, b = view["eyes"], view["brows"]
    xs, ty, by, *_ = eye_geom(e["L"], Face())
    eh = float(np.max(by - ty))
    if "brows" in parts:
        cl = draw_brow(ctx, b["L"], e["L"], ff.left, kit)
        cr = draw_brow(ctx, b["R"], e["R"], ff.right, kit)
        draw_furrow(ctx, cl[-1], cr[-1], eh, max(ff.left.furrow, ff.right.furrow), kit)
    if "eyes" in parts:
        draw_eye(ctx, e["L"], ff.left, kit)
        draw_eye(ctx, e["R"], ff.right, kit)
    nose = view["nose"] + [0.0, ff.nose_dy * eh]
    if "nose" in parts:
        draw_nose(ctx, nose, kit)
        if ff.wrinkle > 0.05:                                  # nose-bridge wrinkle (AU9)
            top = nose[np.argmin(nose[:, 1])]
            for sx in (-1, 1):
                p = np.array([top + [sx * 0.10 * eh, -0.30 * eh], top + [sx * 0.22 * eh, -0.42 * eh],
                              top + [sx * 0.30 * eh, -0.40 * eh]])
                ink.tapered(ctx, ink.catmull(p, 6), INK, 0.42 * kit.sw, alpha=min(1.0, ff.wrinkle))
    if "mouth" in parts:
        draw_mouth(ctx, view["mouth"], ff.mouth, kit, near=t_turn)
        m = view["mouth"]
        for side, amt in (("left", ff.cheek_line[0]), ("right", ff.cheek_line[1])):
            if amt < 0.05:                                     # laugh line (big smile pushes the cheek)
                continue
            c = m[side]
            sx = -1 if side == "left" else 1
            p = np.array([c + [sx * 0.10 * eh, -0.55 * eh], c + [sx * 0.20 * eh, -0.25 * eh],
                          c + [sx * 0.16 * eh, 0.02 * eh]])
            ink.tapered(ctx, ink.catmull(p, 6), INK, 0.40 * kit.sw, alpha=0.75 * amt)


def render_layer(W: int, H: int, T: np.ndarray, fn) -> np.ndarray:
    """premultiplied BGRA layer drawn with cairo through the 2×3 affine T (kit space → frame space)."""
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surf)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_matrix(cairo.Matrix(T[0, 0], T[1, 0], T[0, 1], T[1, 1], T[0, 2], T[1, 2]))
    fn(ctx)
    surf.flush()
    return np.ndarray((H, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :W].copy()


# ================================================================================================= compositor
class FaceRigCompositor:
    def __init__(self, kit_name: str, frames: list, yaw: float | None = None, plate: str | None = None,
                 tracker: TrackerConfig | None = None, level: LevelPolicy | None = None, fps: float = 24.0,
                 occlusion: bool = True, face_scale: float = 1.0, verbose: bool = True):
        self.kit = Kit(kit_name)
        self.frames = frames
        self.H, self.W = frames[0].shape[:2]
        self.level = level or LevelPolicy()
        self.occlusion = occlusion
        self.face_scale = face_scale
        self.verbose = verbose
        self.trk = HeadTracker(tracker or TrackerConfig(fps=fps))
        self.poses = self.trk.run(frames)
        self.skin_bgr, self.bg_bgr = self.trk.skin_bgr, self.trk.bg_bgr
        self.ref_face = self.trk.ref_face
        _, self.ref_ink, _ = segment(frames[self.trk.ref_idx], self.skin_bgr, self.bg_bgr)
        self._calibrate(yaw, plate)
        self.stats: list = []

    # ------------------------------------------------------------------------------------------ calibration
    def _plate_masks(self) -> dict:
        d = self.kit.d
        out = {}
        for v in ("front", "34"):
            img = cv2.imread(os.path.join(PLATES, d["plates"][v]))
            out[v] = load_blank(img, skin_seed=tuple(d["skin_seed"][v]), contour_cut_y=d["contour_cut_y"][v]).face
        out["34m"] = out["34"][:, ::-1].copy()
        return out

    def _fit(self, pmask: np.ndarray, vmask: np.ndarray):
        """symmetric chamfer fit plate → video on the cheek/jaw outline.  params: log s, log aspect, rot, tx, ty."""
        vc, pc = lower_contour(vmask), lower_contour(pmask)
        dt_v = cv2.distanceTransform((edge(vmask) == 0).astype(np.uint8), cv2.DIST_L2, 5)
        dt_p = cv2.distanceTransform((edge(pmask) == 0).astype(np.uint8), cv2.DIST_L2, 5)
        pr, px0, px1 = row_profile(pmask)
        vr, vx0, vx1 = row_profile(vmask)
        s0 = (vx1 - vx0).max() / max(1, (px1 - px0).max())
        pi, vi = int(np.argmax(px1 - px0)), int(np.argmax(vx1 - vx0))
        pa = np.array([0.5 * (px0[pi] + px1[pi]), pr[pi]])
        va = np.array([0.5 * (vx0[vi] + vx1[vi]), vr[vi]])
        vw = float((vx1 - vx0).max())
        tau = 12.0

        def M_of(p):
            ls, la, r, tx, ty = p
            s = math.exp(ls)
            sx, sy = s * math.exp(-0.5 * la), s * math.exp(0.5 * la)
            c, si = math.cos(r), math.sin(r)
            A = np.array([[c * sx, -si * sy], [si * sx, c * sy]])
            return np.c_[A, va + [tx, ty] - A @ pa]

        def samp(dt, pts):
            x = np.clip(pts[:, 0], 0, dt.shape[1] - 1).astype(int)
            y = np.clip(pts[:, 1], 0, dt.shape[0] - 1).astype(int)
            return np.minimum(dt[y, x], tau)

        def cost(p):
            M = M_of(p)
            return (samp(dt_v, apply(M, pc)).mean() + samp(dt_p, apply(invert(M), vc)).mean()
                    + 40.0 * p[1] ** 2 + 20.0 * p[2] ** 2)

        best = None
        for dy0 in (-0.06, 0.0, 0.06):
            r = minimize(cost, np.array([math.log(s0), 0.0, 0.0, 0.0, dy0 * vw]), method="Nelder-Mead",
                         options=dict(xatol=1e-4, fatol=1e-4, maxiter=2500))
            if best is None or r.fun < best.fun:
                best = r
        p = best.x.copy()
        p[1] = float(np.clip(p[1], -0.12, 0.12))
        if abs(math.degrees(p[2])) < self.level.cal_roll_deadband:      # LEVEL: tiny calibration roll → 0
            r = minimize(lambda q: cost(np.r_[q[0], p[1], 0.0, q[1], q[2]]), np.r_[p[0], p[3], p[4]],
                         method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-4, maxiter=1500))
            p = np.r_[r.x[0], p[1], 0.0, r.x[1], r.x[2]]
        return M_of(p), float(cost(p)), p

    def _calibrate(self, yaw, plate):
        pm = self._plate_masks()
        vmask = self.ref_face
        off_v = chin_offset(vmask)
        off_q = max(0.04, chin_offset(pm["34"]))
        cands = [plate] if plate else ["front", "34", "34m"]
        res = {k: self._fit(pm[k], vmask) for k in cands}
        if plate is None:
            side = "34" if off_v > 0 else "34m"
            k = side if (res[side][1] < 0.97 * res["front"][1] or abs(off_v) > 0.6 * off_q) else "front"
        else:
            k = plate
        self.plate = k
        A, self.cal_cost, self.cal_params = res[k]
        if self.face_scale != 1.0:            # optional art-direction: scale features about the face centre
            ys, xs = np.nonzero(pm[k])
            c = apply(A, [[xs.mean(), ys.mean()]])[0]
            S = np.array([[self.face_scale, 0, (1 - self.face_scale) * c[0]],
                          [0, self.face_scale, (1 - self.face_scale) * c[1]]])
            A = compose(S, A)
        self.A = A
        if yaw is None:
            y = 25.0 * off_v / off_q
            yaw = float(np.clip(y, -12.0, 12.0)) if k == "front" else float(np.clip(y, -32.0, 32.0))
            if k != "front" and abs(yaw) < 18:
                yaw = math.copysign(18.0, yaw if yaw != 0 else (1 if k == "34" else -1))
        self.yaw = yaw
        view, t = self.kit.view(yaw, 0.0)
        self.view = self.level.apply(view)
        self.t_turn = t
        self.plate_mask = pm[k]
        if self.verbose:
            print(f"[facecomp] {self.kit.name}: ref={self.trk.ref_idx} plate={k} yaw={yaw:+.1f} chin_off={off_v:+.3f} "
                  f"cal_cost={self.cal_cost:.2f}px scale={math.exp(self.cal_params[0]):.3f} "
                  f"aspect={math.exp(self.cal_params[1]):.3f} rot={math.degrees(self.cal_params[2]):+.2f}deg  costs="
                  + ", ".join(f"{a}:{b[1]:.2f}" for a, b in res.items()))

    # ------------------------------------------------------------------------------------------ per frame
    def transform(self, i: int) -> np.ndarray:
        return compose(self.poses[i].M, self.A)

    def occlusion_mask(self, i: int):
        """→ (prior face mask in frame i, occluder mask, info).  Hands = skin pieces that stick out of the tracked
        face or are bounded by NEW ink (outline not present on the reference face)."""
        M = self.poses[i].M
        prior = cv2.warpAffine(self.ref_face.astype(np.uint8), M, (self.W, self.H), flags=cv2.INTER_NEAREST) > 0
        if not self.occlusion:
            return prior, np.zeros_like(prior), {}
        skin, ink_i, _ = segment(self.frames[i], self.skin_bgr, self.bg_bgr)
        ref_ink_w = cv2.warpAffine(self.ref_ink.astype(np.uint8), M, (self.W, self.H), flags=cv2.INTER_NEAREST)
        ref_ink_w = cv2.dilate(ref_ink_w, np.ones((7, 7), np.uint8)) > 0
        new_ink = ink_i & ~ref_ink_w
        prior_d = cv2.dilate(prior.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        n, lbl, st, _ = cv2.connectedComponentsWithStats(skin.astype(np.uint8), connectivity=4)
        inside = np.bincount(lbl[prior_d], minlength=n)
        area = st[:, cv2.CC_STAT_AREA]
        pa = max(1, int(prior.sum()))
        cand = [j for j in range(1, n) if inside[j] > 0.004 * pa]
        face_ids, hand_ids = [], []
        if cand:
            big = max(cand, key=lambda j: inside[j])
            for j in cand:
                comp = lbl == j
                out_frac = (area[j] - inside[j]) / max(1, area[j])
                ring = cv2.dilate(comp.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~comp
                novelty = (new_ink & ring).sum() / max(1, (ink_i & ring).sum())
                if j == big and out_frac < 0.35:
                    face_ids.append(j)
                elif out_frac > 0.22 or novelty > 0.55:
                    hand_ids.append(j)
                else:
                    face_ids.append(j)
        face_vis = np.isin(lbl, face_ids) if face_ids else np.zeros_like(prior)
        face_vis = cv2.dilate(face_vis.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        prior_e = cv2.erode(prior.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        occ = cv2.morphologyEx((prior_e & ~face_vis).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        no, lo, so, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
        keep = np.zeros(no, bool)
        keep[1:] = so[1:, cv2.CC_STAT_AREA] > 0.004 * pa              # compression specks are not occluders
        occ = keep[lo]
        return prior, occ, dict(occ_frac=float(occ.sum() / pa), n_face=len(face_ids), n_hand=len(hand_ids))

    def render(self, i: int, ff: FaceFrame, debug: bool = False) -> np.ndarray:
        f = self.frames[i]
        T = self.transform(i)
        layer = render_layer(self.W, self.H, T, lambda ctx: draw_face_sided(ctx, self.kit, self.view, self.t_turn, ff))
        prior, occ, info = self.occlusion_mask(i)
        sw = max(2, int(round(self.kit.sw * math.exp(self.cal_params[0]) * self.poses[i].scale)))
        inside = cv2.erode(prior.astype(np.uint8), np.ones((sw, sw), np.uint8)).astype(np.float32)
        hide = cv2.dilate(occ.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
        keep = cv2.GaussianBlur(np.clip(inside - hide, 0, 1), (3, 3), 0)[:, :, None]
        a_before = float(layer[:, :, 3].sum())
        layer = (layer.astype(np.float32) * keep).astype(np.uint8)                 # premultiplied: scale all 4
        a = layer[:, :, 3:4].astype(np.float32) / 255.0
        out = (f.astype(np.float32) * (1 - a) + layer[:, :, :3].astype(np.float32)).clip(0, 255).astype(np.uint8)
        p = self.poses[i]
        self.stats.append(dict(i=i, roll=p.roll_deg, roll_raw=p.roll_raw_deg, scale=p.scale, res=p.residual_px,
                               inl=p.n_inliers, conf=p.region_conf,
                               hidden=1 - float(layer[:, :, 3].sum()) / max(1.0, a_before), **info))
        return self.debug_overlay(out, i, occ) if debug else out

    def debug_overlay(self, img, i, occ):
        o = img.copy()
        o[occ] = (0.55 * o[occ] + 0.45 * np.array([255, 0, 255])).astype(np.uint8)
        T = self.transform(i)
        cv2.polylines(o, [apply(T, contour_pts(self.plate_mask)).astype(np.int32)], True, (0, 200, 0), 1, cv2.LINE_AA)
        p = self.poses[i]
        c = np.array(p.center)
        R = p.M[:, :2] / max(p.scale, 1e-6)
        for v, col in (((60, 0), (0, 0, 255)), ((0, 60), (255, 0, 0))):
            q = c + R @ np.array(v, float)
            cv2.line(o, tuple(c.astype(int)), tuple(q.astype(int)), col, 2, cv2.LINE_AA)
        txt = f"f{i} roll {p.roll_deg:+.1f} (raw {p.roll_raw_deg:+.1f}) s {p.scale:.3f} res {p.residual_px:.2f}px"
        for col, th in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(o, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, th, cv2.LINE_AA)
        y = 52
        for k, v in p.region_conf.items():
            cv2.putText(o, f"{k}:{v:.2f}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 20
        return o

    # ------------------------------------------------------------------------------------------ QA
    def qa(self) -> dict:
        """attachment: tracked face outline vs the real outline (cheek/jaw, unoccluded) in every frame;
        jitter: 2nd difference of the eye-midpoint anchor (px/frame²)."""
        errs, anc = [], []
        e = self.view["eyes"]
        eye_mid = 0.5 * (e["L"]["outline"].mean(0) + e["R"]["outline"].mean(0))
        for i in range(len(self.frames)):
            M = self.poses[i].M
            prior = cv2.warpAffine(self.ref_face.astype(np.uint8), M, (self.W, self.H), flags=cv2.INTER_NEAREST) > 0
            skin, _, _ = segment(self.frames[i], self.skin_bgr, self.bg_bgr)
            fc = ndi.binary_fill_holes(face_component(skin, prior=prior))
            anc.append(apply(self.transform(i), [eye_mid])[0])
            if fc.sum() < 0.5 * prior.sum():
                continue
            dt = cv2.distanceTransform((edge(fc) == 0).astype(np.uint8), cv2.DIST_L2, 5)
            pc = lower_contour(prior, 0.45)
            d = dt[np.clip(pc[:, 1], 0, self.H - 1).astype(int), np.clip(pc[:, 0], 0, self.W - 1).astype(int)]
            d = d[d < 12]
            if len(d) > 20:
                errs.append(float(np.median(d)))
        anc = np.array(anc)
        jit = np.linalg.norm(np.diff(anc, 2, axis=0), axis=1) if len(anc) > 2 else np.zeros(1)
        rolls = np.array([p.roll_deg for p in self.poses])
        rolls_raw = np.array([p.roll_raw_deg for p in self.poses])
        return dict(plate=self.plate, yaw=self.yaw, cal_cost_px=self.cal_cost,
                    feature_scale=float(math.exp(self.cal_params[0])),
                    attach_median_px=float(np.median(errs)) if errs else None,
                    attach_p95_px=float(np.percentile(errs, 95)) if errs else None,
                    anchor_jitter_median=float(np.median(jit)), anchor_jitter_p95=float(np.percentile(jit, 95)),
                    roll_abs_max=float(np.abs(rolls).max()), roll_raw_abs_max=float(np.abs(rolls_raw).max()),
                    frames_level=int((np.abs(rolls) < 1e-6).sum()), n_frames=len(self.poses),
                    occluded_frames=int(sum(1 for s in self.stats if s.get("occ_frac", 0) > 0.01)))


# ================================================================================================= demo / CLI
def demo_script(name: str, fps: float):
    """beat sheets for the two cafe blank-face gesture clips (acting matched to the body language)."""
    from .expressive import Beat, ExpressivePerformer, au
    from .visemes import viseme_track
    if name == "boy":       # open palm explain → scratch head (puzzled) → shrug both hands (no idea)
        beats = [Beat(0.0, "explaining"), Beat(0.55, "friendly"),
                 Beat(1.45, "puzzled"), Beat(2.35, "unsure"),
                 Beat(3.05, {**au(AU1=0.9, AU2=0.8, AU15=0.5, AU17=0.5, AU20=0.3), "AU12_L": 0.25}),
                 Beat(4.30, "smug", 0.7)]
        gaze = [(0.0, (0.25, 0.0)), (0.9, (0.1, 0.05)), (1.5, (0.35, -0.45)), (2.4, (0.2, -0.3)),
                (3.1, (0.0, 0.0)), (4.3, (-0.15, 0.05))]
        words = [("ببین", 0.10, 0.45), ("این", 0.50, 0.70), ("طوری", 0.72, 1.20), ("نمیدونم", 3.10, 3.75),
                 ("واقعا", 3.85, 4.30)]
    else:                   # open palm → three fingers (counting) → hand up → hand on forehead (facepalm)
        beats = [Beat(0.0, "explaining"), Beat(0.60, "friendly", 0.9),
                 Beat(1.40, "determined"), Beat(2.30, "skeptical"),
                 Beat(3.25, "surprised", 0.6), Beat(3.75, "facepalm"), Beat(4.55, "exasperated")]
        gaze = [(0.0, (-0.25, 0.0)), (0.9, (-0.1, 0.0)), (1.5, (-0.3, -0.1)), (2.4, (-0.15, 0.0)),
                (3.3, (-0.1, -0.4)), (3.8, (0.0, 0.35)), (4.6, (0.0, -0.5))]
        words = [("خب", 0.10, 0.35), ("گوش", 0.40, 0.70), ("کن", 0.72, 0.95), ("سه", 1.55, 1.95), ("بار", 2.00, 2.40),
                 ("گفتم", 2.45, 2.95), ("وای", 3.85, 4.30)]
    return ExpressivePerformer(fps=fps, beats=beats, gaze=gaze, visemes=viseme_track(words), seed=7)


def read_frames(path: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out, fps


class VideoWriter:
    """streams BGR frames into ffmpeg (no frame list in RAM — the sandbox has ~1 GB)."""

    def __init__(self, out: str, W: int, H: int, fps: float):
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps),
               "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "veryfast", out]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.out = out

    def write(self, f: np.ndarray):
        self.p.stdin.write(np.ascontiguousarray(f).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()
        return self.out


def main():
    ap = argparse.ArgumentParser(description="track a blank-face cartoon video and composite the rig25 face")
    ap.add_argument("--video", required=True)
    ap.add_argument("--char", required=True, choices=["boy", "girl"])
    ap.add_argument("--script", default=None, help="demo beat sheet: boy | girl (default = --char)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug", action="store_true", help="also write <out>_debug.mp4 with tracker overlay")
    ap.add_argument("--yaw", type=float, default=None)
    ap.add_argument("--plate", default=None, choices=["front", "34", "34m"])
    ap.add_argument("--face-scale", type=float, default=1.0)
    ap.add_argument("--no-occlusion", action="store_true")
    a = ap.parse_args()
    frames, fps = read_frames(a.video)
    comp = FaceRigCompositor(a.char, frames, yaw=a.yaw, plate=a.plate, fps=fps, occlusion=not a.no_occlusion,
                             face_scale=a.face_scale)
    perf = demo_script(a.script or a.char, fps)
    H, W = frames[0].shape[:2]
    vw = VideoWriter(a.out, W, H, fps)
    vd = VideoWriter(os.path.splitext(a.out)[0] + "_debug.mp4", W, H, fps) if a.debug else None
    for i in range(len(frames)):
        img = comp.render(i, perf.frame(i))
        vw.write(img)
        if vd:
            vd.write(comp.debug_overlay(img, i, comp.occlusion_mask(i)[1]))
        if i % 20 == 0:
            print(f"[facecomp] frame {i}/{len(frames)}", flush=True)
    vw.close()
    if vd:
        vd.close()
    q = comp.qa()
    with open(os.path.splitext(a.out)[0] + "_qa.json", "w") as fh:
        json.dump(dict(qa=q, frames=comp.stats), fh, indent=1, default=float)
    print("[facecomp] QA", json.dumps(q, default=float))


if __name__ == "__main__":
    main()
