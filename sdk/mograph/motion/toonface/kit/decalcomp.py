"""
toonface.kit.decalcomp — v33 place artist-drawn expressions (sheetlib decals) on a tracked blank face.

Coordinate chain (all 2×3 affines):
    decal px  --inv(Expression.unit_to_decal)-->  face units  --FaceStyle.unit_to_frame-->  ref-frame px
              --M_track(i)-->  frame i px
Face units are shared by the sheet and the video (x right, y up from chin, 1.0 = widest face row), so an
expression keeps its proportions on any character; per-character tweaks are a small unit-space affine (`adjust`).

Colour: decals are skin-un-mixed (sheetlib), i.e. their colour is the feature ink only, so compositing them on
any skin needs no recolouring.  Optional `ink_bgr` tints near-black ink to the target drawing's line colour.
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from .sheetlib import ExpressionLibrary


def _h(M):
    return np.vstack([np.asarray(M, np.float64), [0.0, 0.0, 1.0]])


def compose(*Ms):
    """compose(A, B, C) = A ∘ B ∘ C  (C applied first)."""
    R = np.eye(3)
    for M in Ms:
        R = R @ _h(M)
    return R[:2]


def invert(M):
    return np.linalg.inv(_h(M))[:2]


def unit_adjust(scale=1.0, dx=0.0, dy=0.0, sx=1.0, sy=1.0):
    """small face-unit affine around the feature centre (0, 0.5): global size, offset, stretch."""
    c = np.array([0.0, 0.5])
    A = np.array([[scale * sx, 0.0], [0.0, scale * sy]])
    t = c - A @ c + np.array([dx, dy])
    return np.hstack([A, t[:, None]])


def warp_decal(rgba: np.ndarray, M_decal_to_frame, W: int, H: int):
    """straight-alpha RGBA → premultiplied (colour, alpha) in frame space."""
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    pm = np.concatenate([rgba[..., :3].astype(np.float32) * a, a * 255.0], 2)
    out = cv2.warpAffine(pm, np.asarray(M_decal_to_frame, np.float64), (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out[..., :3], out[..., 3] / 255.0


def over(frame: np.ndarray, col_pm, alpha, ink_bgr=None, keep=None):
    """composite premultiplied layer over frame; `keep` (0..1) protects occluders (hand in front)."""
    a = np.clip(alpha, 0, 1)
    if keep is not None:
        a = a * (1.0 - keep)
        col_pm = col_pm * (1.0 - keep)[..., None]
    col = col_pm
    if ink_bgr is not None:                              # retint dark ink to the target drawing's line colour
        c = col / np.maximum(a[..., None], 1e-3)
        dark = np.clip((90.0 - c.mean(2)) / 60.0, 0, 1)[..., None]
        c = c * (1 - dark) + np.asarray(ink_bgr, np.float32) * dark
        col = c * a[..., None]
    out = frame.astype(np.float32) * (1.0 - a[..., None]) + col
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


class DecalFace:
    """expression decals on one blank face.

    style   facefind.FaceStyle measured on the reference frame (gives unit_to_frame)
    adjust  2×3 unit-space affine for per-character fitting (unit_adjust)
    """

    def __init__(self, lib: ExpressionLibrary, style, adjust=None, ink_bgr=None):
        self.lib = lib
        self.style = style
        self.adjust = unit_adjust() if adjust is None else np.asarray(adjust, np.float64)
        self.ink_bgr = ink_bgr

    def decal_to_frame(self, name: str, M_track=None):
        e = self.lib.items[name]
        M = compose(self.style.unit_to_frame(), self.adjust, invert(e.unit_to_decal()))
        return M if M_track is None else compose(M_track, M)

    def render(self, frame: np.ndarray, name: str, M_track=None, keep=None, face_mask=None):
        H, W = frame.shape[:2]
        col, a = warp_decal(self.lib.decal(name), self.decal_to_frame(name, M_track), W, H)
        if face_mask is not None:                        # never draw outside the face (hair/background)
            fm = cv2.GaussianBlur(face_mask.astype(np.float32), (5, 5), 0)
            col, a = col * fm[..., None], a * fm
        return over(frame, col, a, self.ink_bgr, keep)


def face_mask_dilated(mask: np.ndarray, px: int = 6):
    return cv2.dilate(mask.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)) > 0


def acceptance_sheet(video: str, lib_dir: str, out: str, names=None, frame_idx: int = 0, crop_pad: float = 0.9,
                     cols: int = 3, adjust=None, clip_to_face: bool = True):
    """still review sheet: each expression placed on the video's blank face (no tracking needed)."""
    from .facefind import find_face, measure_style
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("cannot read frame")
    ff = find_face(frame)
    st = measure_style(frame, ff.mask)
    lib = ExpressionLibrary.load(lib_dir)
    names = names or lib.names()
    dface = DecalFace(lib, st, adjust=adjust)
    fm = face_mask_dilated(ff.mask, max(2, int(st.line_px))) if clip_to_face else None
    w = st.width
    x0 = int(max(0, st.cx - crop_pad * w)); x1 = int(min(frame.shape[1], st.cx + crop_pad * w))
    y0 = int(max(0, st.chin_y - 1.55 * w)); y1 = int(min(frame.shape[0], st.chin_y + 0.35 * w))
    tiles = []
    for nm in names:
        img = dface.render(frame, nm, face_mask=fm)[y0:y1, x0:x1].copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (255, 255, 255), -1)
        cv2.putText(img, nm, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
        tiles.append(img)
    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * th, cols * tw, 3), 255, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    cv2.imwrite(out, sheet)
    return out, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", default="")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--dy", type=float, default=0.0)
    a = ap.parse_args()
    names = [n for n in a.names.split(",") if n] or None
    out, st = acceptance_sheet(a.video, a.lib, a.out, names, a.frame, adjust=unit_adjust(a.scale, dy=a.dy))
    print("[decalcomp]", out, "face width", round(st.width, 1), "chin", round(st.chin_y, 1))


if __name__ == "__main__":
    main()
