"""
Layer extraction for the 3-layer compositor (plates drawn on a flat paper colour with ink outlines).

  * paper_color(img)            dominant border colour (the flat beige / paper of the plate)
  * character_alpha(img)        everything that is NOT paper-coloured open space → the character silhouette;
                                enclosed paper holes (between arm and torso, between legs) are transparent too.
  * props_alpha(img)            foreground props (desk, lamp, book…): paper regions connected to the TOP edge (wall)
                                or BOTTOM edge (floor / under-desk gap) are transparent, everything else is a prop —
                                no hard-coded seed points (v29.2 used 9 fixed seeds tuned to one desk drawing).
  * occlusion_line(img, alpha)  y of the dominant prop surface (longest near-horizontal ink edge on the prop) —
                                the character is IN FRONT of props above it (hands raised over the desk) and BEHIND
                                them below it (legs under the desk). Replaces the v29.2 constants 0.631·H / 0.278·W.
  * split_pose_sheet(img)       auto-detects panel count on a horizontal turnaround / pose sheet (paper gutters).
  * load_layer(path, wh, kind)  RGBA files keep their own alpha; opaque files are segmented by `kind`.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .media import cover_fit, load_image


def paper_color(img_bgr: np.ndarray, border: int = 6) -> np.ndarray:
    """median colour of the outer border band (robust to a few ink lines touching the edge)."""
    b = max(1, min(border, img_bgr.shape[0] // 8, img_bgr.shape[1] // 8))
    strip = np.concatenate([img_bgr[:b].reshape(-1, 3), img_bgr[-b:].reshape(-1, 3),
                            img_bgr[:, :b].reshape(-1, 3), img_bgr[:, -b:].reshape(-1, 3)])
    hsv = cv2.cvtColor(strip[None], cv2.COLOR_BGR2HSV)[0]
    keep = strip[hsv[:, 2] > 120]                    # ignore ink
    return np.median(keep if len(keep) > 10 else strip, axis=0).astype(np.uint8)


def paper_mask(img_bgr: np.ndarray, paper: np.ndarray | None = None, tol: float = 26.0) -> np.ndarray:
    """uint8 0/1 — pixels within `tol` (CIE-Lab ΔE) of the paper colour."""
    paper = paper_color(img_bgr) if paper is None else np.asarray(paper, np.uint8)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    plab = cv2.cvtColor(paper.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    return (np.linalg.norm(lab - plab, axis=2) < tol).astype(np.uint8)


def _soft(mask01: np.ndarray, feather: float = 0.8) -> np.ndarray:
    a = mask01.astype(np.float32)
    return cv2.GaussianBlur(a, (0, 0), feather) if feather > 0 else a


def character_alpha(img_bgr: np.ndarray, paper: np.ndarray | None = None, min_hole_frac: float = 0.0004,
                    keep: str = "largest", feather: float = 0.8) -> np.ndarray:
    """float32 0..1 alpha of the character drawn on flat paper. keep='largest' | 'all' (multiple characters)."""
    H, W = img_bgr.shape[:2]
    pm = paper_mask(img_bgr, paper)
    pm = cv2.morphologyEx(pm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(pm, connectivity=4)
    bg = np.zeros((H, W), np.uint8)
    min_hole = max(30, int(H * W * min_hole_frac))
    for i in range(1, n):
        x, y, w, h, a = st[i]
        touches = x == 0 or y == 0 or x + w >= W or y + h >= H
        if touches or a >= min_hole:
            bg[lbl == i] = 1
    fg = (1 - bg).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return np.zeros((H, W), np.float32)
    if keep == "largest":
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        fg = (lbl == k).astype(np.uint8)
    else:
        big = [i for i in range(1, n) if st[i, 4] >= 0.002 * H * W]
        fg = np.isin(lbl, big).astype(np.uint8)
    return _soft(fg, feather)


def props_alpha(img_bgr: np.ndarray, paper: np.ndarray | None = None, ink_v: int = 150,
                feather: float = 0.8) -> np.ndarray:
    """float32 alpha of foreground props: paper regions reachable from the top / bottom edge are see-through."""
    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    barrier = cv2.dilate((hsv[:, :, 2] < ink_v).astype(np.uint8), np.ones((3, 3), np.uint8))
    pm = paper_mask(img_bgr, paper) & (1 - barrier)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(pm, connectivity=4)
    clear = np.zeros((H, W), np.uint8)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if y == 0 or y + h >= H:
            clear[lbl == i] = 1
    clear = cv2.morphologyEx(clear, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    prop = (1 - clear).astype(np.uint8)
    # drop specks (paper grain, stray ink) — props are big
    n, lbl, st, _ = cv2.connectedComponentsWithStats(prop, connectivity=8)
    keep = [i for i in range(1, n) if st[i, 4] >= 0.0008 * H * W]
    prop = np.isin(lbl, keep).astype(np.uint8)
    return _soft(prop, feather)


def occlusion_line(img_bgr: np.ndarray, alpha: np.ndarray, min_span: float = 0.45) -> int | None:
    """y of the dominant prop TOP surface (≥ min_span·W long near-horizontal ink run on the prop), else None."""
    H, W = img_bgr.shape[:2]
    if alpha.max() < 0.5:
        return None
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ink = ((hsv[:, :, 2] < 150) & (cv2.dilate((alpha > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0))
    horiz = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, max(15, W // 12)), np.uint8))
    rows = horiz.sum(axis=1)
    cand = np.nonzero(rows >= min_span * W)[0]
    if len(cand):
        return int(cand[0])
    ys = np.nonzero((alpha > 0.5).sum(axis=1) >= min_span * W)[0]
    return int(ys[0]) if len(ys) else None


def split_pose_sheet(sheet_bgr: np.ndarray, num_poses: int | str = "auto", min_gutter: float = 0.02,
                     max_poses: int = 8) -> list[np.ndarray]:
    """split a horizontal pose sheet at paper gutters (columns with no ink). num_poses='auto' → detect."""
    H, W = sheet_bgr.shape[:2]
    pm = paper_mask(sheet_bgr)
    col_paper = pm.mean(axis=0)
    is_gut = col_paper > 0.985
    runs, x = [], 0
    while x < W:
        if is_gut[x]:
            x2 = x
            while x2 < W and is_gut[x2]:
                x2 += 1
            if x > 0 and x2 < W and (x2 - x) >= max(3, int(min_gutter * W * 0.25)):
                runs.append((x, x2))
            x = x2
        else:
            x += 1
    cuts = [0] + [(a + b) // 2 for a, b in runs] + [W]
    panels = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] - cuts[i] > 0.08 * W]
    if num_poses != "auto":
        k = int(num_poses)
        if len(panels) != k:                          # fall back to equal slices at the darkest-free columns
            panels = []
            ink_cols = 1.0 - col_paper
            edges = [0]
            for j in range(1, k):
                lo, hi = int(W * (j / k - 0.06)), int(W * (j / k + 0.06))
                edges.append(lo + int(np.argmin(ink_cols[lo:hi])))
            edges.append(W)
            panels = [(edges[i], edges[i + 1]) for i in range(k)]
    panels = panels[:max_poses] or [(0, W)]
    return [np.ascontiguousarray(sheet_bgr[:, a:b]) for a, b in panels]


def fit_panel(panel_bgr: np.ndarray, wh: tuple[int, int], paper: np.ndarray | None = None,
              anchor: str = "bottom") -> np.ndarray:
    """letterbox a pose panel into wh on its own paper colour (keeps the whole character — no cover crop)."""
    W, H = wh
    ph, pw = panel_bgr.shape[:2]
    s = min(W / pw, H / ph)
    nw, nh = max(1, int(round(pw * s))), max(1, int(round(ph * s)))
    r = cv2.resize(panel_bgr, (nw, nh), interpolation=cv2.INTER_LANCZOS4 if s > 1 else cv2.INTER_AREA)
    canvas = np.empty((H, W, 3), np.uint8)
    canvas[:] = paper_color(panel_bgr) if paper is None else paper
    x0 = (W - nw) // 2
    y0 = H - nh if anchor == "bottom" else (H - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas


def load_layer(path: str, wh: tuple[int, int], kind: str) -> dict[str, Any]:
    """load + cover-fit a plate. kind: 'bg' (opaque) | 'fg' (props alpha) | 'char' (character alpha)."""
    bgr, a = load_image(path)
    bgr = cover_fit(bgr, wh)
    if a is not None:
        a = cover_fit(a, wh, interp=cv2.INTER_LINEAR)
    elif kind == "fg":
        a = props_alpha(bgr)
    elif kind == "char":
        a = character_alpha(bgr)
    else:
        a = np.ones(bgr.shape[:2], np.float32)
    return {"bgr": bgr, "alpha": a.astype(np.float32), "path": path}
