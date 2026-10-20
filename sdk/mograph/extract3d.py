"""
mograph.extract3d — Automatic 3D Studio Sprite Sheet Slicer & Alpha Matte Extractor.
Used by `python3 -m mograph extract3d <image_path> --names name1,name2,... [--grid 2x2]`
and dynamically by `@element("tactile_3d")` when given a raw AI-generated image.
"""
from __future__ import annotations
import os
import cv2
import numpy as np

ASSETS_3D_DIR = os.path.join(os.path.dirname(__file__), "assets_3d")


def key_3d_rgba(bgr: np.ndarray, bg_mode: str = "auto") -> np.ndarray:
    """Convert a BGR image of an isolated 3D object on a solid black, green, or white
    background into a clean, defringed RGBA array with tight bounding-box crop."""
    h, w = bgr.shape[:2]
    b, g, r = bgr[:, :, 0].astype(np.float32), bgr[:, :, 1].astype(np.float32), bgr[:, :, 2].astype(np.float32)

    if bg_mode == "auto":
        border = np.concatenate([bgr[0, :], bgr[-1, :], bgr[:, 0], bgr[:, -1]], axis=0).astype(np.float32)
        mb, mg, mr = np.median(border[:, 0]), np.median(border[:, 1]), np.median(border[:, 2])
        if mg > 150 and mg > mr + 60 and mg > mb + 60:
            bg_mode = "green"
        elif mr > 215 and mg > 215 and mb > 215:
            bg_mode = "white"
        else:
            bg_mode = "black"

    if bg_mode == "green":
        excess_g = g - np.maximum(r, b)
        alpha = np.clip(1.0 - (excess_g - 15.0) / 55.0, 0.0, 1.0)
        g_despill = np.minimum(g, np.maximum(r, b) * 0.95 + g * 0.05)
        g = g * alpha + g_despill * (1.0 - alpha)
    elif bg_mode == "white":
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
        dist = np.maximum(255.0 - lum, sat * 1.35)
        alpha = np.clip((dist - 14.0) / 36.0, 0.0, 1.0)
    else:  # black
        lum = np.maximum(np.maximum(r, g), b)
        alpha = np.clip((lum - 14.0) / 34.0, 0.0, 1.0)

    mask_u8 = (alpha > 0.12).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8)
    clean_mask = np.zeros_like(mask_u8)
    min_area = max(120, int(h * w * 0.002))
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 255

    # Fill interior holes inside solid 3D objects
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros_like(clean_mask)
    cv2.drawContours(solid, contours, -1, 255, -1)
    alpha_f = np.where(solid > 0, np.maximum(alpha, 0.96), 0.0).astype(np.float32)

    eroded = cv2.erode(solid, np.ones((5, 5), np.uint8), iterations=1)
    edge_ring = (solid > 0) & (eroded == 0)
    alpha_f[edge_ring] = np.clip(alpha[edge_ring], 0.0, 1.0)
    alpha_f = cv2.GaussianBlur(alpha_f, (3, 3), 0.7)

    ys, xs = np.where(alpha_f > 0.05)
    if len(xs) == 0:
        return np.zeros((64, 64, 4), dtype=np.uint8)
    pad = 14
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad)
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad)

    rgba = np.dstack([
        np.clip(r[y0:y1, x0:x1], 0, 255).astype(np.uint8),
        np.clip(g[y0:y1, x0:x1], 0, 255).astype(np.uint8),
        np.clip(b[y0:y1, x0:x1], 0, 255).astype(np.uint8),
        np.clip(alpha_f[y0:y1, x0:x1] * 255.0, 0, 255).astype(np.uint8),
    ])
    return rgba


def extract_3d_sheet(src_path: str, names: list[str], grid: str = "2x2", bg_mode: str = "auto",
                     out_dir: str | None = None) -> list[str]:
    """Slice a grid sprite sheet (e.g. 2x2, 1x1, 2x3) and save clean RGBA PNGs into assets_3d."""
    out_dir = out_dir or ASSETS_3D_DIR
    os.makedirs(out_dir, exist_ok=True)
    im = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"Cannot read image: {src_path}")
    H, W = im.shape[:2]
    # Trim bottom-right watermark corner if present on 1:1 / 16:9 sheets
    if H >= 800 and W >= 800:
        im[int(H * 0.93):, int(W * 0.88):] = im[0, 0]

    rows, cols = [int(x) for x in grid.lower().split("x")]
    ch, cw = H // rows, W // cols
    margin_y, margin_x = int(ch * 0.04), int(cw * 0.04)
    saved = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(names):
                break
            cell = im[r * ch + margin_y:(r + 1) * ch - margin_y,
                      c * cw + margin_x:(c + 1) * cw - margin_x]
            rgba = key_3d_rgba(cell, bg_mode=bg_mode)
            bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            nm = names[idx].strip()
            if not nm.endswith(".png"):
                nm += ".png"
            dst = os.path.join(out_dir, nm)
            cv2.imwrite(dst, bgra)
            saved.append(dst)
            idx += 1
    return saved
