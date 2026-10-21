"""
mograph.pro.color — professional COLOUR FINISHING for v14 footage edits (Resolve / Premiere Lumetri style).

Spec (v14, all optional):
    "grade": {
        "look":   "teal_orange" | "film_kodak" | "bleach_bypass" | "cinematic_cool" | "warm_doc" | "mono_film" |
                  "vintage_fade" | "clean_commercial" | "night_neon" | "golden_hour",       (built-in, procedural)
        "lut":    "luts/my.cube",            # any Adobe/Resolve .cube (1D or 3D), path relative to the spec
        "lut_strength": 0.8,                 # 0..1 mix (look and lut share it unless "look_strength" is given)
        "look_strength": 0.7,
        "protect_skin": 0.6,                 # 0..1: pull skin pixels back toward the ungraded plate (default 0.6
                                             #       when a look/lut is set) — faces never go green/orange
        "wb":   {"temp": 0.1, "tint": -0.05},      # -1..1 white balance (temp + = warmer, tint + = magenta)
        "lift_gamma_gain": {"lift": [0, 0, 0.02], "gamma": [0, 0, 0], "gain": [0.02, 0, -0.02]}   # RGB, -0.2..0.2
        "contrast": 1.05, "saturation": 1.0, "vibrance": 0.15,
        "grain": 0.25, "grain_size": 1.4,    # film grain (deterministic per frame → chunk-safe)
        "sharpen": 0.3,                      # unsharp mask on the final frame
        "halation": 0.25,                    # red film halation around highlights
        "letterbox": 2.39                    # cinematic bars (aspect ratio of the picture area)
    }
The colour part (look/lut/wb/lgg/contrast/sat/vibrance) is baked into ONE 33³ LUT and applied to the speaker
PLATE (before titles, so graphics keep their designed colours).  grain / halation / sharpen / letterbox are applied
to the FINAL frame like a film print.  The existing v14 "grade.bg / keys / fg / match" keys keep working.

CLI:  python3 -m mograph.pro looks --spec edit.json -o sheet.jpg     contact sheet of every look on YOUR footage
      python3 -m mograph.pro lut-export teal_orange -o teal_orange.cube  (use the look in Premiere / Resolve)
"""
from __future__ import annotations

import os

import cv2
import numpy as np

N = 33                                    # LUT grid size (Resolve default)


# ============================================================================================== .cube I/O
def read_cube(path):
    """→ (lut float32 [N,N,N,3] indexed [r][g][b] → RGB 0..1, domain_min, domain_max).  1D LUTs → expanded 3D."""
    size3 = size1 = None
    dmin, dmax = np.zeros(3, np.float32), np.ones(3, np.float32)
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] == "#":
                continue
            k = s.split()
            u = k[0].upper()
            if u == "LUT_3D_SIZE":
                size3 = int(k[1]); continue
            if u == "LUT_1D_SIZE":
                size1 = int(k[1]); continue
            if u == "DOMAIN_MIN":
                dmin = np.array([float(x) for x in k[1:4]], np.float32); continue
            if u == "DOMAIN_MAX":
                dmax = np.array([float(x) for x in k[1:4]], np.float32); continue
            if u in ("TITLE", "LUT_3D_INPUT_RANGE", "LUT_1D_INPUT_RANGE"):
                continue
            try:
                rows.append([float(x) for x in k[:3]])
            except ValueError:
                continue
    a = np.asarray(rows, np.float32)
    if size3:
        if len(a) != size3 ** 3:
            raise ValueError(f"{path}: expected {size3 ** 3} rows, got {len(a)}")
        # .cube order: R changes fastest → reshape [b][g][r] → transpose to [r][g][b]
        return a.reshape(size3, size3, size3, 3).transpose(2, 1, 0, 3).copy(), dmin, dmax
    if size1:
        v = np.linspace(0, 1, size1, dtype=np.float32)
        r, g, b = np.meshgrid(v, v, v, indexing="ij")
        out = np.stack([np.interp(r, v, a[:, 0]), np.interp(g, v, a[:, 1]), np.interp(b, v, a[:, 2])], -1)
        return out.astype(np.float32), dmin, dmax
    raise ValueError(f"{path}: no LUT_3D_SIZE / LUT_1D_SIZE")


def write_cube(path, lut, title="mograph"):
    n = lut.shape[0]
    with open(path, "w") as f:
        f.write(f'TITLE "{title}"\nLUT_3D_SIZE {n}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n')
        for b in range(n):
            for g in range(n):
                for r in range(n):
                    x = lut[r, g, b]
                    f.write(f"{x[0]:.6f} {x[1]:.6f} {x[2]:.6f}\n")
    return path


def identity(n=N):
    v = np.linspace(0, 1, n, dtype=np.float32)
    r, g, b = np.meshgrid(v, v, v, indexing="ij")
    return np.stack([r, g, b], -1)


def sample(lut, rgb):
    """trilinear lookup: rgb (...,3) 0..1 → (...,3)."""
    n = lut.shape[0]
    x = np.clip(rgb, 0, 1) * (n - 1)
    i0 = np.minimum(np.floor(x).astype(np.int32), n - 2)
    f = x - i0
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    c = 0
    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                w = (fr if dr else 1 - fr) * (fg if dg else 1 - fg) * (fb if db else 1 - fb)
                c = c + w * lut[r0 + dr, g0 + dg, b0 + db]
    return c


def resample(lut, n=N):
    """trilinear resample of any LUT to n³ (so every LUT shares the fast path)."""
    if lut.shape[0] == n:
        return lut
    return sample(lut, identity(n).reshape(-1, 3)).reshape(n, n, n, 3).astype(np.float32)


# ============================================================================================== looks
def _luma(c):
    return c[..., 0:1] * 0.2126 + c[..., 1:2] * 0.7152 + c[..., 2:3] * 0.0722


def _sat(c, s):
    y = _luma(c)
    return y + (c - y) * s


def _contrast(c, k, pivot=0.45):
    return (c - pivot) * k + pivot


def _scurve(c, k):
    """filmic S-curve (k 0..1), keeps 0 and 1 fixed."""
    s = c * c * (3 - 2 * c)
    return c + (s - c) * k


def _split(c, sh, hi, amt=1.0):
    """split toning: shadows → sh RGB offset, highlights → hi offset (luma-weighted)."""
    y = np.clip(_luma(c), 0, 1)
    return c + (np.asarray(sh, np.float32) * (1 - y) ** 2 + np.asarray(hi, np.float32) * y ** 2) * amt


def _fade(c, black=0.06, white=0.96):
    return black + c * (white - black)


def _wb(c, temp=0.0, tint=0.0):
    m = np.array([1 + 0.10 * temp + 0.03 * tint, 1 - 0.06 * tint, 1 - 0.10 * temp + 0.03 * tint], np.float32)
    return c * m


def _lgg(c, lift=(0, 0, 0), gamma=(0, 0, 0), gain=(0, 0, 0)):
    lift, gamma, gain = (np.asarray(x, np.float32) for x in (lift, gamma, gain))
    c = c * (1 + gain) + lift * (1 - c)
    return np.clip(c, 0, None) ** (1.0 / np.maximum(0.2, 1 + gamma))


def _teal_orange(c):
    y = _luma(c)
    warm = np.clip((c[..., 0:1] - c[..., 2:3]) * 3.0 + 0.5, 0, 1)
    teal = np.array([-0.06, 0.02, 0.07], np.float32)
    orng = np.array([0.07, 0.015, -0.06], np.float32)
    c = c + (teal * (1 - warm) * (1 - y) * 1.3 + orng * warm * (0.4 + y))
    return _scurve(np.clip(c, 0, 1), 0.35)


def _kodak(c):
    c = _scurve(np.clip(c, 0, 1), 0.45)
    c = _split(c, [0.0, 0.015, 0.03], [0.035, 0.015, -0.025])
    return _fade(_sat(c, 0.92), 0.025, 0.985)


def _bleach(c):
    y = _luma(c)
    return _scurve(np.clip(_contrast(c * 0.45 + y * 0.55, 1.18), 0, 1), 0.5)


def _cool(c):
    c = _split(_wb(c, -0.35, 0.0), [0.0, 0.02, 0.05], [0.01, 0.01, 0.0])
    return _scurve(np.clip(_sat(c, 0.88), 0, 1), 0.3)


def _warm_doc(c):
    return _fade(_scurve(np.clip(_sat(_wb(c, 0.35, 0.05), 1.05), 0, 1), 0.22), 0.02, 0.99)


def _mono(c):
    y = c[..., 0:1] * 0.30 + c[..., 1:2] * 0.59 + c[..., 2:3] * 0.11
    y = _scurve(np.clip(y, 0, 1), 0.55)
    return np.repeat(_fade(y, 0.03, 0.97), 3, -1) + np.array([0.008, 0.0, -0.01], np.float32)


def _vintage(c):
    c = _split(c, [0.03, 0.02, 0.06], [0.05, 0.03, -0.03])
    return _fade(_sat(_scurve(np.clip(c, 0, 1), 0.2), 0.78), 0.08, 0.94)


def _clean(c):
    return _scurve(np.clip(_lgg(_sat(c, 1.08), gain=(0.02, 0.02, 0.02)), 0, 1), 0.18)


def _night_neon(c):
    c = _split(c, [0.02, -0.01, 0.07], [0.06, -0.01, 0.05])
    return _scurve(np.clip(_sat(c, 1.25) * 0.94, 0, 1), 0.4)


def _golden(c):
    c = _split(_wb(c, 0.6, 0.08), [0.02, 0.0, -0.01], [0.06, 0.03, -0.04])
    return _fade(_scurve(np.clip(c, 0, 1), 0.25), 0.03, 0.99)


LOOKS = {
    "teal_orange": (_teal_orange, "blockbuster teal shadows + orange skin (the most used look on reels)"),
    "film_kodak": (_kodak, "Kodak 2383-like print: soft S-curve, warm highlights, cyan-ish shadows"),
    "bleach_bypass": (_bleach, "desaturated, hard contrast, silver (tension / serious statements)"),
    "cinematic_cool": (_cool, "cool blue-steel corporate / tech look"),
    "warm_doc": (_warm_doc, "warm, natural documentary skin"),
    "mono_film": (_mono, "black & white film (red-filter luma, slight warm tone)"),
    "vintage_fade": (_vintage, "faded blacks, muted colours, nostalgic"),
    "clean_commercial": (_clean, "clean bright commercial (beauty / dental / product)"),
    "night_neon": (_night_neon, "magenta / blue city-night neon"),
    "golden_hour": (_golden, "sunset warmth, glowing highlights"),
}


def look_lut(name, n=N):
    if name not in LOOKS:
        raise KeyError(f"unknown look '{name}'.  Looks: {', '.join(LOOKS)}")
    return np.clip(LOOKS[name][0](identity(n)), 0, 1).astype(np.float32)


# ============================================================================================== grade
def skin_weight(bgr):
    """0..1 skin likelihood: soft ellipse in the CrCb plane (covers light → dark skin), smoothed."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    d = np.sqrt(((cr - 153) / 20) ** 2 + ((cb - 106) / 18) ** 2)
    w = np.clip((1.3 - d) / 0.5, 0, 1) * np.clip((y - 40) / 30, 0, 1)
    return cv2.GaussianBlur(w, (0, 0), 2.0)


class Grader:
    """build once from the spec "grade" dict, then .apply_lut(plate) and .finish(final_frame, frame_index)."""

    KEYS = ("look", "lut", "lut_strength", "look_strength", "protect_skin", "wb", "lift_gamma_gain", "contrast",
            "saturation", "vibrance", "grain", "grain_size", "sharpen", "halation", "letterbox")

    def __init__(self, g, base_dir="."):
        self.g = g = dict(g or {})
        self.active = any(k in g for k in self.KEYS)
        self.is_identity = True
        self.protect = 0.0
        if not self.active:
            return
        lut = identity()
        s_all = float(g.get("lut_strength", 1.0))
        if g.get("look"):
            k = float(g.get("look_strength", s_all))
            lut = lut + (look_lut(g["look"]) - lut) * k
        if g.get("lut"):
            p = g["lut"] if os.path.isabs(g["lut"]) else os.path.join(base_dir, g["lut"])
            L3, dmin, dmax = read_cube(p)
            ext = sample(resample(L3), (lut - dmin) / np.maximum(1e-6, dmax - dmin))
            lut = lut + (ext - lut) * s_all
        c = lut
        wb = g.get("wb") or {}
        if wb:
            c = _wb(c, float(wb.get("temp", 0)), float(wb.get("tint", 0)))
        lgg = g.get("lift_gamma_gain") or {}
        if lgg:
            c = _lgg(c, lgg.get("lift", (0, 0, 0)), lgg.get("gamma", (0, 0, 0)), lgg.get("gain", (0, 0, 0)))
        if "contrast" in g:
            c = _contrast(c, float(g["contrast"]))
        if "saturation" in g:
            c = _sat(c, float(g["saturation"]))
        if g.get("vibrance"):
            mx, mn = c.max(-1, keepdims=True), c.min(-1, keepdims=True)
            c = _sat(c, 1 + float(g["vibrance"]) * (1 - np.clip((mx - mn) * 2.5, 0, 1)))
        self.lut = np.clip(c, 0, 1).astype(np.float32)
        self.is_identity = float(np.abs(self.lut - identity()).max()) < 1e-4
        self.protect = float(g.get("protect_skin", 0.6 if (g.get("look") or g.get("lut")) else 0.0))

    def apply_lut(self, bgr):
        """trilinear 33³ lookup computed at quarter/half resolution as a DELTA (detail stays full-res)."""
        if self.is_identity:
            return bgr
        H, W = bgr.shape[:2]
        div = 4 if W >= 720 else 2
        sm = cv2.resize(bgr, (max(2, W // div), max(2, H // div)), interpolation=cv2.INTER_AREA)
        out = sample(self.lut, sm[..., ::-1].astype(np.float32) / 255.0)[..., ::-1]
        delta = out * 255.0 - sm.astype(np.float32)
        if self.protect > 0:
            delta *= (1 - self.protect * skin_weight(sm))[..., None]
        delta = cv2.resize(delta.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        return np.clip(bgr.astype(np.float32) + delta, 0, 255).astype(np.uint8)

    def finish(self, bgr, fi=0):
        """grain / halation / sharpen / letterbox — the FINAL frame (after titles), like a film print."""
        g = self.g
        out = bgr
        if g.get("halation"):
            H, W = out.shape[:2]
            sm = cv2.resize(out, (max(4, W // 4), max(4, H // 4)), interpolation=cv2.INTER_AREA)
            hl = np.clip((sm.astype(np.float32) / 255.0 - 0.78) * 4.5, 0, 1).max(-1)
            hl = cv2.GaussianBlur(hl, (0, 0), 2.25)
            hl = cv2.resize(hl, (W, H), interpolation=cv2.INTER_LINEAR)
            add = np.stack([hl * 0.12, hl * 0.25, hl * 1.0], -1) * float(g["halation"]) * 255
            out = np.clip(out.astype(np.float32) + add, 0, 255).astype(np.uint8)
        if g.get("sharpen"):
            bl = cv2.GaussianBlur(out, (0, 0), 1.2)
            out = cv2.addWeighted(out, 1 + float(g["sharpen"]), bl, -float(g["sharpen"]), 0)
        if g.get("grain"):
            out = add_grain(out, float(g["grain"]), float(g.get("grain_size", 1.4)), fi)
        if g.get("letterbox"):
            out = letterbox(out, float(g["letterbox"]))
        return out


_GRAIN_TILE = None


def add_grain(bgr, amt, size=1.4, seed=0):
    """luma-weighted gaussian grain, strongest in mid-tones (film), deterministic per frame index (fast cached tile)."""
    global _GRAIN_TILE
    H, W = bgr.shape[:2]
    gh, gw = max(8, int(H / size)), max(8, int(W / size))
    if _GRAIN_TILE is None or _GRAIN_TILE.shape[0] < gh + 128 or _GRAIN_TILE.shape[1] < gw + 128:
        _GRAIN_TILE = np.random.default_rng(1009).standard_normal((max(1600, gh + 256), max(1000, gw + 256))).astype(np.float32)
    rng = np.random.default_rng(1009 + int(seed))
    oy = int(rng.integers(0, _GRAIN_TILE.shape[0] - gh))
    ox = int(rng.integers(0, _GRAIN_TILE.shape[1] - gw))
    sgn = 1.0 if int(rng.integers(0, 2)) == 0 else -1.0
    n = _GRAIN_TILE[oy:oy + gh, ox:ox + gw] * sgn
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)
    y = bgr[:, :, 1].astype(np.float32) * (1.0 / 255.0)
    k = (0.35 + 2.6 * y * (1.0 - y)) * (amt * 14.0)
    return np.clip(bgr.astype(np.float32) + (n * k)[..., None], 0, 255).astype(np.uint8)


def letterbox(bgr, ratio):
    H, W = bgr.shape[:2]
    if ratio <= 0 or W / ratio >= H:
        return bgr
    b = int(round((H - W / ratio) / 2))
    out = bgr.copy()
    out[:b] = 0
    out[H - b:] = 0
    return out


def looks_sheet(src_bgr, out, names=None, cols=4, w=300):
    names = names or list(LOOKS)
    h = int(src_bgr.shape[0] * w / src_bgr.shape[1])
    base = cv2.resize(src_bgr, (w, h), interpolation=cv2.INTER_AREA)
    tiles = []
    for nm in ["original"] + names:
        im = (base if nm == "original" else Grader({"look": nm}).apply_lut(base)).copy()
        cv2.rectangle(im, (0, 0), (w, 26), (0, 0, 0), -1)
        cv2.putText(im, nm, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(base))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out
