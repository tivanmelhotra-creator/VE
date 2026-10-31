"""
toonface.kit.sheetlib — v33 Expression Library from reference sheets (the Swap-Set / Drawing-Substitution way).

Professional 2D pipelines (Adobe Character Animator swap sets, Toon Boom Harmony drawing substitution, Live2D /
Inochi2D) do NOT generate faces procedurally: an artist draws every expression once, the software only places,
swaps and blends those drawings on the moving head.  This module automates the "artist draws once" part:

    lib = build_library("sheets/*.png", out_dir="characters/boy/expressions")     # one call per sheet set
    lib = ExpressionLibrary.load("characters/boy/expressions")
    lib.names()  →  ['roaring', 'star_eyed', 'terrified', 'pleading', ...]        (labels read by OCR)

For every face on a sheet it
  1. finds the blank-face skin region (flat, ink-bounded, egg-shaped — skin-colour agnostic),
  2. cuts out EVERYTHING the artist drew inside the face (eyes, brows, nose, mouth, blush, tears, sweat, stars…)
     as an RGBA decal, removing the face outline and hair locks that hang into the face,
  3. normalises it to FACE UNITS (x right, y up from the chin, 1.0 = widest face row) so it fits any blank face,
  4. splits it into parts (brow_L/R, eye_L/R, nose, mouth, other) with boxes → per-part customisation,
  5. reads the caption under the face with tesseract → the expression name.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi

from .facefind import face_frame_metrics, ink_mask

IGNORE_WORDS = {"FRONT", "VIEW", "SIMPLE", "EXAGGERATED", "EXPRESSIONS", "EXPRESSION", "SHEET", "&", "AND"}


# ================================================================================================ data
@dataclass
class PartBox:
    name: str
    bbox: tuple            # x0, y0, x1, y1 in decal px
    center_u: tuple        # centre in face units (x right, y up from chin)
    area: int


@dataclass
class Expression:
    name: str
    source: str            # sheet file
    cell_bbox: tuple       # face box on the sheet
    px_per_unit: float     # decal pixels per face unit
    origin_px: tuple       # decal px of the face-unit origin (cheek-centre x, chin y)
    skin_bgr: tuple        # sheet skin colour (for harmonising to another skin)
    line_px: float
    parts: list = field(default_factory=list)
    decal: str = ""        # file name of the RGBA png (straight alpha)

    def unit_to_decal(self) -> np.ndarray:
        """2×3: face units → decal px."""
        k = self.px_per_unit
        return np.array([[k, 0.0, self.origin_px[0]], [0.0, -k, self.origin_px[1]]], np.float64)


class ExpressionLibrary:
    def __init__(self, root: str, items: dict):
        self.root = root
        self.items = items
        self._cache = {}

    @classmethod
    def load(cls, root: str) -> "ExpressionLibrary":
        with open(os.path.join(root, "library.json")) as fh:
            d = json.load(fh)
        items = {}
        for k, v in d["expressions"].items():
            v = dict(v)
            v["parts"] = [PartBox(**p) for p in v.get("parts", [])]
            items[k] = Expression(**v)
        return cls(root, items)

    def names(self):
        return list(self.items)

    def decal(self, name: str) -> np.ndarray:
        if name not in self._cache:
            self._cache[name] = cv2.imread(os.path.join(self.root, self.items[name].decal), cv2.IMREAD_UNCHANGED)
        return self._cache[name]

    def save(self):
        d = {"version": 1, "expressions": {k: asdict(v) for k, v in self.items.items()}}
        with open(os.path.join(self.root, "library.json"), "w") as fh:
            json.dump(d, fh, indent=1, default=float)


# ================================================================================================ detection
def _lab(bgr):
    return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def detect_faces(sheet: np.ndarray, min_area: int = 3500):
    """→ (list of dict(label_id, bbox, mask_full), skin_bgr, hair_bgr, lbl).  Face = solid, roughly round flat
    region bounded by line-art, whose colour is NOT the hair colour (hair masses are low-solidity)."""
    g = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
    ink = ink_mask(g, k=9, contrast=45) | (g < 90)
    ink = cv2.dilate(ink.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    n, lbl, st, _ = cv2.connectedComponentsWithStats((~ink).astype(np.uint8), connectivity=4)
    border = set(np.unique(np.r_[lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]]).tolist())
    regs = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if i in border or a < min_area or a > 0.08 * sheet.shape[0] * sheet.shape[1]:
            continue
        comp = lbl[y:y + h, x:x + w] == i
        filled = ndi.binary_fill_holes(comp)
        cs, _ = cv2.findContours(filled.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        sol = cv2.contourArea(c) / max(1.0, cv2.contourArea(cv2.convexHull(c)))
        col = np.median(sheet[lbl == i], axis=0)
        regs.append(dict(i=i, bbox=(int(x), int(y), int(x + w), int(y + h)), area=int(a), sol=float(sol),
                         asp=float(w / max(1, h)), col=col, holes=float(filled.sum() / a)))
    hair = [r for r in regs if r["sol"] < 0.5]
    hair_bgr = np.median([r["col"] for r in hair], axis=0) if hair else np.array([204, 233, 248], np.float32)
    hl = _lab(hair_bgr)
    faces = []
    for r in regs:
        if r["sol"] < 0.5 or not (0.72 <= r["asp"] <= 1.4):
            continue
        if np.linalg.norm(_lab(r["col"]) - hl) < 6.0 and r["holes"] < 1.05:    # a hair-coloured blob, no features
            continue
        faces.append(r)
    if faces:                                                                     # same character → similar sizes
        med = np.median([f["area"] * f["holes"] for f in faces])
        faces = [f for f in faces if 0.45 * med <= f["area"] * f["holes"] <= 2.2 * med]
    skin = np.median([f["col"] for f in faces if f["sol"] > 0.8] or [f["col"] for f in faces], axis=0) \
        if faces else np.array([176, 216, 241], np.float32)
    faces.sort(key=lambda f: (round(f["bbox"][1] / 80), f["bbox"][0]))
    return faces, skin.astype(np.float32), hair_bgr.astype(np.float32), lbl


# ================================================================================================ extraction
def extract_decal(sheet, lbl, face, skin_bgr, hair_bgr, pad: int = 6):
    """cut the artist's drawing out of one face → (rgba crop, face mask crop, offset, line_px)."""
    H, W = sheet.shape[:2]
    comp = lbl == face["i"]
    ys, xs = np.nonzero(comp)
    hull = cv2.convexHull(np.c_[xs, ys].astype(np.int32))
    hm = np.zeros((H, W), np.uint8)
    cv2.drawContours(hm, [hull], -1, 1, -1)
    # the ink outline of the face sits just outside the skin hull: grow by the stroke width
    g = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
    inkm = ink_mask(g, k=9, contrast=45) | (g < 90)
    ring = (cv2.dilate(hm, np.ones((15, 15), np.uint8)) > 0) & (hm == 0) & inkm
    dt = cv2.distanceTransform(ring.astype(np.uint8), cv2.DIST_L2, 3)
    line_px = float(np.clip(2.0 * np.percentile(dt[ring], 85), 2.0, 12.0)) if ring.any() else 4.0
    face_reg = cv2.dilate(hm, np.ones((3, 3), np.uint8)) > 0
    lab = cv2.cvtColor(sheet, cv2.COLOR_BGR2LAB).astype(np.float32)
    # skin reference = THIS face's own skin (sheets shade faces differently: freezing = blue, angry = red)
    loc = np.median(lab[comp], axis=0)
    dsk = np.linalg.norm(lab - loc, axis=2)
    thr = float(np.clip(np.percentile(dsk[comp], 97) + 4.0, 12.0, 26.0))      # adaptive to JPEG noise
    nonskin = face_reg & ((dsk > thr) | inkm)
    nonskin = cv2.morphologyEx(nonskin.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    # the outline band: ink within ~2 strokes of the hull border is the face contour, not a feature
    k_in = int(2 * line_px) | 1
    inner = cv2.erode(hm, np.ones((k_in, k_in), np.uint8)) > 0
    band = face_reg & ~inner
    nonskin &= ~(band & inkm)
    n, cl, cst, _ = cv2.connectedComponentsWithStats(nonskin.astype(np.uint8), connectivity=8)
    dh = np.linalg.norm(lab - _lab(hair_bgr), axis=2)
    yb0, yb1 = ys.min(), ys.max()
    fh = max(1, yb1 - yb0)
    band_wide = face_reg & ~(cv2.erode(hm, np.ones((k_in + 8, k_in + 8), np.uint8)) > 0)
    keep = np.zeros(n, bool)
    for j in range(1, n):
        area = cst[j, cv2.CC_STAT_AREA]
        if area < 12:                                   # specks
            continue
        cj = cl == j
        top_y = (cst[j, cv2.CC_STAT_TOP] - yb0) / fh
        cy = (cst[j, cv2.CC_STAT_TOP] + cst[j, cv2.CC_STAT_HEIGHT] / 2 - yb0) / fh
        touches = (cj & band_wide).any()
        hair_frac = float(((dh < 12) | inkm)[cj].mean())
        band_frac = float(band_wide[cj].mean())
        if band_frac > 0.40:                            # face outline / jaw shading / hair fringe on the border
            continue
        # forelock: hair-coloured/ink, upper half, reaching the face border (it comes from the hair mass)
        if touches and hair_frac > 0.70 and cy < 0.50:
            continue
        keep[j] = True
    feat = keep[cl]
    # alpha: how far a pixel is from skin (anti-aliased edges get partial alpha), only near kept features
    near = cv2.dilate(feat.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    a_lo, a_hi = 0.45 * thr, 2.2 * thr
    alpha = np.clip((dsk - a_lo) / (a_hi - a_lo), 0.0, 1.0)
    alpha = np.where(feat, np.maximum(alpha, 0.85), alpha) * near * face_reg
    alpha = np.where(inkm & near, 1.0, alpha)
    # un-mix the skin out of the edge pixels: C = (I - (1 - a)·skin) / a  → decal works on ANY skin colour
    skin_bgr_loc = cv2.cvtColor(loc.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2BGR)[0, 0].astype(np.float32)
    a3 = np.maximum(alpha, 1e-3)[..., None]
    col = (sheet.astype(np.float32) - (1.0 - a3) * skin_bgr_loc) / a3
    col = np.where(alpha[..., None] > 0.02, col, sheet.astype(np.float32)).clip(0, 255).astype(np.uint8)
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x1, y1 = min(W, xs.max() + pad + 1), min(H, ys.max() + pad + 1)
    rgba = np.dstack([col[y0:y1, x0:x1], (alpha[y0:y1, x0:x1] * 255).astype(np.uint8)])
    return rgba, face_reg[y0:y1, x0:x1], (x0, y0), line_px, feat[y0:y1, x0:x1]


def split_parts(feat: np.ndarray, to_unit) -> list:
    """components of the feature mask → named parts by position in face units."""
    n, cl, st, ce = cv2.connectedComponentsWithStats(cv2.dilate(feat.astype(np.uint8), np.ones((5, 5), np.uint8)),
                                                     connectivity=8)
    parts = []
    for j in range(1, n):
        if st[j, cv2.CC_STAT_AREA] < 25:
            continue
        x, y, w, h, a = st[j]
        u = to_unit(np.array([ce[j]]))[0]
        if u[1] > 0.70:
            nm = "brow"
        elif u[1] > 0.42:
            nm = "eye"
        elif u[1] > 0.30 and abs(u[0]) < 0.12 and a < 0.02 * feat.size:
            nm = "nose"
        else:
            nm = "mouth"
        if nm in ("brow", "eye") and abs(u[0]) > 0.04:
            nm += "_L" if u[0] < 0 else "_R"
        parts.append(PartBox(name=nm, bbox=(int(x), int(y), int(x + w), int(y + h)),
                             center_u=(float(u[0]), float(u[1])), area=int(a)))
    return parts


# ================================================================================================ OCR
def read_words(sheet: np.ndarray):
    """tesseract TSV → [(text, x0, y0, x1, y1)] (upper-case words only)."""
    tmp = "/tmp/_sheet_ocr.png"
    big = cv2.resize(sheet, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(tmp, cv2.cvtColor(big, cv2.COLOR_BGR2GRAY))
    try:
        out = subprocess.run(["tesseract", tmp, "stdout", "--psm", "11", "tsv"], capture_output=True, text=True,
                             timeout=60).stdout
    except Exception:
        return []
    words = []
    for line in out.splitlines()[1:]:
        f = line.split("\t")
        if len(f) < 12 or not f[11].strip():
            continue
        t = re.sub(r"[^A-Z!\-]", "", f[11].upper())
        if len(t) < 3 or float(f[10]) < 30:
            continue
        x, y, w, h = (int(v) // 2 for v in f[6:10])
        words.append((t, x, y, x + w, y + h))
    return words


def _vocab():
    from .expressive import EMOTIONS
    v = set(EMOTIONS) | {"roaring", "star_eyed", "yawning", "thinking", "weeping", "eating", "disgusted", "eureka",
                         "sulking", "whistling", "freezing", "joy", "crying", "sneaky", "excited", "scared", "angry",
                         "shocked", "dizzy", "mischievous", "terrified", "pleading", "shy", "smug"}
    return sorted(v)


def _fuzzy(word: str):
    import difflib
    w = word.lower().replace("-", "_").replace("!", "")
    m = difflib.get_close_matches(w, _vocab(), n=1, cutoff=0.6)
    return (m[0], difflib.SequenceMatcher(None, w, m[0]).ratio()) if m else (w, 0.0)


def label_for(face_bbox, words, used: set):
    """closest caption word under the face (same column), else above; generic sheet words ignored."""
    x0, y0, x1, y1 = face_bbox
    cx = 0.5 * (x0 + x1)
    cands = []
    for k, (t, a, b, c, d) in enumerate(words):
        if t.strip("!-") in IGNORE_WORDS or k in used:
            continue
        wx = 0.5 * (a + c)
        if abs(wx - cx) > 0.75 * (x1 - x0):
            continue
        dy = b - y1 if b >= y1 else (y0 - d) * 2.5 + 400        # prefer below the face
        if dy >= 0:
            cands.append((dy, k, t))
    if not cands:
        return None
    cands.sort()
    best = None
    for dy, k, t in cands[:3]:                                  # nearest captions, best vocabulary match wins
        name, score = _fuzzy(t)
        key = (score >= 0.6, -dy)
        if best is None or key > best[0]:
            best = (key, k, name)
    used.add(best[1])
    return best[2]


# ================================================================================================ build
def build_library(sheets, out_dir: str, unit_px: float = 360.0, verbose: bool = True) -> ExpressionLibrary:
    """sheets: glob pattern or list of files.  Decals are resampled to `unit_px` pixels per face unit."""
    files = sorted(glob.glob(sheets)) if isinstance(sheets, str) else list(sheets)
    os.makedirs(out_dir, exist_ok=True)
    items: dict = {}
    for f in files:
        sheet = cv2.imread(f)
        faces, skin, hair, lbl = detect_faces(sheet)
        words = read_words(sheet)
        used: set = set()
        for k, face in enumerate(faces):
            rgba, freg, (ox, oy), line_px, feat = extract_decal(sheet, lbl, face, skin, hair)
            cx, chin, top, Wf = face_frame_metrics(freg)
            name = label_for(face["bbox"], words, used) or f"{os.path.splitext(os.path.basename(f))[0]}_{k}"
            base, j = name, 2
            while name in items:
                name = f"{base}_{j}"
                j += 1
            s = unit_px / max(1.0, Wf)                                   # resample to a common resolution
            big = cv2.resize(rgba, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            featb = cv2.resize(feat.astype(np.uint8), None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST) > 0
            org = (cx * s, chin * s)
            to_unit = lambda p, org=org: np.c_[(p[:, 0] - org[0]) / unit_px, (org[1] - p[:, 1]) / unit_px]
            fn = f"{name}.png"
            cv2.imwrite(os.path.join(out_dir, fn), big)
            items[name] = Expression(name=name, source=os.path.basename(f), cell_bbox=tuple(face["bbox"]),
                                     px_per_unit=unit_px, origin_px=(float(org[0]), float(org[1])),
                                     skin_bgr=tuple(float(v) for v in skin), line_px=float(line_px * s),
                                     parts=split_parts(featb, to_unit), decal=fn)
            if verbose:
                print(f"[sheetlib] {os.path.basename(f)} face {k}: {name:14s} parts="
                      + ",".join(p.name for p in items[name].parts))
    lib = ExpressionLibrary(out_dir, items)
    lib.save()
    return lib


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="build an expression library from reference sheets")
    ap.add_argument("sheets", help="glob, e.g. 'sheets/*.png'")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build_library(a.sheets, a.out)
