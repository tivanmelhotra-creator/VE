"""
toonface.kit.decalrig — v33 2.5D PART RIG for artist drawings: any head yaw, per-character shape, gender blend.

Pattern (professional cut-out / 2.5D rigs — Live2D "parameter keys", Toon Boom drawing substitution, Spine skins):
  * every expression has KEY DRAWINGS at a few views (front, 3/4, profile) made by the image model in the character's
    style (redesign.py).  Each key drawing is split into PARTS (brow_L/R, eye_L/R, nose, mouth, fx).
  * between keys the parts are moved by a head model (ellipsoid): every part anchor sits on the head surface, the head
    is rotated to the target yaw and the part is re-placed + foreshortened (x-scale = local surface slope).  So 20°,
    37°, 55°… are all valid, not only the drawn angles.  The NEAREST key drawing is used (drawing substitution) with a
    short dissolve band around the switch angle; mirrored keys cover the other side.
  * a FaceShape (per character, saved as JSON) scales / moves parts in canonical head space BEFORE the turn: eye size,
    spacing, roundness, brow height/length/thickness, nose size, mouth width/height, line weight, lashes.
    Gender is not a separate rig: gender_delta() measures the boy and girl kits in face units and returns the
    difference as a FaceShape, so `shape.blend(gender_delta(), t)` switches male→female continuously.

Units: face units (x right, y up from chin, 1.0 = widest face row) for drawings and targets; the head model works in
"front physical" units = face units of the frontal view.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
from dataclasses import asdict, dataclass, field, fields

import cv2
import numpy as np

from .decalcomp import compose, invert
from .sheetlib import ExpressionLibrary, _zone

PARTS_ORDER = ("mouth", "nose", "eye_L", "eye_R", "brow_L", "brow_R", "fx")


# ================================================================================================ head model
@dataclass
class HeadModel:
    a: float = 0.5           # half width of the face (face units of the front view)
    b: float = 0.62          # half height
    yc: float = 0.55         # ellipsoid centre above the chin
    depth: float = 0.42      # z semi-axis: how far features slide when the head turns
    chin_depth: float = 0.21 # chin protrusion (yaw estimate from the drawn chin offset)
    profile_deg: float = 85.0

    def z(self, x, y):
        q = 1.0 - (x / self.a) ** 2 - ((y - self.yc) / self.b) ** 2
        return self.depth * math.sqrt(min(1.0, max(0.02, q)))

    def a_eff(self, th):
        """half width of the visible silhouette at yaw th (rad)."""
        return math.sqrt((self.a * math.cos(th)) ** 2 + (self.depth * math.sin(th)) ** 2)

    def project(self, x, y, th):
        """front point (x, y) → (x', local x-scale, depth toward viewer) at yaw th."""
        z = self.z(x, y)
        s, c = math.sin(th), math.cos(th)
        dz = -(self.depth ** 2) * x / (self.a ** 2 * z)
        return x * c + z * s, c + s * dz, -x * s + z * c

    def unproject_x(self, xp, y, th):
        xs = np.linspace(-0.97 * self.a, 0.97 * self.a, 241)
        best, bx = 1e9, 0.0
        for x in xs:
            p, sx, zf = self.project(x, y, th)
            if zf > -0.02 and abs(p - xp) < best:
                best, bx = abs(p - xp), x
        return float(bx)

    def yaw_from_chin(self, off: float) -> float:
        """drawn chin offset (chin_offset(), units of visible width) → yaw degrees."""
        ths = np.radians(np.linspace(-80, 80, 321))
        err = [abs(off * 2 * self.a_eff(t) - self.chin_depth * math.sin(t)) for t in ths]
        return float(np.degrees(ths[int(np.argmin(err))]))

    def yaw_from_eyes(self, u_mid: float, y_eye: float) -> float:
        """midpoint of the two eyes in a drawing (its own face units) → yaw degrees of that drawing."""
        ths = np.radians(np.linspace(-80, 80, 321))
        z0 = self.z(0.0, y_eye)
        err = [abs(u_mid * 2 * self.a_eff(t) - z0 * math.sin(t)) for t in ths]
        return float(np.degrees(ths[int(np.argmin(err))]))


# ================================================================================================ face shape
@dataclass
class FaceShape:
    """per-character art direction, applied in canonical (front) head space.  Multipliers 1 / offsets 0 = as drawn."""
    eye_scale: float = 1.0
    eye_spacing: float = 1.0
    eye_y: float = 0.0
    eye_round: float = 1.0       # vertical / horizontal eye scale ratio
    brow_y: float = 0.0          # extra brow lift above the eyes
    brow_len: float = 1.0
    brow_thick: float = 1.0
    nose_scale: float = 1.0
    mouth_w: float = 1.0
    mouth_y: float = 0.0
    line: float = 1.0            # ink line weight multiplier
    lashes: float = 0.0          # 0 none … 1 full upper lashes
    brow_layer: str = "under"    # "under" hair (clipped to the face) | "over" hair

    MULT = ("eye_scale", "eye_spacing", "eye_round", "brow_len", "brow_thick", "nose_scale", "mouth_w", "line")

    def blend(self, other: "FaceShape", t: float) -> "FaceShape":
        """multipliers interpolate geometrically, offsets linearly."""
        d = {}
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            if isinstance(a, str):
                d[f.name] = a if t < 0.5 else b
            elif f.name in self.MULT:
                d[f.name] = float(math.exp((1 - t) * math.log(a) + t * math.log(b)))
            else:
                d[f.name] = float((1 - t) * a + t * b)
        return FaceShape(**d)

    def combine(self, other: "FaceShape") -> "FaceShape":
        """apply `other` on top of self (multipliers multiply, offsets add)."""
        d = {}
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            if isinstance(a, str):
                d[f.name] = b if b != f.default else a
            elif f.name in self.MULT:
                d[f.name] = a * b
            elif f.name == "lashes":
                d[f.name] = max(a, b)
            else:
                d[f.name] = a + b
        return FaceShape(**d)

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=1)
        return path

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls(**json.load(fh))


PRESETS = {
    "as_drawn": FaceShape(),
    "big_cute_eyes": FaceShape(eye_scale=1.22, eye_round=1.08, eye_spacing=1.04, nose_scale=0.8, mouth_w=0.9),
    "sharp_narrow": FaceShape(eye_scale=0.9, eye_round=0.78, brow_thick=1.25, brow_y=-0.02, mouth_w=1.08),
    "wide_set": FaceShape(eye_spacing=1.14, eye_scale=0.95, nose_scale=1.15, mouth_w=1.12),
    "brows_over_hair": FaceShape(brow_layer="over", brow_thick=1.15),
}


def _kit_metrics(name: str):
    """front-view feature metrics of a hand-made kit character, in face units of its own blank plate."""
    from .facefind import find_face, measure_style
    here = os.path.dirname(__file__)
    d = json.load(open(os.path.join(here, "data", f"{name}.json")))
    plates = os.path.join(here, "..", "..", "..", "..", "tests", "fixtures", "toonface", "plates")
    img = cv2.imread(os.path.join(plates, d["plates"]["front"]))
    st = measure_style(img, find_face(img).mask)
    W = st.width

    def U(p):
        return np.array([(p[0] - st.cx) / W, (st.chin_y - p[1]) / W])

    v = d["views"]["front"]
    eyes = [np.array(v["eyes"][s]["outline"], float) for s in "LR"]
    brows = [np.array(v["brows"][s]["line"], float) for s in "LR"]
    m = v["mouth"]
    nose = np.array(v["nose"], float)
    return dict(
        eye_w=np.mean([np.ptp(e[:, 0]) for e in eyes]) / W,
        eye_h=np.mean([np.ptp(e[:, 1]) for e in eyes]) / W,
        eye_dx=np.mean([abs(U(e.mean(0))[0]) for e in eyes]),
        eye_y=np.mean([U(e.mean(0))[1] for e in eyes]),
        brow_y=np.mean([U(b.mean(0))[1] for b in brows]),
        brow_len=np.mean([np.linalg.norm(b[-1] - b[0]) for b in brows]) / W,
        brow_th=v["brows"]["L"]["th"] / W,
        nose=math.sqrt(np.ptp(nose[:, 0]) * max(np.ptp(nose[:, 1]), 1.0)) / W,
        mouth_w=abs(m["right"][0] - m["left"][0]) / W,
        mouth_y=U(m["low"])[1],
        stroke=d["stroke"] / W,
        lashes=1.0 if d.get("lashes") else 0.0,
    )


@functools.lru_cache(maxsize=1)
def gender_delta() -> FaceShape:
    """male→female difference measured on the boy / girl kits (same artist, same plates scale) as a FaceShape."""
    b, g = _kit_metrics("boy"), _kit_metrics("girl")
    return FaceShape(
        eye_scale=g["eye_w"] / b["eye_w"],
        eye_spacing=g["eye_dx"] / b["eye_dx"],
        eye_y=g["eye_y"] - b["eye_y"],
        eye_round=(g["eye_h"] / g["eye_w"]) / (b["eye_h"] / b["eye_w"]),
        brow_y=(g["brow_y"] - g["eye_y"]) - (b["brow_y"] - b["eye_y"]),
        brow_len=g["brow_len"] / b["brow_len"],
        brow_thick=g["brow_th"] / b["brow_th"],
        nose_scale=float(np.clip(g["nose"] / b["nose"], 0.7, 1.0)),   # girl nose is a different drawn shape
        mouth_w=g["mouth_w"] / b["mouth_w"],
        mouth_y=g["mouth_y"] - b["mouth_y"],
        line=g["stroke"] / b["stroke"],
        lashes=g["lashes"] - b["lashes"],
    )


def character_shape(base: FaceShape | None = None, femininity: float = 0.0) -> FaceShape:
    """base art direction + gender slider (0 = as drawn / male, 1 = female kit proportions)."""
    base = base or FaceShape()
    if femininity <= 0:
        return base
    return base.combine(FaceShape().blend(gender_delta(), float(np.clip(femininity, 0, 1))))


# ================================================================================================ drawings → parts
@dataclass
class Part:
    name: str
    rgba: np.ndarray          # straight alpha crop
    x0: float                 # crop offset in decal px
    y0: float
    anchor: tuple             # centre in key face units
    line_px: float


@dataclass
class Drawing:
    expr: str
    view: str
    yaw: float                # degrees of the key drawing (+ = looks toward viewer's right)
    unit_px: float
    origin: tuple
    parts: list = field(default_factory=list)

    def unit_to_decal(self):
        k = self.unit_px
        return np.array([[k, 0.0, self.origin[0]], [0.0, -k, self.origin[1]]])

    def mirrored(self) -> "Drawing":
        ox = self.origin[0]
        sw = {"_L": "_R", "_R": "_L"}
        ps = [Part(p.name[:-2] + sw[p.name[-2:]] if p.name[-2:] in sw else p.name, p.rgba[:, ::-1].copy(),
                   2 * ox - (p.x0 + p.rgba.shape[1] - 1), p.y0, (-p.anchor[0], p.anchor[1]), p.line_px)
              for p in self.parts]
        return Drawing(self.expr, self.view + "~m", -self.yaw, self.unit_px, self.origin, ps)


def split_drawing(rgba: np.ndarray, origin, unit_px: float, line_px: float) -> list:
    """decal → Part list (sheetlib zoning, pixel masks).  Left/right is decided relative to the midpoint between the
    two eyes (not x=0): on a turned head the near eye crosses the face centre line and must stay one part."""
    a = rgba[..., 3] > 0
    n, cl, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(a.astype(np.uint8), np.ones((5, 5), np.uint8)), 8)
    lab = np.full(a.shape, -1, np.int32)
    names = list(PARTS_ORDER)
    comps = []
    for j in range(1, n):
        ys, xs = np.nonzero((cl == j) & a)
        if len(xs) < 20:
            continue
        u = np.stack([(xs - origin[0]) / unit_px, (origin[1] - ys) / unit_px], 1)
        comps.append((ys, xs, u))
    eye_c = sorted([(len(c[1]), c[2][:, 0].mean()) for c in comps
                    if 0.42 < c[2][:, 1].mean() <= 0.70 and abs(c[2][:, 0].mean()) < 0.62], reverse=True)[:2]
    bx = float(np.mean([e[1] for e in eye_c])) if len(eye_c) == 2 else 0.0

    def zone(ux, uy, af):
        z = _zone(ux, uy, af)
        if z[-2:] in ("_L", "_R"):
            z = z[:-2] + ("_L" if ux < bx else "_R")
        return z

    for ys, xs, u in comps:
        af = len(xs) / a.size
        sx = np.percentile(u[:, 0], 95) - np.percentile(u[:, 0], 5)
        sy = np.percentile(u[:, 1], 95) - np.percentile(u[:, 1], 5)
        cx, cy = u[:, 0].mean(), u[:, 1].mean()
        if (sx < 0.34 and sy < 0.26) or cy < 0.40:
            lab[ys, xs] = names.index(zone(cx, cy, af))
        else:
            lab[ys, xs] = [names.index(zone(x, y, af)) for x, y in u]
    parts = []
    for k, nm in enumerate(names):
        m = lab == k
        if m.sum() < 20:
            continue
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = rgba[y0:y1, x0:x1].copy()
        crop[..., 3] = np.where(m[y0:y1, x0:x1], crop[..., 3], 0)
        w = crop[..., 3].astype(np.float64) + 1e-6
        gy, gx = np.mgrid[y0:y1, x0:x1]
        ax = ((gx * w).sum() / w.sum() - origin[0]) / unit_px
        ay = (origin[1] - (gy * w).sum() / w.sum()) / unit_px
        parts.append(Part(nm, crop, float(x0), float(y0), (float(ax), float(ay)), line_px))
    return parts


VIEW_YAW = {"front": 0.0, "q34": 40.0, "profile": 85.0}


def load_drawings(lib: ExpressionLibrary, head: HeadModel) -> dict:
    """library → {expr: [Drawing …] sorted by yaw}.  Key yaw = the angle the drawing was ORDERED at (view label ×
    facing side); mirrored copies cover the other side."""
    out = {}
    for key, e in lib.items.items():
        expr, view = key.split("@") if "@" in key else (key, "front")
        parts = split_drawing(lib.decal(key), e.origin_px, e.px_per_unit, e.line_px)
        facing = getattr(e, "facing", 0) or 0
        if view != "front" and not facing:
            facing = 1 if np.mean([p.anchor[0] for p in parts] or [0]) > 0 else -1
        yaw = VIEW_YAW.get(view, 0.0) * (facing if view != "front" else 0)
        if view == "profile":
            near = "_R" if facing < 0 else "_L"
            for p in parts:
                if p.name[-2:] in ("_L", "_R"):
                    p.name = p.name[:-2] + near
        d = Drawing(expr, view, float(yaw), e.px_per_unit, e.origin_px, parts)
        out.setdefault(expr, []).append(d)
        if view != "front":
            out[expr].append(d.mirrored())
    for v in out.values():
        v.sort(key=lambda d: d.yaw)
    return out


# ================================================================================================ part styling
def _thin(rgba: np.ndarray, factor: float, line_px: float) -> np.ndarray:
    """line weight × factor (<1 thinner) by fractional alpha erosion (inner fills are untouched)."""
    if abs(factor - 1.0) < 0.03 or factor > 1.0:
        return rgba
    want = (1.0 - factor) * line_px                    # px of stroke width to remove (both sides)
    out = rgba.copy()
    a = out[..., 3].astype(np.float32)
    while want > 1e-3:
        e = cv2.erode(a, np.ones((3, 3), np.uint8))
        t = min(1.0, want / 2.0)
        a = a * (1 - t) + e * t
        want -= 2.0
    out[..., 3] = a.clip(0, 255).astype(np.uint8)
    return out


def _lashes(rgba: np.ndarray, outer: int, amount: float, line_px: float, unit_px: float):
    """add upper lashes at the outer end of an eye drawing (ink colour + stroke width taken from the drawing).
    Returns (rgba, dx, dy) — the crop grows, (dx, dy) = offset of the old crop inside the new one."""
    if amount <= 0.02:
        return rgba, 0, 0
    a = rgba[..., 3] > 128
    if a.sum() < 30:
        return rgba, 0, 0
    L = int(0.05 * unit_px * (0.6 + 0.4 * amount)) + 2
    pad = L + 2
    big = cv2.copyMakeBorder(rgba, pad, 0, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    A = big[..., 3] > 128
    cols = np.where(A.any(0))[0]
    x_lo, x_hi = cols.min(), cols.max()
    top = {x: int(np.argmax(A[:, x])) for x in cols}
    dark = big[..., :3].astype(np.float32).mean(2) < 90
    ink = np.median(big[..., :3][dark & A], axis=0) if (dark & A).any() else np.array([20, 20, 20])
    lw = max(2.0, 0.9 * line_px)
    w = x_hi - x_lo
    n = 3 if amount > 0.5 else 2
    sup = 4
    layer = np.zeros((big.shape[0] * sup, big.shape[1] * sup), np.uint8)
    for k in range(n):
        f = 0.80 + 0.18 * k / max(1, n - 1)
        x = int(x_lo + f * w) if outer > 0 else int(x_hi - f * w)
        x = int(np.clip(x, x_lo, x_hi))
        y = top[x] + 0.3 * lw
        ang = math.radians(100 - 22 * (k + 1)) if outer > 0 else math.radians(80 + 22 * (k + 1))
        ln = L * (0.75 + 0.25 * k / max(1, n - 1))
        tip = (x + ln * math.cos(ang), y - ln * math.sin(ang))
        nx, ny = -math.sin(ang), -math.cos(ang)
        mid = (x + 0.55 * ln * math.cos(ang) - outer * 0.12 * ln, y - 0.55 * ln * math.sin(ang))
        poly = [(x + nx * lw / 2, y + ny * lw / 2), (mid[0] + nx * lw / 3, mid[1] + ny * lw / 3), tip,
                (mid[0] - nx * lw / 3, mid[1] - ny * lw / 3), (x - nx * lw / 2, y - ny * lw / 2)]
        cv2.fillPoly(layer, [np.round(np.array(poly) * sup).astype(np.int32)], 255, cv2.LINE_AA)
    la = cv2.resize(layer, (big.shape[1], big.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    la *= min(1.0, amount * 1.5)
    out = big.astype(np.float32)
    a0 = out[..., 3:4] / 255.0
    col = out[..., :3] * a0 * (1 - la[..., None]) + ink * la[..., None]
    an = la[..., None] + a0 * (1 - la[..., None])
    out[..., :3] = col / np.maximum(an, 1e-4)
    out[..., 3:4] = an * 255
    return out.clip(0, 255).astype(np.uint8), pad, pad


# ================================================================================================ the rig
class DecalRig:
    """expression drawings + head model + character shape → premultiplied face layer at any yaw."""

    def __init__(self, lib: ExpressionLibrary | str, head: HeadModel | None = None, shape: FaceShape | None = None,
                 dissolve_deg: float = 0.0, hysteresis_deg: float = 3.0):
        self.lib = ExpressionLibrary.load(lib) if isinstance(lib, str) else lib
        self.head = head or HeadModel()
        self.shape = shape or FaceShape()
        self.dissolve = dissolve_deg
        self.hysteresis_deg = hysteresis_deg
        self._swap = {}
        self.drawings = load_drawings(self.lib, self.head)
        self._styled = {}

    def expressions(self):
        return sorted(self.drawings)

    # ------------------------------------------------------------------------------------------ keyed interpolation
    @staticmethod
    def _psize(d: Drawing, p: Part):
        return p.rgba.shape[1] / d.unit_px, p.rgba.shape[0] / d.unit_px

    def _synth(self, d_from: Drawing, p: Part, yaw_to: float):
        """part missing in a key: carry it from the other key with the head model (anchor, size, vis=0..1)."""
        h = self.head
        th0, th1 = math.radians(d_from.yaw), math.radians(yaw_to)
        s0, s1 = 2 * h.a_eff(th0), 2 * h.a_eff(th1)
        X, Y = p.anchor[0] * s0, p.anchor[1] * s0
        xc = h.unproject_x(X, Y, th0)
        _, sx0, _ = h.project(xc, Y, th0)
        Xt, sx1, zf = h.project(xc, Y, th1)
        w, hh = self._psize(d_from, p)
        k = s0 / s1
        vis = float(np.clip((zf + 0.02) / 0.08, 0, 1) * np.clip((sx1 - 0.12) / 0.15, 0, 1))
        return (Xt / s1, Y / s1), (w * k * max(0.08, sx1) / max(0.15, sx0), hh * k), vis

    def bracket(self, expr: str, yaw: float):
        """→ (d0, d1, t): neighbouring key drawings and the position between them (t in 0..1)."""
        ds = self.drawings[expr]
        if yaw <= ds[0].yaw:
            return ds[0], ds[0], 0.0
        if yaw >= ds[-1].yaw:
            return ds[-1], ds[-1], 0.0
        for d0, d1 in zip(ds, ds[1:]):
            if d0.yaw <= yaw <= d1.yaw:
                return d0, d1, (yaw - d0.yaw) / max(1e-6, d1.yaw - d0.yaw)
        return ds[0], ds[0], 0.0

    def targets(self, expr: str, yaw: float):
        """per part: interpolated anchor (target face units), size, visibility, and the drawings to show with
        their dissolve weights.  {name: dict(anchor, size, vis, draws=[(Drawing, Part, w)])}"""
        d0, d1, t = self.bracket(expr, yaw)
        span = max(1.0, abs(d1.yaw - d0.yaw))
        if self.dissolve > 0:                                        # optional soft swap (off by default)
            band = self.dissolve / span
            sw = float(np.clip((t - 0.5) / max(1e-3, band) + 0.5, 0, 1))
            sw = sw * sw * (3 - 2 * sw)
        else:                                                        # hard swap at mid-interval + hysteresis
            k = (expr, d0.view, d1.view)
            hy = self.hysteresis_deg / span
            prev = self._swap.get(k, 1.0 if t >= 0.5 else 0.0)
            sw = 1.0 if t > 0.5 + hy else 0.0 if t < 0.5 - hy else prev
            self._swap[k] = sw
        out = {}
        names = {p.name for p in d0.parts} | {p.name for p in d1.parts}
        for nm in names:
            p0 = next((p for p in d0.parts if p.name == nm), None)
            p1 = next((p for p in d1.parts if p.name == nm), None)
            if d0 is d1:                                             # outside the key range: head-model carry
                a, sz, v = self._synth(d0, p0, yaw)
                out[nm] = dict(anchor=a, size=sz, vis=v, draws=[(d0, p0, 1.0)])
                continue
            if p0 is not None:
                a0, sz0, v0 = p0.anchor, self._psize(d0, p0), 1.0
            else:
                a0, sz0, v0 = self._synth(d1, p1, d0.yaw)
            if p1 is not None:
                a1, sz1, v1 = p1.anchor, self._psize(d1, p1), 1.0
            else:
                a1, sz1, v1 = self._synth(d0, p0, d1.yaw)
            lerp = lambda x, y: (1 - t) * np.asarray(x, float) + t * np.asarray(y, float)
            draws = []
            if p0 is not None and 1 - sw > 1e-3:
                draws.append((d0, p0, 1 - sw if p1 is not None else 1.0))
            if p1 is not None and sw > 1e-3:
                draws.append((d1, p1, sw if p0 is not None else 1.0))
            v = float(lerp(v0, v1))
            v = float(np.clip((v - 0.4) / 0.2, 0, 1))
            out[nm] = dict(anchor=tuple(lerp(a0, a1)), size=tuple(lerp(sz0, sz1)), vis=v, draws=draws)
        self._apply_shape(out)
        return out

    def _apply_shape(self, tg: dict):
        """character FaceShape on the interpolated layout (spacing about the eye-pair midpoint, sizes about anchors)."""
        sh = self.shape
        eyes = [tg[n]["anchor"] for n in ("eye_L", "eye_R") if n in tg]
        mid = np.mean(eyes, axis=0) if eyes else np.array([0.0, 0.5])
        disp = {}
        for side in ("L", "R"):
            n = "eye_" + side
            if n in tg:
                a = np.asarray(tg[n]["anchor"], float)
                na = np.array([mid[0] + (a[0] - mid[0]) * sh.eye_spacing, a[1] + sh.eye_y])
                disp[side] = na - a
                w, h = tg[n]["size"]
                tg[n].update(anchor=tuple(na), size=(w * sh.eye_scale, h * sh.eye_scale * sh.eye_round))
        for side in ("L", "R"):
            n = "brow_" + side
            if n in tg:
                d = disp.get(side, np.array([(np.asarray(tg[n]["anchor"])[0] - mid[0]) * (sh.eye_spacing - 1),
                                             sh.eye_y]))
                a = np.asarray(tg[n]["anchor"], float) + d + np.array([0.0, sh.brow_y])
                w, h = tg[n]["size"]
                tg[n].update(anchor=tuple(a), size=(w * sh.brow_len, h * sh.brow_thick))
        if "nose" in tg:
            w, h = tg["nose"]["size"]
            tg["nose"]["size"] = (w * sh.nose_scale, h * sh.nose_scale)
        if "mouth" in tg:
            a = tg["mouth"]["anchor"]
            w, h = tg["mouth"]["size"]
            tg["mouth"].update(anchor=(a[0], a[1] + sh.mouth_y), size=(w * sh.mouth_w, h))

    def _styled_part(self, d: Drawing, p: Part):
        sh = self.shape
        key = (d.expr, d.view, p.name, round(sh.line, 3), round(sh.lashes, 3))
        if key not in self._styled:
            rgba = _thin(p.rgba, sh.line, p.line_px)
            dx = dy = 0
            if p.name.startswith("eye") and sh.lashes > 0:
                rgba, dx, dy = _lashes(rgba, 1 if p.anchor[0] > 0 else -1, sh.lashes, p.line_px, d.unit_px)
            self._styled[key] = (rgba, dx, dy)
        return self._styled[key]

    def part_unit_affine(self, d: Drawing, p: Part, tgt: dict):
        """2×3 key-face-units → target-face-units: the drawn part's anchor → target anchor, size → target size."""
        w0, h0 = self._psize(d, p)
        w1, h1 = tgt["size"]
        A = np.array([[w1 / max(1e-4, w0), 0.0], [0.0, h1 / max(1e-4, h0)]])
        t = np.asarray(tgt["anchor"], float) - A @ np.asarray(p.anchor, float)
        return np.hstack([A, t[:, None]])

    def layer(self, W: int, H: int, expr: str, yaw: float, style, M_track=None, face_mask=None):
        """→ (premultiplied colour float32 H×W×3, alpha float32 H×W)."""
        col = np.zeros((H, W, 3), np.float32)
        alp = np.zeros((H, W), np.float32)
        U = style.unit_to_frame()
        fm = None
        if face_mask is not None:
            fm = cv2.GaussianBlur(face_mask.astype(np.float32), (5, 5), 0)
        tg = self.targets(expr, yaw)
        for nm in PARTS_ORDER:
            if nm not in tg or tg[nm]["vis"] < 0.01:
                continue
            for d, p, wk in tg[nm]["draws"]:
                Dinv = invert(d.unit_to_decal())
                Au = self.part_unit_affine(d, p, tg[nm])
                vis = tg[nm]["vis"]
                rgba, dx, dy = self._styled_part(d, p)
                T = np.array([[1.0, 0.0, p.x0 - dx], [0.0, 1.0, p.y0 - dy]])
                M = compose(U, Au, Dinv, T)
                if M_track is not None:
                    M = compose(M_track, M)
                a = rgba[..., 3:4].astype(np.float32) / 255.0
                pm = np.concatenate([rgba[..., :3].astype(np.float32) * a, a * 255.0], 2)
                w = cv2.warpAffine(pm, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                pa = w[..., 3] / 255.0 * vis * wk
                pc = w[..., :3] * (vis * wk)
                if fm is not None and not (nm.startswith("brow") and self.shape.brow_layer == "over"):
                    pa, pc = pa * fm, pc * fm[..., None]
                col = pc + col * (1 - pa[..., None])
                alp = pa + alp * (1 - pa)
        return col, alp

    def render(self, frame, expr, yaw, style, M_track=None, face_mask=None, keep=None):
        H, W = frame.shape[:2]
        col, a = self.layer(W, H, expr, yaw, style, M_track, face_mask)
        if keep is not None:
            a, col = a * keep, col * keep[..., None]
        out = frame.astype(np.float32) * (1 - a[..., None]) + col
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def over_layers(frame, layers):
    """composite [(col_pm, alpha, weight)] — used for time dissolves between expressions."""
    out = frame.astype(np.float32)
    col = np.zeros_like(out)
    a = np.zeros(frame.shape[:2], np.float32)
    for c, al, w in layers:
        col += c * w
        a += al * w
    return np.clip(out * (1 - a[..., None]) + col + 0.5, 0, 255).astype(np.uint8)


# ================================================================================================ video
def parse_script(spec: str, fps: float):
    """'0:smug,1.5:pleading,3:sulking' → [(frame, expr)] sorted."""
    out = []
    for tok in spec.split(","):
        t, e = tok.split(":")
        out.append((int(round(float(t) * fps)), e.strip()))
    return sorted(out)


def render_video(video: str, lib: str, out: str, script: str, femininity: float = 0.0, preset: str = "as_drawn",
                 shape_json: str | None = None, yaw: float | None = None, swap_frames: int = 2, debug: bool = False):
    """track the blank face (v32 tracker + occlusion), place the expression rig, stream to mp4."""
    from .facecomp import FaceRigCompositor, VideoWriter, chin_offset, read_frames
    frames, fps = read_frames(video)
    comp = FaceRigCompositor("boy", frames, fps=fps, verbose=False)        # tracking + occlusion only
    st = comp.trk.style
    base = FaceShape.load(shape_json) if shape_json else PRESETS[preset]
    rig = DecalRig(lib, shape=character_shape(base, femininity))
    if yaw is None:
        yaw = rig.head.yaw_from_chin(chin_offset(comp.ref_face))
    keys = parse_script(script, fps)
    H, W = frames[0].shape[:2]
    vw = VideoWriter(out, W, H, fps)
    for i, f in enumerate(frames):
        cur = [e for k, e in keys if k <= i]
        cur = cur[-1] if cur else keys[0][1]
        prev_k = [(k, e) for k, e in keys if k <= i]
        prior, occ, _ = comp.occlusion_mask(i)
        sw = max(2, int(round(st.line_px)))
        inside = cv2.erode(prior.astype(np.uint8), np.ones((sw, sw), np.uint8)) > 0
        keep = np.clip(1.0 - cv2.dilate(occ.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32), 0, 1)
        M = comp.poses[i].M
        layers = []
        if len(prev_k) >= 2 and i - prev_k[-1][0] < swap_frames:          # short dissolve between expressions
            t = (i - prev_k[-1][0] + 1) / (swap_frames + 1)
            for e, w in ((prev_k[-2][1], 1 - t), (cur, t)):
                c, a = rig.layer(W, H, e, yaw, st, M, inside)
                layers.append((c * keep[..., None], a * keep, w))
        else:
            c, a = rig.layer(W, H, cur, yaw, st, M, inside)
            layers.append((c * keep[..., None], a * keep, 1.0))
        img = over_layers(f, layers)
        if debug:
            cv2.putText(img, f"{i} {cur} yaw={yaw:+.0f} fem={femininity:.1f}", (10, 30), 0, 0.8, (0, 0, 255), 2)
        vw.write(img)
        frames[i] = None
    vw.close()
    return out


def main():
    ap = argparse.ArgumentParser(description="expression rig on a blank-face video")
    ap.add_argument("--video", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--script", default="0:smug,1.7:pleading,3.4:sulking")
    ap.add_argument("--femininity", type=float, default=0.0)
    ap.add_argument("--preset", default="as_drawn", choices=sorted(PRESETS))
    ap.add_argument("--shape", default=None, help="FaceShape json (per-character art direction)")
    ap.add_argument("--yaw", type=float, default=None)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    print("[decalrig]", render_video(a.video, a.lib, a.out, a.script, a.femininity, a.preset, a.shape, a.yaw,
                                     debug=a.debug))


if __name__ == "__main__":
    main()
