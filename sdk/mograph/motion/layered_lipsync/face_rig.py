"""
Parametric 3D-sphere FACE RIG — eyes, brows and mouth are pure vector math, never pixels cut from references.

Model (exact for round-headed characters):
  * features are curves on the FACE PLANE (unit disk, x right, y down, radius 1 = head radius)
  * each point is LIFTED onto the head sphere  z = sqrt(1 − x² − y²)
  * rotated by R = Rz(roll) · Rx(pitch) · Ry(yaw)   (yaw > 0 = face turns to the viewer's right, pitch > 0 = looks
    down, roll = eye-line tilt)  and orthographically projected  →  screen = centre + radius · (X, Y)
  * points with z < 0 are behind the head: strokes are cut there, fills are pushed onto the silhouette
  → the far eye squashes, the mouth slides toward the cheek and wraps around the limb in 3/4 and profile
    automatically, with no per-view drawings.

Calibration (fit_rig): head circle from the detected head hull; yaw / pitch / roll + eye spacing / height solved by
least squares from the detected eye centres and the far-eye squash (1 eye → profile solve); eye radius / aspect,
pupil size and gaze measured from the art; mouth centre solved from the detected mouth. Colours: line ink + face
colour sampled from the art; the style palette (mouth interior, tongue, teeth) is a parameter.

Drawing: feature ops → projected → cairo (anti-aliased, round caps) in a head-sized ROI → composited through a
clip mask = head interior minus anything that is in front of the face (raised hands / fingers).

API: fit_rig(img, anatomy) → RigParams · prepare_face(img, anatomy) → (clean_img, clip_mask) ·
     face_ops(P, viseme_geom, eyes=, mouth=, blink=) → ops · draw_face(frame, P, ops, clip) ·
     EYE_EXPRESSIONS / MOUTH_EXPRESSIONS presets · project(P, pts)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np

from .anatomy import fill_mask, kernel

try:                                          # pycairo is already a motion-kit dependency
    import cairo
except Exception:  # noqa: BLE001
    cairo = None

# --------------------------------------------------------------------------------------------- expression presets
# eye params: lu upper-lid closure 0..1 · ll lower lid · inner_drop (+ angry, − sad) · er_mult · pupil_r · gaze ·
#             style normal|closed|happy|x|spiral|star|dot · brow {raise, tilt (+ inner down), bow, thick} · shine
EYE_EXPRESSIONS: dict[str, dict[str, Any]] = {
    "neutral": {},
    "wide": {"er_mult": 1.2, "pupil_r": 0.24},
    "shocked": {"er_mult": 1.18, "pupil_r": 0.12},
    "sad": {"lu": 0.2, "inner_drop": -0.9, "brow": {"raise": 0.08, "tilt": -0.6, "bow": 0.2}},
    "worried": {"brow": {"raise": 0.13, "tilt": -0.5, "bow": 0.3}},
    "angry": {"lu": 0.3, "inner_drop": 1.0, "brow": {"raise": -0.03, "tilt": 0.75, "bow": -0.15, "thick": 1.6}},
    "suspicious": {"lu": 0.45, "ll": 0.12, "brow": {"raise": 0.0, "tilt": 0.2, "bow": 0.0}},
    "bored": {"lu": 0.5},
    "sleepy": {"lu": 0.66, "gaze": (0.0, 0.35)},
    "squint": {"lu": 0.38, "ll": 0.3},
    "closed": {"style": "closed"},
    "happy": {"style": "happy"},
    "pain": {"style": "x"},
    "dizzy": {"style": "spiral"},
    "starry": {"style": "star"},
    "dot": {"style": "dot"},
    "teary": {"pupil_r": 0.6, "shine": True, "brow": {"raise": 0.1, "tilt": -0.5, "bow": 0.2}},
}
# mouth params: smile (+ up / − frown) · open_add (rest opening) · teeth · teeth_lines · lower_teeth · w_mult ·
#               line wavy|zigzag|cat (closed shape style) · tongue_out
MOUTH_EXPRESSIONS: dict[str, dict[str, Any]] = {
    "neutral": {},
    "smile": {"smile": 0.55},
    "big_smile": {"smile": 0.8, "open_add": 0.32, "teeth": 1.0},
    "grin": {"smile": 0.75, "open_add": 0.3, "teeth": 1.0, "teeth_lines": True, "lower_teeth": True, "w_mult": 1.2},
    "frown": {"smile": -0.6},
    "grimace": {"smile": -0.1, "open_add": 0.26, "teeth": 1.0, "teeth_lines": True, "lower_teeth": True,
                "w_mult": 1.2},
    "nervous": {"line": "wavy"},
    "zigzag": {"line": "zigzag"},
    "cat": {"line": "cat"},
    "shout": {"open_add": 0.7, "w_mult": 1.05, "teeth": 0.6},
    "tongue_out": {"smile": 0.45, "open_add": 0.12, "tongue_out": True},
}

DEFAULT_PALETTE = {
    "sclera": (250, 251, 251), "mouth_in": (22, 28, 58), "tongue": (96, 152, 214), "teeth": (247, 248, 248),
}


@dataclass
class RigParams:
    """everything needed to draw a face; lengths in face units (head radius = 1) unless noted."""
    cx: float
    cy: float
    R: float                                   # head radius (px)
    yaw: float = 0.0                           # radians
    pitch: float = 0.0
    roll: float = 0.0
    ex: float = 0.34                           # half eye spacing
    ey: float = -0.06                          # eye height (− = above centre)
    er: float = 0.17                           # eye horizontal radius
    eye_aspect: float = 1.0                    # eye height / width
    pupil_r: float = 0.36                      # fraction of er
    gaze: tuple[float, float] = (0.0, 0.0)     # fraction of er
    mx: float = 0.0
    my: float = 0.42
    mw: float = 0.42                           # mouth width at viseme w = 1
    stroke: float = 4.0                        # px (constant on screen, like an ink pen)
    ink: tuple[int, int, int] = (12, 10, 10)   # BGR
    face: tuple[int, int, int] = (252, 252, 252)
    palette: dict = field(default_factory=lambda: dict(DEFAULT_PALETTE))
    base_eyes: str = "neutral"
    fit_err: float = 0.0
    view: str = "front"

    def summary(self) -> dict[str, Any]:
        return {"yaw_deg": round(math.degrees(self.yaw), 1), "pitch_deg": round(math.degrees(self.pitch), 1),
                "roll_deg": round(math.degrees(self.roll), 1), "R": round(self.R, 1), "ex": round(self.ex, 3),
                "ey": round(self.ey, 3), "er": round(self.er, 3), "eye_aspect": round(self.eye_aspect, 2),
                "mouth": [round(self.mx, 3), round(self.my, 3)], "mw": round(self.mw, 3),
                "fit_err": round(self.fit_err, 4), "view": self.view, "base_eyes": self.base_eyes}


# --------------------------------------------------------------------------------------------- projection
def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy_, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Ry = np.array([[cy_, 0, sy], [0, 1, 0], [-sy, 0, cy_]])
    Rx = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def lift(pts: np.ndarray) -> np.ndarray:
    """face-plane points (N, 2) → unit-sphere points (N, 3) (points outside the disk are put on its rim)."""
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    r2 = (p ** 2).sum(1)
    out = np.empty((len(p), 3))
    s = np.where(r2 > 1.0, 1.0 / np.sqrt(np.maximum(r2, 1e-12)), 1.0)
    out[:, :2] = p * s[:, None]
    out[:, 2] = np.sqrt(np.clip(1.0 - (out[:, :2] ** 2).sum(1), 0.0, 1.0))
    return out


def project(P: RigParams, pts: np.ndarray, clamp: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """→ (screen xy (N, 2) px, depth z (N,)); clamp=True pushes hidden points onto the silhouette."""
    Q = lift(pts) @ rotation(P.yaw, P.pitch, P.roll).T
    xy = Q[:, :2].copy()
    z = Q[:, 2]
    if clamp:
        h = z < 0
        if h.any():
            n = np.linalg.norm(xy[h], axis=1, keepdims=True)
            xy[h] = xy[h] / np.maximum(n, 1e-9)
    return np.c_[P.cx + P.R * xy[:, 0], P.cy + P.R * xy[:, 1]], z


# --------------------------------------------------------------------------------------------- feature ops
def _ellipse(cx, cy, rx, ry, n=72):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[cx + rx * np.cos(t), cy + ry * np.sin(t)]


def _eye_ops(P: RigParams, s: int, e: dict, blink: float) -> list:
    st = e.get("style", "normal")
    er = P.er * e.get("er_mult", 1.0)
    asp = P.eye_aspect
    cx, cy = s * P.ex, P.ey
    ry = er * asp
    lu = float(np.clip(max(e.get("lu", 0.0), blink), 0, 1))
    ll = float(e.get("ll", 0.0))
    if st == "normal" and lu >= 0.95:
        st = "closed"
    ops: list = []
    t = np.linspace(-1, 1, 25)
    if st == "normal":
        ell = _ellipse(cx, cy, er, ry)
        ops.append(("fill", ell, "sclera"))
        H, top = 2 * ry, cy - ry
        xs = np.linspace(cx - 1.4 * er, cx + 1.4 * er, 29)
        inner = np.clip((xs - cx) * (-s) / er, -1.4, 1.4)
        drop = e.get("inner_drop", 0.0) * min(1.0, lu * 3.0)
        ylid = top + lu * H + drop * 0.32 * H * inner
        ybot = top + H - ll * H
        below = np.r_[np.c_[xs, ylid], [[xs[-1], top + 3 * H], [xs[0], top + 3 * H]]]
        above = np.r_[np.c_[xs, ylid], [[xs[-1], top - 2 * H], [xs[0], top - 2 * H]]]
        g = e.get("gaze", P.gaze)
        pr = er * float(np.clip(e.get("pupil_r", P.pupil_r), 0.05, 0.9))
        gx, gy = float(np.clip(g[0], -1, 1)), float(np.clip(g[1], -1, 1))
        lim = max(0.0, 1.0 - pr / er)
        gn = math.hypot(gx, gy)
        if gn > lim and gn > 0:
            gx, gy = gx * lim / gn, gy * lim / gn
        pcx, pcy = cx + gx * er, cy + gy * ry
        ops += [("clip", ell)]
        if lu > 0.01:
            ops.append(("clip", below))
        ops.append(("fill", _ellipse(pcx, pcy, pr, pr, 40), "ink"))
        if e.get("shine"):
            ops.append(("fill", _ellipse(pcx - 0.35 * pr, pcy - 0.38 * pr, 0.3 * pr, 0.3 * pr, 24), "sclera"))
            ops.append(("fill", _ellipse(pcx + 0.3 * pr, pcy + 0.35 * pr, 0.14 * pr, 0.14 * pr, 16), "sclera"))
        ops.append(("unclip",))
        inside = lambda x, y: ((x - cx) / er) ** 2 + ((y - cy) / ry) ** 2 <= 1.0  # noqa: E731
        if lu > 0.01:
            ops += [("clip", ell), ("fill", above, "face"), ("unclip",)]
            m = inside(xs, ylid)
            if m.sum() >= 2:
                ops.append(("stroke", np.c_[xs[m], ylid[m]], "ink", 1.0, False))
        if ll > 0.01:
            low = np.r_[np.c_[xs, np.full_like(xs, ybot)], [[xs[-1], ybot + 2 * H], [xs[0], ybot + 2 * H]]]
            ops += [("clip", ell), ("fill", low, "face"), ("unclip",)]
            m = inside(xs, np.full_like(xs, ybot))
            if m.sum() >= 2:
                ops.append(("stroke", np.c_[xs[m], np.full(m.sum(), ybot)], "ink", 1.0, False))
        ops.append(("stroke", ell, "ink", 1.0, True))
    elif st == "closed":
        ops.append(("stroke", np.c_[cx + er * t, cy + 0.2 * ry + 0.22 * er * (1 - t * t)], "ink", 1.15, False))
    elif st == "happy":
        ops.append(("stroke", np.c_[cx + er * t, cy + 0.15 * ry - 0.45 * er * (1 - t * t)], "ink", 1.25, False))
    elif st == "x":
        o = s * 0.85 * er
        ops.append(("stroke", np.array([[cx + o, cy - 0.7 * er], [cx - o, cy], [cx + o, cy + 0.7 * er]]),
                    "ink", 1.35, False))
    elif st == "dot":
        ops.append(("fill", _ellipse(cx, cy, 0.3 * er, 0.3 * er, 32), "ink"))
    elif st == "spiral":
        th = np.linspace(0, 6 * np.pi, 160)
        r = 0.9 * er * th / th[-1]
        ops.append(("stroke", np.c_[cx + r * np.cos(th + s), cy + r * np.sin(th + s) * asp], "ink", 0.75, False))
        ops.append(("stroke", _ellipse(cx, cy, er, ry), "ink", 1.0, True))
    elif st == "star":
        ops.append(("fill", _ellipse(cx, cy, er, ry), "ink"))
        a = np.linspace(0, 2 * np.pi, 9)[:-1] - np.pi / 2
        rr = np.where(np.arange(8) % 2 == 0, 0.78 * er, 0.2 * er)
        ops.append(("fill", np.c_[cx + rr * np.cos(a), cy + rr * np.sin(a)], "sclera"))
        ops.append(("fill", _ellipse(cx + 0.45 * er, cy + 0.45 * er, 0.1 * er, 0.1 * er, 16), "sclera"))
    b = e.get("brow")
    if b:
        u = np.linspace(-1, 1, 21)
        bx = cx + 1.05 * er * u
        inner = u * (-s)
        by = (cy - ry - 0.08 - b.get("raise", 0.0)) - b.get("bow", 0.0) * 0.1 * (1 - u * u) \
            + b.get("tilt", 0.0) * 0.32 * er * inner
        ops.append(("stroke", np.c_[bx, by], "ink", 1.5 * b.get("thick", 1.0), False))
    return ops


def _mouth_ops(P: RigParams, g: dict, m: dict, talking: bool) -> list:
    w = float(g.get("w", 0.8)) * m.get("w_mult", 1.0)
    a = 0.5 * P.mw * w
    h = 0.5 * P.mw * float(g.get("h", 0.0))
    if not talking:
        h += 0.5 * P.mw * m.get("open_add", 0.0)
    c = -m.get("smile", 0.0) * 0.42 * a
    t = np.linspace(-1, 1, 49)
    x = P.mx + a * t
    base = P.my + c * (t * t - 0.35)
    ops: list = []
    if h < 0.012:
        line = m.get("line") if not talking else None
        press = float(g.get("press", 0.0))
        if line == "wavy":
            y = base + 0.045 * P.mw * np.sin(t * np.pi * 3.5)
        elif line == "zigzag":
            y = base + 0.06 * P.mw * (2 * np.abs(((t * 3.5) % 2) - 1) - 1)
        elif line == "cat":
            y = base + 0.08 * P.mw * np.abs(np.sin(t * np.pi)) - 0.04 * P.mw
        else:
            y = base + 0.03 * P.mw * (1 + press) * (1 - t * t)
        ops.append(("stroke", np.c_[x, y], "ink", 1.0 + 0.45 * press, False))
        if press > 0.5:
            for sy in (-1, 1):
                yy = P.my + sy * 0.055 * P.mw
                ops.append(("stroke", np.array([[P.mx - 0.18 * a, yy], [P.mx + 0.18 * a, yy]]), "ink", 0.5, False))
        return ops
    e = 0.5 + 0.35 * (1.0 - float(g.get("round", 0.0)))
    env = (1 - t * t) ** e
    top = base - 0.42 * h * env
    bot = base + 0.58 * h * env
    poly = np.r_[np.c_[x, top], np.c_[x[::-1], bot[::-1]]]
    ops += [("fill", poly, "mouth_in"), ("clip", poly)]
    teeth = max(float(g.get("teeth", 0.0)), float(m.get("teeth", 0.0)))
    if teeth > 0.05:
        tb = h * (0.2 + 0.26 * teeth)
        band = np.r_[np.c_[x, top - 0.2 * h], np.c_[x[::-1], (top + tb)[::-1]]]
        ops.append(("fill", band, "teeth"))
        if g.get("bite", 0) > 0.5:
            ops.append(("fill", np.r_[np.c_[x, top + tb], np.c_[x[::-1], (bot + 0.2 * h)[::-1]]], "face"))
        if m.get("teeth_lines"):
            for tk in np.linspace(-0.72, 0.72, 6):
                y0 = P.my + c * (tk * tk - 0.35) - 0.42 * h * (1 - tk * tk) ** e
                ops.append(("stroke", np.array([[P.mx + a * tk, y0], [P.mx + a * tk, y0 + tb]]), "ink", 0.55, False))
        if m.get("lower_teeth"):
            lb = np.r_[np.c_[x, bot - tb], np.c_[x[::-1], (bot + 0.2 * h)[::-1]]]
            ops.append(("fill", lb, "teeth"))
            if m.get("teeth_lines"):
                for tk in np.linspace(-0.6, 0.6, 5):
                    y1 = P.my + c * (tk * tk - 0.35) + 0.58 * h * (1 - tk * tk) ** e
                    ops.append(("stroke", np.array([[P.mx + a * tk, y1 - tb], [P.mx + a * tk, y1]]), "ink", 0.55,
                                False))
    tg = float(g.get("tongue", 0.0))
    if tg > 0.05 and not m.get("tongue_out"):
        tcy = P.my + c * -0.35 + 0.58 * h * 0.9 - float(g.get("tongue_up", 0.0)) * 0.5 * h
        ops.append(("fill", _ellipse(P.mx, tcy, 0.5 * a, 0.42 * h + 0.01, 40), "tongue"))
    ops.append(("unclip",))
    if m.get("tongue_out"):
        tt = np.linspace(0, np.pi, 25)
        tw, tl = 0.32 * a + 0.02, 0.28 * P.mw + 0.6 * h
        y0 = P.my + c * -0.35 + 0.3 * h
        tongue = np.r_[[[P.mx - tw, y0]], np.c_[P.mx - tw * np.cos(tt), y0 + tl + tw * 0.9 * np.sin(tt)],
                       [[P.mx + tw, y0]]]
        ops += [("fill", tongue, "tongue"), ("stroke", tongue[1:-1], "ink", 0.9, False),
                ("stroke", np.array([[P.mx, y0 + 0.2 * tl], [P.mx, y0 + 0.75 * tl]]), "ink", 0.5, False)]
    ops.append(("stroke", poly, "ink", 1.0, True))
    return ops


def face_ops(P: RigParams, geom: dict | None = None, eyes: str | dict | None = None,
             mouth: str | dict | None = None, blink: float = 0.0) -> list:
    """all drawing ops for one frame. geom = viseme geometry (lipsync.MOUTHS / shape_at), None = rest."""
    from .. import lipsync as LS
    g = dict(LS.MOUTHS["rest"]) if geom is None else geom
    e = EYE_EXPRESSIONS.get(eyes or P.base_eyes, {}) if not isinstance(eyes, dict) else eyes
    m = MOUTH_EXPRESSIONS.get(mouth or "neutral", {}) if not isinstance(mouth, dict) else mouth
    talking = g.get("v", "rest") not in ("rest",) and float(g.get("h", 0.0)) > 0.0 or g.get("v") == "MBP"
    ops: list = []
    order = (-1, 1) if P.yaw >= 0 else (1, -1)          # far eye first (painter's order)
    for s in order:
        ops += _eye_ops(P, s, e, blink)
    ops += _mouth_ops(P, g, m, talking)
    return ops


# --------------------------------------------------------------------------------------------- drawing
def _rgb(c):
    return c[2] / 255.0, c[1] / 255.0, c[0] / 255.0


def render_ops(P: RigParams, ops: list, box: tuple[int, int, int, int]) -> np.ndarray:
    """rasterise ops into a BGRA premultiplied float32 array of size box (x0, y0, w, h)."""
    if cairo is None:
        raise RuntimeError("pycairo is required for the face rig (pip install pycairo)")
    x0, y0, w, h = box
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(w), int(h))
    ctx = cairo.Context(surf)
    ctx.translate(-x0, -y0)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    colors = {"ink": P.ink, "face": P.face, **P.palette}
    clips = 0

    def path(pts, close=True):
        ctx.new_path()
        ctx.move_to(*pts[0])
        for q in pts[1:]:
            ctx.line_to(*q)
        if close:
            ctx.close_path()

    for op in ops:
        k = op[0]
        if k == "fill":
            pts, _ = project(P, op[1], clamp=True)
            path(pts)
            ctx.set_source_rgb(*_rgb(colors[op[2]]))
            ctx.fill()
        elif k == "stroke":
            pts3 = op[1]
            if op[4]:
                pts3 = np.r_[pts3, pts3[:1]]
            pts, z = project(P, pts3, clamp=False)
            vis = z > -0.01
            ctx.set_source_rgb(*_rgb(colors[op[2]]))
            ctx.set_line_width(max(0.8, P.stroke * op[3]))
            i, n = 0, len(pts)
            while i < n:
                if not vis[i]:
                    i += 1
                    continue
                j = i
                while j < n and vis[j]:
                    j += 1
                if j - i >= 2:
                    path(pts[i:j], close=False)
                    ctx.stroke()
                i = j
        elif k == "clip":
            ctx.save()
            clips += 1
            pts, _ = project(P, op[1], clamp=True)
            path(pts)
            ctx.clip()
        elif k == "unclip":
            for _ in range(clips):
                ctx.restore()
            clips = 0
    surf.flush()
    stride = surf.get_stride()
    arr = np.frombuffer(surf.get_data(), np.uint8).reshape(int(h), stride)[:, :int(w) * 4].reshape(int(h), int(w), 4)
    return arr.astype(np.float32) / 255.0


def face_box(P: RigParams, shape: tuple[int, int], pad: float = 0.15) -> tuple[int, int, int, int]:
    H, W = shape[:2]
    r = P.R * (1 + pad) + 2 * P.stroke
    x0, y0 = max(0, int(P.cx - r)), max(0, int(P.cy - r))
    x1, y1 = min(W, int(P.cx + r) + 1), min(H, int(P.cy + r) + 1)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def draw_face(frame: np.ndarray, P: RigParams, ops: list, clip: np.ndarray | None = None) -> np.ndarray:
    """composite the rig onto frame IN PLACE through `clip` (float 0..1 full-frame mask, None = no clip)."""
    x0, y0, w, h = face_box(P, frame.shape)
    bgra = render_ops(P, ops, (x0, y0, w, h))
    a = bgra[:, :, 3:4]
    col = bgra[:, :, :3]
    if clip is not None:
        c = clip[y0:y0 + h, x0:x0 + w, None]
        a = a * c
        col = col * c
    roi = frame[y0:y0 + h, x0:x0 + w].astype(np.float32) / 255.0
    frame[y0:y0 + h, x0:x0 + w] = np.clip((col + roi * (1 - a)) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return frame


# --------------------------------------------------------------------------------------------- calibration
def _measure_eyes(img: np.ndarray, info: dict) -> list[dict]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = []
    for ef in info.get("eye_fills", []):
        x, y, w, h = ef["box"]
        m = ef["mask"].astype(np.uint8)
        ys, xs = np.nonzero(m)
        if len(xs) < 5:
            continue
        cx, cy = x + xs.mean(), y + ys.mean()
        d = {"c": (float(cx), float(cy)), "w": float(w), "h": float(h), "kind": ef["kind"]}
        if ef["kind"] == "sclera":
            me = cv2.erode(m, kernel(max(3, info["stroke_px"] * 1.2)))
            g = gray[y:y + h, x:x + w]
            pm = (me > 0) & (g < 110)
            if pm.sum() >= 3:
                py, px = np.nonzero(pm)
                d["pupil"] = ((x + px.mean() - cx) / (w / 2.0), (y + py.mean() - cy) / (h / 2.0))
                d["pupil_r"] = float(math.sqrt(pm.sum() / math.pi) / (h / 2.0))
        out.append(d)
    out.sort(key=lambda d: d["c"][0])
    return out


def _sample_ink(img: np.ndarray, info: dict) -> tuple[int, int, int]:
    hm = info["head_mask_full"] > 0
    ring = cv2.dilate(hm.astype(np.uint8), kernel(2 * info["stroke_px"] + 3)) > 0
    ring &= ~hm
    px = img[ring]
    g = px.mean(1) if len(px) else np.array([])
    dark = px[g < 90] if len(px) else px
    if len(dark) < 10:
        return (12, 10, 10)
    return tuple(int(v) for v in np.median(dark, axis=0))


def fit_rig(img: np.ndarray, info: dict, palette: dict | None = None, prev: RigParams | None = None) -> RigParams:
    """calibrate a RigParams to one drawn pose (head circle, yaw / pitch / roll, eyes, mouth, colours)."""
    from scipy.optimize import least_squares

    hm = (info["head_mask_full"] > 0).astype(np.uint8)
    cs, _ = cv2.findContours(hm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    if len(c) >= 5:
        (hcx, hcy), (A, B), _ = cv2.fitEllipse(c)
        R = 0.25 * (A + B)
    else:
        (hcx, hcy), R = cv2.minEnclosingCircle(c)
    P = RigParams(cx=float(hcx), cy=float(hcy), R=float(R), stroke=float(info["stroke_px"]),
                  ink=_sample_ink(img, info), face=tuple(int(v) for v in info["face_color"]),
                  palette=dict(DEFAULT_PALETTE, **(palette or {})))
    eyes = _measure_eyes(img, info)
    sclera = [e for e in eyes if e["kind"] == "sclera"]
    if info.get("eye_kind") == "ink":
        P.base_eyes = "closed"
    if sclera:
        fills = [img[int(e["c"][1]), int(e["c"][0])] for e in sclera]
        P.palette["sclera"] = tuple(int(v) for v in np.max(np.array(fills), axis=0))

    def pt(p, x, y):
        Q = lift(np.array([[x, y]])) @ rotation(p[0], p[1], p[2]).T
        return np.array([hcx + R * Q[0, 0], hcy + R * Q[0, 1]]), Q[0, 2]

    def width(p, x, y, r):
        a, _ = pt(p, x - r, y)
        b, _ = pt(p, x + r, y)
        return float(np.linalg.norm(b - a))

    best = None
    if len(eyes) >= 2:
        L, Rr = eyes[0], eyes[1]
        er0 = 0.5 * (0.5 * (L["h"] + Rr["h"])) / R
        ow = (L["w"], Rr["w"])

        def res(p):
            yaw, pitch, roll, ex, ey = p
            pl, zl = pt(p, -ex, ey)
            pr, zr = pt(p, ex, ey)
            r = list((pl - np.array(L["c"])) / R) + list((pr - np.array(Rr["c"])) / R)
            wl, wr = width(p, -ex, ey, er0), width(p, ex, ey, er0)
            r.append(0.6 * (math.log(max(wl, 1e-3) / max(wr, 1e-3)) - math.log(ow[0] / ow[1])))
            r += [0.35 * pitch, 0.8 * (ex - 0.34), 0.35 * (ey + 0.06)]
            r += [4.0 * max(0.0, 0.08 - zl), 4.0 * max(0.0, 0.08 - zr)]
            return r

        roll0 = math.atan2(Rr["c"][1] - L["c"][1], Rr["c"][0] - L["c"][0])
        starts = [prev.yaw] if prev is not None else []
        starts += [0.0, -0.45, 0.45, -0.9, 0.9]
        for y0 in starts:
            sol = least_squares(res, [y0, 0.0, roll0, 0.34, -0.06],
                                bounds=([-1.45, -0.7, -1.2, 0.12, -0.65], [1.45, 0.7, 1.2, 0.62, 0.45]))
            if best is None or sol.cost < best.cost:
                best = sol
        P.yaw, P.pitch, P.roll, P.ex, P.ey = (float(v) for v in best.x)
        big = L if L["w"] * L["h"] >= Rr["w"] * Rr["h"] else Rr
        sx = -P.ex if big is L else P.ex
        pw = width(best.x, sx, P.ey, er0)
        P.er = float(np.clip(er0 * big["w"] / max(pw, 1e-3), 0.06, 0.4)) + 0.5 * P.stroke / R
        P.eye_aspect = float(np.clip(pw / max(big["w"], 1e-3), 0.6, 1.8))
        P.fit_err = float(math.sqrt(2 * best.cost / max(1, len(best.fun))))
    elif len(eyes) == 1:
        E = eyes[0]
        side = -1.0 if info["view_mode"] == "profile_left" or E["c"][0] < hcx else 1.0
        x_eye = -side * 0.34                       # facing left → the visible eye is the face's right eye
        er0 = 0.5 * E["h"] / R

        def res1(p):
            yaw, ey = p
            q, z = pt([yaw, 0.0, 0.0], x_eye, ey)
            return list((q - np.array(E["c"])) / R) + [0.3 * (abs(yaw) - 1.15), 0.35 * (ey + 0.06),
                                                         4.0 * max(0.0, 0.1 - z)]
        lo, hi = ((-1.5, -0.35) if side < 0 else (0.35, 1.5))
        best = least_squares(res1, [side * 1.1, -0.06], bounds=([lo, -0.65], [hi, 0.45]))
        P.yaw, P.ey = float(best.x[0]), float(best.x[1])
        P.er = float(np.clip(er0, 0.06, 0.4)) + 0.5 * P.stroke / R
        pw = width([P.yaw, 0, 0], x_eye, P.ey, P.er)
        P.eye_aspect = float(np.clip(pw / max(E["w"], 1e-3), 0.6, 1.8)) if E["w"] > 2 else 1.0
        P.fit_err = float(math.sqrt(2 * best.cost / max(1, len(best.fun))))
    else:
        P.yaw = 0.0
    pupils = [e for e in sclera if "pupil" in e]
    if pupils:
        P.gaze = (float(np.clip(np.mean([e["pupil"][0] for e in pupils]), -0.8, 0.8)),
                  float(np.clip(np.mean([e["pupil"][1] for e in pupils]), -0.8, 0.8)))
        P.pupil_r = float(np.clip(np.mean([e["pupil_r"] for e in pupils]), 0.12, 0.7))

    # mouth: solve its face-plane position from the detected mouth centre
    P.my = P.ey + 0.5
    if info.get("mouth_found"):
        tgt = np.array(info["mouth_center"], float)

        def resm(p):
            q, z = pt([P.yaw, P.pitch, P.roll], p[0], p[1])
            return list((q - tgt) / R) + [0.25 * p[0], 3.0 * max(0.0, 0.05 - z)]
        sm = least_squares(resm, [0.0, P.ey + 0.5], bounds=([-0.6, P.ey + 0.12], [0.6, 0.85]))
        P.mx, P.my = float(sm.x[0]), float(sm.x[1])
        mx_, my_, mw_, mh_ = info["mouth_box"]
        unit = width([P.yaw, P.pitch, P.roll], P.mx, P.my, 0.5) / R    # projected length of 1 face unit
        if abs(P.yaw) < 0.9 and unit > 0.2:
            w_face = mw_ / R / unit
            closed = mh_ < 0.3 * mw_
            P.mw = float(np.clip(w_face / (0.8 if closed else 0.95), 0.26, 0.6))
    view = "front" if abs(P.yaw) < math.radians(12) else ("three_quarter_right" if P.yaw > 0 else
                                                           "three_quarter_left")
    if abs(P.yaw) > math.radians(58):
        view = "profile_right" if P.yaw > 0 else "profile_left"
    P.view = view
    return P


# --------------------------------------------------------------------------------------------- clean plate
def prepare_face(img: np.ndarray, info: dict) -> tuple[np.ndarray, np.ndarray]:
    """erase every drawn facial feature (eyes, pupils, mouth, teeth, brows, blush) → (clean image, draw clip).
    Anything that enters the head from OUTSIDE (raised hands / fingers) is kept and protected from the rig."""
    H, W = img.shape[:2]
    stroke = float(info["stroke_px"])
    head = (info["head_mask_full"] > 0).astype(np.uint8)
    fm = fill_mask(img)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(fm, connectivity=4)
    core = cv2.erode(head, kernel(max(5, 0.25 * math.sqrt(head.sum() / math.pi))))
    ids, cnt = np.unique(lbl[(core > 0) & (lbl > 0)], return_counts=True)
    face_id = int(ids[np.argmax(cnt)]) if len(ids) else -1
    x0, y0, w, h = cv2.boundingRect(head)
    pad = int(3 * stroke + 4)
    xa, ya, xb, yb = max(0, x0 - pad), max(0, y0 - pad), min(W, x0 + w + pad), min(H, y0 + h + pad)
    keep = np.zeros_like(head)
    for i in np.unique(lbl[ya:yb, xa:xb]):
        if i in (0, face_id):
            continue
        comp = lbl == i
        inside = head[comp].mean()
        if inside < 0.75:                               # mostly outside the head → a hand / sleeve / prop
            keep[comp] = 1
    keep_band = cv2.dilate(keep, kernel(2 * stroke + 3)) > 0
    inner = cv2.erode(head, kernel(max(3, 0.6 * stroke))) > 0
    erase = inner & (lbl != face_id) & ~keep_band
    a = cv2.GaussianBlur(erase.astype(np.float32), (0, 0), 0.8)[:, :, None]
    face = np.empty_like(img)
    face[:] = info["face_color"]
    clean = (face * a + img * (1 - a) + 0.5).astype(np.uint8)
    clip = (cv2.erode(head, kernel(max(3, 0.4 * stroke))) > 0) & ~keep_band
    clip = cv2.GaussianBlur(clip.astype(np.float32), (0, 0), 0.7)
    return clean, clip


def smooth_params(prev: RigParams | None, cur: RigParams, k: float = 0.6) -> RigParams:
    """exponential smoothing of the continuous rig parameters (video mode)."""
    if prev is None:
        return cur
    out = replace(cur)
    for f in ("cx", "cy", "R", "yaw", "pitch", "roll", "ex", "ey", "er", "eye_aspect", "mx", "my", "mw"):
        setattr(out, f, k * getattr(prev, f) + (1 - k) * getattr(cur, f))
    out.gaze = tuple(k * a + (1 - k) * b for a, b in zip(prev.gaze, cur.gaze))
    return out
