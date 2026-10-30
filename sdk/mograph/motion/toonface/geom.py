"""
toonface.geom — the 3D head frame every feature is attached to.

Canonical face space (u, v) is measured in head radii on the FRONT of a unit sphere: u → screen-right at front view,
v ↓.  A feature is NEVER stored as pixels: it is a set of canonical points lifted onto the sphere (z = sqrt(1-u²-v²)),
rotated by yaw / pitch / roll and orthographically projected.

  wrap(uv)          per-point surface projection — brows, mouth, nose, cheek FX wrap around the skull
  local(anchor, ab) rigid "decal" through the Jacobian at an anchor, with the animator's foreshortening cheat
                    (min singular value ≥ cheat·max) — eyes keep a drawable shape even at 70° of yaw

yaw > 0 → the face turns toward screen-right.  pitch > 0 → looks down.  visibility(anchor) fades features that rotate
behind the head.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def rot_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], float)
    rx = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]], float)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], float)
    return rz @ rx @ ry


@dataclass
class HeadFrame:
    cx: float
    cy: float
    R: float
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    sx: float = 1.0
    sy: float = 1.0
    depth: float = 1.0
    cheat: float = 0.46
    _rot: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._rot = rot_matrix(self.yaw, self.pitch, self.roll)

    def replace(self, **kw) -> "HeadFrame":
        d = {k: getattr(self, k) for k in ("cx", "cy", "R", "yaw", "pitch", "roll", "sx", "sy", "depth", "cheat")}
        d.update(kw)
        return HeadFrame(**d)

    def lift(self, uv: np.ndarray, bulge: float = 0.0) -> np.ndarray:
        p = np.asarray(uv, float).reshape(-1, 2)
        r2 = p[:, 0] ** 2 + p[:, 1] ** 2
        z = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
        over = r2 > 1.0
        if over.any():
            r = np.sqrt(r2[over])
            ang = (r - 1.0) * 1.2
            k = np.cos(ang) / r
            p = p.copy()
            p[over] *= k[:, None]
            z = z.copy()
            z[over] = -np.sin(ang)
        z = z * self.depth + bulge
        return np.c_[p[:, 0] * self.sx, p[:, 1] * self.sy, z]

    def rotate(self, pts3: np.ndarray) -> np.ndarray:
        return pts3 @ self._rot.T

    def to_screen(self, q: np.ndarray) -> np.ndarray:
        return np.c_[self.cx + self.R * q[:, 0], self.cy + self.R * q[:, 1]]

    def wrap(self, uv, bulge: float = 0.0):
        q = self.rotate(self.lift(uv, bulge))
        return self.to_screen(q), q[:, 2]

    def normal_z(self, uv) -> float:
        _, z = self.wrap(np.asarray(uv, float).reshape(1, 2))
        return float(z[0])

    def visibility(self, uv, lo: float = -0.05, hi: float = 0.22) -> float:
        return float(np.clip((self.normal_z(uv) - lo) / (hi - lo), 0.0, 1.0))

    def jacobian(self, uv, eps: float = 1e-3) -> np.ndarray:
        uv = np.asarray(uv, float)
        s, _ = self.wrap(np.array([uv, uv + [eps, 0], uv + [0, eps]]))
        return np.c_[(s[1] - s[0]) / eps, (s[2] - s[0]) / eps]

    def decal(self, anchor_uv, cheat: float | None = None):
        anchor_uv = np.asarray(anchor_uv, float)
        o, _ = self.wrap(anchor_uv.reshape(1, 2))
        J = self.jacobian(anchor_uv)
        U, S, Vt = np.linalg.svd(J)
        c = self.cheat if cheat is None else cheat
        S2 = S.copy()
        S2[1] = max(S[1], c * S[0])
        M = U @ np.diag(S2) @ Vt
        if np.linalg.det(M) * np.linalg.det(J) < 0:
            M = J
        return o[0], M

    def local(self, anchor_uv, ab, cheat: float | None = None) -> np.ndarray:
        o, M = self.decal(anchor_uv, cheat)
        return o[None, :] + np.asarray(ab, float).reshape(-1, 2) @ M.T


def fit_circle(pts: np.ndarray) -> tuple[float, float, float]:
    x, y = pts[:, 0].astype(float), pts[:, 1].astype(float)
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    (a, c, d), *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    return float(a), float(c), float(math.sqrt(max(d + a * a + c * c, 1e-6)))
