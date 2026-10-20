"""
mograph.obj3d — v14.4 core 3D object renderer (pure NumPy + OpenCV + Cairo, CPU, no GPU).

Replaces the v13.9 prototypes in sam3d_bridge.py (which are kept as thin wrappers) with ONE correct,
fast renderer:

  render_extruded_sprite()   any RGBA sprite (assets_3d/*.png, image_gen output) → solid 3D object:
                             • correct perspective (front face nearer than back face — v13.9 had it inverted)
                             • front / back face visibility (past 90° you see the BACK, not a mirrored front)
                             • gap-free side wall: slice count follows the projected depth in px (no stripes)
                             • rounded profile (bevel → cushion) by distance-field inset (no Lanczos moiré)
                             • per-pixel wall normals + key/rim light + moving specular sheen on the face
                             • works at OUTPUT resolution → 5-10× faster than v13.9 (≈40-90 ms @ 300 px)
  render_multiview()         2..N real camera views of one object (e.g. laptop_yaw_m45/m15/p15/p45.png) →
                             continuous turntable: each real view is held + turned slightly, neighbours dissolve
                             at the midpoint (v13.9 hard-snapped between views → visible pops)
  render_flat_sprite()       homography-only billboard (pitch/roll tilt) used by the multiview path
  resolve_asset()            src lookup: absolute → scene base_dir → cwd → mograph/assets_3d

Conventions: screen space (x right, y DOWN, z toward the viewer).  R = Rz(roll)·Rx(pitch)·Ry(yaw).
pitch > 0 → we look slightly from above.  yaw > 0 → the object's right side turns away from us.
"""
from __future__ import annotations

import glob
import math
import os
import re
from functools import lru_cache

import cairo
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS_3D = os.path.join(_HERE, "assets_3d")

LIGHT_KEY = np.array([0.45, -0.55, 0.70], np.float32)
LIGHT_KEY /= np.linalg.norm(LIGHT_KEY)
LIGHT_RIM = np.array([-0.70, -0.25, -0.35], np.float32)
LIGHT_RIM /= np.linalg.norm(LIGHT_RIM)
FOCAL = 3.4            # camera distance in units of the sprite half-size (moderate, product-shot perspective)


# ============================================================================================ assets
def resolve_asset(src, base_dir=None):
    """Find an asset file: absolute path, relative to the scene/project dir, cwd, or mograph/assets_3d."""
    if not src:
        return None
    src = str(src)
    cands = [src] if os.path.isabs(src) else []
    if base_dir:
        cands.append(os.path.join(base_dir, src))
    cands += [src, os.path.join(ASSETS_3D, src), os.path.join(ASSETS_3D, os.path.basename(src))]
    for c in cands:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


@lru_cache(maxsize=64)
def load_rgba(path):
    """BGRA uint8 (straight alpha).  3-channel images are keyed (black / white / green background)."""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 3:
        try:
            from .extract3d import key_3d_rgba
            im = key_3d_rgba(im)
            if im.shape[2] == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
        except Exception:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    return np.ascontiguousarray(im[:, :, :4])


def find_views(src, base_dir=None):
    """Auto-detect multi-view siblings:  laptop_3d.png → laptop_yaw_m45.png, laptop_yaw_p15.png …
    Returns sorted [(angle_deg, path), ...] (≥ 2 views) or None."""
    p = resolve_asset(src, base_dir)
    if not p:
        return None
    d, b = os.path.split(p)
    stem = re.sub(r"(_3d)?\.(png|webp|jpg|jpeg)$", "", b, flags=re.I)
    stem = re.sub(r"_yaw_[mp]\d+$", "", stem)
    out = []
    for f in glob.glob(os.path.join(d, f"{stem}_yaw_*.*")):
        m = re.search(r"_yaw_([mp])(\d+)\.", os.path.basename(f))
        if m:
            out.append(((-1 if m.group(1) == "m" else 1) * float(m.group(2)), f))
    out.sort()
    return out if len(out) >= 2 else None


# ============================================================================================ maths
def rot_matrix(yaw=0.0, pitch=0.0, roll=0.0):
    y, p, r = math.radians(yaw), math.radians(pitch), math.radians(roll)
    cy, sy, cp, sp, cr, sr = math.cos(y), math.sin(y), math.cos(p), math.sin(p), math.cos(r), math.sin(r)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], np.float32)
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], np.float32)
    return Rz @ Rx @ Ry


def _project(P, F=FOCAL):
    """object-space points (N,3) (already rotated) → screen units (N,2), depth factor."""
    zc = np.maximum(F * 0.2, F - P[:, 2])
    k = F / zc
    return P[:, :2] * k[:, None], k


def _plane_H(R, z0, src_px, out_half, unit_px, F=FOCAL, ex=1.0, ey=1.0):
    """Homography: padded sprite pixels (src_px × src_px, object units [-1,1]) at depth z0 → output pixels."""
    c = np.array([[-ex, -ey, z0], [ex, -ey, z0], [ex, ey, z0], [-ex, ey, z0]], np.float32)
    P = c @ R.T
    if np.any(F - P[:, 2] < F * 0.25):
        return None
    xy, _ = _project(P, F)
    dst = (xy * unit_px + out_half).astype(np.float32)
    s0, s1 = src_px * (1 - ex) / 2, src_px * (1 + ex) / 2
    t0, t1 = src_px * (1 - ey) / 2, src_px * (1 + ey) / 2
    srcc = np.array([[s0, t0], [s1, t0], [s1, t1], [s0, t1]], np.float32)
    return cv2.getPerspectiveTransform(srcc, dst)


# ============================================================================================ prep (cached)
@lru_cache(maxsize=48)
def _prep_sprite(path, res):
    """Square-padded sprite at working res + everything that does not depend on the view angle."""
    bgra = load_rgba(path)
    if bgra is None:
        return None
    h0, w0 = bgra.shape[:2]
    fill = 0.86
    sc = res * fill / max(h0, w0)
    nw, nh = max(8, int(round(w0 * sc))), max(8, int(round(h0 * sc)))
    im = cv2.resize(bgra, (nw, nh), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
    pad = np.zeros((res, res, 4), np.uint8)
    ox, oy = (res - nw) // 2, (res - nh) // 2
    pad[oy:oy + nh, ox:ox + nw] = im
    a = pad[:, :, 3].astype(np.float32) / 255.0
    mask = a > 0.5
    dist = distance_transform_edt(mask).astype(np.float32)          # px to the silhouette (inside)
    # outward contour normal (sprite plane), valid near the edge
    sm = gaussian_filter(a, 1.6)
    gy, gx = np.gradient(sm)
    g = np.maximum(1e-6, np.hypot(gx, gy))
    nx, ny = -gx / g, -gy / g
    # wall colour = sprite colour near the rim propagated inward (so the side looks like the same material)
    band = (mask & (dist < max(3.0, res * 0.02))).astype(np.float32)
    rgb = pad[:, :, :3].astype(np.float32)
    num = gaussian_filter(rgb * band[..., None], (res * 0.04, res * 0.04, 0))
    den = gaussian_filter(band, res * 0.04)[..., None]
    mean_rim = (rgb * band[..., None]).sum((0, 1)) / max(1.0, band.sum())
    wall = np.where(den > 1e-3, num / np.maximum(den, 1e-3), mean_rim)
    wall = np.clip(wall * 0.80 + 10.0, 0, 255).astype(np.float32)
    # dome height (for cushion bulge + face normals)
    dmax = max(1.0, float(dist.max()))
    dome = np.sin(0.5 * math.pi * np.clip(dist / dmax, 0, 1) ** 0.6)
    dome = gaussian_filter(dome, res * 0.012) * mask
    hy, hx = np.gradient(dome)
    # premultiplied straight copy for the face
    pm = pad.astype(np.float32)
    pm[:, :, :3] *= a[..., None]
    lum = (0.114 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.299 * rgb[..., 2]) / 255.0
    return dict(pad=pad, pm=pm, a=a, mask=mask, dist=dist, nx=nx.astype(np.float32), ny=ny.astype(np.float32),
                wall=wall, mean_rim=mean_rim, dome=dome.astype(np.float32), hx=hx.astype(np.float32),
                hy=hy.astype(np.float32), dmax=dmax, lum=lum.astype(np.float32), res=res)


# ============================================================================================ painting helpers
def paint_premul(ctx, pm, x0, y0, alpha=1.0):
    """Paint a premultiplied float/uint8 BGRA array at (x0, y0) on a cairo context."""
    arr = np.clip(pm * float(alpha), 0, 255).astype(np.uint8) if pm.dtype != np.uint8 or alpha != 1.0 else pm
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    surf = cairo.ImageSurface.create_for_data(arr, cairo.FORMAT_ARGB32, w, h)
    ctx.save()
    ctx.set_source_surface(surf, x0, y0)
    ctx.get_source().set_filter(cairo.FILTER_GOOD)
    ctx.paint()
    ctx.restore()
    surf.finish()


def contact_shadow(ctx, cx, cy, w, alpha=1.0, strength=0.42, squash=0.22):
    if alpha <= 0.01 or w <= 2:
        return
    ctx.save()
    ctx.translate(cx, cy)
    ctx.scale(1.0, squash)
    g = cairo.RadialGradient(0, 0, 0, 0, 0, w * 0.5)
    g.add_color_stop_rgba(0, 0, 0, 0, strength * alpha)
    g.add_color_stop_rgba(0.55, 0, 0, 0, strength * 0.38 * alpha)
    g.add_color_stop_rgba(1, 0, 0, 0, 0)
    ctx.set_source(g)
    ctx.arc(0, 0, w * 0.5, 0, 2 * math.pi)
    ctx.fill()
    ctx.restore()


def _over(acc, src):
    """acc, src: premultiplied float32 BGRA — src OVER acc (in place)."""
    k = 1.0 - src[:, :, 3:4] / 255.0
    acc *= k
    acc += src


# ============================================================================================ extruded sprite
def render_extruded_sprite_array(path, size, yaw=0.0, pitch=10.0, roll=0.0, depth=0.16, round_=0.25,
                                 bulge=0.0, specular=0.7, rim_color=(1.0, 1.0, 1.0), back="solid",
                                 F=FOCAL, res=None):
    """Return (premultiplied float32 BGRA D×D, D) of the sprite turned into a solid 3D object.
    depth   total thickness (fraction of the sprite size)       round_  0 = sharp slab edge … 1 = full half-round
    bulge   front-face dome (0 = flat badge, 0.5 = plump cushion) back    "solid" | "mirror" (show sprite mirrored)
    """
    D = max(24, int(round(size * 1.30)))
    res = int(res or min(420, max(160, int(size * 0.95))))
    pr = _prep_sprite(path, res)
    if pr is None:
        return None, D
    R = rot_matrix(yaw, pitch, roll)
    Rt = R.T
    half = D * 0.5
    unit = size * 0.5                            # px per object unit (sprite spans [-1, 1])
    T = max(0.0, float(depth))
    zf, zb = T * 0.5 * 2, -T * 0.5 * 2          # object units: sprite spans [-1,1] → depth fraction × 2
    # ---- visibility of the faces (vector from face centre to the camera vs. face normal)
    n_front = R[:, 2]
    cf = R @ np.array([0, 0, zf], np.float32)
    cb = R @ np.array([0, 0, zb], np.float32)
    cam = np.array([0, 0, F], np.float32)
    front_vis = float(np.dot(n_front, cam - cf)) > 0
    back_vis = float(np.dot(-n_front, cam - cb)) > 0
    # ---- light in object space
    Lk = Rt @ LIGHT_KEY
    Lr = Rt @ LIGHT_RIM
    V = Rt @ np.array([0, 0, 1], np.float32)
    Hk = Lk + V
    Hk /= max(1e-6, np.linalg.norm(Hk))
    acc = np.zeros((D, D, 4), np.float32)
    # ---- side wall: slices from far to near
    if T > 1e-4:
        p_f, _ = _project(cf[None])
        p_b, _ = _project(cb[None])
        disp_px = float(np.linalg.norm(p_f - p_b)) * unit
        # silhouette growth with depth (perspective) also opens gaps → add the radius change
        k_f = F / max(F * 0.2, F - cf[2])
        k_b = F / max(F * 0.2, F - cb[2])
        disp_px += abs(k_f - k_b) * unit
        n_sl = int(np.clip(math.ceil(disp_px * 1.15) + 2, 3, 120))
        r_px = float(np.clip(round_, 0, 1)) * min(T * res * 0.5, pr["dmax"] * 0.6)   # rounding radius (src px)
        nx, ny, dist, wall = pr["nx"], pr["ny"], pr["dist"], pr["wall"]
        rimc = np.array(rim_color[::-1], np.float32) * 255.0      # BGR
        order = range(n_sl) if front_vis or not back_vis else range(n_sl - 1, -1, -1)
        # the near cap is drawn last; walls in between
        for i in order:
            s = -1.0 + 2.0 * i / (n_sl - 1)            # -1 back … +1 front
            z = zb + (zf - zb) * (s + 1) * 0.5
            # rounded profile: inset & normal tilt near the two caps
            e = max(0.0, (abs(s) - (1 - min(1.0, round_ * 1.0))) / max(1e-3, min(1.0, round_ * 1.0))) if round_ > 0 else 0.0
            e = min(1.0, e)
            inset = r_px * (1 - math.sqrt(max(0.0, 1 - e * e)))
            sz = math.copysign(e, s)                    # normal z component (toward the cap)
            cxy = math.sqrt(max(0.0, 1 - sz * sz))
            m = np.clip(dist - inset, 0, 1).astype(np.float32)          # dist = 0 outside, ≥ 1 inside
            if not m.any():
                continue
            ndl = np.clip(cxy * (nx * Lk[0] + ny * Lk[1]) + sz * Lk[2], 0, 1)
            ndr = np.clip(cxy * (nx * Lr[0] + ny * Lr[1]) + sz * Lr[2], 0, 1)
            ndh = np.clip(cxy * (nx * Hk[0] + ny * Hk[1]) + sz * Hk[2], 0, 1)
            ao = 0.72 + 0.28 * (s + 1) * 0.5 if front_vis else 0.72 + 0.28 * (1 - s) * 0.5
            shade = (0.30 + 0.62 * ndl) * ao
            col = wall * shade[..., None] + (ndh ** 28)[..., None] * (170.0 * specular) \
                + (ndr ** 3)[..., None] * rimc * (0.35 * specular)
            sl = np.empty((res, res, 4), np.float32)
            sl[:, :, :3] = np.clip(col, 0, 255) * m[..., None]
            sl[:, :, 3] = m * 255.0
            ex = ey = 1.0
            H = _plane_H(R, z, res, half, unit, F, ex, ey)
            if H is None:
                continue
            w = cv2.warpPerspective(sl, H, (D, D), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            _over(acc, w)
    # ---- caps
    a = pr["a"]
    if front_vis:
        acc_face = _face_layer(pr, R, Lk, Hk, zf, D, half, unit, F, specular, bulge, yaw, pitch, front=True)
        if acc_face is not None:
            _over(acc, acc_face)
    elif back_vis:
        if back == "mirror":
            acc_face = _face_layer(pr, R, Lk, Hk, zb, D, half, unit, F, specular * 0.6, bulge, yaw, pitch,
                                   front=False)
        else:
            # solid back plate: uniform rim material (brushed), lit by the key light, bevel highlight on the edge
            ndl = max(0.0, float(-Lk[2]))
            ndh = max(0.0, float(-Hk[2]))
            res_ = pr["res"]
            yy, xx = np.mgrid[0:res_, 0:res_].astype(np.float32) / res_
            grad = 0.85 + 0.30 * (1 - yy) * 0.5 + 0.10 * xx            # soft studio gradient
            base = np.array(pr["mean_rim"], np.float32) * 0.85 + 12.0
            edge = np.clip(1 - pr["dist"] / max(2.0, res_ * 0.025), 0, 1)
            col = base[None, None, :] * ((0.35 + 0.55 * ndl) * grad)[..., None] \
                + (edge * (60 + 120 * ndh ** 8))[..., None] * specular
            sl = np.empty((res, res, 4), np.float32)
            sl[:, :, :3] = np.clip(col, 0, 255) * a[..., None]
            sl[:, :, 3] = a * 255.0
            H = _plane_H(R, zb, res, half, unit, F)
            acc_face = None if H is None else cv2.warpPerspective(sl, H, (D, D), flags=cv2.INTER_LINEAR)
        if acc_face is not None:
            _over(acc, acc_face)
    return acc, D


def _face_layer(pr, R, Lk, Hk, z0, D, half, unit, F, specular, bulge, yaw, pitch, front=True):
    res = pr["res"]
    pm = pr["pm"]
    a = pr["a"]
    sgn = 1.0 if front else -1.0
    # face normals (flat or dome)
    k = float(bulge) * res * 0.30
    nx = -pr["hx"] * k
    ny = -pr["hy"] * k
    nz = np.ones_like(nx)
    nl = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / nl, ny / nl, sgn * nz / nl
    ndl = np.clip(nx * Lk[0] + ny * Lk[1] + nz * Lk[2], 0, 1)
    ndl0 = float(LIGHT_KEY[2])                          # what the sprite was lit for (frontal)
    ndh = np.clip(nx * Hk[0] + ny * Hk[1] + nz * Hk[2], 0, 1)
    mod = 1.0 + 0.40 * (ndl - ndl0)                      # relative relight keeps the baked studio look
    # moving sheen: broad diagonal band that sweeps with the angle + a tight highlight on domes
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32)
    u = (xx / res - 0.5) * 2
    v = (yy / res - 0.5) * 2
    sweep = math.sin(math.radians(yaw)) * 1.5 - math.sin(math.radians(pitch)) * 0.8
    band = np.exp(-((u * 0.80 + v * 0.60 - sweep) ** 2) / (2 * 0.20 ** 2))
    spec = (band * 55.0 + (ndh ** 40) * (120.0 if bulge > 0 else 40.0)) * specular
    col = pm[:, :, :3] * mod[..., None] + (spec * a)[..., None]
    if not front:
        col *= 0.55
    face = np.empty_like(pm)
    face[:, :, :3] = np.clip(col, 0, 255)
    face[:, :, 3] = pm[:, :, 3]
    # bulge parallax: the dome top shifts toward the viewer's side (cheap ray-march, 2 steps)
    if bulge > 0:
        par = res * 0.10 * float(bulge)
        dx = math.sin(math.radians(yaw)) * par
        dy = -math.sin(math.radians(pitch)) * par
        dome = pr["dome"]
        mx = xx + dome * dx
        my = yy + dome * dy
        d2 = cv2.remap(dome, mx, my, cv2.INTER_LINEAR)
        mx = xx + d2 * dx
        my = yy + d2 * dy
        face = cv2.remap(face, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    H = _plane_H(R, z0 + (0.0 if bulge <= 0 else sgn * 0.0), res, half, unit, F)
    if H is None:
        return None
    return cv2.warpPerspective(face, H, (D, D), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)


def render_extruded_sprite(ctx, src, cx, cy, size, yaw=0.0, pitch=10.0, roll=0.0, depth=0.16, round_=0.25,
                           bulge=0.0, alpha=1.0, specular=0.7, shadow=True, rim_color=(1, 1, 1), back="solid",
                           base_dir=None):
    path = resolve_asset(src, base_dir)
    if not path or alpha <= 0.01 or size < 4:
        return None
    acc, D = render_extruded_sprite_array(path, size, yaw, pitch, roll, depth, round_, bulge, specular,
                                          rim_color, back)
    if acc is None:
        return None
    acc = np.clip(acc, 0, 255)
    acc[:, :, :3] = np.minimum(acc[:, :, :3], acc[:, :, 3:4])     # keep premultiplied invariant
    if shadow:
        contact_shadow(ctx, cx, cy + size * 0.47, size * 0.95, alpha)
    paint_premul(ctx, acc, cx - D * 0.5, cy - D * 0.5, alpha)
    return (cx - size * 0.5, cy - size * 0.5, cx + size * 0.5, cy + size * 0.5)


# ============================================================================================ flat billboard
def render_flat_sprite_array(bgra, size, yaw=0.0, pitch=0.0, roll=0.0, F=FOCAL):
    D = max(24, int(round(size * 1.30)))
    h0, w0 = bgra.shape[:2]
    res = int(min(600, max(h0, w0)))
    sc = res * 0.86 / max(h0, w0)
    im = cv2.resize(bgra, (max(4, int(w0 * sc)), max(4, int(h0 * sc))), interpolation=cv2.INTER_AREA)
    pad = np.zeros((res, res, 4), np.float32)
    oy, ox = (res - im.shape[0]) // 2, (res - im.shape[1]) // 2
    pad[oy:oy + im.shape[0], ox:ox + im.shape[1]] = im
    pad[:, :, :3] *= pad[:, :, 3:4] / 255.0
    H = _plane_H(rot_matrix(yaw, pitch, roll), 0.0, res, D * 0.5, size * 0.5, F)
    if H is None:
        return None, D
    return cv2.warpPerspective(pad, H, (D, D), flags=cv2.INTER_CUBIC), D


# ============================================================================================ multi-view turntable
@lru_cache(maxsize=48)
def _view_img(path, res):
    b = load_rgba(path)
    h0, w0 = b.shape[:2]
    sc = res * 0.86 / max(h0, w0)
    im = cv2.resize(b, (max(4, int(w0 * sc)), max(4, int(h0 * sc))), interpolation=cv2.INTER_AREA)
    pad = np.zeros((res, res, 4), np.uint8)
    oy, ox = (res - im.shape[0]) // 2, (res - im.shape[1]) // 2
    pad[oy:oy + im.shape[0], ox:ox + im.shape[1]] = im
    return pad


def view_weights(angs, yaw, dissolve=0.08, switch=0.62):
    """Which real views are visible at `yaw` → [(index, weight, residual_yaw)].
    Each captured view is held and turned slightly (billboard residual, ≤ half the gap × 0.5) toward its
    neighbour; around the midpoint the two views dissolve over `dissolve` × gap.  (Optical-flow morphs between
    views 30° apart ghost badly on flat surfaces such as screens — measured, v14.4.)"""
    if yaw <= angs[0]:
        return [(0, 1.0, yaw - angs[0])]
    if yaw >= angs[-1]:
        return [(len(angs) - 1, 1.0, yaw - angs[-1])]
    j = int(np.searchsorted(angs, yaw)) - 1
    a0, a1 = angs[j], angs[j + 1]
    f = (yaw - a0) / max(1e-6, a1 - a0)
    # switch point is OFF the midpoint: with views at ±15° the midpoint is yaw 0 — the most common rest pose —
    # and a static 50/50 dissolve there is a visible double image.  A short dissolve (8 % of the gap ≈ 2.4°) is
    # crossed in < 2 frames by any moving turntable.
    c = float(np.clip((f - switch) / dissolve + 0.5, 0.0, 1.0))
    c = c * c * (3 - 2 * c)
    out = []
    if c < 1:
        out.append((j, 1 - c, (yaw - a0) * 0.5))
    if c > 0:
        out.append((j + 1, c, (yaw - a1) * 0.5))
    return out


def render_multiview(ctx, views, cx, cy, size, yaw=0.0, pitch=0.0, roll=0.0, alpha=1.0, shadow=True,
                     base_dir=None):
    vv = []
    for a, p in views:
        rp = resolve_asset(p, base_dir)
        if rp:
            vv.append((float(a), rp))
    vv.sort()
    if not vv:
        return None
    res = int(min(560, max(200, size * 1.1)))
    acc = None
    D = None
    for i, w, resid in view_weights([a for a, _ in vv], float(yaw)):
        resid = float(np.clip(resid, -24.0, 24.0))      # beyond the captured range: turn a little, then stop
        arr, D = render_flat_sprite_array(_view_img(vv[i][1], res), size, yaw=resid, pitch=pitch * 0.30, roll=roll)
        if arr is None:
            continue
        acc = arr * w if acc is None else acc + arr * w
    if acc is None:
        return None
    if shadow:
        contact_shadow(ctx, cx, cy + size * 0.40, size * 0.95, alpha)
    paint_premul(ctx, np.clip(acc, 0, 255), cx - D * 0.5, cy - D * 0.5, alpha)
    return (cx - size * 0.5, cy - size * 0.5, cx + size * 0.5, cy + size * 0.5)


# ============================================================================================ motion presets
def motion_angles(el, lt, yaw0=0.0, pitch0=10.0):
    """Shared yaw/pitch motion presets for every 3D element.
    turntable_osc (default) · spin_360 · spring_entry · hover · none"""
    mode = el.get("motion", "turntable_osc")
    if mode == "spin_360":
        return yaw0 + lt * float(el.get("yaw_speed", 75.0)), pitch0 + math.sin(lt * 2.2) * float(el.get("pitch_amp", 8.0))
    if mode == "spring_entry":
        e = math.exp(-3.2 * lt)
        return yaw0 + e * float(el.get("yaw_amp", 58.0)) * math.cos(lt * 6.5), \
            pitch0 + e * float(el.get("pitch_amp", 24.0)) * math.sin(lt * 5.0)
    if mode == "hover":
        return yaw0 + math.sin(lt * 1.1) * float(el.get("yaw_amp", 14.0)), \
            pitch0 + math.sin(lt * 1.7 + 0.6) * float(el.get("pitch_amp", 5.0))
    if mode == "none":
        return yaw0, pitch0
    sp = float(el.get("speed", 2.4))
    return yaw0 + math.sin(lt * sp) * float(el.get("yaw_amp", 46.0)), \
        pitch0 + math.cos(lt * sp * 0.7) * float(el.get("pitch_amp", 14.0))


# ============================================================================================ audit
def audit_sheet(src, out_png, angles=(-75, -45, -20, 0, 20, 45, 75, 110, 150, 180), cell=260, pitch=12.0):
    """3D QA sheet: one row per geometry (slab / cushion / multiview) for an image, or the sculpt3d preset
    rotating.  Use it BEFORE putting a 3D object in an edit — the user's rule: check every angle that will be
    on screen (the turn must never show a broken / mirrored / striped frame)."""
    rows = []
    presets = ("laptop_pro_3d", "ozempic_pen_3d", "trophy_3d")
    is_tree = src in presets or str(src).endswith(".json") or os.path.isfile(
        os.path.join(_HERE, "sculpts", f"{src}.json"))
    if is_tree:
        from . import sculpt3d as S
        tree = S.get_tree(src if str(src).endswith(".json") else None, src, open_dur=0.01)
        rows.append(("sculpt3d", lambda c, x, y, s, a: S.render_sculpt3d_scene(c, tree, x, y, s, t=5.0, yaw_deg=a,
                                                                               pitch_deg=pitch)))
    else:
        rows.append(("slab", lambda c, x, y, s, a: render_extruded_sprite(c, src, x, y, s, yaw=a, pitch=pitch,
                                                                         depth=0.12, round_=0.3)))
        rows.append(("cushion", lambda c, x, y, s, a: render_extruded_sprite(c, src, x, y, s, yaw=a, pitch=pitch,
                                                                            depth=0.22, round_=1.0, bulge=0.45)))
        vv = find_views(src)
        if vv:
            rows.append(("multiview", lambda c, x, y, s, a: render_multiview(c, vv, x, y, s, yaw=a, pitch=pitch)))
    W, H = cell * len(angles), cell * len(rows) + 40
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surf)
    ctx.set_source_rgb(0.08, 0.09, 0.12)
    ctx.paint()
    ctx.set_source_rgb(0.7, 0.75, 0.8)
    ctx.set_font_size(15)
    for i, a in enumerate(angles):
        ctx.move_to(i * cell + 8, 24)
        ctx.show_text(f"yaw {a}°")
    for r, (name, fn) in enumerate(rows):
        for i, a in enumerate(angles):
            fn(ctx, i * cell + cell / 2, 40 + r * cell + cell / 2, cell * 0.62, a)
        ctx.set_source_rgb(0.7, 0.75, 0.8)
        ctx.move_to(8, 40 + r * cell + 18)
        ctx.show_text(name)
    surf.write_to_png(out_png)
    return out_png
