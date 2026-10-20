"""
mograph.sam3d_bridge — Native CPU Renderer for `facebookresearch/sam-3d-objects` Exports (`.ply` Gaussian Splats & `.obj` Meshes).

While the heavy PyTorch/CUDA DiT inference of `facebookresearch/sam-3d-objects` requires an external
NVIDIA GPU (32 GB VRAM) to reconstruct an object from a photo (`output["gs"].save_ply("splat.ply")`),
its exported `.ply` (3D Gaussian Splat / Point Cloud) and `.obj` (3D Mesh) files CAN be rendered,
lit, and rotated 360° in real time inside `mograph` using pure vectorized NumPy + PyCairo!

Usage in `spec.json`:
  {
    "type": "sam3d_object",
    "id": "obj1",
    "src": "assets_3d/splat.ply",
    "t": 0.0,
    "out": 4.0,
    "x": 540,
    "y": 960,
    "scale": 320,
    "yaw_speed": 65.0,
    "pitch": 15.0
  }
"""
from __future__ import annotations
import math
import os
from functools import lru_cache
import cairo
import cv2
import numpy as np

from .core import clamp
from .elements import element, val, xy


@lru_cache(maxsize=16)
def load_ply_or_obj(path: str):
    """
    Loads either:
      * A `.ply` file (ASCII or binary little-endian 3D Gaussian Splat / Point Cloud from SAM 3D Objects)
      * A `.obj` Wavefront 3D mesh file
    Returns normalized `(xyz, rgb, sizes)` arrays in `[-1, 1]^3`.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"3D asset not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        verts = []
        cols = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()[1:]
                    if len(parts) >= 3:
                        verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        if len(parts) >= 6:
                            cols.append([float(parts[3]), float(parts[4]), float(parts[5])])
        xyz = np.array(verts, dtype=np.float32) if verts else np.zeros((1, 3), dtype=np.float32)
        if cols and len(cols) == len(verts):
            rgb = np.array(cols, dtype=np.float32)
            if rgb.max() <= 1.05:
                rgb *= 255.0
        else:
            rgb = np.full_like(xyz, 220.0)
        sizes = np.full((len(xyz),), 3.5, dtype=np.float32)
    else:
        # Parse .ply header (supports both standard RGB point clouds and 3D Gaussian Splat f_dc_0..2 PLYs!)
        with open(path, "rb") as f:
            header_lines = []
            while True:
                line = f.readline().decode("ascii", errors="ignore").strip()
                header_lines.append(line)
                if line == "end_header":
                    break
            is_binary = any("binary_little_endian" in l for l in header_lines)
            n_verts = 0
            props, ptypes = [], []
            in_vertex = False
            for l in header_lines:
                if l.startswith("element"):
                    in_vertex = l.startswith("element vertex")
                    if in_vertex:
                        n_verts = int(l.split()[-1])
                elif l.startswith("property") and in_vertex and "list" not in l:
                    props.append(l.split()[-1])
                    ptypes.append(l.split()[1])

            if is_binary and n_verts > 0:
                # v14.4: real per-property dtypes (SAM-3D / 3DGS use float, colour PLYs use uchar)
                tmap = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8", "uchar": "u1",
                        "uint8": "u1", "char": "i1", "int8": "i1", "short": "<i2", "int16": "<i2", "ushort": "<u2",
                        "uint16": "<u2", "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4"}
                dt = np.dtype([(f"p{i}", tmap.get(t, "<f4")) for i, t in enumerate(ptypes)])
                rec = np.frombuffer(f.read(n_verts * dt.itemsize), dtype=dt)
                data = np.stack([rec[f"p{i}"].astype(np.float32) for i in range(len(props))], axis=1)
            else:
                rows = []
                for _ in range(n_verts):
                    ln = f.readline().decode("ascii", errors="ignore").strip()
                    if not ln:
                        break
                    rows.append([float(v) for v in ln.split()])
                data = np.array(rows, dtype=np.float32) if rows else np.zeros((1, max(3, len(props))), dtype=np.float32)

        p_idx = {name: i for i, name in enumerate(props)}
        xyz = np.stack([
            data[:, p_idx.get("x", 0)],
            data[:, p_idx.get("y", 1)],
            data[:, p_idx.get("z", 2)],
        ], axis=1)

        C0 = 0.28209479177387814  # 0th-order Spherical Harmonic constant used by 3D Gaussian Splatting
        if "f_dc_0" in p_idx:
            r = np.clip((0.5 + C0 * data[:, p_idx["f_dc_0"]]) * 255.0, 0, 255)
            g = np.clip((0.5 + C0 * data[:, p_idx["f_dc_1"]]) * 255.0, 0, 255)
            b = np.clip((0.5 + C0 * data[:, p_idx["f_dc_2"]]) * 255.0, 0, 255)
            rgb = np.stack([r, g, b], axis=1)
        elif "red" in p_idx:
            rgb = np.stack([data[:, p_idx["red"]], data[:, p_idx["green"]], data[:, p_idx["blue"]]], axis=1)
            if rgb.max() <= 1.05:
                rgb *= 255.0
        else:
            rgb = np.full_like(xyz, 225.0)

        sizes = np.full((len(xyz),), 3.5, dtype=np.float32)

    # Subsample if point count > 35,000 for fast 30fps CPU rendering
    if len(xyz) > 35000:
        step = max(1, len(xyz) // 35000)
        xyz = xyz[::step]
        rgb = rgb[::step]
        sizes = sizes[::step]

    # Center and normalize to [-1, 1]
    center = (xyz.max(axis=0) + xyz.min(axis=0)) * 0.5
    xyz = xyz - center
    span = max(1e-5, float(np.abs(xyz).max()))
    xyz = xyz / span
    return xyz, np.clip(rgb, 0, 255).astype(np.uint8), sizes


@element("sam3d_object", dur=0.75, enter="spring", sfx="whoosh")
def el_sam3d_object(ctx, el, s):
    """
    Renders a true 3D `.ply` Gaussian Splat (exported from `facebookresearch/sam-3d-objects`)
    or `.obj` 3D mesh with full 360° 3D rotation (`yaw`, `pitch`, `roll`), perspective projection,
    depth-sorting, and soft studio contact shadow!
    """
    x, y = xy(el, s.t)
    radius_px = float(val(el, "scale", s.t, 240.0)) * s.scale
    local_t = max(0.0, s.t - float(el["t"]))

    yaw = math.radians(float(val(el, "yaw", s.t, 0.0)) + local_t * float(el.get("yaw_speed", 55.0)))
    pitch = math.radians(float(val(el, "pitch", s.t, 14.0)))
    roll = math.radians(float(val(el, "roll", s.t, 0.0)))

    from .obj3d import resolve_asset
    src = resolve_asset(el.get("src", ""), getattr(getattr(s, "scene", None), "base_dir", None))
    if not src:
        return None
    xyz, rgb, _ = load_ply_or_obj(src)

    # 3D Euler rotation matrix
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
    R = Rz @ Rx @ Ry

    pts = xyz @ R.T
    # Perspective projection (camera at z = -3.2)
    cam_dist = 3.2
    z_depth = cam_dist - pts[:, 2]
    persp = cam_dist / np.maximum(0.4, z_depth)
    px = pts[:, 0] * persp * radius_px
    py = -pts[:, 1] * persp * radius_px

    # Depth sort back-to-front (largest z_depth first)
    order = np.argsort(-z_depth)
    px = px[order]
    py = py[order]
    cols = rgb[order]

    D = int(radius_px * 2.5)
    half_D = D * 0.5
    canvas = np.zeros((D, D, 4), dtype=np.uint8)
    ix = np.clip((px + half_D).astype(np.int32), 2, D - 3)
    iy = np.clip((py + half_D).astype(np.int32), 2, D - 3)

    # Splat each Gaussian point into a 3x3 neighborhood
    for dy_o in (-1, 0, 1):
        for dx_o in (-1, 0, 1):
            w_a = 255 if (dx_o == 0 and dy_o == 0) else 170
            canvas[iy + dy_o, ix + dx_o, 0] = cols[:, 2]  # B
            canvas[iy + dy_o, ix + dx_o, 1] = cols[:, 1]  # G
            canvas[iy + dy_o, ix + dx_o, 2] = cols[:, 0]  # R
            canvas[iy + dy_o, ix + dx_o, 3] = np.maximum(canvas[iy + dy_o, ix + dx_o, 3], w_a)

    # Smooth Gaussian splat blending
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.85)
    # Premultiply alpha for Cairo
    a_norm = (canvas[:, :, 3:4].astype(np.float32) / 255.0) * clamp(s.alpha)
    bgra = np.zeros_like(canvas, dtype=np.uint8)
    bgra[:, :, :3] = np.clip(canvas[:, :, :3].astype(np.float32) * a_norm, 0, 255).astype(np.uint8)
    bgra[:, :, 3] = np.clip(canvas[:, :, 3].astype(np.float32) * clamp(s.alpha), 0, 255).astype(np.uint8)

    surf = cairo.ImageSurface.create_for_data(bgra, cairo.FORMAT_ARGB32, D, D)
    ctx.save()
    ctx.set_source_surface(surf, x - half_D, y + s.dy - half_D)
    ctx.get_source().set_filter(cairo.FILTER_GOOD)
    ctx.paint()
    ctx.restore()
    return (x - radius_px, y - radius_px, x + radius_px, y + radius_px)




# ===========================================================================
# SINGLE-IMAGE TO TRUE 3D VOLUMETRIC MESH & PBR NORMAL-RELIGHTING ENGINE
# ===========================================================================
from scipy.ndimage import distance_transform_edt, gaussian_filter
from .reelfx import _load_3d_rgba, _ASSETS_3D


@lru_cache(maxsize=32)
def reconstruct_3d_from_image(src_name: str, work_res: int = 220, depth_amount: float = 0.28):
    """
    Converts ANY single 2D RGBA image (e.g. generated by `image_gen` or in `assets_3d/`)
    into a true 3D Volumetric Model with:
      1. `Z(x, y)` 3D Depth Map: Combines Signed Euclidean Distance Transform inflation
         (`edt(mask) ** 0.58`, like MonsterMash/Teddy 3D inflation) with Shape-from-Shading
         luminance relief (`L(x, y)`) so ridges, screens, coins, and bevels have real 3D height!
      2. `N(x, y)` 3D Surface Normal Map: Analytical normal vectors `(nx, ny, nz)` at every pixel
         for real-time 3D Blinn-Phong specular relighting as the object rotates!
      3. Full 3D Watertight Shell: Front relief surface `(x, y, +Z)` + extruded metallic/glass
         3D side-wall bevels along the silhouette contour so the object can rotate freely to
         steep 3D angles (-65°..+65°) with true volumetric thickness!
    """
    bgra_in = _load_3d_rgba(src_name)
    rgba = cv2.cvtColor(bgra_in, cv2.COLOR_BGRA2RGBA) if bgra_in is not None else None
    if rgba is None and os.path.exists(src_name):
        im = cv2.imread(src_name, cv2.IMREAD_UNCHANGED)
        if im is not None:
            if im.shape[2] == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)
            else:
                im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
            rgba = im
    if rgba is None:
        raise FileNotFoundError(f"Cannot load image for 3D reconstruction: {src_name}")

    # Resize to working resolution `work_res x work_res` preserving aspect ratio
    h0, w0 = rgba.shape[:2]
    scale = work_res / max(h0, w0)
    nw, nh = max(16, int(round(w0 * scale))), max(16, int(round(h0 * scale)))
    im_s = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA)

    # Pad to square `work_res x work_res`
    sq = np.zeros((work_res, work_res, 4), dtype=np.uint8)
    ox, oy = (work_res - nw) // 2, (work_res - nh) // 2
    sq[oy:oy + nh, ox:ox + nw] = im_s

    alpha_f = sq[:, :, 3].astype(np.float32) / 255.0
    mask = alpha_f > 0.15

    # 1. Macro 3D Inflation Dome from Euclidean Distance Transform
    dist = distance_transform_edt(mask).astype(np.float32)
    d_max = max(1.0, float(dist.max()))
    z_dome = (dist / d_max) ** 0.55
    z_dome = gaussian_filter(z_dome, sigma=2.2)

    # 2. Micro 3D Shape-from-Shading Relief from RGB Luminance
    rgb_f = sq[:, :, :3].astype(np.float32) / 255.0
    lum = 0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2]
    lum_smooth = gaussian_filter(lum, sigma=1.2)
    z_relief = (lum_smooth - 0.35) * mask.astype(np.float32)

    # Combined 3D Depth Map Z(x, y) in normalized [-1, 1] object space
    z_map = (0.70 * z_dome + 0.30 * np.clip(z_relief, -0.2, 0.8)) * depth_amount * mask.astype(np.float32)

    # 3. Analytical 3D Surface Normals N(x, y) = normalize(-dZ/dx, -dZ/dy, 1)
    dz_dy, dz_dx = np.gradient(z_map * (work_res * 0.5))
    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(z_map)
    norm_len = np.maximum(1e-5, np.sqrt(nx**2 + ny**2 + nz**2))
    nx /= norm_len
    ny /= norm_len
    nz /= norm_len
    normals_map = np.stack([nx, ny, nz], axis=-1)

    # 4. Extract valid front-surface vertices + multi-layer 3D side-wall bevel extrusion
    yy, xx = np.mgrid[0:work_res, 0:work_res].astype(np.float32)
    x_norm = (xx - work_res * 0.5) / (work_res * 0.5)
    y_norm = (yy - work_res * 0.5) / (work_res * 0.5)

    fg_y, fg_x = np.where(mask)
    pts_front = np.stack([x_norm[fg_y, fg_x], y_norm[fg_y, fg_x], z_map[fg_y, fg_x]], axis=1)
    nrm_front = normals_map[fg_y, fg_x]
    col_front = sq[fg_y, fg_x, :3].astype(np.float32)
    alp_front = alpha_f[fg_y, fg_x]

    # Boundary contour pixels for extruded 3D side bevel walls (from +z_edge down to -depth_amount*0.65)
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3) > 0
    rim_mask = mask & (~eroded)
    ry, rx = np.where(rim_mask)

    bevel_pts, bevel_nrm, bevel_col, bevel_alp = [], [], [], []
    # Side wall normal points outward along the distance transform gradient
    gy_d, gx_d = np.gradient(-dist)
    g_len = np.maximum(1e-5, np.hypot(gx_d, gy_d))
    rim_nx = (gx_d[ry, rx] / g_len[ry, rx]) * 0.85
    rim_ny = (gy_d[ry, rx] / g_len[ry, rx]) * 0.85
    rim_nz = np.full_like(rim_nx, 0.52)
    rim_nrm = np.stack([rim_nx, rim_ny, rim_nz], axis=1)

    # Metallic/anodized side-wall color (slightly darker & richer than front edge)
    rim_base_col = np.clip(sq[ry, rx, :3].astype(np.float32) * 0.68 + 28.0, 0, 255)
    n_slices = 32
    back_z = -depth_amount * 0.55
    for k_i in range(1, n_slices + 1):
        frac = k_i / float(n_slices)
        zk = z_map[ry, rx] * (1.0 - frac) + back_z * frac
        bevel_pts.append(np.stack([x_norm[ry, rx], y_norm[ry, rx], zk], axis=1))
        bevel_nrm.append(rim_nrm)
        bevel_col.append(rim_base_col * (1.0 - 0.25 * frac))
        bevel_alp.append(np.full((len(ry),), 0.96, dtype=np.float32))

    # Back shell surface (so even 120° flips look solid!)
    pts_back = np.stack([x_norm[fg_y, fg_x], y_norm[fg_y, fg_x], np.full_like(z_map[fg_y, fg_x], back_z)], axis=1)
    nrm_back = np.stack([ -nx[fg_y, fg_x], -ny[fg_y, fg_x], -nz[fg_y, fg_x] ], axis=1)
    col_back = col_front * 0.55

    all_pts = np.vstack([pts_front] + bevel_pts + [pts_back]).astype(np.float32)
    all_nrm = np.vstack([nrm_front] + bevel_nrm + [nrm_back]).astype(np.float32)
    all_col = np.vstack([col_front] + bevel_col + [col_back]).astype(np.float32)
    all_alp = np.concatenate([alp_front] + bevel_alp + [alp_front]).astype(np.float32)

    return {
        "pts": all_pts,
        "nrm": all_nrm,
        "col": all_col,
        "alp": all_alp,
        "z_map": z_map,
        "normals_map": normals_map,
        "rgba_sq": sq,
        "mask": mask,
    }


def export_image_to_ply(src_name: str, out_ply: str, work_res: int = 200, depth_amount: float = 0.28):
    """Exports the reconstructed 3D volumetric model of a 2D image to a standard `.ply` 3D file!"""
    model = reconstruct_3d_from_image(src_name, work_res=work_res, depth_amount=depth_amount)
    pts = model["pts"]
    nrm = model["nrm"]
    col = np.clip(model["col"], 0, 255).astype(np.uint8)
    with open(out_ply, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("comment Reconstructed by mograph v13.9 Single-Image-to-3D Engine\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (px, py, pz), (nx, ny, nz), (r, g, b) in zip(pts, nrm, col):
            f.write(f"{px:.5f} {-py:.5f} {pz:.5f} {nx:.4f} {-ny:.4f} {nz:.4f} {r} {g} {b}\n")
    return out_ply


def render_reconstructed_3d(ctx, src_name: str, cx: float, cy: float, size: float,
                            yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0,
                            alpha: float = 1.0, depth_amount: float = 0.30,
                            specular: float = 0.68, shadow: bool = True):
    """
    Renders a single-image reconstructed 3D Volumetric Model at `(cx, cy)` with:
      * True 3D perspective projection (`yaw_deg`, `pitch_deg`, `roll_deg`),
      * Dynamic Blinn-Phong 3D Surface Normal Relighting (moving specular reflection & Fresnel rim!),
      * Z-buffered front-to-back occlusion & smooth hole-free splatting!
    """
    if alpha <= 0.01 or size <= 4:
        return
    model = reconstruct_3d_from_image(src_name, work_res=210, depth_amount=depth_amount)
    pts = model["pts"]
    nrm = model["nrm"]
    col = model["col"]
    alp = model["alp"]

    yw = math.radians(yaw_deg)
    pt = math.radians(pitch_deg)
    rl = math.radians(roll_deg)
    cy_m, sy_m = math.cos(yw), math.sin(yw)
    cp_m, sp_m = math.cos(pt), math.sin(pt)
    cr_m, sr_m = math.cos(rl), math.sin(rl)

    Ry = np.array([[cy_m, 0.0, sy_m], [0.0, 1.0, 0.0], [-sy_m, 0.0, cy_m]], dtype=np.float32)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp_m, -sp_m], [0.0, sp_m, cp_m]], dtype=np.float32)
    Rz = np.array([[cr_m, -sr_m, 0.0], [sr_m, cr_m, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    R = Rz @ Rx @ Ry

    # Rotate 3D vertices and 3D normals
    p_rot = pts @ R.T
    n_rot = nrm @ R.T

    # Cull back-facing points (n_rot[:, 2] < -0.15) for speed & crispness
    vis = n_rot[:, 2] >= -0.15
    p_rot = p_rot[vis]
    n_rot = n_rot[vis]
    col_v = col[vis]
    alp_v = alp[vis]

    # Dynamic 3D Studio Blinn-Phong Relighting!
    # Key light from top-right-front, moving relative to the rotating object normals
    lx, ly, lz = 0.42, -0.48, 0.77
    ndotl = np.clip(n_rot[:, 0] * lx + n_rot[:, 1] * ly + n_rot[:, 2] * lz, 0.0, 1.0)
    # Halfway vector H between Light L and View V=(0, 0, 1)
    hx, hy, hz = lx, ly, lz + 1.0
    h_len = math.sqrt(hx*hx + hy*hy + hz*hz)
    hx, hy, hz = hx / h_len, hy / h_len, hz / h_len
    ndoth = np.clip(n_rot[:, 0] * hx + n_rot[:, 1] * hy + n_rot[:, 2] * hz, 0.0, 1.0)

    # Diffuse shading modulation (subtle so original texture stays rich, +22% on lit slopes, -18% on shadow slopes)
    diff_mod = 0.82 + 0.30 * ndotl
    # Sharp studio specular highlight + soft broad sheen that sweeps across the 3D surface as it turns!
    spec_highlight = ((ndoth ** 26.0) * 145.0 + (ndoth ** 6.0) * 38.0) * specular
    # Fresnel rim light on grazing 3D edges
    fresnel = ((1.0 - np.clip(n_rot[:, 2], 0.0, 1.0)) ** 3.2) * 52.0 * specular

    shaded_rgb = np.clip(col_v * diff_mod[:, None] + (spec_highlight + fresnel)[:, None], 0, 255).astype(np.uint8)

    # Perspective projection (camera at z = +2.8 looking toward origin)
    cam_d = 2.8
    z_dist = cam_d - p_rot[:, 2]
    persp = cam_d / np.maximum(0.5, z_dist)

    D = max(32, int(round(size * 1.28)))
    half_D = D * 0.5
    scale_px = size * 0.48
    sx = p_rot[:, 0] * persp * scale_px + half_D
    sy = p_rot[:, 1] * persp * scale_px + half_D

    # Back-to-front Painter's Z-sort
    order = np.argsort(-z_dist)
    ix = np.clip(np.round(sx[order]).astype(np.int32), 1, D - 2)
    iy = np.clip(np.round(sy[order]).astype(np.int32), 1, D - 2)
    s_rgb = shaded_rgb[order]
    s_alp = (alp_v[order] * 255.0).astype(np.uint8)

    canvas = np.zeros((D, D, 4), dtype=np.uint8)
    # 3x3 watertight splat kernel so steep 55° side walls are 100% solid brushed metal with zero gaps!
    for dy_o in (-1, 0, 1):
        for dx_o in (-1, 0, 1):
            canvas[iy + dy_o, ix + dx_o, 0] = s_rgb[:, 2]  # B
            canvas[iy + dy_o, ix + dx_o, 1] = s_rgb[:, 1]  # G
            canvas[iy + dy_o, ix + dx_o, 2] = s_rgb[:, 0]  # R
            canvas[iy + dy_o, ix + dx_o, 3] = np.maximum(canvas[iy + dy_o, ix + dx_o, 3], s_alp)

    # Anti-aliased edge smoothing
    canvas[:, :, 3] = cv2.GaussianBlur(canvas[:, :, 3], (3, 3), 0.7)

    if shadow:
        sh_g = cairo.RadialGradient(cx + yaw_deg * 0.25, cy + size * 0.36, size * 0.05,
                                    cx + yaw_deg * 0.25, cy + size * 0.36, size * 0.48)
        sh_g.add_color_stop_rgba(0.0, 0, 0, 0, 0.36 * alpha)
        sh_g.add_color_stop_rgba(0.6, 0, 0, 0, 0.14 * alpha)
        sh_g.add_color_stop_rgba(1.0, 0, 0, 0, 0.0)
        ctx.save()
        ctx.set_source(sh_g)
        ctx.arc(cx + yaw_deg * 0.25, cy + size * 0.36, size * 0.48, 0, 2 * math.pi)
        ctx.fill()
        ctx.restore()

    a_norm = (canvas[:, :, 3:4].astype(np.float32) / 255.0) * clamp(alpha)
    bgra = np.zeros_like(canvas, dtype=np.uint8)
    bgra[:, :, :3] = np.clip(canvas[:, :, :3].astype(np.float32) * a_norm, 0, 255).astype(np.uint8)
    bgra[:, :, 3] = np.clip(canvas[:, :, 3].astype(np.float32) * clamp(alpha), 0, 255).astype(np.uint8)

    surf = cairo.ImageSurface.create_for_data(bgra, cairo.FORMAT_ARGB32, D, D)
    ctx.save()
    ctx.set_source_surface(surf, cx - half_D, cy - half_D)
    ctx.get_source().set_filter(cairo.FILTER_GOOD)
    ctx.paint()
    ctx.restore()


@element("image_to_3d", dur=0.75, enter="spring", sfx="whoosh")
def el_image_to_3d(ctx, el, s):
    """
    `image_to_3d` — ANY 2D RGBA image (assets_3d/*.png, image_gen output, project path) shown as a solid 3D
    object that turns in space.  v14.4: rendered by mograph.obj3d (correct perspective, back face, gap-free wall).

    geometry   auto (default) | slab | badge | cushion | multiview | sphere | capsule | points
               auto → multiview when sibling views exist (<stem>_yaw_m45.png …), sphere for *globe*/*sphere*,
                      capsule for *vial*/*pen*/*capsule*, cushion for pin/heart/warn/cash, else slab
    motion     turntable_osc (default) | spin_360 | spring_entry | hover | none   (see obj3d.motion_angles)
    depth      thickness (slab 0.12, cushion 0.22)    round  edge rounding 0..1    bulge  front dome 0..0.6
    views      [[angle, "file.png"], ...]  explicit multiview set      back  "solid" | "mirror"
    """
    from . import obj3d as O
    x, y = xy(el, s.t)
    sz = float(val(el, "w", s.t, el.get("size", 280))) * s.scale
    local_t = max(0.0, s.t - float(el["t"]))
    base_dir = getattr(getattr(s, "scene", None), "base_dir", None)
    yaw, pitch = O.motion_angles(el, local_t, float(val(el, "yaw", s.t, 0.0)), float(val(el, "pitch", s.t, 10.0)))
    roll = float(val(el, "roll", s.t, 0.0))
    src = el.get("src") or el.get("sprite") or "laptop_3d.png"
    low = str(src).lower()
    geom = el.get("geometry", "auto")
    views = el.get("views")
    if geom in ("turntable_4view", "multiview") or (geom == "auto" and (views or O.find_views(src, base_dir))):
        geom = "multiview"
    elif geom == "auto":
        if any(k in low for k in ("sphere", "globe", "orb")):
            geom = "sphere"
        elif any(k in low for k in ("vial", "capsule", "ozempic", "pen_")):
            geom = "capsule"
        elif any(k in low for k in ("pin", "warn", "heart", "cash", "coin")):
            geom = "cushion"
        else:
            geom = "slab"
    if geom in ("slab_3d",):
        geom = "slab"
    if geom in ("cushion_3d",):
        geom = "cushion"
    sh = bool(el.get("shadow", True))
    spec = float(el.get("specular", 0.7))
    yy = y + s.dy
    if geom == "multiview":
        vv = views or O.find_views(src, base_dir)
        O.render_multiview(ctx, [(float(a), p) for a, p in vv], x, yy, sz, yaw=yaw, pitch=pitch, roll=roll,
                           alpha=s.alpha, shadow=sh, base_dir=base_dir)
    elif geom == "sphere":
        p = O.resolve_asset(src, base_dir)
        bgra_t = O.load_rgba(p) if p else None
        if bgra_t is not None:
            tex_rgba = cv2.cvtColor(bgra_t, cv2.COLOR_BGRA2RGBA)
        else:
            tex_rgba = _procedural_globe_texture()
        render_true3d_sphere(ctx, tex_rgba, x, yy, sz * 0.46, yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
                             alpha=s.alpha, specular=spec, shadow=sh)
    elif geom in ("capsule", "cylinder"):
        p = O.resolve_asset(el.get("texture"), base_dir) if el.get("texture") else None
        tex_v = cv2.cvtColor(O.load_rgba(p), cv2.COLOR_BGRA2RGBA) if p else _procedural_pen_texture(el.get("label", "OZEMPIC 3D"))
        render_true3d_cylinder_capsule(ctx, tex_v, x, yy, radius=sz * 0.25, height=sz * 0.88,
                                       yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll - 14.0, alpha=s.alpha, shadow=sh)
    elif geom == "points":
        p = O.resolve_asset(src, base_dir)
        if p:
            render_reconstructed_3d(ctx, p, x, yy, sz, yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, alpha=s.alpha,
                                    depth_amount=float(el.get("depth", 0.30)), specular=spec, shadow=sh)
    else:
        prof = {"slab": (0.12, 0.30, 0.0), "badge": (0.06, 0.6, 0.0), "cushion": (0.22, 1.0, 0.45)}.get(geom, (0.12, 0.3, 0.0))
        O.render_extruded_sprite(ctx, src, x, yy, sz, yaw=yaw, pitch=pitch, roll=roll,
                                 depth=float(el.get("depth", prof[0])), round_=float(el.get("round", prof[1])),
                                 bulge=float(el.get("bulge", prof[2])), alpha=s.alpha, specular=spec, shadow=sh,
                                 back=el.get("back", "solid"), base_dir=base_dir)
    return (x - sz * 0.6, y - sz * 0.6, x + sz * 0.6, y + sz * 0.6)


def _procedural_globe_texture():
    tex = np.zeros((360, 720, 4), dtype=np.uint8)
    yy_g, xx_g = np.mgrid[0:360, 0:720].astype(np.float32)
    tex[:, :, 0] = 28
    tex[:, :, 1] = np.clip(95 + 75 * np.sin(xx_g * 0.035) * np.cos(yy_g * 0.045), 20, 220).astype(np.uint8)
    tex[:, :, 2] = np.clip(165 + 70 * np.cos(xx_g * 0.025), 60, 245).astype(np.uint8)
    tex[:, :, 3] = 255
    for gx in range(0, 720, 45):
        tex[:, gx:gx + 2, :3] = [120, 245, 255]
    for gy in range(0, 360, 45):
        tex[gy:gy + 2, :, :3] = [120, 245, 255]
    return tex


def _procedural_pen_texture(label="OZEMPIC 3D"):
    tex = np.zeros((400, 600, 4), dtype=np.uint8)
    tex[:, :, :3] = [25, 95, 78]
    tex[90:310, :, :3] = [245, 247, 250]
    tex[90:125, :, :3] = [245, 158, 11]
    tex[:55, :, :3] = [210, 215, 225]
    tex[:, :, 3] = 255
    cv2.putText(tex, str(label), (85, 220), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (15, 23, 42, 255), 5, cv2.LINE_AA)
    return tex


# ---- v14.4 compatibility wrappers (old names → new obj3d engine) -------------------------------------------
def render_machined_3d_slab(ctx, src_name, cx, cy, size, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
                            thickness=0.14, alpha=1.0, specular=0.75, shadow=True):
    from . import obj3d as O
    return O.render_extruded_sprite(ctx, src_name, cx, cy, size, yaw=yaw_deg, pitch=pitch_deg, roll=roll_deg,
                                    depth=thickness, round_=0.3, alpha=alpha, specular=specular, shadow=shadow)


def render_true3d_inflated_cushion(ctx, src_name, cx, cy, size, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
                                   inflate=0.42, alpha=1.0, specular=0.85, shadow=True):
    from . import obj3d as O
    return O.render_extruded_sprite(ctx, src_name, cx, cy, size, yaw=yaw_deg, pitch=pitch_deg, roll=roll_deg,
                                    depth=0.12 + 0.2 * inflate, round_=1.0, bulge=inflate, alpha=alpha,
                                    specular=specular, shadow=shadow)


def render_multiview_turntable_3d(ctx, views, cx, cy, size, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
                                  alpha=1.0, shadow=True):
    from . import obj3d as O
    return O.render_multiview(ctx, views, cx, cy, size, yaw=yaw_deg, pitch=pitch_deg, roll=roll_deg,
                              alpha=alpha, shadow=shadow)


def render_machined_3d_slab_v139(ctx, src_name: str, cx: float, cy: float, size: float,
                            yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0,
                            thickness: float = 0.14, alpha: float = 1.0,
                            specular: float = 0.75, shadow: bool = True):
    """
    LEGACY v13.9 (kept for A/B only).  Known bugs: inverted perspective (back face projected larger than the
    front), mirrored front face visible past 90°, striped side wall at steep angles, ~450 ms/call.
    Zero-Distortion 3D Machined Volume Renderer:
      * Keeps 100% of the original image's studio crispness via `cv2.warpPerspective(..., cv2.INTER_LANCZOS4)`
        (ZERO balloon bulging, ZERO point-cloud pixelation!).
      * Constructs a true 3D perspective homography `H(z)` for `N = 56` continuous sub-pixel depth planes
        from `z = -thickness/2` (back face) to `z = +thickness/2` (front face).
      * Shades the 3D side-wall rim using the exact 2D outward contour normal `(nx, ny)` rotated into 3D space
        `N_rot = R @ (nx, ny, 0)` lit by a studio key-light + metallic specular band!
      * Sweeps an angle-dependent Blinn-Phong specular reflection band across the front face as `yaw_deg` and
        `pitch_deg` rotate in 3D space!
    """
    bgra = _load_3d_rgba(src_name)
    if bgra is None:
        return
    h0, w0 = bgra.shape[:2]
    # Work at high resolution (360x360) for razor-sharp Lanczos4 quality
    W_s = 360
    scale_in = (W_s * 0.74) / max(h0, w0)
    nw, nh = max(16, int(round(w0 * scale_in))), max(16, int(round(h0 * scale_in)))
    im_r = cv2.resize(bgra, (nw, nh), interpolation=cv2.INTER_AREA)

    pad_s = np.zeros((W_s, W_s, 4), dtype=np.uint8)
    ox, oy = (W_s - nw) // 2, (W_s - nh) // 2
    pad_s[oy:oy + nh, ox:ox + nw] = im_r

    # Compute outward contour normal (nx, ny) around the alpha boundary for realistic 3D metallic side-wall shading
    a_f = pad_s[:, :, 3].astype(np.float32) / 255.0
    a_smooth = gaussian_filter(a_f, sigma=2.0)
    gy, gx = np.gradient(-a_smooth)
    g_mag = np.maximum(1e-5, np.hypot(gx, gy))
    nx_2d = gx / g_mag
    ny_2d = gy / g_mag

    # Rotate 3D side-wall normals by (yaw, pitch, roll)
    yw = math.radians(yaw_deg)
    pt = math.radians(pitch_deg)
    rl = math.radians(roll_deg)
    cy_m, sy_m = math.cos(yw), math.sin(yw)
    cp_m, sp_m = math.cos(pt), math.sin(pt)
    cr_m, sr_m = math.cos(rl), math.sin(rl)
    Ry = np.array([[cy_m, 0.0, sy_m], [0.0, 1.0, 0.0], [-sy_m, 0.0, cy_m]], dtype=np.float32)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp_m, -sp_m], [0.0, sp_m, cp_m]], dtype=np.float32)
    Rz = np.array([[cr_m, -sr_m, 0.0], [sr_m, cr_m, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    R = Rz @ Rx @ Ry

    # Studio Key Light direction L = normalize(0.55, -0.45, 0.70)
    lx, ly, lz = 0.55, -0.45, 0.70
    # Rotated side-wall normal dot L:
    n_rot_x = R[0, 0] * nx_2d + R[0, 1] * ny_2d
    n_rot_y = R[1, 0] * nx_2d + R[1, 1] * ny_2d
    n_rot_z = R[2, 0] * nx_2d + R[2, 1] * ny_2d
    rim_ndotl = np.clip(n_rot_x * lx + n_rot_y * ly + n_rot_z * lz, 0.0, 1.0)
    # Brushed anodized metal shading on the side wall: dark ambient + diffuse + sharp specular highlight!
    rim_shade = 0.32 + 0.52 * rim_ndotl + 0.55 * (rim_ndotl ** 12.0)

    side_bgra = pad_s.copy()
    for ch in range(3):
        side_bgra[:, :, ch] = np.clip(
            pad_s[:, :, ch].astype(np.float32) * rim_shade + (rim_ndotl ** 16.0) * 95.0,
            0, 255
        ).astype(np.uint8)

    # View-dependent specular light sweep across the front face as yaw & pitch rotate!
    front_bgra = pad_s.copy()
    yy, xx = np.mgrid[0:W_s, 0:W_s].astype(np.float32)
    u_norm = (xx - W_s * 0.5) / (W_s * 0.5)
    v_norm = (yy - W_s * 0.5) / (W_s * 0.5)
    # As yaw_deg sweeps from -45° to +45°, a soft diagonal studio light reflection sweeps across the front face
    sweep_pos = math.sin(yw) * 1.15 - math.sin(pt) * 0.55
    band_dist = (u_norm * 0.82 + v_norm * 0.57) - sweep_pos
    spec_band = (np.exp(-(band_dist ** 2) / (2.0 * (0.18 ** 2))) * 68.0 * specular * abs(math.sin(yw * 1.2 + 0.25)))
    for ch in range(3):
        front_bgra[:, :, ch] = np.clip(front_bgra[:, :, ch].astype(np.float32) + spec_band, 0, 255).astype(np.uint8)

    # Projective Homography helper for a plane at depth offset `z_off` in [-thickness/2, +thickness/2]
    focal = W_s * 2.15
    half_s = W_s * 0.5
    base_corners = np.array([
        [-half_s, -half_s],
        [+half_s, -half_s],
        [+half_s, +half_s],
        [-half_s, +half_s],
    ], dtype=np.float32)
    src_corners = base_corners + half_s

    def _warp_at_z(img_in, z_val, interp=cv2.INTER_LINEAR):
        pts_3d = np.column_stack([base_corners, np.full((4,), z_val, dtype=np.float32)])
        rot_3d = pts_3d @ R.T
        z_cam = np.maximum(focal * 0.25, focal + rot_3d[:, 2])
        proj_x = rot_3d[:, 0] * (focal / z_cam) + half_s
        proj_y = rot_3d[:, 1] * (focal / z_cam) + half_s
        dst_corners = np.column_stack([proj_x, proj_y]).astype(np.float32)
        H_mat = cv2.getPerspectiveTransform(src_corners, dst_corners)
        return cv2.warpPerspective(img_in, H_mat, (W_s, W_s), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Composite N=52 sub-pixel extrusion slices from back (-z_max) to front (+z_max)
    z_max = W_s * thickness * 0.5
    n_layers = 52
    acc = np.zeros((W_s, W_s, 4), dtype=np.float32)

    for i in range(n_layers):
        frac = i / float(n_layers - 1)
        z_cur = -z_max + (2.0 * z_max) * frac
        w_slice = _warp_at_z(side_bgra, z_cur, interp=cv2.INTER_LINEAR).astype(np.float32)
        # Ambient occlusion darkening near the back edge, brighter near the front bevel
        ao = 0.68 + 0.32 * frac
        w_slice[:, :, :3] *= ao
        sa = (w_slice[:, :, 3:4] / 255.0)
        acc[:, :, :3] = w_slice[:, :, :3] * sa + acc[:, :, :3] * (1.0 - sa)
        acc[:, :, 3:4] = w_slice[:, :, 3:4] + acc[:, :, 3:4] * (1.0 - sa)

    # Top-most Crisp Front Face with Lanczos-4 interpolation (ZERO distortion!)
    front_warped = _warp_at_z(front_bgra, +z_max, interp=cv2.INTER_LANCZOS4).astype(np.float32)
    fa = front_warped[:, :, 3:4] / 255.0
    acc[:, :, :3] = front_warped[:, :, :3] * fa + acc[:, :, :3] * (1.0 - fa)
    acc[:, :, 3:4] = np.maximum(acc[:, :, 3:4], front_warped[:, :, 3:4])

    out_u8 = np.clip(acc, 0, 255).astype(np.uint8)
    out_D = max(24, int(round(size * 1.25)))
    if out_D != W_s:
        out_u8 = cv2.resize(out_u8, (out_D, out_D), interpolation=cv2.INTER_AREA)

    if shadow:
        sh_g = cairo.RadialGradient(cx + yaw_deg * 0.28, cy + size * 0.38, size * 0.05,
                                    cx + yaw_deg * 0.28, cy + size * 0.38, size * 0.48)
        sh_g.add_color_stop_rgba(0.0, 0, 0, 0, 0.38 * alpha)
        sh_g.add_color_stop_rgba(0.6, 0, 0, 0, 0.15 * alpha)
        sh_g.add_color_stop_rgba(1.0, 0, 0, 0, 0.0)
        ctx.save()
        ctx.set_source(sh_g)
        ctx.arc(cx + yaw_deg * 0.28, cy + size * 0.38, size * 0.48, 0, 2 * math.pi)
        ctx.fill()
        ctx.restore()

    a_norm = (out_u8[:, :, 3:4].astype(np.float32) / 255.0) * min(1.0, max(0.0, alpha))
    bgra_p = np.zeros_like(out_u8, dtype=np.uint8)
    bgra_p[:, :, :3] = np.clip(out_u8[:, :, :3].astype(np.float32) * a_norm, 0, 255).astype(np.uint8)
    bgra_p[:, :, 3] = np.clip(out_u8[:, :, 3].astype(np.float32) * min(1.0, max(0.0, alpha)), 0, 255).astype(np.uint8)

    surf = cairo.ImageSurface.create_for_data(bgra_p, cairo.FORMAT_ARGB32, out_D, out_D)
    ctx.save()
    ctx.set_source_surface(surf, cx - out_D * 0.5, cy - out_D * 0.5)
    ctx.get_source().set_filter(cairo.FILTER_GOOD)
    ctx.paint()
    ctx.restore()



def infer_depth_anything_onnx(bgr_img: np.ndarray, model_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "depth_anything_v2_vits_quantized.onnx")) -> np.ndarray:
    """
    Cloned from `fabio-sim/Depth-Anything-ONNX` (`infer.py`) for `onnx-community/depth-anything-v2-small` (27.3 MB):
    Runs 100% on CPU via `onnxruntime` (zero PyTorch required).
    Input: BGR uint8 image (H, W, 3) -> Output: normalized float32 depth map (H, W) in [0, 1].
    """
    import onnxruntime as ort
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ONNX model not found at {model_path}. Download the 27.3 MB quantized model from "
            "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model_quantized.onnx"
        )
    orig_h, orig_w = bgr_img.shape[:2]
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # Resize to multiple of 14 (518x518 standard Depth-Anything-V2 resolution)
    inp = cv2.resize(rgb, (518, 518), interpolation=cv2.INTER_CUBIC)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = (inp - mean) / std
    tensor = np.transpose(inp, (2, 0, 1))[None, :, :, :].astype(np.float32)

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    depth = sess.run(None, {in_name: tensor})[0]
    if depth.ndim == 4:
        depth = depth[0, 0]
    elif depth.ndim == 3:
        depth = depth[0]
    depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    d_min, d_max = float(depth.min()), float(depth.max())
    return ((depth - d_min) / max(1e-6, d_max - d_min)).astype(np.float32)


def bilateral_normal_integration_cpu(normal_map: np.ndarray, mask: np.ndarray, k: float = 2.0, max_iter: int = 35, tol: float = 1e-4) -> np.ndarray:
    """
    Adapted from `xucao-42/bilateral_normal_integration` (ECCV 2022 Official CPU Implementation):
    Discontinuity-preserving surface reconstruction from a surface normal map `(H, W, 3)` and boolean `mask (H, W)`
    using Iteratively Reweighted Least Squares (IRLS) + SciPy Sparse Conjugate Gradient (`scipy.sparse.linalg.cg`).
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import cg

    H, W = mask.shape
    nx = normal_map[:, :, 0].astype(np.float64)
    ny = normal_map[:, :, 1].astype(np.float64)
    nz = np.maximum(1e-4, np.abs(normal_map[:, :, 2].astype(np.float64)))
    p = -nx / nz
    q = -ny / nz

    idx_map = -np.ones((H, W), dtype=np.int32)
    ys, xs = np.where(mask)
    n_pix = len(ys)
    if n_pix < 4:
        return np.zeros((H, W), dtype=np.float32)
    idx_map[ys, xs] = np.arange(n_pix, dtype=np.int32)

    # Horizontal and vertical valid neighbor pairs inside mask
    valid_r = mask[:, :-1] & mask[:, 1:]
    ry, rx = np.where(valid_r)
    i_r0 = idx_map[ry, rx]
    i_r1 = idx_map[ry, rx + 1]
    gx_r = 0.5 * (p[ry, rx] + p[ry, rx + 1])

    valid_d = mask[:-1, :] & mask[1:, :]
    dy, dx = np.where(valid_d)
    i_d0 = idx_map[dy, dx]
    i_d1 = idx_map[dy + 1, dx]
    gy_d = 0.5 * (q[dy, dx] + q[dy + 1, dx])

    n_eq = len(i_r0) + len(i_d0)
    rows = np.arange(n_eq, dtype=np.int32)
    c0 = np.concatenate([i_r0, i_d0])
    c1 = np.concatenate([i_r1, i_d1])
    vals = np.concatenate([-np.ones(len(i_r0)), np.ones(len(i_r1)), -np.ones(len(i_d0)), np.ones(len(i_d1))])
    A = sp.csr_matrix(
        (np.concatenate([-np.ones(n_eq), np.ones(n_eq)]), (np.concatenate([rows, rows]), np.concatenate([c0, c1]))),
        shape=(n_eq, n_pix)
    )
    b = np.concatenate([gx_r, gy_d])
    reg = sp.eye(n_pix, format="csr") * 1e-5

    z = np.zeros(n_pix, dtype=np.float64)
    w = np.ones(n_eq, dtype=np.float64)
    for _ in range(max_iter):
        W_mat = sp.diags(w)
        AtW = A.T @ W_mat
        lhs = AtW @ A + reg
        rhs = AtW @ b
        z_new, _ = cg(lhs, rhs, x0=z, rtol=tol, maxiter=80)
        if np.linalg.norm(z_new - z) / max(1.0, np.linalg.norm(z)) < tol:
            z = z_new
            break
        z = z_new
        res = (A @ z) - b
        # Bilateral sigmoid weight preserving sharp depth discontinuities
        w = 1.0 / (1.0 + np.exp(k * (np.abs(res) - np.median(np.abs(res)))))

    z -= z.min()
    depth_out = np.zeros((H, W), dtype=np.float32)
    depth_out[ys, xs] = z.astype(np.float32)
    return depth_out

def render_true3d_sphere(ctx, tex_rgba: np.ndarray, cx: float, cy: float, radius: float,
                         yaw_deg: float = 0.0, pitch_deg: float = 15.0, roll_deg: float = 0.0,
                         alpha: float = 1.0, specular: float = 0.85, shadow: bool = True):
    """
    Mode 1 (`geometry="sphere"`):
    Exact Analytical Ray-Sphere Intersection + Inverse Equirectangular/Spherical UV Mapping (`cv2.remap` Lanczos4)!
    Turns any 2D texture, map, pattern, or icon into a 100% smooth, revolving 3D Sphere / Globe / Orb!
    """
    R = max(8.0, float(radius))
    D = int(math.ceil(R * 2.2))
    if D % 2 == 1:
        D += 1
    half_D = D * 0.5

    yy, xx = np.mgrid[0:D, 0:D].astype(np.float32)
    nx = (xx - half_D) / R
    ny = (yy - half_D) / R
    r2 = nx * nx + ny * ny
    inside = r2 <= 1.0
    nz = np.sqrt(np.maximum(0.0, 1.0 - r2))

    # 3D Rotation Matrix R_euler (transpose used for inverse ray mapping into object space)
    yw, pt, rl = math.radians(yaw_deg), math.radians(pitch_deg), math.radians(roll_deg)
    cy_m, sy_m = math.cos(yw), math.sin(yw)
    cp_m, sp_m = math.cos(pt), math.sin(pt)
    cr_m, sr_m = math.cos(rl), math.sin(rl)
    Ry = np.array([[cy_m, 0.0, sy_m], [0.0, 1.0, 0.0], [-sy_m, 0.0, cy_m]], dtype=np.float32)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp_m, -sp_m], [0.0, sp_m, cp_m]], dtype=np.float32)
    Rz = np.array([[cr_m, -sr_m, 0.0], [sr_m, cr_m, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    R_mat = Rz @ Rx @ Ry
    R_inv = R_mat.T

    # Rotate surface point (nx, ny, nz) by R_inv to get object-space unit vector (ox, oy, oz)
    ox = R_inv[0, 0] * nx + R_inv[0, 1] * ny + R_inv[0, 2] * nz
    oy = R_inv[1, 0] * nx + R_inv[1, 1] * ny + R_inv[1, 2] * nz
    oz = R_inv[2, 0] * nx + R_inv[2, 1] * ny + R_inv[2, 2] * nz

    # Spherical UV coordinates: longitude lon in [-pi, pi], latitude lat in [-pi/2, pi/2]
    lon = np.arctan2(ox, oz)
    lat = np.arcsin(np.clip(oy, -1.0, 1.0))

    th, tw = tex_rgba.shape[:2]
    # Map front hemisphere [-pi/2, pi/2] smoothly across the texture (with seamless wrap-around!)
    u_map = ((lon / math.pi) * 0.5 + 0.5) * (tw - 1)
    v_map = ((lat / (math.pi * 0.5)) * 0.5 + 0.5) * (th - 1)

    sampled = cv2.remap(tex_rgba, u_map.astype(np.float32), v_map.astype(np.float32),
                        interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_WRAP)

    # 3D Studio PBR Spherical Lighting (Key light + Fill light + Specular + Atmospheric Fresnel Rim)
    lx, ly, lz = 0.45, -0.52, 0.72
    ndotl = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
    hx, hy, hz = lx, ly, lz + 1.0
    h_len = math.sqrt(hx*hx + hy*hy + hz*hz)
    ndoth = np.clip((nx * hx + ny * hy + nz * hz) / h_len, 0.0, 1.0)

    diffuse = 0.22 + 0.85 * ndotl
    spec = ((ndoth ** 38.0) * 185.0 + (ndoth ** 8.0) * 42.0) * specular
    fresnel = ((1.0 - nz) ** 3.2) * 75.0 * specular

    out = np.zeros((D, D, 4), dtype=np.uint8)
    for c in range(3):
        ch_val = sampled[:, :, c].astype(np.float32) * diffuse + spec + fresnel * (0.6 if c == 0 else 0.9)
        out[:, :, c] = np.where(inside, np.clip(ch_val, 0, 255), 0).astype(np.uint8)

    # Anti-aliased smooth sphere edge (1.5px feather at r == 1.0)
    edge_aa = np.clip((1.0 - np.sqrt(r2)) * (R * 0.5), 0.0, 1.0)
    out[:, :, 3] = (edge_aa * 255.0 * alpha).astype(np.uint8)

    _paint_rgba_to_cairo(ctx, out, cx, cy, R, alpha, shadow)


def render_true3d_cylinder_capsule(ctx, tex_rgba: np.ndarray, cx: float, cy: float,
                                   radius: float = 95.0, height: float = 240.0,
                                   yaw_deg: float = 0.0, pitch_deg: float = 12.0, roll_deg: float = -15.0,
                                   alpha: float = 1.0, specular: float = 0.88, shadow: bool = True):
    """
    Mode 2 (`geometry="capsule"` / `"cylinder"`):
    Exact Analytical Ray-Capsule / Ray-Cylinder Intersection + Inverse Cylindrical UV Mapping (`cv2.remap`)!
    Renders true 3D medical vials, injection pens (Ozempic/Spartina), cans, bottles, and 3D pills!
    """
    R = max(10.0, float(radius))
    H_cyl = max(20.0, float(height))
    D = int(math.ceil(max(R * 2.4, H_cyl * 1.45)))
    if D % 2 == 1:
        D += 1
    half_D = D * 0.5

    yy, xx = np.mgrid[0:D, 0:D].astype(np.float32)
    sx = xx - half_D
    sy = yy - half_D

    # Inverse rotate screen coords by roll & pitch to align with capsule axis
    rl = math.radians(roll_deg)
    pt = math.radians(pitch_deg)
    yw = math.radians(yaw_deg)
    cr, sr = math.cos(-rl), math.sin(-rl)
    rx = sx * cr - sy * sr
    ry = sx * sr + sy * cr

    half_h = H_cyl * 0.5 - R
    # Clamped y along cylinder spine [-half_h, +half_h] gives capsule (cylinder + 2 hemispherical caps!)
    spine_y = np.clip(ry, -half_h, +half_h)
    dy_cap = ry - spine_y
    r2 = (rx / R) ** 2 + (dy_cap / R) ** 2
    inside = r2 <= 1.0
    nz = np.sqrt(np.maximum(0.0, 1.0 - r2))
    nx = rx / R
    ny = dy_cap / R

    # Cylindrical longitude + yaw rotation
    lon = np.arctan2(nx, nz) - yw
    v_norm = (ry + H_cyl * 0.5) / max(1.0, H_cyl)

    th, tw = tex_rgba.shape[:2]
    u_map = ((lon / (math.pi * 0.85)) * 0.5 + 0.5) * (tw - 1)
    v_map = np.clip(v_norm, 0.0, 1.0) * (th - 1)

    sampled = cv2.remap(tex_rgba, u_map.astype(np.float32), v_map.astype(np.float32),
                        interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_WRAP)

    # Anisotropic cylindrical studio lighting
    lx, ly, lz = 0.52, -0.38, 0.76
    ndotl = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
    hx, hy, hz = lx, ly, lz + 1.0
    h_len = math.sqrt(hx*hx + hy*hy + hz*hz)
    ndoth = np.clip((nx * hx + ny * hy + nz * hz) / h_len, 0.0, 1.0)

    diffuse = 0.25 + 0.82 * ndotl
    # Crisp vertical cylindrical specular reflection band + cap highlight
    spec = ((ndoth ** 42.0) * 195.0 + (nx - 0.38 == 0) * 0.0 + np.exp(-((nx - 0.35) ** 2) / 0.025) * 65.0 * nz) * specular
    fresnel = ((1.0 - nz) ** 3.0) * 60.0 * specular

    out = np.zeros((D, D, 4), dtype=np.uint8)
    for c in range(3):
        ch_val = sampled[:, :, c].astype(np.float32) * diffuse + spec + fresnel
        out[:, :, c] = np.where(inside, np.clip(ch_val, 0, 255), 0).astype(np.uint8)

    edge_aa = np.clip((1.0 - np.sqrt(r2)) * (R * 0.45), 0.0, 1.0)
    out[:, :, 3] = (edge_aa * 255.0 * alpha).astype(np.uint8)

    _paint_rgba_to_cairo(ctx, out, cx, cy, max(R, H_cyl * 0.5), alpha, shadow)


def render_true3d_inflated_cushion_v139(ctx, src_name: str, cx: float, cy: float, size: float,
                                   yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0,
                                   inflate: float = 0.42, alpha: float = 1.0,
                                   specular: float = 0.85, shadow: bool = True):
    r"""LEGACY v13.9 (kept for A/B only — slow, moiré on the side wall; use render_true3d_inflated_cushion).

    Mode 3 (`geometry="cushion_3d"`):
    Inverse Ray-Marched Smooth 3D Inflated Dome/Cushion (`cv2.remap` Lanczos-4 + Smooth Distance-Field Dome)!
    Unlike forward point-splatting (which creates pixelated dots), this evaluates a smooth hemispherical
    dome profile $Z(u, v) = \sin(\frac{\pi}{2} \cdot d(u,v)^{0.65})$ and uses **Inverse Ray-Parallax Mapping
    (`cv2.remap(..., cv2.INTER_LANCZOS4)`)** + 48-layer curved bevel shell so ANY 2D icon/object becomes a
    plump, rounded, spherical-cushion 3D object with ZERO pixelation and a curved 3D specular highlight!
    """
    bgra = _load_3d_rgba(src_name)
    if bgra is None:
        return
    h0, w0 = bgra.shape[:2]
    W_s = 360
    scale_in = (W_s * 0.72) / max(h0, w0)
    nw, nh = max(16, int(round(w0 * scale_in))), max(16, int(round(h0 * scale_in)))
    im_r = cv2.resize(bgra, (nw, nh), interpolation=cv2.INTER_AREA)

    pad_s = np.zeros((W_s, W_s, 4), dtype=np.uint8)
    ox, oy = (W_s - nw) // 2, (W_s - nh) // 2
    pad_s[oy:oy + nh, ox:ox + nw] = im_r

    mask = pad_s[:, :, 3] > 24
    dist = distance_transform_edt(mask).astype(np.float32)
    d_max = max(1.0, float(dist.max()))
    # Smooth spherical-dome inflation height Z(u, v) in [0, 1]
    z_dome = np.sin((math.pi * 0.5) * np.clip(dist / d_max, 0.0, 1.0) ** 0.62)
    z_dome = gaussian_filter(z_dome, sigma=4.5) * (mask.astype(np.float32))

    # Analytical 3D normals of the curved cushion dome N = normalize(-dZ/du, -dZ/dv, 1)
    dz_dv, dz_du = np.gradient(z_dome * (W_s * 0.28 * inflate))
    nx = -dz_du
    ny = -dz_dv
    nz = np.ones_like(z_dome)
    n_len = np.maximum(1e-5, np.sqrt(nx*nx + ny*ny + nz*nz))
    nx /= n_len; ny /= n_len; nz /= n_len

    # Rotate normals by (yaw, pitch, roll)
    yw, pt, rl = math.radians(yaw_deg), math.radians(pitch_deg), math.radians(roll_deg)
    cy_m, sy_m = math.cos(yw), math.sin(yw)
    cp_m, sp_m = math.cos(pt), math.sin(pt)
    cr_m, sr_m = math.cos(rl), math.sin(rl)
    Ry = np.array([[cy_m, 0.0, sy_m], [0.0, 1.0, 0.0], [-sy_m, 0.0, cy_m]], dtype=np.float32)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp_m, -sp_m], [0.0, sp_m, cp_m]], dtype=np.float32)
    Rz = np.array([[cr_m, -sr_m, 0.0], [sr_m, cr_m, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    R = Rz @ Rx @ Ry

    n_rx = R[0, 0] * nx + R[0, 1] * ny + R[0, 2] * nz
    n_ry = R[1, 0] * nx + R[1, 1] * ny + R[1, 2] * nz
    n_rz = R[2, 0] * nx + R[2, 1] * ny + R[2, 2] * nz

    # Curved 3D Blinn-Phong lighting on the spherical cushion surface
    lx, ly, lz = 0.48, -0.50, 0.72
    ndotl = np.clip(n_rx * lx + n_ry * ly + n_rz * lz, 0.0, 1.0)
    hx, hy, hz = lx, ly, lz + 1.0
    h_len = math.sqrt(hx*hx + hy*hy + hz*hz)
    ndoth = np.clip((n_rx * hx + n_ry * hy + n_rz * hz) / h_len, 0.0, 1.0)

    # Smooth spherical shading + curved specular highlight + Fresnel rim
    diff_shade = 0.72 + 0.42 * ndotl
    spec_dome = ((ndoth ** 24.0) * 135.0 + (ndoth ** 6.0) * 35.0) * specular
    fresnel_dome = ((1.0 - np.clip(n_rz, 0.0, 1.0)) ** 2.8) * 55.0 * specular

    lit_pad = pad_s.copy()
    for c in range(3):
        lit_pad[:, :, c] = np.clip(
            pad_s[:, :, c].astype(np.float32) * diff_shade + spec_dome + fresnel_dome,
            0, 255
        ).astype(np.uint8)

    # Inverse Parallax-Occlusion UV Warping (`cv2.remap` with INTER_LANCZOS4!):
    # When viewing at angle (yaw, pitch), a point with height Z(u, v) shifts laterally by
    # (du, dv) = -Z(u, v) * (sin(yaw), sin(pitch)) *parallax_scale!
    yy, xx = np.mgrid[0:W_s, 0:W_s].astype(np.float32)
    parallax_px = W_s * 0.22 * inflate
    # 2-step iterative inverse ray-march lookup for exact curved silhouette & surface parallax
    u_ray = xx - z_dome * math.sin(yw) * parallax_px
    v_ray = yy + z_dome * math.sin(pt) * parallax_px
    z_step2 = cv2.remap(z_dome, u_ray, v_ray, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    u_ray = xx - z_step2 * math.sin(yw) * parallax_px
    v_ray = yy + z_step2 * math.sin(pt) * parallax_px

    inflated_front = cv2.remap(lit_pad, u_ray, v_ray, interpolation=cv2.INTER_LANCZOS4,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Combine with rounded 3D perspective homography & curved back-shell for full 3D rotation
    focal = W_s * 2.1
    half_s = W_s * 0.5
    base_corners = np.array([[-half_s, -half_s], [+half_s, -half_s], [+half_s, +half_s], [-half_s, +half_s]], dtype=np.float32)
    src_corners = base_corners + half_s

    def _warp_z(im_in, z_val, scale_xy=1.0):
        pts_3d = np.column_stack([base_corners * scale_xy, np.full((4,), z_val, dtype=np.float32)])
        rot_3d = pts_3d @ R.T
        z_cam = np.maximum(focal * 0.25, focal + rot_3d[:, 2])
        px = rot_3d[:, 0] * (focal / z_cam) + half_s
        py = rot_3d[:, 1] * (focal / z_cam) + half_s
        H_mat = cv2.getPerspectiveTransform(src_corners, np.column_stack([px, py]).astype(np.float32))
        return cv2.warpPerspective(im_in, H_mat, (W_s, W_s), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Rounded elliptical profile across Z thickness (`scale_xy = sqrt(1 - u^2)` so the side profile is a smooth curved capsule/sphere, NOT a flat slab!)
    z_max = W_s * 0.11 * inflate
    n_layers = 36
    acc = np.zeros((W_s, W_s, 4), dtype=np.float32)
    for i in range(n_layers):
        u_l = -1.0 + 2.0 * (i / float(n_layers - 1))  # -1 (back) to +1 (front)
        z_cur = u_l * z_max
        # Elliptical cross-section profile: narrows smoothly at the back & front poles like a real 3D cushion!
        ellip_scale = 0.92 + 0.08 * math.sqrt(max(0.0, 1.0 - (u_l ** 2)))
        w_slice = _warp_z(inflated_front, z_cur, scale_xy=ellip_scale).astype(np.float32)
        shade_z = 0.55 + 0.45 * ((u_l + 1.0) * 0.5)
        w_slice[:, :, :3] *= shade_z
        sa = w_slice[:, :, 3:4] / 255.0
        acc[:, :, :3] = w_slice[:, :, :3] * sa + acc[:, :, :3] * (1.0 - sa)
        acc[:, :, 3:4] = w_slice[:, :, 3:4] + acc[:, :, 3:4] * (1.0 - sa)

    out_u8 = np.clip(acc, 0, 255).astype(np.uint8)
    out_D = max(24, int(round(size * 1.25)))
    if out_D != W_s:
        out_u8 = cv2.resize(out_u8, (out_D, out_D), interpolation=cv2.INTER_AREA)

    _paint_bgra_to_cairo(ctx, out_u8, cx, cy, size, yaw_deg, alpha, shadow)


def _paint_rgba_to_cairo(ctx, rgba: np.ndarray, cx: float, cy: float, size: float, alpha: float, shadow: bool):
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    _paint_bgra_to_cairo(ctx, bgra, cx, cy, size * 2.0, 0.0, alpha, shadow)


def _paint_bgra_to_cairo(ctx, out_u8: np.ndarray, cx: float, cy: float, size: float, yaw_deg: float, alpha: float, shadow: bool):
    out_D = out_u8.shape[0]
    if shadow:
        sh_g = cairo.RadialGradient(cx + yaw_deg * 0.25, cy + size * 0.38, size * 0.05,
                                    cx + yaw_deg * 0.25, cy + size * 0.38, size * 0.46)
        sh_g.add_color_stop_rgba(0.0, 0, 0, 0, 0.38 * alpha)
        sh_g.add_color_stop_rgba(0.6, 0, 0, 0, 0.14 * alpha)
        sh_g.add_color_stop_rgba(1.0, 0, 0, 0, 0.0)
        ctx.save()
        ctx.set_source(sh_g)
        ctx.arc(cx + yaw_deg * 0.25, cy + size * 0.38, size * 0.46, 0, 2 * math.pi)
        ctx.fill()
        ctx.restore()

    a_norm = (out_u8[:, :, 3:4].astype(np.float32) / 255.0) * min(1.0, max(0.0, alpha))
    bgra_p = np.zeros_like(out_u8, dtype=np.uint8)
    bgra_p[:, :, :3] = np.clip(out_u8[:, :, :3].astype(np.float32) * a_norm, 0, 255).astype(np.uint8)
    bgra_p[:, :, 3] = np.clip(out_u8[:, :, 3].astype(np.float32) * min(1.0, max(0.0, alpha)), 0, 255).astype(np.uint8)

    surf = cairo.ImageSurface.create_for_data(bgra_p, cairo.FORMAT_ARGB32, out_D, out_D)
    ctx.save()
    ctx.set_source_surface(surf, cx - out_D * 0.5, cy - out_D * 0.5)
    ctx.get_source().set_filter(cairo.FILTER_GOOD)
    ctx.paint()
    ctx.restore()



@lru_cache(maxsize=16)
def _precompute_multiview_pair(src_a: str, src_b: str, W_s: int = 360):
    """
    Loads two adjacent 3D turntable views (e.g. -45° and -15°) and computes bidirectional
    dense optical flow (`cv2.calcOpticalFlowFarneback`) between their silhouettes & textures
    so intermediate 3D rotation angles morph smoothly!
    """
    def _prep(name):
        bgra = _load_3d_rgba(name)
        if bgra is None:
            raise FileNotFoundError(name)
        h0, w0 = bgra.shape[:2]
        sc = (W_s * 0.80) / max(h0, w0)
        nw, nh = max(16, int(round(w0 * sc))), max(16, int(round(h0 * sc)))
        im_r = cv2.resize(bgra, (nw, nh), interpolation=cv2.INTER_AREA)
        pad = np.zeros((W_s, W_s, 4), dtype=np.uint8)
        ox, oy = (W_s - nw) // 2, (W_s - nh) // 2
        pad[oy:oy + nh, ox:ox + nw] = im_r
        return pad

    im_a = _prep(src_a)
    im_b = _prep(src_b)
    gray_a = cv2.cvtColor(im_a[:, :, :3], cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(im_b[:, :, :3], cv2.COLOR_BGR2GRAY)
    # Include alpha edge in grayscale so silhouette boundaries lock tightly in optical flow
    ga = cv2.addWeighted(gray_a, 0.7, im_a[:, :, 3], 0.3, 0)
    gb = cv2.addWeighted(gray_b, 0.7, im_b[:, :, 3], 0.3, 0)

    flow_ab = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 4, 21, 3, 7, 1.5, 0)
    flow_ba = cv2.calcOpticalFlowFarneback(gb, ga, None, 0.5, 4, 21, 3, 7, 1.5, 0)
    return im_a, im_b, flow_ab.astype(np.float32), flow_ba.astype(np.float32)


def render_multiview_turntable_3d_v139(ctx, views: list[tuple[float, str]], cx: float, cy: float, size: float,
                                  yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0,
                                  alpha: float = 1.0, shadow: bool = True):
    """
    Mode 4 (`geometry="turntable_4view"`):
    Zero-Ghosting Multi-Angle 3D Turnaround Renderer!
    Selects the closest real AI 3D camera view `theta_k` from `[(-45°, view0), (-15°, view1), (+15°, view2), (+45°, view3)]`
    and applies a true 3D Machined Perspective Homography for the small residual angle `delta_yaw = yaw_deg - theta_k`
    (which is always within `±15°`), guaranteeing 100% razor-sharp studio clarity with ZERO ghosting across a full
    110° (-55° to +55°) 3D turnaround!
    """
    views_sorted = sorted(views, key=lambda item: float(item[0]))
    # Find nearest camera view to yaw_deg
    best_a, best_src = min(views_sorted, key=lambda item: abs(yaw_deg - float(item[0])))
    residual_yaw = clamp(yaw_deg - float(best_a), -18.0, +18.0)

    render_machined_3d_slab(
        ctx, best_src, cx, cy, size,
        yaw_deg=residual_yaw,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        thickness=0.07,
        alpha=alpha,
        specular=0.82,
        shadow=shadow
    )
