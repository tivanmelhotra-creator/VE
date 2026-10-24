"""
mograph.mocap.props — line-art props that live in the SAME 3D space as the mocap actors (so bodies occlude them
and the camera moves them correctly): open toy box, blocks, bench, floor line, candle + flame, script title.
"""
from __future__ import annotations

import math

import numpy as np

from ..human.lineart import MeshTopo, Part, Stroke3D


def smooth(u):
    u = min(1.0, max(0.0, u))
    return u * u * u * (u * (6 * u - 15) + 10)


def _roty(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def box(center, size, pid, yaw=0.0, weight=0.75):
    """open-top box → (occluder Part, edge strokes, corner points)."""
    w, h, d = [s / 2 for s in size]
    corners = np.array([[x, y, z] for y in (0, 2 * h) for z in (-d, d) for x in (-w, w)], float)
    P = corners @ _roty(yaw).T + np.asarray(center, float)
    quads = [(0, 1, 3, 2), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    tris = np.array([t for q in quads for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))], np.int32)
    part = Part("box", P.astype(np.float32), tris, MeshTopo(tris, len(P)), pid=pid, contour=False)
    edges = [(4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7), (0, 1), (2, 3), (0, 2), (1, 3)]
    strokes = [Stroke3D(np.linspace(P[a], P[b], 14), weight, "prop", 0.12) for a, b in edges]
    return part, strokes, P


def toys(P, seed=3):
    """little blocks + a ball peeking over the rim of the box."""
    rng = np.random.default_rng(seed)
    lo, hi = P.min(0), P.max(0)
    out = []
    for k in range(5):
        c = np.array([lo[0] + (k + 0.7) * (hi[0] - lo[0]) / 5.6, hi[1] - 0.2 + rng.uniform(0, 0.3),
                      rng.uniform(lo[2] + 0.5, hi[2] - 0.5)])
        s = rng.uniform(0.3, 0.45)
        if k == 2:
            th = np.linspace(0, 2 * math.pi, 28)
            out.append(Stroke3D(np.c_[c[0] + s * 1.2 * np.cos(th), c[1] + 0.3 + s * 1.2 * np.sin(th),
                                      np.full(28, c[2])], 0.55, "prop", 0.05))
            continue
        q = np.array([[-s, 0, -s], [s, 0, -s], [s, 0, s], [-s, 0, s], [-s, 0, -s]]) @ _roty(rng.uniform(0, 90)).T + c
        out.append(Stroke3D(q + [0, 2 * s, 0], 0.5, "prop", 0.05))
        for v in q[:4]:
            out.append(Stroke3D(np.array([v, v + [0, 2 * s, 0]]), 0.45, "prop", 0.05))
    return out


def floor_line(z, x0, x1, seed=1, weight=0.42):
    """the storyboard's loose hand-drawn ground line, ON the floor (bodies occlude it)."""
    xs = np.linspace(x0, x1, 160)
    ph = np.random.default_rng(seed).uniform(0, 6, 2)
    zz = z + 0.8 * np.sin(xs * 0.09 + ph[0]) + 0.35 * np.sin(xs * 0.23 + ph[1])
    return Stroke3D(np.c_[xs, np.full_like(xs, 0.01), zz], weight, "ground", 0.35)


def bench(x0, x1, z, h, d=2.6, pid=95):
    """simple bench: seat occluder + edges + 4 legs."""
    P = np.array([[x0, h, z - d / 2], [x1, h, z - d / 2], [x1, h, z + d / 2], [x0, h, z + d / 2]], float)
    tris = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    part = Part("bench", P.astype(np.float32), tris, MeshTopo(tris, 4), pid=pid, contour=False)
    st = [Stroke3D(np.linspace(P[a], P[b], 30), 0.7, "prop", 0.08) for a, b in ((0, 1), (3, 2), (0, 3), (1, 2))]
    for x in (x0 + 0.5, x1 - 0.5):
        for zz in (z - d / 2 + 0.3, z + d / 2 - 0.3):
            st.append(Stroke3D(np.array([[x, h, zz], [x, 0.02, zz]]), 0.6, "prop", 0.08))
    return part, st


def candle(ctx, cam, base, t, alpha, ink, r=0.55, h=2.4):
    import cairo
    th = np.linspace(0, 2 * math.pi, 64)

    def ring(y, rr=r):
        return cam.project(np.c_[base[0] + rr * np.cos(th), np.full(64, base[1] + y), base[2] + rr * np.sin(th)])[0]
    top, bot, plate = ring(h), ring(0.03), ring(0.01, r * 2.3)
    ctx.set_source_rgba(*ink, alpha)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_width(2.4)
    ctx.new_path()
    for p in top:
        ctx.line_to(*p)
    ctx.close_path()
    ctx.stroke()
    for idx in (bot[:, 0].argmin(), bot[:, 0].argmax()):
        ctx.move_to(*top[idx])
        ctx.line_to(*bot[idx])
        ctx.stroke()
    front = np.sin(th) > -0.02
    pts = bot[front][np.argsort(bot[front][:, 0])]
    ctx.new_path()
    for p in pts:
        ctx.line_to(*p)
    ctx.stroke()
    ctx.set_line_width(1.6)
    pts = plate[np.sin(th) > -0.02]
    pts = pts[np.argsort(pts[:, 0])]
    ctx.new_path()
    for p in pts:
        ctx.line_to(*p)
    ctx.stroke()
    wick = cam.project(np.array([[base[0], base[1] + h, base[2]], [base[0], base[1] + h + 0.35, base[2]]]))[0]
    ctx.set_line_width(2.0)
    ctx.move_to(*wick[0])
    ctx.line_to(*wick[1])
    ctx.stroke()
    px = abs(wick[1][1] - wick[0][1]) / 0.35
    fh = (1.2 + 0.12 * math.sin(t * 13.1) + 0.06 * math.sin(t * 29.7)) * px
    fw = 0.42 * px
    sway = 0.08 * px * math.sin(t * 7.3) + 0.04 * px * math.sin(t * 17.0)
    x0, y0 = wick[1][0], wick[1][1] + 0.12 * px
    g = cairo.RadialGradient(x0, y0 - fh * 0.45, 0, x0, y0 - fh * 0.45, fh * 2.8)
    g.add_color_stop_rgba(0, 1.0, 0.86, 0.58, 0.32 * alpha)
    g.add_color_stop_rgba(1, 1.0, 0.8, 0.5, 0.0)
    ctx.set_source(g)
    ctx.arc(x0, y0 - fh * 0.45, fh * 2.8, 0, 2 * math.pi)
    ctx.fill()
    ctx.set_source_rgba(1.0, 0.97, 0.9, alpha)
    ctx.move_to(x0, y0)
    ctx.curve_to(x0 + fw, y0 - fh * 0.15, x0 + fw * 0.6 + sway, y0 - fh * 0.7, x0 + sway * 1.6, y0 - fh)
    ctx.curve_to(x0 - fw * 0.6 + sway, y0 - fh * 0.7, x0 - fw, y0 - fh * 0.15, x0, y0)
    ctx.fill()


def script_title(ctx, t, t0, x, y, ink, lines=(("Happy Birthday", 76), ("my little sister", 58)), font="Dancing Script"):
    """handwritten title revealed left → right like a pen, then a small heart + flourish."""
    import cairo
    for k, (txt, size) in enumerate(lines):
        u = (t - t0 - k * 1.2) / 1.7
        if u <= 0:
            continue
        u = smooth(min(1.0, u))
        ctx.save()
        ctx.select_font_face(font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(size)
        e = ctx.text_extents(txt)
        tx, ty = x - e.width / 2 + k * 36, y + k * size * 1.15
        ctx.rectangle(tx - 12, ty - size * 1.3, (e.width + 34) * u, size * 2.1)
        ctx.clip()
        ctx.set_source_rgba(*ink, 1.0)
        ctx.move_to(tx, ty)
        ctx.show_text(txt)
        ctx.restore()
    u = smooth(min(1.0, max(0.0, (t - t0 - 3.0) / 0.9)))
    if u > 0:
        ctx.set_source_rgba(*ink, u)
        ctx.set_line_width(2.2)
        cx, cy, s = x + 40, y + lines[0][1] * 2.25, 12
        ctx.move_to(cx, cy + s)
        ctx.curve_to(cx - 2 * s, cy - 0.2 * s, cx - 0.9 * s, cy - 1.5 * s, cx, cy - 0.4 * s)
        ctx.curve_to(cx + 0.9 * s, cy - 1.5 * s, cx + 2 * s, cy - 0.2 * s, cx, cy + s)
        ctx.stroke()


# ================================================================================================ detailed props
def _poly(P, w=0.6, kind="prop", taper=0.06, closed=False):
    P = np.asarray(P, float)
    if closed:
        P = np.vstack([P, P[:1]])
    return Stroke3D(P, w, kind, taper)


def _quad_part(name, P, pid):
    tris = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    return Part(name, np.asarray(P, np.float32), tris, MeshTopo(tris, 4), pid=pid, contour=False)


def _mesh_part(name, P, quads, pid):
    tris = np.array([t for q in quads for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))], np.int32)
    return Part(name, np.asarray(P, np.float32), tris, MeshTopo(tris, len(P)), pid=pid, contour=False)


def snack_box(center, size=(4.2, 1.15, 3.0), yaw=0.0, lid_open=105.0, pid=90, seed=5, wall=0.12):
    """the storyboard's treats box: rounded rim, thick walls (double line), hinged lid standing open at the back,
    a 3×2 grid of compartments, and every compartment filled with cookies / wrapped candies / round biscuits.
    → (occluder parts, strokes)"""
    Rm = _roty(yaw)
    C = np.asarray(center, float)
    w, h, d = size[0] / 2, size[1], size[2] / 2

    def T(p):
        return (np.asarray(p, float) @ Rm.T) + C

    parts, st = [], []
    # outer shell with slightly rounded corners -------------------------------------------------
    r = 0.22
    ring = []
    for cx, cz, a0 in ((w - r, d - r, 0), (-w + r, d - r, 90), (-w + r, -d + r, 180), (w - r, -d + r, 270)):
        for a in np.linspace(a0, a0 + 90, 5):
            ring.append((cx + r * math.cos(math.radians(a)), cz + r * math.sin(math.radians(a))))
    ring = np.array(ring)
    inner = ring * np.array([(w - wall) / w, (d - wall) / d])
    n = len(ring)
    top_o = np.c_[ring[:, 0], np.full(n, h), ring[:, 1]]
    bot_o = np.c_[ring[:, 0], np.full(n, 0.02), ring[:, 1]]
    top_i = np.c_[inner[:, 0], np.full(n, h), inner[:, 1]]
    floor_i = np.c_[inner[:, 0], np.full(n, 0.18), inner[:, 1]]
    V = np.vstack([top_o, bot_o, top_i, floor_i])
    quads = []
    for i in range(n):
        j = (i + 1) % n
        quads.append((i, j, n + j, n + i))                               # outer wall
        quads.append((2 * n + i, 2 * n + j, 3 * n + j, 3 * n + i))       # inner wall
        quads.append((i, j, 2 * n + j, 2 * n + i))                       # rim
    fl = [3 * n + i for i in range(n)]
    for i in range(1, n - 1):
        quads.append((fl[0], fl[i], fl[i + 1], fl[i + 1]))
    parts.append(_mesh_part("box", T(V), quads, pid))
    st.append(_poly(T(top_o), 0.85, closed=True))
    st.append(_poly(T(top_i), 0.55, closed=True))
    st.append(_poly(T(bot_o), 0.8, closed=True))
    for i in range(0, n, 5):                                             # rounded corner verticals
        st.append(_poly(T([top_o[i + 2], bot_o[i + 2]]), 0.75))
    # a soft band around the box (embossed line) -----------------------------------------------
    st.append(_poly(T(np.c_[ring[:, 0] * 1.003, np.full(n, h * 0.78), ring[:, 1] * 1.003]), 0.35, closed=True))
    # dividers: 3 × 2 compartments --------------------------------------------------------------
    iw, idp = w - wall, d - wall
    for fx in (-1 / 3, 1 / 3):
        x = fx * 2 * iw
        st.append(_poly(T([[x, h * 0.92, -idp], [x, h * 0.92, idp]]), 0.5))
        st.append(_poly(T([[x + 0.06, h * 0.92, -idp], [x + 0.06, h * 0.92, idp]]), 0.3))
    st.append(_poly(T([[-iw, h * 0.92, 0], [iw, h * 0.92, 0]]), 0.5))
    st.append(_poly(T([[-iw, h * 0.92, 0.06], [iw, h * 0.92, 0.06]]), 0.3))
    # treats ------------------------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    cells = [((-2 / 3 + k * 2 / 3) * iw, (-0.5 + m) * idp) for m in (0, 1) for k in range(3)]
    kinds = ["cookie", "candy", "biscuit", "candy", "cookie", "chocolate"]
    for (cx, cz), kind in zip(cells, kinds):
        cw, cd = iw * 2 / 3 * 0.42, idp * 0.42
        y = h * 0.8
        if kind == "cookie":                                            # round cookie with chips
            for k in range(2):
                ox, oz = (k - 0.5) * cw * 0.9, rng.uniform(-0.1, 0.1)
                rr = min(cw, cd) * 0.62
                th = np.linspace(0, 2 * math.pi, 30)
                st.append(_poly(T(np.c_[cx + ox + rr * np.cos(th), np.full(30, y + 0.06 * k),
                                        cz + oz + rr * np.sin(th)]), 0.5))
                for _ in range(4):
                    a, q = rng.uniform(0, 6.28), rng.uniform(0.2, 0.65) * rr
                    p = np.array([cx + ox + q * math.cos(a), y + 0.07, cz + oz + q * math.sin(a)])
                    st.append(_poly(T([p, p + [0.06, 0, 0.04]]), 0.45))
        elif kind == "candy":                                           # wrapped candies (twist ends)
            for k in range(3):
                ox = (k - 1) * cw * 0.65
                oz = (k % 2 - 0.5) * cd * 0.5
                L, R_ = cw * 0.3, cd * 0.22
                th = np.linspace(0, 2 * math.pi, 24)
                st.append(_poly(T(np.c_[cx + ox + L * np.cos(th), np.full(24, y + 0.05), cz + oz + R_ * np.sin(th)]),
                                0.5))
                for sgn in (-1, 1):
                    e = np.array([cx + ox + sgn * L, y + 0.05, cz + oz])
                    st.append(_poly(T([e, e + [sgn * 0.28, 0, 0.16], e + [sgn * 0.28, 0, -0.16], e]), 0.45))
                st.append(_poly(T([[cx + ox - L * 0.3, y + 0.06, cz + oz - R_ * 0.7],
                                   [cx + ox - L * 0.3, y + 0.06, cz + oz + R_ * 0.7]]), 0.3))
        elif kind == "biscuit":                                         # rectangular biscuits with dots
            for k in range(2):
                oz = (k - 0.5) * cd * 0.95
                q = np.array([[-cw, 0, -cd * 0.38], [cw, 0, -cd * 0.38], [cw, 0, cd * 0.38], [-cw, 0, cd * 0.38]]) * 0.82
                q = q + [cx, y + 0.05 * k, cz + oz]
                st.append(_poly(T(q), 0.5, closed=True))
                for ix in np.linspace(-0.6, 0.6, 4):
                    p = np.array([cx + ix * cw * 0.8, y + 0.06 + 0.05 * k, cz + oz])
                    st.append(_poly(T([p, p + [0.03, 0, 0.02]]), 0.6))
        else:                                                           # chocolate bar with a grid
            q = np.array([[-cw, 0, -cd], [cw, 0, -cd], [cw, 0, cd], [-cw, 0, cd]]) * 0.85 + [cx, y + 0.05, cz]
            st.append(_poly(T(q), 0.55, closed=True))
            for fx in (-1 / 3, 1 / 3):
                st.append(_poly(T([[cx + fx * 2 * cw * 0.85, y + 0.06, cz - cd * 0.85],
                                   [cx + fx * 2 * cw * 0.85, y + 0.06, cz + cd * 0.85]]), 0.3))
            st.append(_poly(T([[cx - cw * 0.85, y + 0.06, cz], [cx + cw * 0.85, y + 0.06, cz]]), 0.3))
    # hinged lid standing open at the back --------------------------------------------------------
    a = math.radians(lid_open)
    hinge_z = -d
    lid = []
    for (x, z) in ring:
        dz = z - hinge_z                                                # distance from the hinge
        lid.append((x, h + dz * math.sin(a), hinge_z + dz * math.cos(a)))
    lid = np.array(lid)
    lid_in = lid.copy()
    nrm = np.array([0, -math.cos(a), math.sin(a)])                      # lid inner face normal (towards the box)
    lid_in = lid + nrm * 0.16
    Vl = np.vstack([lid, lid_in])
    ql = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    ctr = [i for i in range(n)]
    for i in range(1, n - 1):
        ql.append((ctr[0], ctr[i], ctr[i + 1], ctr[i + 1]))
    parts.append(_mesh_part("lid", T(Vl), ql, pid + 1))
    st.append(_poly(T(lid), 0.85, closed=True))
    st.append(_poly(T(lid_in), 0.45, closed=True))
    inset = (lid_in - lid_in.mean(0)) * 0.82 + lid_in.mean(0) + nrm * 0.02
    st.append(_poly(T(inset), 0.35, closed=True))
    for xh in (-w * 0.6, w * 0.6):                                     # hinges
        p = np.array([xh, h, -d])
        st.append(_poly(T([p + [-0.25, 0, 0.02], p + [0.25, 0, 0.02]]), 0.7))
    return parts, st


def room_shelf(x0, x1, y, z, items=True, seed=2):
    """back-wall shelf with a picture frame, a vase and books (storyboard panel 1 background)."""
    st = [_poly([[x0, y, z], [x1, y, z]], 0.6), _poly([[x0, y - 0.25, z], [x1, y - 0.25, z]], 0.4)]
    if items:
        fx = x0 + (x1 - x0) * 0.15
        st.append(_poly([[fx, y, z], [fx + 2.2, y, z], [fx + 2.2, y + 2.6, z], [fx, y + 2.6, z]], 0.55, closed=True))
        st.append(_poly([[fx + 0.3, y + 0.3, z], [fx + 1.9, y + 0.3, z], [fx + 1.9, y + 2.3, z], [fx + 0.3, y + 2.3, z]],
                        0.35, closed=True))
        vx = x0 + (x1 - x0) * 0.78
        th = np.linspace(-1, 1, 30)
        prof = 0.55 + 0.35 * np.cos(th * 2.2) - 0.25 * (th > 0.6)
        ys = y + (th + 1) * 1.1
        st.append(_poly(np.c_[vx - prof, ys, np.full(30, z)], 0.5))
        st.append(_poly(np.c_[vx + prof, ys, np.full(30, z)], 0.5))
        st.append(_poly([[vx - prof[-1], ys[-1], z], [vx + prof[-1], ys[-1], z]], 0.45))
        bx = x0 + (x1 - x0) * 0.45
        rng = np.random.default_rng(seed)
        for k in range(5):
            hh = rng.uniform(1.4, 2.0)
            st.append(_poly([[bx + k * 0.38, y, z], [bx + k * 0.38, y + hh, z], [bx + k * 0.38 + 0.34, y + hh, z],
                             [bx + k * 0.38 + 0.34, y, z]], 0.4))
    return st


def table(x0, x1, z0, z1, h, pid=96, legs=True):
    P = np.array([[x0, h, z0], [x1, h, z0], [x1, h, z1], [x0, h, z1]], float)
    st = [_poly(P, 0.75, closed=True), _poly(P + [0, -0.3, 0], 0.4, closed=True)]
    if legs:
        for x in (x0 + 0.5, x1 - 0.5):
            for z in (z0 + 0.5, z1 - 0.5):
                st.append(_poly([[x, h - 0.3, z], [x, 0.02, z]], 0.6))
    return [_quad_part("table", P, pid)], st


def nail_kit(center, seed=4):
    """nail polish bottles + a cushion + a small file on the table (adult sister's job)."""
    cx, y, cz = center
    st = []
    for k, ox in enumerate((-2.4, -1.7, -1.0)):
        th = np.linspace(0, 2 * math.pi, 20)
        r = 0.28
        base = np.array([cx + ox, y, cz + 0.6 * (k % 2)])
        st.append(_poly(np.c_[base[0] + r * np.cos(th), np.full(20, y + 0.75), base[2] + r * np.sin(th)], 0.45))
        st.append(_poly([[base[0] - r, y, base[2]], [base[0] - r, y + 0.75, base[2]]], 0.5))
        st.append(_poly([[base[0] + r, y, base[2]], [base[0] + r, y + 0.75, base[2]]], 0.5))
        st.append(_poly([[base[0] - 0.12, y + 0.75, base[2]], [base[0] - 0.12, y + 1.35, base[2]],
                         [base[0] + 0.12, y + 1.35, base[2]], [base[0] + 0.12, y + 0.75, base[2]]], 0.5))
    # cushion (soft rounded pillow)
    th = np.linspace(0, 2 * math.pi, 40)
    st.append(_poly(np.c_[cx + 1.2 + 1.3 * np.cos(th), y + 0.45 + 0.35 * np.sin(th) * (np.sin(th) > 0), cz + 0.2 +
                          0.0 * th], 0.6))
    st.append(_poly([[cx + 2.6, y + 0.05, cz + 1.4], [cx + 4.2, y + 0.05, cz + 1.0]], 0.6))
    return st


def thought_scribble(ctx, x, y, size, t, t0, ink, alpha=1.0):
    """the storyboard's tangled 'overthinking' scribble above her head, drawn on and slowly turning."""
    import cairo
    u = smooth((t - t0) / 1.4)
    if u <= 0:
        return
    rng = np.random.default_rng(9)
    n = 260
    s = np.linspace(0, 1, n)
    ang = s * 2 * math.pi * 5.5 + (t - t0) * 0.6
    rr = size * (0.55 + 0.35 * np.sin(s * 17 + 1.0) + 0.12 * np.sin(s * 41))
    px = x + rr * np.cos(ang) + rng.normal(0, 0.6, n)
    py = y + rr * 0.72 * np.sin(ang * 1.07)
    m = max(2, int(n * u))
    ctx.set_source_rgba(*ink, alpha)
    ctx.set_line_width(1.6)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.new_path()
    ctx.move_to(px[0], py[0])
    for i in range(1, m):
        ctx.line_to(px[i], py[i])
    ctx.stroke()
    for k, (dx, dy, r) in enumerate(((-size * 0.5, size * 1.0, size * 0.12), (-size * 0.75, size * 1.45, size * 0.07))):
        if u > 0.6 + 0.15 * k:
            ctx.arc(x + dx, y + dy, r, 0, 2 * math.pi)
            ctx.stroke()
