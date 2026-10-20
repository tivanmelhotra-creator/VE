"""
mograph.geometry — compiled SVG paths with EXACT trim-path (stroke-dashoffset).

Why this replaces v3's draw_svg_path:
  * v3 re-tokenised the SVG string on every frame and guessed the path length
    (`approx_len = 460 * scale`), so Trim-Path speed was wrong for every icon
    (short icons finished instantly, long ones never completed until p=1).
  * v3 ignored S/T/A and most relative commands (silently broke imported SVGs).
  * cairo resets dashes at every sub-path, so multi-part icons trimmed oddly.

Here every path string is parsed ONCE (lru cache) into cairo.Path objects
(one per sub-path) together with the true arc length of each sub-path
(computed from cairo's flattened polyline). Drawing is then just
append_path + stroke, and trim uses the real length.

Public API
----------
compile_path(d, fillet=0.0) -> Compiled
draw_path(ctx, d_or_compiled, x, y, scale, trim=(0,1) | p, stroke=..., width=..., fill=None,
          view_box=100, mode="sequential"|"parallel", dashed=None)
path_bbox(d, view_box) -> (x0,y0,x1,y1) in view-box units
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

import cairo

from .core import clamp, color as _color

_TOKEN_RE = re.compile(r"[MLHVCSQTAZmlhvcsqtaz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_SCRATCH = cairo.Context(cairo.ImageSurface(cairo.FORMAT_A8, 1, 1))


# ---------------------------------------------------------------------------
# parsing → absolute segments
# ---------------------------------------------------------------------------

def _parse(d: str):
    """Returns list of sub-paths; each = {"segs": [...], "closed": bool}.
    seg kinds: ("L", x, y) ("C", x1,y1,x2,y2,x,y)  start point stored in sub["start"]."""
    toks = _TOKEN_RE.findall(d)
    i, n = 0, len(toks)
    subs, cur = [], None
    x = y = sx = sy = 0.0
    last_c2 = None   # last cubic control point (for S)
    last_q = None    # last quad control (for T)
    cmd = None

    def num():
        nonlocal i
        v = float(toks[i]); i += 1
        return v

    def new_sub(px, py):
        nonlocal cur
        cur = {"start": (px, py), "segs": [], "closed": False}
        subs.append(cur)

    while i < n:
        t = toks[i]
        if t.isalpha():
            cmd = t; i += 1
            if cmd in "Zz":
                if cur is not None:
                    cur["closed"] = True
                x, y = sx, sy
                cur = None
                last_c2 = last_q = None
                continue
        if cmd is None:
            i += 1
            continue
        rel = cmd.islower()
        c = cmd.upper()
        try:
            if c == "M":
                nx, ny = num(), num()
                if rel: nx += x; ny += y
                x, y, sx, sy = nx, ny, nx, ny
                new_sub(x, y)
                cmd = "l" if rel else "L"
                last_c2 = last_q = None
                continue
            if cur is None:
                new_sub(x, y)
            if c == "L":
                nx, ny = num(), num()
                if rel: nx += x; ny += y
                cur["segs"].append(("L", nx, ny)); x, y = nx, ny
                last_c2 = last_q = None
            elif c == "H":
                nx = num() + (x if rel else 0)
                cur["segs"].append(("L", nx, y)); x = nx
                last_c2 = last_q = None
            elif c == "V":
                ny = num() + (y if rel else 0)
                cur["segs"].append(("L", x, ny)); y = ny
                last_c2 = last_q = None
            elif c == "C":
                a = [num() for _ in range(6)]
                if rel: a = [a[k] + (x if k % 2 == 0 else y) for k in range(6)]
                cur["segs"].append(("C", *a)); last_c2 = (a[2], a[3]); x, y = a[4], a[5]
                last_q = None
            elif c == "S":
                a = [num() for _ in range(4)]
                if rel: a = [a[k] + (x if k % 2 == 0 else y) for k in range(4)]
                c1 = (2 * x - last_c2[0], 2 * y - last_c2[1]) if last_c2 else (x, y)
                cur["segs"].append(("C", c1[0], c1[1], a[0], a[1], a[2], a[3]))
                last_c2 = (a[0], a[1]); x, y = a[2], a[3]; last_q = None
            elif c in "QT":
                if c == "Q":
                    q = [num() for _ in range(4)]
                    if rel: q = [q[k] + (x if k % 2 == 0 else y) for k in range(4)]
                    qx, qy, ex, ey = q
                else:
                    ex, ey = num(), num()
                    if rel: ex += x; ey += y
                    qx, qy = (2 * x - last_q[0], 2 * y - last_q[1]) if last_q else (x, y)
                c1 = (x + 2 / 3 * (qx - x), y + 2 / 3 * (qy - y))
                c2 = (ex + 2 / 3 * (qx - ex), ey + 2 / 3 * (qy - ey))
                cur["segs"].append(("C", c1[0], c1[1], c2[0], c2[1], ex, ey))
                last_q = (qx, qy); x, y = ex, ey; last_c2 = None
            elif c == "A":
                rx, ry, phi, fa, fs, ex, ey = (num() for _ in range(7))
                if rel: ex += x; ey += y
                for seg in _arc_to_cubics(x, y, rx, ry, phi, int(fa), int(fs), ex, ey):
                    cur["segs"].append(seg)
                x, y = ex, ey; last_c2 = last_q = None
            else:
                i += 1
        except (IndexError, ValueError):
            break
    return [s for s in subs if s["segs"]]


def _arc_to_cubics(x1, y1, rx, ry, phi_deg, fa, fs, x2, y2):
    """SVG elliptical arc → list of ("C", ...) segments (W3C implementation notes F.6)."""
    if rx == 0 or ry == 0 or (x1 == x2 and y1 == y2):
        return [("L", x2, y2)]
    phi = math.radians(phi_deg)
    cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = (x1 - x2) / 2, (y1 - y2) / 2
    x1p, y1p = cp * dx + sp * dy, -sp * dx + cp * dy
    rx, ry = abs(rx), abs(ry)
    lam = x1p ** 2 / rx ** 2 + y1p ** 2 / ry ** 2
    if lam > 1:
        s = math.sqrt(lam); rx *= s; ry *= s
    num = rx ** 2 * ry ** 2 - rx ** 2 * y1p ** 2 - ry ** 2 * x1p ** 2
    den = rx ** 2 * y1p ** 2 + ry ** 2 * x1p ** 2
    co = math.sqrt(max(0.0, num / den)) * (-1 if fa == fs else 1)
    cxp, cyp = co * rx * y1p / ry, -co * ry * x1p / rx
    cx = cp * cxp - sp * cyp + (x1 + x2) / 2
    cy = sp * cxp + cp * cyp + (y1 + y2) / 2

    def ang(ux, uy, vx, vy):
        a = math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
        return a
    t1 = ang(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dt = ang((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not fs and dt > 0: dt -= 2 * math.pi
    if fs and dt < 0: dt += 2 * math.pi
    nseg = max(1, int(math.ceil(abs(dt) / (math.pi / 2))))
    d = dt / nseg
    k = 4 / 3 * math.tan(d / 4)
    out = []
    for s in range(nseg):
        a0, a1 = t1 + s * d, t1 + (s + 1) * d
        e0 = (math.cos(a0), math.sin(a0)); e1 = (math.cos(a1), math.sin(a1))
        p = [(e0[0] - k * e0[1], e0[1] + k * e0[0]), (e1[0] + k * e1[1], e1[1] - k * e1[0]), e1]
        pts = []
        for ux, uy in p:
            X, Y = rx * ux, ry * uy
            pts += [cp * X - sp * Y + cx, sp * X + cp * Y + cy]
        out.append(("C", *pts))
    return out


# ---------------------------------------------------------------------------
# fillet (rounded corners) for line-only sub-paths
# ---------------------------------------------------------------------------

def _fillet_points(pts, closed, r):
    n = len(pts)
    items = []
    for i in range(n):
        if not closed and (i == 0 or i == n - 1):
            items.append(("P", pts[i])); continue
        p0, p1, p2 = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
        v1 = (p0[0] - p1[0], p0[1] - p1[1]); v2 = (p2[0] - p1[0], p2[1] - p1[1])
        d1, d2 = math.hypot(*v1), math.hypot(*v2)
        if d1 < 1e-6 or d2 < 1e-6:
            items.append(("P", p1)); continue
        cosang = (v1[0] * v2[0] + v1[1] * v2[1]) / (d1 * d2)
        if cosang < -0.995:  # straight line, no corner
            items.append(("P", p1)); continue
        re_ = min(r, d1 * 0.45, d2 * 0.45)
        a = (p1[0] + v1[0] / d1 * re_, p1[1] + v1[1] / d1 * re_)
        b = (p1[0] + v2[0] / d2 * re_, p1[1] + v2[1] / d2 * re_)
        items.append(("F", a, p1, b))
    return items


def _emit_sub(ctx, sub, fillet):
    segs = sub["segs"]
    line_only = all(s[0] == "L" for s in segs)
    if fillet > 0 and line_only and len(segs) >= 2:
        pts = [sub["start"]] + [(s[1], s[2]) for s in segs]
        closed = sub["closed"]
        if closed and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 1e-6:
            pts = pts[:-1]
        items = _fillet_points(pts, closed, fillet)
        first = True
        for it in items:
            if it[0] == "P":
                (ctx.move_to if first else ctx.line_to)(*it[1])
            else:
                a, c, b = it[1], it[2], it[3]
                (ctx.move_to if first else ctx.line_to)(*a)
                ctx.curve_to(a[0] + 2 / 3 * (c[0] - a[0]), a[1] + 2 / 3 * (c[1] - a[1]),
                             b[0] + 2 / 3 * (c[0] - b[0]), b[1] + 2 / 3 * (c[1] - b[1]), b[0], b[1])
            first = False
        if closed:
            if items and items[0][0] == "F":
                ctx.line_to(*items[0][1])
            ctx.close_path()
        return
    ctx.move_to(*sub["start"])
    for s in segs:
        if s[0] == "L":
            ctx.line_to(s[1], s[2])
        else:
            ctx.curve_to(*s[1:])
    if sub["closed"]:
        ctx.close_path()


class Compiled:
    __slots__ = ("subs", "lengths", "total", "bbox", "whole")

    def __init__(self, subs, lengths, bbox, whole):
        self.subs, self.lengths, self.bbox, self.whole = subs, lengths, bbox, whole
        self.total = sum(lengths)


def _flat_length(path):
    L = 0.0
    px = py = sx = sy = None
    for kind, pts in path:
        if kind == cairo.PATH_MOVE_TO:
            px, py = pts; sx, sy = pts
        elif kind == cairo.PATH_LINE_TO:
            L += math.hypot(pts[0] - px, pts[1] - py); px, py = pts
        elif kind == cairo.PATH_CLOSE_PATH:
            if sx is not None:
                L += math.hypot(sx - px, sy - py); px, py = sx, sy
    return L


@lru_cache(maxsize=4096)
def compile_path(d: str, fillet: float = 0.0) -> Compiled:
    ctx = _SCRATCH
    ctx.identity_matrix()
    ctx.set_tolerance(0.05)
    subs, lengths = [], []
    for sub in _parse(d):
        ctx.new_path()
        _emit_sub(ctx, sub, fillet)
        subs.append(ctx.copy_path())
        lengths.append(_flat_length(ctx.copy_path_flat()))
    ctx.new_path()
    for p in subs:
        ctx.append_path(p)
    bbox = ctx.path_extents() if subs else (0, 0, 0, 0)
    whole = ctx.copy_path()
    ctx.new_path()
    return Compiled(subs, lengths, bbox, whole)


def path_bbox(d, fillet=0.0):
    return compile_path(d, fillet).bbox


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def _trim_window(trim):
    if isinstance(trim, (tuple, list)):
        return clamp(trim[0]), clamp(trim[1])
    return 0.0, clamp(trim)


def stroke_trimmed(ctx, comp: Compiled, trim=1.0, mode="sequential"):
    """Strokes `comp` using the CURRENT source/line width, honouring trim.
    mode='sequential': sub-paths are drawn one after another (like a pen).
    mode='parallel'  : every sub-path trims simultaneously."""
    s, e = _trim_window(trim)
    if e - s <= 1e-4 or comp.total <= 1e-6:
        return
    if s <= 1e-4 and e >= 0.9999:
        ctx.set_dash([], 0)
        ctx.new_path(); ctx.append_path(comp.whole); ctx.stroke()
        return
    if mode == "parallel":
        for p, L in zip(comp.subs, comp.lengths):
            _dash_stroke(ctx, p, L, s * L, e * L)
        return
    a, b = s * comp.total, e * comp.total
    acc = 0.0
    for p, L in zip(comp.subs, comp.lengths):
        lo, hi = max(a - acc, 0.0), min(b - acc, L)
        if hi > lo + 1e-4:
            _dash_stroke(ctx, p, L, lo, hi)
        acc += L
        if acc >= b:
            break
    ctx.set_dash([], 0)


def _dash_stroke(ctx, p, L, lo, hi):
    if hi - lo <= 1e-4:
        return
    ctx.new_path(); ctx.append_path(p)
    if lo <= 1e-4 and hi >= L - 1e-4:
        ctx.set_dash([], 0)
    else:
        big = L * 3 + 10
        total = (hi - lo) + big
        ctx.set_dash([hi - lo, big], total - lo)
    ctx.stroke()


def draw_path(ctx, d, x=0.0, y=0.0, scale=1.0, trim=1.0, stroke="#ffffff", width=2.4,
              fill=None, view_box=100.0, mode="sequential", dashed=None, fillet=0.0,
              alpha=1.0, rotate=0.0, cap="round", anchor="center"):
    """Draw an SVG path string (or Compiled) placed with its view-box centre at (x, y).

    width is in OUTPUT pixels (independent of `scale`), so icons keep a uniform
    line weight at any size. `dashed=(on, off)` draws a dashed line (pixels).
    `fill` paints the interior (e.g. '#000' to mask lines behind)."""
    comp = d if isinstance(d, Compiled) else compile_path(d, fillet)
    if alpha <= 0.002:
        return
    s, e = _trim_window(trim)
    if e - s <= 1e-4 and fill is None:
        return
    ctx.save()
    ctx.translate(x, y)
    if rotate:
        ctx.rotate(math.radians(rotate))
    ctx.scale(scale, scale)
    if anchor == "center":
        ctx.translate(-view_box / 2, -view_box / 2)
    if fill is not None and e > 0.25:
        ctx.new_path(); ctx.append_path(comp.whole)
        ctx.set_source_rgba(*_color(fill, alpha * clamp((e - 0.25) / 0.3)))
        ctx.fill()
    if stroke is not None and width > 0:
        ctx.set_source_rgba(*_color(stroke, alpha))
        ctx.set_line_width(width / max(scale, 1e-6))
        ctx.set_line_cap(cairo.LINE_CAP_ROUND if cap == "round" else cairo.LINE_CAP_BUTT)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        if dashed:
            on, off = dashed
            ctx.set_dash([on / scale, off / scale], 0)
            ctx.new_path(); ctx.append_path(comp.whole); ctx.stroke()
            ctx.set_dash([], 0)
        else:
            stroke_trimmed(ctx, comp, (s, e), mode)
    ctx.restore()


# ---------------------------------------------------------------------------
# primitive path builders (return SVG strings → cached by compile_path)
# ---------------------------------------------------------------------------

def rounded_rect_d(x0, y0, w, h, r=14.0):
    r = max(0.0, min(float(r), w / 2, h / 2))
    k = 0.5522847498 * r
    x1, y1 = x0 + w, y0 + h
    if r <= 0.01:
        return f"M {x0} {y0} H {x1} V {y1} H {x0} Z"
    # starts top-centre so trim draws symmetrically-looking
    xm = x0 + w / 2
    return (f"M {xm:.3f} {y0:.3f} H {x1 - r:.3f} C {x1 - r + k:.3f} {y0:.3f} {x1:.3f} {y0 + r - k:.3f} {x1:.3f} {y0 + r:.3f} "
            f"V {y1 - r:.3f} C {x1:.3f} {y1 - r + k:.3f} {x1 - r + k:.3f} {y1:.3f} {x1 - r:.3f} {y1:.3f} "
            f"H {x0 + r:.3f} C {x0 + r - k:.3f} {y1:.3f} {x0:.3f} {y1 - r + k:.3f} {x0:.3f} {y1 - r:.3f} "
            f"V {y0 + r:.3f} C {x0:.3f} {y0 + r - k:.3f} {x0 + r - k:.3f} {y0:.3f} {x0 + r:.3f} {y0:.3f} Z")


def circle_d(cx, cy, r, start_deg=-90.0):
    """Circle as 4 cubic arcs starting at start_deg (default: 12 o'clock) clockwise."""
    k = 0.5522847498
    out = []
    a0 = math.radians(start_deg)
    for q in range(4):
        a = a0 + q * math.pi / 2; b = a + math.pi / 2
        p0 = (cx + r * math.cos(a), cy + r * math.sin(a))
        p3 = (cx + r * math.cos(b), cy + r * math.sin(b))
        c1 = (p0[0] - k * r * math.sin(a), p0[1] + k * r * math.cos(a))
        c2 = (p3[0] + k * r * math.sin(b), p3[1] - k * r * math.cos(b))
        if q == 0:
            out.append(f"M {p0[0]:.3f} {p0[1]:.3f}")
        out.append(f"C {c1[0]:.3f} {c1[1]:.3f} {c2[0]:.3f} {c2[1]:.3f} {p3[0]:.3f} {p3[1]:.3f}")
    return " ".join(out) + " Z"


def polyline_d(pts, closed=False):
    s = "M " + " L ".join(f"{x:.3f} {y:.3f}" for x, y in pts)
    return s + (" Z" if closed else "")


def arrow_d(x0, y0, x1, y1, head=12.0, curve=0.0):
    """Arrow from (x0,y0) to (x1,y1). curve≠0 bends it (fraction of length, sign = side)."""
    if curve:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        cx, cy = mx - dy * curve, my + dx * curve
        body = f"M {x0} {y0} Q {cx:.3f} {cy:.3f} {x1} {y1}"
        tx, ty = x1 - cx, y1 - cy
    else:
        body = f"M {x0} {y0} L {x1} {y1}"
        tx, ty = x1 - x0, y1 - y0
    L = math.hypot(tx, ty) or 1
    ux, uy = tx / L, ty / L
    a = (x1 - ux * head - uy * head * 0.6, y1 - uy * head + ux * head * 0.6)
    b = (x1 - ux * head + uy * head * 0.6, y1 - uy * head - ux * head * 0.6)
    return body + f" M {a[0]:.3f} {a[1]:.3f} L {x1} {y1} L {b[0]:.3f} {b[1]:.3f}"


# ---------------------------------------------------------------------------
# topological vector MORPH  (icon A → icon B with real shape interpolation)
# ---------------------------------------------------------------------------

def _resample_flat(path, n):
    """Flattened cairo path → n points evenly spaced by arc length (+closed flag)."""
    pts = []
    closed = False
    for kind, p in path:
        if kind in (cairo.PATH_MOVE_TO, cairo.PATH_LINE_TO):
            pts.append(p)
        elif kind == cairo.PATH_CLOSE_PATH:
            closed = True
    if len(pts) < 2:
        p0 = pts[0] if pts else (50.0, 50.0)
        return [p0] * n, closed
    if closed and pts[0] != pts[-1]:
        pts.append(pts[0])
    seg = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])]
    total = sum(seg) or 1e-6
    out, acc, k = [], 0.0, 0
    for i in range(n):
        target = total * i / (n if closed else max(1, n - 1))
        while k < len(seg) - 1 and acc + seg[k] < target:
            acc += seg[k]; k += 1
        f = (target - acc) / (seg[k] or 1e-6)
        f = min(1.0, max(0.0, f))
        a, b = pts[k], pts[k + 1]
        out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
    return out, closed


def _centroid(pts):
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


@lru_cache(maxsize=512)
def _morph_rings(ds_tuple, n=64):
    """Compile a list of path strings into resampled point rings (one per sub-path)."""
    rings = []
    ctx = _SCRATCH
    for d in ds_tuple:
        comp = compile_path(d)
        for p in comp.subs:
            ctx.new_path(); ctx.identity_matrix(); ctx.append_path(p)
            flat = ctx.copy_path_flat()
            ctx.new_path()
            pts, closed = _resample_flat(flat, n)
            L = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
            rings.append((tuple(pts), closed, L))
    rings.sort(key=lambda r: -r[2])  # biggest first → matched pairwise
    return tuple(rings)


def _align_ring(a, b, closed):
    """Rotate ring b (closed) so its start best matches a; also try reversed direction."""
    if not closed:
        d1 = math.hypot(a[0][0] - b[0][0], a[0][1] - b[0][1]) + math.hypot(a[-1][0] - b[-1][0], a[-1][1] - b[-1][1])
        d2 = math.hypot(a[0][0] - b[-1][0], a[0][1] - b[-1][1]) + math.hypot(a[-1][0] - b[0][0], a[-1][1] - b[0][1])
        return list(b) if d1 <= d2 else list(b[::-1])
    n = len(b)
    best, best_b = 1e18, list(b)
    step = max(1, n // 32)
    for cand in (list(b), list(b[::-1])):
        for sft in range(0, n, step):
            rb = cand[sft:] + cand[:sft]
            e = sum((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2 for pa, pb in zip(a[::4], rb[::4]))
            if e < best:
                best, best_b = e, rb
    return best_b


def _ring_feat(r):
    pts = r[0]
    c = _centroid(pts)
    xs = [q[0] for q in pts]; ys = [q[1] for q in pts]
    return c, max(max(xs) - min(xs), max(ys) - min(ys))


@lru_cache(maxsize=256)
def morph_plan(ds_a, ds_b, n=64):
    """Pair sub-paths of A and B greedily by (centroid distance + size difference).
    Unpaired sub-paths collapse to / grow from their own centroid (clean appear/disappear)."""
    ra, rb = list(_morph_rings(tuple(ds_a), n)), list(_morph_rings(tuple(ds_b), n))
    fa, fb = [_ring_feat(r) for r in ra], [_ring_feat(r) for r in rb]
    cand = []
    for i, (ca, sa) in enumerate(fa):
        for j, (cb, sb) in enumerate(fb):
            cost = math.hypot(ca[0] - cb[0], ca[1] - cb[1]) + 0.6 * abs(sa - sb) + (25 if ra[i][1] != rb[j][1] else 0)
            cand.append((cost, i, j))
    cand.sort()
    used_a, used_b, pairs = set(), set(), []
    for cost, i, j in cand:
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j)
        a, b = ra[i], rb[j]
        closed = a[1] and b[1]
        pairs.append((a[0], tuple(_align_ring(a[0], b[0], closed)), closed, a[1] or b[1]))
    for i, a in enumerate(ra):
        if i not in used_a:
            c = _centroid(a[0]); pairs.append((a[0], tuple([c] * n), a[1], a[1]))
    for j, b in enumerate(rb):
        if j not in used_b:
            c = _centroid(b[0]); pairs.append((tuple([c] * n), b[0], b[1], b[1]))
    return tuple(pairs)


def draw_morph(ctx, ds_a, ds_b, p, x, y, scale=1.0, stroke="#ffffff", width=2.6, alpha=1.0, view_box=100.0,
               n=64, twist=0.0):
    """Draw the in-between shape at progress p (0 = A, 1 = B). Optional `twist` (deg) for flair."""
    plan = morph_plan(tuple(ds_a), tuple(ds_b), n)
    ctx.save()
    ctx.translate(x, y)
    if twist:
        ctx.rotate(math.radians(twist * math.sin(math.pi * p)))
    ctx.scale(scale, scale)
    ctx.translate(-view_box / 2, -view_box / 2)
    ctx.new_path()
    for a, b, closed, _ in plan:
        pts = [(pa[0] + (pb[0] - pa[0]) * p, pa[1] + (pb[1] - pa[1]) * p) for pa, pb in zip(a, b)]
        # smooth with Catmull-Rom → cubic Bezier for organic in-betweens
        m = len(pts)
        ctx.move_to(*pts[0])
        rng = range(m if closed else m - 1)
        for i in rng:
            p0 = pts[(i - 1) % m] if (closed or i > 0) else pts[i]
            p1 = pts[i]
            p2 = pts[(i + 1) % m]
            p3 = pts[(i + 2) % m] if (closed or i + 2 < m) else p2
            c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
            c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
            ctx.curve_to(c1[0], c1[1], c2[0], c2[1], p2[0], p2[1])
        if closed:
            ctx.close_path()
    from .core import color as _c
    ctx.set_source_rgba(*_c(stroke, alpha))
    ctx.set_line_width(width / max(scale, 1e-6))
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.stroke()
    ctx.restore()


def point_on_path(d, p, fillet=0.0):
    """(x, y, angle_rad) at fraction p of path d (absolute coords) — for particles along curves."""
    comp = compile_path(d, fillet)
    target = comp.total * (p % 1.0)
    acc = 0.0
    ctx = _SCRATCH
    for path, L in zip(comp.subs, comp.lengths):
        if acc + L >= target:
            ctx.new_path(); ctx.identity_matrix(); ctx.append_path(path)
            flat = ctx.copy_path_flat(); ctx.new_path()
            pts = [pp for k, pp in flat if k in (cairo.PATH_MOVE_TO, cairo.PATH_LINE_TO)]
            rem = target - acc
            for a, b in zip(pts, pts[1:]):
                sl = math.hypot(b[0] - a[0], b[1] - a[1])
                if rem <= sl:
                    f = rem / (sl or 1e-6)
                    return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, math.atan2(b[1] - a[1], b[0] - a[0]))
                rem -= sl
            return (*pts[-1], 0.0)
        acc += L
    return (0.0, 0.0, 0.0)
