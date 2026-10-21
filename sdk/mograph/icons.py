"""
mograph.icons — data-driven multi-layer vector icon library (100×100 view-box).

Icons live in `icons.json` (primary + detail layers + Persian/English tags)
so the agent can ADD icons without touching code, and can SEARCH them:

    find_icons("دارو")        -> ['vial', 'pill', 'syringe', ...]
    draw_icon(ctx, "vial", x, y, size=160, p=0.6)          # trim-path reveal
    draw_icon(ctx, "vial", x, y, size=160, p=1, style="ghost")
    add_icon("rocket2", primary=[...], detail=[...], tags=["موشک"], save=True)
    import_svg_icon("/usr/share/icons/.../x.svg", "name")   # normalises to 100 box

`size` is the on-screen size in pixels of the 100-unit box. Line weight is
in pixels and does NOT scale with size (consistent stroke across the video).
"""
from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET

from .core import clamp, color as _color, ease_out
from .geometry import compile_path, draw_path, stroke_trimmed

_HERE = os.path.dirname(os.path.abspath(__file__))
ICON_FILE = os.path.join(_HERE, "icons.json")

with open(ICON_FILE, encoding="utf-8") as _f:
    ICONS: dict = json.load(_f)

STYLES = {
    # name: (primary color, detail color, primary width px, detail width px)
    "default": ("#ffffff", "#8a8a90", 2.6, 1.6),
    "dim": ("#5a5a5e", "#3a3a3e", 2.2, 1.4),
    "ghost": ("#ffffff", "#ffffff", 1.6, 1.0),
    "bold": ("#ffffff", "#a0a0a6", 3.6, 2.0),
}


def icon_names():
    return sorted(ICONS)


def find_icons(query: str, limit=10):
    """Fuzzy search by name or Persian/English tag."""
    q = query.strip().lower()
    scored = []
    for k, v in ICONS.items():
        tags = [k.lower()] + [t.lower() for t in v.get("tags", [])]
        s = 0
        for t in tags:
            if q == t:
                s = max(s, 3)
            elif q in t or t in q:
                s = max(s, 2)
        if s:
            scored.append((-s, k))
    return [k for _, k in sorted(scored)[:limit]]


def draw_icon(ctx, name, x, y, size=120.0, p=1.0, style="default", color=None, detail_color=None,
              width=None, detail=True, alpha=1.0, rotate=0.0, fill=None, dashed=None, mode="sequential"):
    """
    Draw icon `name` centred at (x, y), `size` px wide.  p = trim progress (0..1).
    Primary layers draw first, details follow slightly later (layered reveal).
    fill='#000' masks lines behind the icon (useful when overlapping things).
    """
    ic = ICONS.get(name)
    if ic is None:
        ic = ICONS["question"]
    if alpha <= 0.002 or p <= 0.0005:
        return
    pc, dc, pw, dw = STYLES.get(style, STYLES["default"])
    pc = color or pc
    dc = detail_color or (dc if color is None else _color(color, 0.55))
    pw = width or pw
    dw = (width * 0.6) if width else dw
    scale = size / 100.0
    prim = ic["primary"]
    dets = ic.get("detail", []) if detail else []
    # primaries over first 80% of p, details over 35%..100%
    pp = clamp(p / 0.8)
    dp = clamp((p - 0.35) / 0.65)
    if fill is not None and prim:
        draw_path(ctx, prim[0], x, y, scale, trim=0.0, stroke=None, fill=fill, alpha=alpha * clamp(p * 3),
                  rotate=rotate)
    n = len(prim)
    for i, d in enumerate(prim):
        # each primary sub-layer starts slightly after the previous one
        sp = clamp((pp * (n + 0.6) - i * 0.6) / (n * 0.4 + 0.6)) if n > 1 else pp
        draw_path(ctx, d, x, y, scale, trim=sp, stroke=pc, width=pw, alpha=alpha, rotate=rotate,
                  dashed=dashed, mode=mode)
    for d in dets:
        draw_path(ctx, d, x, y, scale, trim=dp, stroke=dc, width=dw, alpha=alpha, rotate=rotate, mode="parallel")


def icon_bbox(name, x, y, size=120.0):
    ic = ICONS.get(name, ICONS["question"])
    x0 = y0 = 1e9; x1 = y1 = -1e9
    for d in ic["primary"] + ic.get("detail", []):
        b = compile_path(d).bbox
        x0, y0, x1, y1 = min(x0, b[0]), min(y0, b[1]), max(x1, b[2]), max(y1, b[3])
    s = size / 100.0
    return (x + (x0 - 50) * s, y + (y0 - 50) * s, x + (x1 - 50) * s, y + (y1 - 50) * s)


# ---------------------------------------------------------------------------
# growing the library
# ---------------------------------------------------------------------------

def add_icon(name, primary, detail=(), tags=(), save=False):
    """Register an icon (100×100 box). save=True persists it into icons.json (evolution!)."""
    for d in list(primary) + list(detail):
        compile_path(d)  # validate now, fail early
    ICONS[name] = {"primary": list(primary), "detail": list(detail), "tags": list(tags)}
    if save:
        with open(ICON_FILE, "w", encoding="utf-8") as f:
            json.dump(ICONS, f, ensure_ascii=False, indent=1)


def _svg_transform_paths(svg_path):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    vb = root.attrib.get("viewBox")
    if vb:
        vx, vy, vw, vh = [float(v) for v in vb.replace(",", " ").split()]
    else:
        vx = vy = 0.0
        vw = float(str(root.attrib.get("width", "24")).rstrip("px"))
        vh = float(str(root.attrib.get("height", "24")).rstrip("px"))
    ds = []
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "path" and el.attrib.get("d"):
            ds.append(el.attrib["d"])
        elif tag == "circle":
            from .geometry import circle_d
            ds.append(circle_d(float(el.attrib.get("cx", 0)), float(el.attrib.get("cy", 0)),
                               float(el.attrib.get("r", 0))))
        elif tag == "rect":
            from .geometry import rounded_rect_d
            ds.append(rounded_rect_d(float(el.attrib.get("x", 0)), float(el.attrib.get("y", 0)),
                                     float(el.attrib.get("width", 0)), float(el.attrib.get("height", 0)),
                                     float(el.attrib.get("rx", 0))))
        elif tag in ("line",):
            a = el.attrib
            ds.append(f"M {a.get('x1',0)} {a.get('y1',0)} L {a.get('x2',0)} {a.get('y2',0)}")
        elif tag in ("polyline", "polygon"):
            pts = el.attrib.get("points", "").replace(",", " ").split()
            if len(pts) >= 4:
                d = "M " + " ".join(pts[:2]) + " L " + " ".join(pts[2:])
                ds.append(d + (" Z" if tag == "polygon" else ""))
    return ds, (vx, vy, vw, vh)


def import_svg_icon(svg_path, name, tags=(), save=False, margin=10.0):
    """Import an outline SVG (Lucide/Tabler/Feather/your own) as a library icon.
    Coordinates are rewritten into the 100 box using cairo (so every command type works)."""
    import cairo
    ds, (vx, vy, vw, vh) = _svg_transform_paths(svg_path)
    s = (100 - 2 * margin) / max(vw, vh)
    ox = margin + (100 - 2 * margin - vw * s) / 2 - vx * s
    oy = margin + (100 - 2 * margin - vh * s) / 2 - vy * s
    out = []
    ctx = cairo.Context(cairo.ImageSurface(cairo.FORMAT_A8, 1, 1))
    for d in ds:
        comp = compile_path(d)
        ctx.new_path()
        ctx.identity_matrix()
        ctx.translate(ox, oy)
        ctx.scale(s, s)
        ctx.append_path(comp.whole)
        ctx.identity_matrix()
        segs = []
        for kind, pts in ctx.copy_path():
            if kind == cairo.PATH_MOVE_TO:
                segs.append(f"M {pts[0]:.2f} {pts[1]:.2f}")
            elif kind == cairo.PATH_LINE_TO:
                segs.append(f"L {pts[0]:.2f} {pts[1]:.2f}")
            elif kind == cairo.PATH_CURVE_TO:
                segs.append("C " + " ".join(f"{v:.2f}" for v in pts))
            else:
                segs.append("Z")
        out.append(" ".join(segs))
    add_icon(name, out, (), tags, save=save)
    return name
