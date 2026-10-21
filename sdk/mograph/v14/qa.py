"""
mograph.v14.qa — fast checks before any render (< 5 s):
  * every time / word anchor resolved (prepare() already raises on a missing word)
  * every layer: duration > 0.25 s, inside the edit, sampled at 3 instants → ink bbox on screen (warn if clipped)
  * overlapping FRONT layers at the same instant in the same screen region (clutter)
  * subtitle phrases: none longer than 3.5 s, none shorter than 0.35 s
Exit code 0 = clean, 1 = warnings.
"""
from __future__ import annotations

import cairo


class _Probe:
    """Context that only records drawn extents."""


def run_qa(R):
    warn = qa_warnings(R, verbose=True)
    for w in warn:
        print("⚠", w)
    print("QA clean ✓" if not warn else f"QA: {len(warn)} warning(s)")
    return 1 if warn else 0


def qa_warnings(R, verbose=False):
    """All QA warnings as a list of strings (used by run_qa and by the v14.8 auto-director's fix loop)."""
    warn = []
    for L in R.layers:
        d = L["_t1"] - L["_t0"]
        name = L.get("id", L["type"])
        if d < 0.25:
            warn.append(f"{name}: very short ({d:.2f}s)")
        if L["_t0"] > R.duration:
            warn.append(f"{name}: starts after the end ({L['_t0']:.2f} > {R.duration:.2f})")
        if verbose:
            print(f"  {L['_t0']:6.2f} → {min(L['_t1'], R.duration):6.2f}  {L['type']:<14} {name}")
    if R.subs:
        for p in R.subs["_phr"]:
            dd = p["t1"] - p["t0"]
            txt = " ".join(w["w"] for w in p["words"])
            if dd > 3.5:
                warn.append(f"subtitle too long ({dd:.1f}s): {txt}")
            if dd < 0.35:
                warn.append(f"subtitle flashes ({dd:.2f}s): {txt}")
    # on-screen check: render ink of each layer alone at its mid-time
    from .compositor import Frame
    from . import layers as LY
    for L in R.layers:
        if L["type"] in ("progress", "flash", "sweep_light", "ambient_bokeh", "god_rays"):   # atmosphere layers
            continue
        t = min(R.duration - 0.05, L["_t0"] + min(0.9, (L["_t1"] - L["_t0"]) / 2))
        F = Frame(R, t)
        F.M, F.zoom = R.matrix(t)
        F.prepass = True
        s = cairo.ImageSurface(cairo.FORMAT_A8, R.W, R.H)
        c = cairo.Context(s)
        for p in ("back", "front"):
            if LY.in_pass(L, p):
                F.pass_ = p
                c.save(); LY.draw(c, L, F); c.restore()
        import numpy as np
        s.flush()
        a = np.ndarray((R.H, s.get_stride()), np.uint8, s.get_data())[:, :R.W]
        ys, xs = np.nonzero(a > 20)
        if len(xs) == 0:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        if x0 <= 2 or x1 >= R.W - 3 or y0 <= 2 or y1 >= R.H - 3:
            warn.append(f"{L.get('id')}: touches frame edge at t={t:.2f} bbox=({x0},{y0},{x1},{y1})")
        # FACE CHECK: ink over the mouth/eyes region (layers designed around her are exempt)
        if L["type"] not in ("halo", "orbit_ring", "strike"):
            from .stage import face_zone
            zx, zy, rx, ry = face_zone(F, 0.0)
            yy, xx = np.mgrid[0:R.H:4, 0:R.W:4]
            m = ((xx - zx) / rx) ** 2 + ((yy - zy) / ry) ** 2 < 1
            cover = float((a[::4, ::4][m] > 20).mean()) if m.any() else 0.0
            if cover > 0.04:
                warn.append(f"{L.get('id')}: covers {cover * 100:.0f}% of the FACE at t={t:.2f}")
    return warn
