"""
mograph.media — real footage inside the vector world.

1) element "video" — a clip placed like any element (follows camera, enters/exits, QA bbox):
   {"type":"video","src":"media/coin.mp4","x":540,"y":900,"w":720,      # or "fill":true (whole screen)
    "t":"w:سکه","dur":0.4,"out":"w:ولی","clip_start":0.0,"speed":1.0,"loop":false,
    "blend":"normal|screen|multiply","grade":"none|mono|noir","radius":24,"border":true}
   blend "screen" on a black background = only the bright parts show → footage glows into the line art.
   grade "noir" = mono + crushed blacks + soft contrast → matches the ENSAN noir style.

2) plates (used by mograph.omni) — a clip that REPLACES (or sits UNDER) the vector world between two
   absolute times; HUD/subtitles/post-FX stay vector-crisp on top.

Frames are read sequentially with OpenCV (one decoder per process, seek only on jumps) → cheap.
"""
from __future__ import annotations

import math

import cairo
import cv2
import numpy as np

from .core import clamp, color as _color, ease_in_out, lerp, prog
from .elements import element, xy, _apply_xform
from .geometry import compile_path, draw_path, rounded_rect_d

_READERS = {}


class _Reader:
    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video '{path}'")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 24.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.pos = -1
        self.last = None

    @property
    def duration(self):
        return self.n / self.fps if self.n else 0.0

    def frame(self, t, loop=False):
        """BGR uint8 frame at clip time t (seconds)."""
        i = int(round(t * self.fps))
        if self.n:
            i = i % self.n if loop else max(0, min(self.n - 1, i))
        if i == self.pos and self.last is not None:
            return self.last
        if i != self.pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = self.cap.read()
        if not ok:
            return self.last if self.last is not None else np.zeros((max(self.h, 2), max(self.w, 2), 3), np.uint8)
        self.pos, self.last = i, fr
        return fr


def reader(path):
    r = _READERS.get(path)
    if r is None:
        r = _READERS[path] = _Reader(path)
    return r


def grade(fr, mode):
    if mode in (None, "none"):
        return fr
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    if mode == "noir":
        g = cv2.convertScaleAbs(g, alpha=1.18, beta=-22)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def to_surface(fr_bgr):
    h, w = fr_bgr.shape[:2]
    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_RGB24, w)
    buf = np.zeros((h, stride // 4, 4), np.uint8)
    buf[:, :w, :3] = fr_bgr
    buf[:, :w, 3] = 255
    return cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, w, h, stride), buf


_BLEND = {"normal": cairo.OPERATOR_OVER, "screen": cairo.OPERATOR_SCREEN, "multiply": cairo.OPERATOR_MULTIPLY,
          "add": cairo.OPERATOR_ADD, "lighten": cairo.OPERATOR_LIGHTEN}


def paint_frame(ctx, fr, x, y, w, h=None, alpha=1.0, blend="normal", radius=0, cover=True):
    """Paint a BGR frame centred at (x, y) with width w (height from aspect, or cover-crop into w×h)."""
    fh, fw = fr.shape[:2]
    if h is None:
        h = w * fh / fw
    k = max(w / fw, h / fh) if cover else min(w / fw, h / fh)
    surf, _keep = to_surface(fr)
    ctx.save()
    ctx.rectangle(x - w / 2, y - h / 2, w, h) if not radius else \
        ctx.append_path(compile_path(rounded_rect_d(x - w / 2, y - h / 2, w, h, radius)).whole)
    ctx.clip()
    ctx.translate(x, y)
    ctx.scale(k, k)
    ctx.set_source_surface(surf, -fw / 2, -fh / 2)
    ctx.get_source().set_filter(cairo.FILTER_GOOD)
    ctx.set_operator(_BLEND.get(blend, cairo.OPERATOR_OVER))
    ctx.paint_with_alpha(clamp(alpha))
    ctx.restore()
    return w, h


@element("video", dur=0.4, enter="fade", sfx="none")
def el_video(ctx, el, s):
    """Footage clip (see module doc)."""
    r = reader(el["_src_abs"] if "_src_abs" in el else el["src"])
    lt = max(0.0, s.t - float(el["t"])) * float(el.get("speed", 1.0)) + float(el.get("clip_start", 0.0))
    fr = grade(r.frame(lt, loop=bool(el.get("loop"))), el.get("grade", "none"))
    if el.get("fill"):
        sc = s.scene
        x, y, w, h = sc.W / 2, sc.H / 2, float(sc.W), float(sc.H)
    else:
        x, y = xy(el, s.t)
        w = float(el.get("w", 640))
        h = float(el["h"]) if el.get("h") else None
    ctx.save()
    if s.scale != 1.0:
        ctx.translate(x, y + s.dy); ctx.scale(s.scale, s.scale); ctx.translate(-x, -(y + s.dy))
    w, h = paint_frame(ctx, fr, x, y + s.dy, w, h, alpha=s.alpha, blend=el.get("blend", "normal"),
                       radius=float(el.get("radius", 0)))
    if el.get("border"):
        draw_path(ctx, rounded_rect_d(x - w / 2, y + s.dy - h / 2, w, h, float(el.get("radius", 0))), 0, 0, 1,
                  anchor="none", stroke=s.color, width=2.4, alpha=s.alpha, trim=s.trim if isinstance(s.trim, float) else 1)
    ctx.restore()
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


# ---------------------------------------------------------------------------
# plates (Omni outputs)
# ---------------------------------------------------------------------------
def active_plate(scene, t):
    for p in getattr(scene, "plates", []):
        if p["t0"] - p["fade"] <= t < p["t1"] + p["fade"]:
            return p
    return None


def plate_alpha(p, t):
    f = max(1e-3, p["fade"])
    return clamp(min((t - (p["t0"] - f)) / f, ((p["t1"] + f) - t) / f, 1.0))


def draw_plate(ctx, scene, p, t):
    r = reader(p["src"])
    # map scene time → clip time (clip is stretched/squeezed to exactly cover t0..t1)
    span = max(1e-3, p["t1"] - p["t0"])
    ct = clamp((t - p["t0"]) / span) * (r.duration or span) * 0.999
    fr = grade(r.frame(ct), p.get("grade", "none"))
    paint_frame(ctx, fr, scene.W / 2, scene.H / 2, scene.W, scene.H, alpha=plate_alpha(p, t),
                blend=p.get("blend", "normal"))


# ---------------------------------------------------------------------------
# v9 — talking_head: 3-layer Z-sandwich (backdrop → behind-speaker climax → segmented speaker) + PIP morph
#      + editorial punch-in on every voice-edit splice (scene.cuts)
# ---------------------------------------------------------------------------
def punch_zoom(scene, el, t):
    """1.0 ↔ 1+punch alternating at every jump cut (spec._cuts) → no visible jump cut."""
    cuts = [c for c in (getattr(scene, "cuts", None) or []) if c <= t]
    amt = float(el.get("punch", 0.12))
    if not amt or not cuts:
        return 1.0
    k = len(cuts)
    target = 1.0 + (amt if k % 2 else 0.0)
    prev = 1.0 + (amt if (k - 1) % 2 else 0.0)
    q = clamp((t - cuts[-1]) / 0.08)          # near-instant (a cut, not a zoom): 2–3 frames ease
    return lerp(prev, target, ease_in_out(q))


@element("talking_head", dur=0.4, enter="fade", sfx="none")
def el_talking_head(ctx, el, s):
    from .matte import cached_matte, premultiplied_surface, rim_surface
    from .elements import REGISTRY, compute_state
    sc = s.scene
    r = reader(el["_src_abs"] if "_src_abs" in el else el["src"])
    lt = max(0.0, s.t - float(el["t"])) * float(el.get("speed", 1.0)) + float(el.get("clip_start", 0.0))
    if getattr(sc, "cut_map", None) is not None and el.get("follow_cuts", True):
        lt = sc.cut_map(lt)                    # source time of the edited timeline (skips excised video)
    fr = r.frame(lt, loop=bool(el.get("loop", True)))
    fr = grade(fr, el.get("grade", "none"))
    idx = int(round(lt * r.fps))
    x, y = xy(el, s.t)
    W = float(el.get("w", sc.W if sc else 1080)); H = float(el.get("h", sc.H if sc else 1920))
    # matte at reduced resolution (fast), upsampled by cairo
    mh = int(el.get("matte_h", 640))
    fh, fw = fr.shape[:2]
    ms = min(1.0, mh / fh)
    small = cv2.resize(fr, (int(fw * ms), int(fh * ms)), interpolation=cv2.INTER_AREA) if ms < 1 else fr
    alpha = cached_matte(r.path, idx, small, el.get("matte", "auto"), el.get("key"))
    fb = float(el.get("feather_bottom", 0.08))          # soft lower edge: the torso never ends in a hard cut
    if fb > 0:
        n = max(1, int(alpha.shape[0] * fb))
        alpha = alpha.copy(); alpha[-n:] *= np.linspace(1, 0, n, dtype=np.float32)[:, None]
    # layout: cover-fit the plate into W×H centred at (x, y), punch-in on cuts
    k = max(W / small.shape[1], H / small.shape[0]) * punch_zoom(sc, el, s.t) * float(el.get("zoom", 1.0))
    # PIP morph
    pip_t = el.get("pip_at")
    pp = ease_in_out(prog(s.t, float(pip_t), float(el.get("pip_dur", 0.8)))) if pip_t is not None else 0.0
    pip = el.get("pip") or {}
    px, py, pr = float(pip.get("x", W * 0.78)), float(pip.get("y", H * 0.76)), float(pip.get("r", 150))
    a = s.alpha * (1 - s.out_p)
    ctx.save()
    # (a) backdrop: the original plate (fades out as the PIP forms) unless bg == "world"
    bgm = el.get("bg", "keep")
    if bgm == "keep" and pp < 1:
        paint_frame(ctx, small, x, y, W, H, alpha=a * (1 - pp))
    # (b) behind-speaker climax layer
    for j, b in enumerate(el.get("behind") or []):
        bel = b if "_c" in b else el.setdefault("_behind", {}).get(j)
        if bel is None:
            bel = dict(b)
            bel.setdefault("id", f"{el['id']}_behind{j}")
            bel["_chapter"] = el.get("_chapter")
            for kk in ("t", "out"):
                if kk in bel and sc is not None:
                    ch = sc.chapters[el["_chapter"]]
                    bel[kk] = sc.resolve_time(bel[kk], ch["t"], ch["end"])
            bel.setdefault("t", float(el["t"]))
            el["_behind"][j] = bel
        if bel.get("type") in REGISTRY:
            bs = compute_state(bel, s.t, sc)
            if bs.ok():
                bs.alpha *= a
                REGISTRY[bel["type"]]["fn"](ctx, bel, bs)
    # (c) segmented speaker (+ rim) — full frame, morphing into a circular PIP
    surf, _keep = premultiplied_surface(small, alpha)
    sw, sh = small.shape[1], small.shape[0]
    if pp > 0:
        cx, cy = lerp(x, px, pp), lerp(y, py, pp)
        rad = lerp(math.hypot(W, H) / 2, pr, pp)
        ctx.save()
        ctx.arc(cx, cy, rad, 0, 2 * math.pi); ctx.clip()
        kk = lerp(k, k * pr * 2.4 / H, pp)
        fy = float(pip.get("face_y", 0.32))           # keep the face centred in the PIP
        oy = lerp(0, (0.5 - fy) * sh * kk, pp)
        ctx.translate(cx, cy + oy); ctx.scale(kk, kk)
        ctx.set_source_surface(surf if pp < 0.999 else to_surface(small)[0], -sw / 2, -sh / 2)
        ctx.paint_with_alpha(a)
        ctx.restore()
        if pp > 0.5:                                    # VO-reactive ring
            env = sc.vo_env(s.t) if sc else 0.0
            rc = _color(el.get("rim") or "#ffffff")
            ctx.set_source_rgba(rc[0], rc[1], rc[2], a * (pp - 0.5) * 2)
            ctx.set_line_width(6 + 6 * env)
            ctx.arc(cx, cy, rad + 4 + 10 * env, 0, 2 * math.pi); ctx.stroke()
        bb = (cx - rad, cy - rad, cx + rad, cy + rad)
    else:
        ctx.save(); ctx.translate(x, y); ctx.scale(k, k)
        if el.get("rim"):
            rs, _k2 = rim_surface(alpha, el["rim"], int(el.get("rim_w", 8)))
            ctx.set_source_surface(rs, -sw / 2, -sh / 2); ctx.paint_with_alpha(a * float(el.get("rim_alpha", 0.9)))
        ctx.set_source_surface(surf, -sw / 2, -sh / 2)
        ctx.get_source().set_filter(cairo.FILTER_GOOD)
        ctx.paint_with_alpha(a)
        ctx.restore()
        bb = (x - W / 2, y - H / 2, x + W / 2, y + H / 2)
    ctx.restore()
    return bb
