"""
mograph.v14.compositor — the v14 "Studio Depth" frame pipeline for talking-head reels.

WHY A NEW PIPELINE (v13.9 audit, see V14_AUDIT.md):
  * v13.9 drew everything on ONE cairo canvas → no true depth, glass could not blur what is under it, glows
    never touched the speaker, light spill was a fake gradient, behind-speaker text had halo edges.
  * v14 renders every frame as a physically ordered sandwich:

     1  plate       source frame → virtual camera (warpAffine) → speaker matte (MODNet, cached)
     2  background  mask-normalised DOF blur + grade (dim / desat / tint, keyframable) – no person ghost
     3  BACK pass   layers with pass "back" (titles behind the head, halos, domes, back half of 3D rings)
                    + their EMISSION (glow) on a half-res light buffer
     4  speaker     graded, LIGHT-WRAPPED by what is behind her, LIGHT-SPILLED by every glowing layer
                    (red disc rim tints the collar, gold title tints the shoulder …)
     5  FRONT pass  layers with pass "front" (glass reads the real composite under it → true frosted glass)
     6  bloom       two-radius bloom from the light buffers, vignette, grain

SPEC (JSON) — see V14_SPEC.md for every key:
  {"meta": {"w":1080,"h":1920,"fps":30},
   "source": {"src":"media/raw.mp4","words":"words.json","segments":[[1.0,7.1],[7.45,17.6]]},
   "grade": {...}, "camera": [...], "subtitles": {...}, "layers": [...], "audio": {...}}
Times anywhere: seconds (output timeline) | "w:word" | "w:word#2" | "we:word" (word end) | "+0.2" offsets.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time

import cairo
import cv2
import numpy as np

from . import physics as ph
from .track import Tracks

# =====================================================================================================  timing
_NORM = re.compile(r"[\s\u200c.,!?؟،:;«»\"'()\-]")


def norm_word(w):
    return _NORM.sub("", w).replace("ي", "ی").replace("ك", "ک")


class Timeline:
    """EDL: output time ↔ source time, and word anchors in OUTPUT time."""

    def __init__(self, segments, words):
        self.seg = [(float(a), float(b)) for a, b in segments]
        self.cum = [0.0]
        for a, b in self.seg:
            self.cum.append(self.cum[-1] + (b - a))
        self.duration = self.cum[-1]
        self.cuts = self.cum[1:-1]
        self.words = []
        for w in words:
            ot = self.to_out(w["t"])
            if ot is None:
                continue
            oe = self.to_out(min(w["end"], self._seg_end(w["t"]))) or ot
            self.words.append({"w": w["w"], "n": norm_word(w["w"]), "t": ot, "end": max(ot + 0.05, oe),
                               "st": w["t"]})

    def _seg_end(self, s):
        for a, b in self.seg:
            if a <= s <= b:
                return b
        return s

    def to_out(self, s):
        for i, (a, b) in enumerate(self.seg):
            if a - 1e-6 <= s <= b + 1e-6:
                return self.cum[i] + (s - a)
        return None

    def to_src(self, t):
        t = min(max(0.0, t), self.duration - 1e-4)
        for i, (a, b) in enumerate(self.seg):
            if t <= self.cum[i + 1]:
                return a + (t - self.cum[i])
        return self.seg[-1][1]

    def word(self, key, end=False):
        occ = 1
        if "#" in key:
            key, o = key.rsplit("#", 1)
            occ = int(o)
        n = norm_word(key)
        hits = [w for w in self.words if w["n"] == n] or [w for w in self.words if w["n"].startswith(n)]
        if not hits:
            raise KeyError(f"word anchor '{key}' not found in transcript")
        w = hits[min(len(hits), occ) - 1]
        return w["end"] if end else w["t"]

    def resolve(self, v, default=None):
        if v is None:
            return default
        if isinstance(v, (int, float)):
            return float(v)
        v = str(v).strip()
        m = re.match(r"^(we?):(.+?)([+-]\d*\.?\d+)?$", v)
        if m:
            base = self.word(m.group(2), end=(m.group(1) == "we"))
            return base + (float(m.group(3)) if m.group(3) else 0.0)
        return float(v)


def load_words(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, dict):
        d = d.get("words", [])
    out = []
    for w in d:
        if isinstance(w, (list, tuple)):
            out.append({"w": w[0], "t": float(w[1]), "end": float(w[2])})
        else:
            out.append({"w": w.get("w") or w.get("text"), "t": float(w.get("t", w.get("start", 0))),
                        "end": float(w.get("end", 0))})
    return out


# =====================================================================================================  camera
class Camera:
    """Virtual camera on the source plate. keys: [t, zoom, cx, cy, ease?, rot?]  (cx, cy in source uv)."""

    def __init__(self, keys, shakes=None):
        self.keys = sorted([list(k) for k in keys], key=lambda k: k[0]) or [[0, 1.0, 0.5, 0.5]]
        self.shakes = shakes or []

    def at(self, t):
        ks = self.keys
        if t <= ks[0][0]:
            k = ks[0]
            z, cx, cy, rot = k[1], k[2], k[3], (k[5] if len(k) > 5 else 0.0)
        elif t >= ks[-1][0]:
            k = ks[-1]
            z, cx, cy, rot = k[1], k[2], k[3], (k[5] if len(k) > 5 else 0.0)
        else:
            for i in range(1, len(ks)):
                if t <= ks[i][0]:
                    a, b = ks[i - 1], ks[i]
                    e = b[4] if len(b) > 4 else "silk"
                    p = (t - a[0]) / max(1e-6, b[0] - a[0])
                    if e == "step":
                        p = 0.0
                    elif e == "spring":
                        p = ph.sp(t - a[0], "bouncy")
                    elif e == "jelly":
                        p = ph.sp(t - a[0], "jelly")
                    elif e == "expo":
                        p = ph.EXPO_OUT(p)
                    elif e == "linear":
                        pass
                    else:
                        p = ph.SILK(p)
                    ra = a[5] if len(a) > 5 else 0.0
                    rb = b[5] if len(b) > 5 else 0.0
                    z = a[1] + (b[1] - a[1]) * p
                    cx = a[2] + (b[2] - a[2]) * p
                    cy = a[3] + (b[3] - a[3]) * p
                    rot = ra + (rb - ra) * p
                    break
        dx = dy = 0.0
        for s in self.shakes:            # [t, dur, amp_px]
            lt = t - s[0]
            if 0 <= lt <= s[1]:
                e = (1 - lt / s[1]) ** 2 * s[2]
                dx += e * math.sin(lt * 71.0) * 0.8 + e * math.sin(lt * 43.0) * 0.4
                dy += e * math.cos(lt * 59.0) * 0.7
                rot += e * 0.04 * math.sin(lt * 37.0)
        return z, cx, cy, rot, dx, dy


# =====================================================================================================  frame ctx
class Frame:
    """Everything a layer needs for one frame."""

    def __init__(self, R, t):
        self.R, self.t = R, t
        self.W, self.H = R.W, R.H
        self.src_t = R.tl.to_src(t)
        self.M = None
        self.zoom = 1.0
        self.pass_ = "front"
        self.prepass = False
        self.surf = None          # current canvas surface (for glass)
        self.ectx = None          # emission context (half res, already scaled 0.5)
        self.events = []

    def anchor(self, name, src_t=None):
        u, v = self.R.tracks.anchor(name, self.src_t if src_t is None else src_t)
        return self.uv(u, v)

    def uv(self, u, v):
        M = self.M
        x, y = u * self.R.sw, v * self.R.sh
        return (M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2])

    def head_px(self):
        """Head width in screen px (scales layers with the speaker + camera zoom)."""
        return self.R.tracks.head_width(self.src_t) * self.R.sw * self.M[0, 0]

    def shoulder_px(self):
        return self.R.tracks.shoulder_width(self.src_t) * self.R.sw * self.M[0, 0]


# =====================================================================================================  glass
def surface_array(surf):
    surf.flush()
    h, w, st = surf.get_height(), surf.get_width(), surf.get_stride()
    return np.ndarray((h, st // 4, 4), np.uint8, surf.get_data())[:, :w]


def glass(ctx, F, path_fn, bbox, blur=16.0, tint=(0.06, 0.07, 0.09, 0.42), lift=0.06, sat=0.85, alpha=1.0):
    """TRUE frosted glass: blurs the real composite under the shape (not a flat fill)."""
    if F.prepass or alpha <= 0.003:
        return
    x0, y0, x1, y1 = [int(v) for v in bbox]
    pad = int(blur * 2.5) + 2
    X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
    X1, Y1 = min(F.W, x1 + pad), min(F.H, y1 + pad)
    if X1 - X0 < 4 or Y1 - Y0 < 4:
        return
    arr = surface_array(F.surf)
    crop = arr[Y0:Y1, X0:X1].copy()
    small = cv2.resize(crop, ((X1 - X0) // 2 + 1, (Y1 - Y0) // 2 + 1), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(1.0, blur / 2))
    crop = cv2.resize(small, (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_LINEAR)
    if sat != 1.0 or lift:
        f = crop[:, :, :3].astype(np.float32)
        g = f.mean(axis=2, keepdims=True)
        f = g + (f - g) * sat
        f = f + (255 - f) * lift
        crop[:, :, :3] = np.clip(f, 0, 255).astype(np.uint8)
    crop[:, :, 3] = 255
    crop = np.ascontiguousarray(crop)
    gs = cairo.ImageSurface.create_for_data(crop, cairo.FORMAT_ARGB32, X1 - X0, Y1 - Y0,
                                            cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, X1 - X0))
    ctx.save()
    ctx.new_path()
    path_fn(ctx)
    ctx.clip()
    ctx.set_source_surface(gs, X0, Y0)
    ctx.paint_with_alpha(alpha)
    ctx.set_source_rgba(tint[0], tint[1], tint[2], tint[3] * alpha)
    ctx.paint()
    ctx.restore()
    del gs


# =====================================================================================================  renderer
class Renderer:
    def __init__(self, spec, base_dir="."):
        if isinstance(spec, str):
            base_dir = os.path.dirname(os.path.abspath(spec))
            with open(spec, encoding="utf-8") as f:
                spec = json.load(f)
        self.spec, self.base = spec, base_dir
        m = spec.get("meta", {})
        self.W, self.H, self.fps = int(m.get("w", 1080)), int(m.get("h", 1920)), float(m.get("fps", 30))
        src = spec["source"]
        self.src = os.path.join(base_dir, src["src"])
        self.cap = cv2.VideoCapture(self.src)
        self.sfps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.sw, self.sh = int(self.cap.get(3)), int(self.cap.get(4))
        self.sn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._pos, self._last = -1, None
        words = load_words(os.path.join(base_dir, src["words"])) if src.get("words") else []
        segs = src.get("segments") or [[0.0, self.sn / self.sfps]]
        self.tl = Timeline(segs, words)
        self.duration = float(m.get("duration", self.tl.duration))
        self.tracks = Tracks(self.src)
        cam = spec.get("camera", {})
        keys = [[self.tl.resolve(k[0])] + list(k[1:]) for k in cam.get("keys", [[0, 1, .5, .5]])]
        shakes = [[self.tl.resolve(s[0])] + list(s[1:]) for s in cam.get("shakes", [])]
        for c in self.tl.cuts if cam.get("punch_cuts", True) else []:
            pass
        self.cam = Camera(keys, shakes)
        from . import layers as LY
        from . import glyphs as _G
        _G.DEFAULT_THEME[0] = m.get("theme", "doran")
        pm = m.get("palette", "auto")             # colours DESIGNED from this footage (palette.py), not copied
        if pm == "auto":
            from .palette import auto_palette
            _G.set_palette(auto_palette(self.src))
        elif isinstance(pm, dict):
            _G.set_palette(pm)
        self.palette = dict(_G.PALETTE)
        self.LY = LY
        self.layers = [LY.prepare(L, self) for L in spec.get("layers", [])]
        subs = spec.get("subtitles")
        self.subs = LY.prepare_subs(subs, self) if subs else None
        g = spec.get("grade", {})
        self.grade = g
        self.grade_keys = [[self.tl.resolve(k[0]), k[1]] for k in g.get("keys", [])]
        self.post = spec.get("post", {})
        self._vig = None

    # ------------------------------------------------------------------ source
    def src_frame(self, st):
        i = max(0, min(self.sn - 1, int(round(st * self.sfps))))
        if i == self._pos and self._last is not None:
            return self._last, i
        if i != self._pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = self.cap.read()
        if not ok:
            fr = self._last if self._last is not None else np.zeros((self.sh, self.sw, 3), np.uint8)
        self._pos, self._last = i, fr
        return fr, i

    def matrix(self, t):
        z, cx, cy, rot, dx, dy = self.cam.at(t)
        k0 = max(self.W / self.sw, self.H / self.sh)
        k = k0 * max(1.0, z)
        hw, hh = self.W / (2 * k), self.H / (2 * k)
        cxp = min(max(cx * self.sw, hw), self.sw - hw)
        cyp = min(max(cy * self.sh, hh), self.sh - hh)
        M = np.array([[k, 0, self.W / 2 - k * cxp + dx], [0, k, self.H / 2 - k * cyp + dy]], np.float64)
        if abs(rot) > 1e-3:
            Rm = cv2.getRotationMatrix2D((self.W / 2, self.H / 2), rot, 1.0)
            M = np.vstack([Rm, [0, 0, 1]]) @ np.vstack([M, [0, 0, 1]])
            M = M[:2]
        return M, z

    def grade_at(self, t):
        base = dict(self.grade.get("bg", {}))
        ks = self.grade_keys
        if not ks:
            return base
        cur = dict(base)
        for i, (kt, kv) in enumerate(ks):
            if t < kt:
                prev = ks[i - 1] if i > 0 else None
                if prev is None:
                    return cur
                p = ph.SILK((t - prev[0]) / max(1e-6, kt - prev[0]))
                a, b = dict(base, **prev[1]), dict(base, **kv)
                return {k: (a.get(k, 0) + (b.get(k, 0) - a.get(k, 0)) * p) if isinstance(b.get(k), (int, float))
                        else b.get(k) for k in set(a) | set(b)}
            cur = dict(base, **kv)
        return cur

    # ------------------------------------------------------------------ frame
    def render(self, t):
        W, H = self.W, self.H
        F = Frame(self, t)
        st = F.src_t
        fr, idx = self.src_frame(st)
        M, z = self.matrix(t)
        F.M, F.zoom = M, z
        plate = cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        A = cv2.warpAffine(self.tracks.matte(idx), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        A = np.clip((A - 0.04) / 0.92, 0, 1).astype(np.float32)
        # ---------------- background: mask-normalised DOF blur (no person ghost) + grade
        g = self.grade_at(t)
        q = 4
        sm = cv2.resize(plate, (W // q, H // q), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        am = cv2.resize(A, (W // q, H // q), interpolation=cv2.INTER_AREA)[..., None]
        inv = 1 - np.clip(am * 1.6, 0, 1)
        sig = max(0.5, float(g.get("blur", 6)) / q)
        num = cv2.GaussianBlur(sm * inv, (0, 0), sig)
        den = cv2.GaussianBlur(inv, (0, 0), sig)[..., None] + 1e-4
        bg = num / den
        far = cv2.GaussianBlur(sm * inv, (0, 0), sig * 4) / (cv2.GaussianBlur(inv, (0, 0), sig * 4)[..., None] + 1e-4)
        hole = np.clip((den - 0.02) * 6, 0, 1)
        bg = bg * hole + far * (1 - hole)
        lum = bg.mean(axis=2, keepdims=True)
        bg = lum + (bg - lum) * (1 - float(g.get("desat", 0.35)))
        tint = np.array(_hex_bgr(g.get("tint", "#2a1a10")), np.float32)
        bg = bg * (1 - float(g.get("dim", 0.45))) + tint * float(g.get("tint_amt", 0.25))
        if g.get("vignette_bg", 0.0):
            bg = bg * self._vignette(W // q, H // q, float(g["vignette_bg"]))[..., None]
        bg = cv2.resize(np.clip(bg, 0, 1), (W, H), interpolation=cv2.INTER_LINEAR)
        canvas = (bg * 255).astype(np.uint8)
        surf = _bgr_to_surface(canvas)
        ctx = cairo.Context(surf)
        esb = cairo.ImageSurface(cairo.FORMAT_ARGB32, W // 2, H // 2)
        esf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W // 2, H // 2)
        ectx_b = cairo.Context(esb); ectx_b.scale(0.5, 0.5)
        ectx_f = cairo.Context(esf); ectx_f.scale(0.5, 0.5)
        F.surf = surf
        # ---------------- STAGE pass (local scrim + grid + coloured light behind each element; stage.py)
        F.pass_, F.ectx = "back", ectx_b
        self._stage_pass(F)
        # ---------------- BACK pass
        self._draw_pass(ctx, F, "back")
        # ---------------- FRONT emission pre-pass (so front glows can light the speaker)
        dummy = cairo.ImageSurface(cairo.FORMAT_A8, 1, 1)
        F.pass_, F.ectx, F.prepass = "front", ectx_f, True
        self._draw_pass(cairo.Context(dummy), F, "front", events=False)
        F.prepass = False
        # ---------------- speaker: grade + light wrap + spill
        cv = surface_array(surf)[:, :, :3].astype(np.float32) / 255
        # environment light: every glowing layer (back + front) throws a wide, soft pool of coloured light onto
        # the BACKGROUND (ground, trees) — brighter surfaces catch more, like real bounce light (reference look).
        envk = float(self.post.get("env_light", 0.9))
        if envk > 0:
            Eall = _surf_rgb(esb) + _surf_rgb(esf)
            q2 = 4
            Es = cv2.resize(Eall, (W // (2 * q2), H // (2 * q2)), interpolation=cv2.INTER_AREA)
            pool = cv2.GaussianBlur(Es, (0, 0), 22) * 0.9 + cv2.GaussianBlur(Es, (0, 0), 60) * 0.7
            pool = cv2.resize(pool, (W, H), interpolation=cv2.INTER_LINEAR) * envk
            cl = cv.mean(axis=2, keepdims=True)
            cv = np.clip(cv + pool * (0.35 + 1.4 * cl), 0, 1)
        P = plate.astype(np.float32) / 255
        fg = self.grade.get("fg", {})
        P = (P - 0.5) * float(fg.get("contrast", 1.06)) + 0.5 + float(fg.get("lift", 0.0))
        warm = float(fg.get("warm", 0.03))
        P[..., 2] *= 1 + warm; P[..., 0] *= 1 - warm
        Pl = P.mean(axis=2, keepdims=True)
        P = Pl + (P - Pl) * float(fg.get("sat", 1.0))
        P = np.clip(P, 0, 1)
        Eb = _surf_rgb(esb)          # half-res linear light (BGR float)
        Ef = _surf_rgb(esf)
        a3 = A[..., None]
        # light wrap: the background (incl. glowing back layers) bleeds around the speaker's silhouette
        Ah = cv2.resize(A, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        edge = np.clip(Ah - cv2.GaussianBlur(Ah, (0, 0), 7), 0, 1) * 2.2
        edge = cv2.resize(np.clip(edge, 0, 1), (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
        cvh = cv2.resize(cv, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        wrap_src = cv2.GaussianBlur(cvh, (0, 0), 10) * float(self.post.get("wrap_bg", 0.55)) + \
            cv2.GaussianBlur(Eb, (0, 0), 9) * float(self.post.get("wrap_glow", 1.6))
        wrap = cv2.resize(wrap_src, (W, H), interpolation=cv2.INTER_LINEAR)
        P = 1 - (1 - P) * (1 - np.clip(wrap * edge, 0, 1))          # screen
        # light spill: every glowing layer lights the speaker (strongest on bright clothes / skin)
        spill_src = cv2.GaussianBlur(Eb + Ef, (0, 0), 38) + cv2.GaussianBlur(Ef, (0, 0), 12) * 0.6
        spill = cv2.resize(spill_src, (W, H), interpolation=cv2.INTER_LINEAR) * float(self.post.get("spill", 1.25))
        Pl = P.mean(axis=2, keepdims=True)
        P = np.clip(P + spill * (0.25 + 0.95 * Pl) * (0.5 + 0.5 * P), 0, 1)
        comp = cv * (1 - a3) + P * a3
        arr = surface_array(surf)
        arr[:, :, :3] = (comp * 255 + 0.5).astype(np.uint8)
        surf.mark_dirty()
        # ---------------- FRONT pass
        F.pass_, F.ectx = "front", None
        ctx = cairo.Context(surf)
        self._draw_pass(ctx, F, "front", emission=False)
        # ---------------- bloom + post
        out = surface_array(surf)[:, :, :3].astype(np.float32) / 255
        E = Eb * (1 - cv2.resize(A, (W // 2, H // 2), interpolation=cv2.INTER_AREA)[..., None] * 0.8) + Ef
        bl = cv2.GaussianBlur(E, (0, 0), 6) * 0.55 + cv2.GaussianBlur(E, (0, 0), 22) * 0.75 + \
            cv2.GaussianBlur(E, (0, 0), 60) * 0.45
        bl = cv2.resize(bl, (W, H), interpolation=cv2.INTER_LINEAR) * float(self.post.get("bloom", 1.0))
        out = 1 - (1 - out) * (1 - np.clip(bl, 0, 1))
        v = float(self.post.get("vignette", 0.28))
        if v:
            out *= self._vignette(W, H, v)[..., None]
        gr = float(self.post.get("grain", 0.018))
        if gr:
            rng = np.random.default_rng(int(t * 1000))
            out += (rng.standard_normal((H // 2, W // 2, 1)).astype(np.float32) * gr).repeat(2, 0).repeat(2, 1)
        fade = self.post.get("fade")
        if fade:
            fi, fo = float(fade[0]), float(fade[1])
            k = min(1.0, t / fi if fi > 0 else 1.0, (self.duration - t) / fo if fo > 0 else 1.0)
            out *= max(0.0, k)
        return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), F

    def _vignette(self, W, H, amt):
        key = (W, H, amt)
        if self._vig is None or self._vig[0] != key:
            y, x = np.mgrid[0:H, 0:W].astype(np.float32)
            d = np.sqrt(((x - W / 2) / (W / 2)) ** 2 * 0.8 + ((y - H / 2) / (H / 2)) ** 2)
            self._vig = (key, np.clip(1 - amt * np.clip(d - 0.45, 0, 1) ** 1.6, 0, 1).astype(np.float32))
        return self._vig[1]

    def _stage_pass(self, F):
        from .stage import draw_stage, element_alpha, STAGE_DEFAULT
        for L in self.layers:
            if not L.get("stage", STAGE_DEFAULT.get(L["type"], False)):
                continue
            if not (L["_t0"] - 0.05 <= F.t <= L["_t1"] + 0.4):
                continue
            if L["type"] == "kinetic_title" and not any(w["sname"] in ("hero", "hero_white", "alert", "callig")
                                                         for w in self.LY.blk(L).words):
                continue
            try:
                x, y, _ = self.LY.pos(L, F)
                r = self.LY.element_radius(L, F, F.head_px()) or 0.6 * F.head_px()
            except Exception:  # noqa: BLE001
                continue
            col = L.get("stage_color") or {"halo": L.get("color", "@alert"), "disc": L.get("neon", "@alert"),
                                           "glass_shape": L.get("stroke", "@accent")}.get(L["type"])
            if col is None and L.get("text"):      # light = colour of the block's hero word (red title → red light)
                b = self.LY.blk(L)
                hs = [g.style for g in b.glyphs if g.sname in ("hero", "alert", "callig", "hero_white")]
                col = hs[0].get("glow_color") or hs[0].get("color") if hs else "@accent"
            col = col or "@accent"
            draw_stage(F, L, x, y, r, col, element_alpha(L, F))

    def _draw_pass(self, ctx, F, pass_, emission=True, events=True):
        items = [L for L in self.layers if self.LY.in_pass(L, pass_) and L["_t0"] - 0.05 <= F.t <= L["_t1"] + 0.05]
        if pass_ == "front" and self.subs:
            items.append(self.subs)
        items.sort(key=lambda L: L.get("z", 0))
        for L in items:
            ctx.save()
            try:
                self.LY.draw(ctx, L, F)
            finally:
                ctx.restore()

    # ------------------------------------------------------------------ video
    def render_video(self, out, start=0.0, end=None, scale=1.0, log=print, crf=17):
        end = self.duration if end is None else min(end, self.duration)
        n = int(round((end - start) * self.fps))
        W, H = int(self.W * scale) // 2 * 2, int(self.H * scale) // 2 * 2
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
               "-r", str(self.fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-threads", "2", "-x264-params", "rc-lookahead=10:ref=2",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        t0 = time.time()
        for i in range(n):
            t = start + i / self.fps
            img, _ = self.render(t)
            if scale != 1.0:
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            p.stdin.write(img.tobytes())
            if i % 30 == 0:
                el = time.time() - t0
                log(f"[v14 render] {i}/{n}  {el / (i + 1):.2f}s/frame", flush=True)
        p.stdin.close()
        p.wait()
        return out


# =====================================================================================================  helpers
def _hex_bgr(h):
    h = h.lstrip("#")
    return [int(h[4:6], 16) / 255, int(h[2:4], 16) / 255, int(h[0:2], 16) / 255]


def _bgr_to_surface(bgr):
    H, W = bgr.shape[:2]
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    a = surface_array(s)
    a[:, :, :3] = bgr
    a[:, :, 3] = 255
    s.mark_dirty()
    return s


def _surf_rgb(s):
    return surface_array(s)[:, :, :3].astype(np.float32) / 255.0
