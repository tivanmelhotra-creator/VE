"""
mograph.hbc — zero-wheel HarfBuzz: the system C library `libharfbuzz.so.0` through ctypes, plus fontTools
glyph outlines.  v9 Studio Apex.

Why: the `uharfbuzz` pip wheel is often missing in offline sandboxes, but the native shaper that every
Linux desktop stack (Pango, Chromium, ffmpeg/libass) depends on is always there.  This module exposes the
tiny subset of the uharfbuzz API that `mograph.text` uses, so text.py runs the SAME vector code path
(true glyph outlines → cairo paths, exact ink boxes, karaoke per-word geometry) without any wheel:

    Blob.from_file_path(p) · Face(blob) · face.upem · face.unicodes
    Font(face) · font.get_font_extents("ltr") · font.draw_glyph_with_pen(gid, pen)
    Buffer() · buf.add_str(s) · buf.guess_segment_properties() · buf.direction = "rtl"
    shape(font, buf, features) · buf.glyph_infos[i].codepoint/.cluster · buf.glyph_positions[i].x_advance …

Clusters are CODEPOINT indices (text is added as UTF-32), exactly like uharfbuzz.add_str.
Glyph outlines come from fontTools `TTFont.getGlyphSet()[name].draw(pen)` (glyf + CFF), so the pen
protocol is the standard fontTools one (moveTo / lineTo / curveTo / qCurveTo / closePath / endPath).

available() → True when the C library loads.  Nothing here allocates per frame: faces/fonts are created
once per font file and buffers are recycled by the caller's lru caches.
"""
from __future__ import annotations

import ctypes
import ctypes.util

_LIB = None
_ERR = None


def _load():
    global _LIB, _ERR
    if _LIB is not None or _ERR is not None:
        return _LIB
    for name in ("libharfbuzz.so.0", ctypes.util.find_library("harfbuzz") or "", "libharfbuzz.so"):
        if not name:
            continue
        try:
            _LIB = ctypes.CDLL(name)
            break
        except OSError as e:  # noqa: PERF203
            _ERR = e
    if _LIB is None:
        return None
    L = _LIB
    vp, u32, i32, cp = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32, ctypes.c_char_p
    sig = {
        "hb_blob_create_from_file": ([cp], vp),
        "hb_blob_destroy": ([vp], None),
        "hb_face_create": ([vp, ctypes.c_uint], vp),
        "hb_face_get_upem": ([vp], ctypes.c_uint),
        "hb_face_destroy": ([vp], None),
        "hb_font_create": ([vp], vp),
        "hb_font_destroy": ([vp], None),
        "hb_font_get_h_extents": ([vp, vp], ctypes.c_int),
        "hb_font_get_nominal_glyph": ([vp, u32, ctypes.POINTER(u32)], ctypes.c_int),
        "hb_buffer_create": ([], vp),
        "hb_buffer_destroy": ([vp], None),
        "hb_buffer_reset": ([vp], None),
        "hb_buffer_add_utf32": ([vp, ctypes.POINTER(u32), ctypes.c_int, ctypes.c_uint, ctypes.c_int], None),
        "hb_buffer_guess_segment_properties": ([vp], None),
        "hb_buffer_set_direction": ([vp, ctypes.c_int], None),
        "hb_buffer_get_direction": ([vp], ctypes.c_int),
        "hb_buffer_get_length": ([vp], ctypes.c_uint),
        "hb_buffer_get_glyph_infos": ([vp, ctypes.POINTER(ctypes.c_uint)], vp),
        "hb_buffer_get_glyph_positions": ([vp, ctypes.POINTER(ctypes.c_uint)], vp),
        "hb_shape": ([vp, vp, vp, ctypes.c_uint], None),
        "hb_version_string": ([], cp),
    }
    for fn, (args, res) in sig.items():
        f = getattr(L, fn)
        f.argtypes = args
        f.restype = res
    return L


def available():
    return _load() is not None


def version():
    L = _load()
    return L.hb_version_string().decode() if L else None


class _GlyphInfo(ctypes.Structure):
    _fields_ = [("codepoint", ctypes.c_uint32), ("mask", ctypes.c_uint32), ("cluster", ctypes.c_uint32),
                ("var1", ctypes.c_uint32), ("var2", ctypes.c_uint32)]


class _GlyphPos(ctypes.Structure):
    _fields_ = [("x_advance", ctypes.c_int32), ("y_advance", ctypes.c_int32), ("x_offset", ctypes.c_int32),
                ("y_offset", ctypes.c_int32), ("var", ctypes.c_uint32)]


class _FontExtents(ctypes.Structure):
    _fields_ = [("ascender", ctypes.c_int32), ("descender", ctypes.c_int32), ("line_gap", ctypes.c_int32)] + \
               [(f"r{i}", ctypes.c_int32) for i in range(9)]


_DIR = {"ltr": 4, "rtl": 5, "ttb": 6, "btt": 7}
_DIR_R = {v: k for k, v in _DIR.items()}


class Blob:
    def __init__(self, ptr, path):
        self.ptr, self.path = ptr, path

    @classmethod
    def from_file_path(cls, path):
        L = _load()
        p = L.hb_blob_create_from_file(str(path).encode())
        return cls(p, str(path))


class Face:
    def __init__(self, blob, index=0):
        L = _load()
        self.blob = blob
        self.path = blob.path
        self.ptr = L.hb_face_create(blob.ptr, index)
        self.upem = int(L.hb_face_get_upem(self.ptr))
        self._tt = None
        self._unicodes = None

    def ttfont(self):
        if self._tt is None:
            from fontTools.ttLib import TTFont
            self._tt = TTFont(self.path, lazy=True)
        return self._tt

    @property
    def unicodes(self):
        if self._unicodes is None:
            self._unicodes = set(self.ttfont().getBestCmap().keys())
        return self._unicodes


class _Extents:
    __slots__ = ("ascender", "descender", "line_gap")


class Font:
    def __init__(self, face):
        L = _load()
        self.face = face
        self.ptr = L.hb_font_create(face.ptr)      # scale defaults to upem → positions are font units
        self._gs = None
        self._order = None

    def get_font_extents(self, direction="ltr"):
        e = _FontExtents()
        _load().hb_font_get_h_extents(self.ptr, ctypes.byref(e))
        out = _Extents()
        out.ascender, out.descender, out.line_gap = e.ascender, e.descender, e.line_gap
        return out

    def get_nominal_glyph(self, cp):
        g = ctypes.c_uint32(0)
        ok = _load().hb_font_get_nominal_glyph(self.ptr, int(cp), ctypes.byref(g))
        return int(g.value) if ok else None

    def glyph_name(self, gid):
        if self._order is None:
            self._order = self.face.ttfont().getGlyphOrder()
        return self._order[gid] if 0 <= gid < len(self._order) else None

    def draw_glyph_with_pen(self, gid, pen):
        """Outline of glyph `gid` in font units (y up), via fontTools (works for glyf and CFF)."""
        if self._gs is None:
            self._gs = self.face.ttfont().getGlyphSet()
        name = self.glyph_name(gid)
        if name is None or name not in self._gs:
            return
        if hasattr(pen, "glyphSet"):
            pen.glyphSet = self._gs          # composite glyphs (ligatures / Arabic forms) decompose via components
        self._gs[name].draw(pen)

    def advance(self, gid):
        name = self.glyph_name(gid)
        try:
            return self.face.ttfont()["hmtx"][name][0]
        except Exception:  # noqa: BLE001
            return 0


class Buffer:
    def __init__(self):
        self.ptr = _load().hb_buffer_create()
        self._infos = None
        self._pos = None

    def __del__(self):
        try:
            if self.ptr and _LIB is not None:
                _LIB.hb_buffer_destroy(self.ptr)
        except Exception:  # noqa: BLE001
            pass

    def add_str(self, text):
        cps = [ord(c) for c in text]
        arr = (ctypes.c_uint32 * max(1, len(cps)))(*cps)
        _load().hb_buffer_add_utf32(self.ptr, arr, len(cps), 0, len(cps))
        self._infos = self._pos = None

    def guess_segment_properties(self):
        _load().hb_buffer_guess_segment_properties(self.ptr)

    @property
    def direction(self):
        return _DIR_R.get(_load().hb_buffer_get_direction(self.ptr), "ltr")

    @direction.setter
    def direction(self, d):
        _load().hb_buffer_set_direction(self.ptr, _DIR.get(d, 4))

    def _fetch(self):
        L = _load()
        n = ctypes.c_uint(0)
        pi = L.hb_buffer_get_glyph_infos(self.ptr, ctypes.byref(n))
        pp = L.hb_buffer_get_glyph_positions(self.ptr, ctypes.byref(n))
        k = n.value
        infos = ctypes.cast(pi, ctypes.POINTER(_GlyphInfo * k)).contents if k else []
        pos = ctypes.cast(pp, ctypes.POINTER(_GlyphPos * k)).contents if k else []
        # copy out (the C arrays belong to the buffer)
        self._infos = [_GI(g.codepoint, g.cluster) for g in infos]
        self._pos = [_GP(p.x_advance, p.y_advance, p.x_offset, p.y_offset) for p in pos]

    @property
    def glyph_infos(self):
        if self._infos is None:
            self._fetch()
        return self._infos

    @property
    def glyph_positions(self):
        if self._pos is None:
            self._fetch()
        return self._pos


class _GI:
    __slots__ = ("codepoint", "cluster")

    def __init__(self, cp, cl):
        self.codepoint, self.cluster = cp, cl


class _GP:
    __slots__ = ("x_advance", "y_advance", "x_offset", "y_offset")

    def __init__(self, xa, ya, xo, yo):
        self.x_advance, self.y_advance, self.x_offset, self.y_offset = xa, ya, xo, yo


def shape(font, buf, features=None):
    """features are HarfBuzz defaults (kern, liga, Arabic init/medi/fina/rlig are always on)."""
    _load().hb_shape(font.ptr, buf.ptr, None, 0)
    buf._infos = buf._pos = None
