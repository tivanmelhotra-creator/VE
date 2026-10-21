"""
mograph.scene — compile a JSON/dict spec into a renderable Scene.

Spec skeleton (see examples/demo_spec.json and SPEC_REFERENCE.md):
{
  "meta": {"w":1080,"h":1920,"fps":30,"duration":40,"bg":"#000000","bloom":0.4,"out":"out.mp4"},
  "hud": {...}, "audio": {...}, "subtitles": [...],
  "plugins": ["my_elements.py"],
  "chapters": [
     {"t": 0, "title": "من و شما",
      "transition": {"type": "pan", "dir": "down", "dur": 0.9},   # pan | fade | cut
      "camera": [[0, [540, 960, 1.0]], [4, [540, 760, 1.35], "smooth"]],   # LOCAL coords/time
      "elements": [ {...}, ... ]}
  ]
}

TIME VALUES inside a chapter are LOCAL seconds (0 = chapter start), OR word anchors
locked to the narration:
     "w:ممکنه"        → the moment the word «ممکنه» starts being spoken (first one in this chapter)
     "w:ممکنه#2"      → second occurrence
     "w:ممکنه+0.15"   → plus offset (also "-0.2")
     "we:ممکنه"       → the moment that word ENDS
     "end" / "end-0.5"→ chapter end
This makes every animation hit exactly on the spoken word, even when the VO timing changes.
POSITIONS inside a chapter are LOCAL to the chapter origin (0..W, 0..H is "the chapter's screen").
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import re

from .core import Camera, Track
from .elements import REGISTRY
from . import widgets as _widgets  # noqa: F401  (registers v5 compound widgets)
from . import media as _media  # noqa: F401  (registers the "video" element)
from . import typo as _typo  # noqa: F401  (v7 swiss + kinetic elements)
from . import horizon as _horizon  # noqa: F401  (v8 swiss data widgets, waveform, odometer)
from . import promo as _promo  # noqa: F401
from . import uiphysics as _uiphysics  # noqa: F401  (v9 promo / 3D / UI mockup widgets)
from . import fx as _fx  # noqa: F401  (v11 footage editing / tracking / compositing)
from . import glowfx as _glowfx  # noqa: F401  (v11 saber / flash / glow ring / tracked callout / emitter …)
from .aecomp import bridge as _aecomp  # noqa: F401  (v17 AE-Comp shape elements / lower thirds)
from . import motion as _motion  # noqa: F401,E402  (v18 motion kit: saber / character / mouth + easing specs)
from . import showcase as _showcase  # noqa: F401  (v11 device_frame / backdrop / timeline / formula …)
from . import genimage as _genimage  # noqa: F401  (v11 auto_trace / depth_plate / sprite_rig)
from . import reelfx as _reelfx
from . import visual_dna as _visual_dna
from . import creative_reels as _creative_reels  # noqa: F401  (v13.8 wireframe_badge / circular_dial / glass_hud_frame / zigzag_path_3d / before_after)  # noqa: F401  (v13.5 dna_stage / dna_connector / tactile_3d / hud_loading_bar)  # noqa: F401  (v12 title_card / logo_chip / route_arc / price_tag …)
from . import sam3d_bridge as _sam3d_bridge  # noqa: F401  (v14.4 3D: sam3d_object / image_to_3d)
from . import sculpt3d as _sculpt3d  # noqa: F401  (v14.4 3D: procedural sculpt3d meshes)
from . import real3d as _real3d  # noqa: F401  (v14.5 true 3D textured product models)
from . import pbr3d as _pbr3d  # noqa: F401  (v14.6 photoreal scanned PBR models via EffectCraft)
from . import three3d as _three3d  # noqa: F401  (v20 Procedural Three.js 3D Engine & JS bridge)
from .hud import merged_hud, prepare_subtitles

_TIME_KEYS = ("t", "out", "dim_at", "undim_at", "highlight_at", "cross_at", "morph_at", "scan_at", "drill_at",
              "stamp_at", "flip_at", "fill_at", "sweep_at",
              # v9
              "kashida_at", "kashida_back", "stroke_reveal_at", "type_at", "strike_at", "morph_t", "burst_at",
              "rec_at", "send_at", "flex_at", "cut_at", "pip_at")
# keys inside list-of-dicts that hold times (widgets: highlights, drops, rows, flips...)
_NESTED_TIME_LISTS = ("highlights", "drops", "rows", "items_t", "events", "items", "nodes")
_TARGET_KEYS = ("anchor", "target", "from", "to")
_PUNCT = "«»\"'.,،؛:!?؟()[]…"


def _norm_word(w):
    import unicodedata
    w = "".join(c for c in w if unicodedata.category(c) != "Mn")  # drop kasra/fatha... (منِ == من)
    return w.strip(_PUNCT).replace("\u200c", "").replace("ي", "ی").replace("ك", "ک")


class Scene:
    def __init__(self, spec: dict, base_dir="."):
        from .styles import apply as _apply_style
        from .text import apply_style_roles
        spec = copy.deepcopy(spec)
        if spec.get("edit"):                        # v12: EDL (jump cuts, edited subtitles, cut transitions)
            from .cut import apply_edit
            apply_edit(spec, base_dir)
        spec = _apply_style(spec)
        apply_style_roles(spec["meta"].get("fonts"))
        self.style = spec["meta"].get("style", "lineart")
        self.palette = spec["meta"].get("_palette", {})
        self.spec = spec
        self.base_dir = base_dir
        m = spec.get("meta", {})
        from .apex import ASPECTS
        if m.get("aspect") in ASPECTS and "w" not in m:          # v9 multi-aspect
            m["w"], m["h"] = ASPECTS[m["aspect"]]
        self.W, self.H = int(m.get("w", 1080)), int(m.get("h", 1920))
        self.aspect = m.get("aspect") or ("16:9" if self.W > self.H * 1.2 else "1:1" if abs(self.W - self.H) < 40
                                          else "9:16")
        self.fps = float(m.get("fps", 30))
        self.bg = m.get("bg", "#000000")
        self.bloom = float(m.get("bloom", 0.4))
        self.hud = merged_hud(spec)
        self.worlds = spec["meta"].get("_worlds") or []
        self.subs = prepare_subtitles(spec.get("subtitles"))
        b = spec.get("beats") or []
        if isinstance(b, dict):
            from .analyze import beat_grid
            b = beat_grid(b.get("bpm", 120), float(m.get("duration") or 120), float(b.get("offset", 0.0)))
        self.beats = [float(x) for x in b]
        self._load_plugins()
        self._build_chapters()
        dur = m.get("duration")
        if dur is None:
            last = 5.0
            if self.chapters:
                last = self.chapters[-1]["end"] or self.chapters[-1]["t"] + 3.0
            dur = max(last, max((s["end"] for s in self.subs), default=0) + 0.5)
        self.duration = float(dur)
        if self.chapters and self.chapters[-1]["end"] is None:
            self.chapters[-1]["end"] = self.duration
        self._fix_open_ends()
        self.by_id = {}
        for ch in self.chapters:
            for el in ch["elements"]:
                if "id" in el:
                    self.by_id[el["id"]] = el
        self._build_threads()
        _glowfx.prepare_scene_fx(self)
        self.camera = self._build_camera()
        self._has_whip_pan = any(c["transition"].get("type") == "whip_pan" for c in self.chapters)
        self.warnings = []
        for ch in self.chapters:            # media paths are relative to the spec file
            for el in ch["elements"]:
                if el.get("type") in ("video", "talking_head", "footage", "image_card") and el.get("src") and not os.path.isabs(el["src"]):
                    el["_src_abs"] = os.path.join(self.base_dir, el["src"])
        from .omni import load_plates
        self.plates, self.missing_plates = load_plates(self)
        self._vo_env = None
        # v9: jump cuts from the voice editor (spec._cuts, edited timeline) + optional source map for footage
        self.cuts = sorted(float(c) for c in (spec.get("_cuts") or []))
        self.cut_map = None
        if spec.get("_cut_source"):          # [[edited_t, source_t], ...] piecewise → talking-head footage
            import numpy as _np
            pts = _np.array(spec["_cut_source"], float)
            self.cut_map = lambda t, p=pts: float(_np.interp(t, p[:, 0], p[:, 1]))

    # ------------------------------------------------------------------
    VO_HZ = 60.0

    def vo_env(self, t):
        """v8: voice-over amplitude envelope 0..1 at time t (60 Hz table, linear interp).
        From audio.vo when present (RMS, fast attack / slower release, normalised to the 98th pct);
        offline → a synthetic syllable cadence from the subtitle word timings (so vo_pulse / waveform
        still dance in previews and in draft-voiced specs)."""
        if self._vo_env is None:
            self._vo_env = self._build_vo_env()
        env = self._vo_env
        f = t * self.VO_HZ
        i = int(f)
        if i < 0 or i >= len(env) - 1:
            return 0.0
        k = f - i
        return float(env[i] * (1 - k) + env[i + 1] * k)

    def _build_vo_env(self):
        import numpy as np
        n = int(self.duration * self.VO_HZ) + 2
        vo = (self.spec.get("audio") or {}).get("vo")
        path = os.path.join(self.base_dir, vo) if vo else None
        if path and os.path.exists(path):
            try:
                from .audio import read_audio, SR
                a = read_audio(path, self.duration)
                hop = int(SR / self.VO_HZ)
                m = len(a) // hop
                rms = np.sqrt((a[:m * hop].reshape(m, hop) ** 2).mean(axis=1) + 1e-12)
                ref = float(np.percentile(rms, 98)) or 1.0
                x = np.clip(rms / ref, 0, 1) ** 0.8
                out = np.zeros_like(x)
                lvl = 0.0
                for j, v in enumerate(x):              # attack 1 frame, release ~120 ms
                    lvl = v if v > lvl else lvl + (v - lvl) * 0.14
                    out[j] = lvl
                env = np.zeros(n, np.float32); env[:min(n, len(out))] = out[:n]
                return env
            except Exception:  # noqa: BLE001
                pass
        env = np.zeros(n, np.float32)
        tt = np.arange(n) / self.VO_HZ
        for sub in self.subs:
            for w in sub["words"]:
                d = max(0.08, w["end"] - w["t"])
                syl = max(1, round(len(_norm_word(w["w"])) / 2.4))
                m_ = (tt >= w["t"]) & (tt <= w["end"])
                ph = (tt[m_] - w["t"]) / d
                shape = np.abs(np.sin(np.pi * ph * syl)) ** 0.7 * np.sin(np.pi * ph) ** 0.35
                env[m_] = np.maximum(env[m_], (0.95 if w.get("emph") else 0.75) * shape)
        return env

    # ------------------------------------------------------------------
    def _load_plugins(self):
        for p in self.spec.get("plugins", []):
            path = p if os.path.isabs(p) else os.path.join(self.base_dir, p)
            name = "mograph_plugin_" + re.sub(r"\W", "_", os.path.basename(path))
            sp = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(mod)

    # ------------------------------------------------------------------
    def word_time(self, word, after=0.0, occurrence=1, end=False):
        target = _norm_word(word)
        k = 0
        for s in self.subs:
            for w in s["words"]:
                if w["t"] + 1e-6 < after:
                    continue
                if _norm_word(w["w"]) == target:
                    k += 1
                    if k == occurrence:
                        return w["end"] if end else w["t"]
        # fallback: partial match
        for s in self.subs:
            for w in s["words"]:
                if w["t"] + 1e-6 >= after and target and target in _norm_word(w["w"]):
                    return w["end"] if end else w["t"]
        raise KeyError(f"word anchor '{word}' not found in subtitles after t={after:.2f}s")

    def resolve_time(self, v, ch_t, ch_end):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return ch_t + float(v)
        s = str(v).strip()
        off = 0.0
        mm = re.match(r"^(.*?)([+-]\d+(?:\.\d+)?)$", s)
        if mm and not s.startswith(("+", "-")):
            s, off = mm.group(1), float(mm.group(2))
        if s == "end":
            return (ch_end if ch_end is not None else ch_t + 9999) + off
        if s.startswith(("w:", "we:")):
            is_end = s.startswith("we:")
            word = s.split(":", 1)[1]
            occ = 1
            if "#" in word:
                word, o = word.rsplit("#", 1)
                occ = int(o)
            return self.word_time(word, after=ch_t - 0.05, occurrence=occ, end=is_end) + off
        if s.startswith("abs:"):
            return float(s[4:]) + off
        if s.startswith(("b:", "bar:")):              # v9 musical beat anchors (1-based): b:12, bar:3 (4/4)
            bt = self.beats
            if not bt:
                raise KeyError(f"beat anchor '{v}' but the spec has no beats (run `mograph analyze ref.mp4 --spec`)"
                               " or set \"beats\": {\"bpm\": 120}")
            n = int(float(s.split(":", 1)[1]))
            k = (n - 1) * (4 if s.startswith("bar:") else 1)
            k = max(0, min(len(bt) - 1, k))
            return float(bt[k]) + off
        return ch_t + float(s) + off

    # ------------------------------------------------------------------
    def _build_chapters(self):
        raw = self.spec.get("chapters") or []
        chs = []
        ox, oy = 0.0, 0.0
        for i, c in enumerate(raw):
            c = copy.deepcopy(c)
            t = c.get("t")
            if isinstance(t, str):
                t = self.resolve_time(t, 0.0, None)
            c["t"] = float(t if t is not None else (chs[-1]["end"] if chs else 0.0))
            w = self.worlds[i] if i < len(self.worlds) else None
            if w is None:
                from .styles import world_for
                w = world_for(self.spec.get("meta", {}), self.spec.get("hud") or {}, c)
            # v8: no explicit transition → the chapter's OWN style default (kinetic = cut, swiss = wipe …)
            tr = c.get("transition") or w.get("transition") or self.spec.get("meta", {}).get("transition") or \
                {"type": "fade", "dur": 0.6}
            if isinstance(tr, str):
                tr = {"type": tr}
            tr = dict(tr)
            tr.setdefault("dur", {"pan": 0.9, "thread": 1.3, "wipe": 0.55, "slide": 0.6, "cut": 0.0,
                                  "portal": 0.75, "whip_pan": 0.42}.get(tr.get("type"), 0.6))
            if tr.get("type") == "whip_pan":
                tr.setdefault("edge", 0)
                tr.setdefault("ease", "in_out_expo")
            c["transition"] = tr
            c["_world"] = w        # v8: the chapter's own style world (palette / bg / fonts / sound / hud / FX)
            c["bg"] = w["bg"]
            c["_hud"] = merged_hud({"hud": w["hud"]})
            if i > 0 and tr.get("type") in ("pan", "thread") and "origin" not in c:
                d = tr.get("dir", "down")
                gap = float(tr.get("gap", 1.0))
                if d == "down": oy += self.H * gap
                elif d == "up": oy -= self.H * gap
                elif d == "left": ox -= self.W * gap
                elif d == "right": ox += self.W * gap
            if "origin" in c:
                ox, oy = c["origin"]
            c["origin"] = (float(ox), float(oy))
            chs.append(c)
        for i, c in enumerate(chs):
            c["end"] = chs[i + 1]["t"] if i + 1 < len(chs) else c.get("end")
        self.chapters = chs

    def _fix_open_ends(self):
        for idx, c in enumerate(self.chapters):
            ct, ce = c["t"], c["end"]
            els = []
            for j, el in enumerate(_showcase.expand_children([dict(e, id=e.get("id") or f"c{idx + 1}_{e.get('type', 'el')}_{jj}")
                                                              for jj, e in enumerate(c.get("elements", []))])):
                el = dict(el)
                el.setdefault("id", f"c{idx + 1}_{el.get('type', 'el')}_{j}")
                if "col" in el or "row" in el:
                    _typo.grid_place(el, self)
                if el.get("type") not in REGISTRY:
                    el["_bad"] = f"unknown element type '{el.get('type')}'"
                for k in _TIME_KEYS:
                    if k in el and not (k in ("from", "to") and el.get("type") == "morph"):
                        el[k] = self.resolve_time(el[k], ct, ce)
                for k in _NESTED_TIME_LISTS:
                    if isinstance(el.get(k), list):
                        el[k] = [dict(it, t=self.resolve_time(it["t"], ct, ce)) if isinstance(it, dict) and "t" in it
                                 else it for it in el[k]]
                for k in ("stamp", "total"):
                    if isinstance(el.get(k), dict) and "t" in el[k]:
                        el[k] = dict(el[k], t=self.resolve_time(el[k]["t"], ct, ce))
                if isinstance(el.get("at"), list):  # morph chain times
                    el["_chain_t"] = [self.resolve_time(v, ct, ce) for v in el["at"]]
                if isinstance(el.get("flips"), list):
                    el["flips"] = [[self.resolve_time(f[0], ct, ce), f[1]] for f in el["flips"]]
                if "tracks" in el:
                    el["tracks"] = {k: [[self.resolve_time(kk[0], ct, ce)] + list(kk[1:]) for kk in v]
                                    for k, v in el["tracks"].items()}
                el.setdefault("t", ct)
                if el.get("type") == "footage":
                    _fx.normalize_footage(el, self, ct, ce)
                el["_chapter"] = idx
                els.append(el)
            self._cascade(els, c)
            c["elements"] = sorted(els, key=lambda e: e.get("layer", 0))

    def _cascade(self, els, ch):
        """v5: (1) beat groups — elements sharing "group" inherit the group's out (first explicit
        "out" in the group, or "group_out" on the chapter);  (2) dependents (callout/arrow/circle/line
        targeting an id) inherit their target's out automatically → no orphan leader lines."""
        groups = {}
        for el in els:
            g = el.get("group")
            if g and el.get("out") is not None:
                groups.setdefault(g, (el["out"], el.get("out_dur", 0.45)))
        gouts = ch.get("group_out") or {}
        for g, v in gouts.items():
            groups[g] = (self.resolve_time(v, ch["t"], ch["end"]), 0.45)
        for el in els:
            g = el.get("group")
            if g in groups and el.get("out") is None:
                el["out"], od = groups[g]
                el.setdefault("out_dur", od)
                el["_inherited_out"] = f"group:{g}"
        by = {e["id"]: e for e in els}
        for _ in range(3):  # propagate through chains (callout → callout → icon)
            for el in els:
                if el.get("out") is not None:
                    continue
                for k in _TARGET_KEYS:
                    v = el.get(k)
                    name = v.get("id") if isinstance(v, dict) else (v.split("+")[0] if isinstance(v, str) else None)
                    tgt = by.get(name)
                    if tgt is not None and tgt.get("out") is not None and el.get("type") != "morph":
                        el["out"] = tgt["out"]
                        el.setdefault("out_dur", tgt.get("out_dur", 0.45))
                        el["_inherited_out"] = f"target:{name}"
                        break

    def _build_threads(self):
        """transition {"type":"thread","from":"<id in previous chapter>","to":"<id in this chapter>"}
        → a glowing leader line is drawn from the focal element of chapter N down into chapter N+1
        while the camera glides along it (chapters physically wired on one master canvas)."""
        self.threads = []
        for i, c in enumerate(self.chapters):
            tr = c["transition"]
            if i == 0 or tr.get("type") != "thread":
                continue
            prev = self.chapters[i - 1]
            src = self.by_id.get(tr.get("from")) if tr.get("from") else None
            dst = self.by_id.get(tr.get("to")) if tr.get("to") else None
            px, py = prev["origin"]; cx, cy = c["origin"]
            if src is not None:
                sx, sy = px + float(src.get("x", self.W / 2)), py + float(src.get("y", self.H / 2)) + tr.get("from_dy", 120)
            else:
                sx, sy = px + self.W / 2, py + self.H * 0.72
            if dst is not None:
                ex, ey = cx + float(dst.get("x", self.W / 2)), cy + float(dst.get("y", self.H / 2)) - tr.get("to_dy", 140)
            else:
                ex, ey = cx + self.W / 2, cy + self.H * 0.2
            self.threads.append({"t0": c["t"] - 0.15, "dur": tr["dur"], "a": (sx, sy), "b": (ex, ey),
                                 "hold": tr.get("hold", 1.6), "ch": i, "swing": tr.get("swing", 140)})

    def _build_camera(self):
        keys = []
        cx0, cy0 = self.W / 2, self.H / 2
        for i, c in enumerate(self.chapters):
            ox, oy = c["origin"]
            tr = c["transition"]
            arrive = c["t"] + (tr["dur"] if (i > 0 and tr["type"] in ("pan", "thread")) else 0.0)
            cam = c.get("camera")
            if i > 0 and keys:
                keys.append((c["t"], keys[-1][1], "linear"))  # hold previous view until transition
            if cam:
                first = True
                for k in cam:
                    lt = self.resolve_time(k[0], c["t"], c["end"])
                    v = k[1]
                    ang = v[3] if len(v) > 3 else (k[3] if len(k) > 3 and isinstance(k[3], (int, float)) else
                                                   c.get("cam_angle"))
                    view = (ox + v[0], oy + v[1], v[2] if len(v) > 2 else 1.0) + ((float(ang),) if ang is not None else ())
                    ease = k[2] if len(k) > 2 else "in_out"
                    if first and lt < arrive:
                        lt = arrive
                    keys.append((lt, view, "in_out" if first and i > 0 else ease))
                    first = False
            else:
                keys.append((arrive, (ox + cx0, oy + cy0, 1.0), "in_out"))
        if not keys:
            keys = [(0.0, (cx0, cy0, 1.0))]
        # normalise view lengths
        L = max(len(k[1]) for k in keys)
        keys = [(k[0], tuple(k[1]) + (0.0,) * (L - len(k[1])), k[2]) for k in keys]
        drift = float(self.spec.get("meta", {}).get("camera_drift", 0.0))
        return Camera(self.W, self.H, keys=keys, drift=drift)

    # ------------------------------------------------------------------
    def chapter_index(self, t):
        idx = 0
        for i, c in enumerate(self.chapters):
            if t >= c["t"]:
                idx = i
        return idx

    REGION_TRANSITIONS = ("wipe", "slide", "portal", "whip_pan")

    def wipe_state(self, t):
        """(p 0..1, dir, type, colour, chapter_index) while a wipe/slide/portal transition runs, else None."""
        r = self.region_state(t)
        return None if r is None else (r["p"], r["dir"], r["type"], r["color"], r["ci"])

    def region_state(self, t):
        """v8: a structural transition that splits the frame into OLD and NEW regions.
        {"type","p","dir","color","ci","cx","cy","shape"} — cx/cy = portal focal point (chapter-local)."""
        if not self.chapters:
            return None
        i = self.chapter_index(t)
        cur = self.chapters[i]
        tr = cur["transition"]
        if i == 0 or tr["type"] not in self.REGION_TRANSITIONS or tr["dur"] <= 0:
            return None
        lt = t - cur["t"]
        if lt >= tr["dur"]:
            return None
        from .core import get_ease
        from .styles import resolve_color
        pal = cur["_world"]["palette"]
        p = get_ease(tr.get("ease", "in_out_expo" if tr["type"] != "portal" else "in_out_quart"))(max(0.0, lt / tr["dur"]))
        col = resolve_color(tr.get("color", "@accent"), pal)
        out = {"type": tr["type"], "p": p, "dir": tr.get("dir", "left"), "color": col, "ci": i,
               "shape": tr.get("shape", "circle"), "cx": self.W / 2, "cy": self.H / 2,
               "edge": float(tr.get("edge", 14 if tr["type"] == "portal" else 90))}
        if tr["type"] == "portal":
            fx, fy = tr.get("x"), tr.get("y")
            src = tr.get("from")
            if src and src in self.by_id:
                el = self.by_id[src]
                from .elements import xy
                ex, ey = xy(el, t)
                po = self.chapters[el["_chapter"]]["origin"]; co = cur["origin"]
                fx = ex + po[0] - co[0] if fx is None else fx
                fy = ey + po[1] - co[1] if fy is None else fy
            out["cx"] = float(fx if fx is not None else self.W / 2)
            out["cy"] = float(fy if fy is not None else self.H / 2)
        return out

    def chapter_world(self, t):
        return self.chapters[self.chapter_index(t)]["_world"] if self.chapters else None

    def look_at(self, t):
        """Background colour + post-FX amounts at time t, blended across chapter transitions (v8).
        Region transitions (wipe/slide/portal) paint each chapter's bg inside its own region instead."""
        from .core import color as _c, lerp
        if not self.chapters:
            return {"bg": _c(self.bg), "bloom": self.bloom, "vignette": 0.0, "grain": 0.0, "slam_pulse": True,
                    "region": False}
        i = self.chapter_index(t)
        cur = self.chapters[i]
        w = cur["_world"]
        res = {"bg": _c(w["bg"]), "bloom": float(w.get("bloom") or 0), "vignette": float(w.get("vignette") or 0),
               "grain": float(w.get("grain") or 0), "slam_pulse": bool(w.get("slam_pulse", True)), "region": False}
        tr = cur["transition"]
        lt = t - cur["t"]
        if i > 0 and lt < tr["dur"] and tr["type"] != "cut":
            pw = self.chapters[i - 1]["_world"]
            p = max(0.0, min(1.0, lt / tr["dur"]))
            from .core import ease_in_out
            q = ease_in_out(p)
            for k in ("bloom", "vignette", "grain"):
                res[k] = lerp(float(pw.get(k) or 0), res[k], q)
            if tr["type"] in self.REGION_TRANSITIONS:
                res["region"] = True
                from .contrast import is_light
                if is_light(w["bg"]) or is_light(pw["bg"]):       # bloom never glows a paper ground
                    res["bloom"] = min(float(pw.get("bloom") or 0), float(w.get("bloom") or 0))
            else:
                a, b = _c(pw["bg"]), res["bg"]
                res["bg"] = tuple(lerp(a[k], b[k], q) for k in range(4))
        return res

    def visible_chapters(self, t):
        """[(chapter, alpha)] — current chapter plus the previous one while transitioning."""
        if not self.chapters:
            return []
        i = self.chapter_index(t)
        cur = self.chapters[i]
        out = []
        tr = cur["transition"]
        lt = t - cur["t"]
        if i > 0 and lt < tr["dur"] and tr["type"] != "cut":
            prev = self.chapters[i - 1]
            p = lt / tr["dur"]
            if tr["type"] in ("pan", "thread"):
                out.append((prev, 1.0 - min(1.0, max(0.0, (p - 0.15) / 0.6))))  # fades while sliding away
            elif tr["type"] in self.REGION_TRANSITIONS:
                out.append((prev, 1.0))          # hard structural reveal: clipped in render.draw_world
            else:
                out.append((prev, 1.0 - p))
        out.append((cur, 1.0))
        return out


def load_spec(path_or_dict):
    if isinstance(path_or_dict, dict):
        return Scene(path_or_dict, ".")
    with open(path_or_dict, encoding="utf-8") as f:
        spec = json.load(f)
    return Scene(spec, os.path.dirname(os.path.abspath(path_or_dict)))
