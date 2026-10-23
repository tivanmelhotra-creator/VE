"""
mograph.pro.lint — v14 SPEC LINTER: the mistakes agents actually make, caught in ~1 s before any render.

    python3 -m mograph.pro lint edit.json      → ✗ errors (exit 2) / ⚠ warnings (exit 1) / clean (exit 0)
    python3 -m mograph.pro lint-all            → every projects/**/edit_*.json

Checks
  * unknown layer "type" (did-you-mean) · unknown keys per layer (typos: "ofset", "enterr" …) — the known-key set is
    EXTRACTED FROM THE ENGINE SOURCE (every L.get("…") / L["…"] in the layer modules), so it never goes stale
  * files exist: source.src / words, broll src, grade.lut, grade.match.ref, audio.music.file
  * names: grade.look · retime ops · broll mode / transitions · saber preset
  * every time resolves (word anchors exist), out > t, layer starts before the end
  * two FRONT hero layers on screen together for > 0.8 s (pro rule: one idea at a time)
  * v22 static SPATIAL collisions: real placement (anchors + offset + follow + guard) of every pair on screen together,
    sampled every 0.25 s without rendering a pixel → "spatial collision A × B t0–t1 (IoU)"
  * lower-third layers (disc / glass / tag) while subtitles are visible in the same zone
"""
from __future__ import annotations

import difflib
import glob
import json
import os
import re

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))         # sdk/mograph
_SCAN = ["v14/layers.py", "v14/craftlayers.py", "v14/subs.py", "v14/object3d.py", "v14/glassbadge.py", "v14/stage.py",
         "v14/bokeh.py", "v14/motionblur.py", "v14/compositor.py", "v14/glyphs.py", "v14/director.py",
         "motion/bridge.py", "motion/saber.py", "motion/character.py", "motion/lipsync.py", "aecomp/bridge.py",
         "aefx/bridge.py", "aefx/textanim.py", "pro/broll.py", "three3d.py"]
HERO = {"kinetic_title", "halo", "disc", "glass_shape", "frame_box", "shatter_text"}   # props (object_3d) accompany
LOWER = {"disc", "glass_shape", "tracked_tag"}
TOP = {"type", "id", "t", "out", "pass", "z", "anchor", "offset", "size", "text", "styles", "clamp_y", "clamp_x",
       "sfx", "sfx_gain", "fx", "palette", "guard", "stage", "depth", "follow", "tilt_follow", "motion_blur",
       "preset", "preset_dur", "preset_out", "preset_out_dur", "selectors", "ease", "chapter", "comment", "note",
       "_comment", "group", "track", "x", "y", "w", "h", "color", "opacity", "font", "seed", "quality", "dur", "at",
       # read through loops in layers.prepare / bokeh defaults (invisible to the regex)
       "land", "flip_at", "hero_hit", "pulse_at", "turn_on", "break_at", "words_at", "lines_at", "front", "back",
       # v20.1 three_anim (read through three_anim.DEFAULTS / config)
       "anim", "flips_at", "close_at", "explode_at", "pour_dur", "flip_dur", "hold", "settle", "start_fill",
       "flip_axis", "flips", "open_dur", "open_delay", "assemble_dur", "explode_from", "explode_dur", "explode_to"}
_KEYS = None


def known_keys():
    global _KEYS
    if _KEYS is None:
        ks = set(TOP)
        pat = re.compile(r"""\b(?:L|el|spec|sp|P|kb|cfg|o|p|g|ls|C|L2)\s*(?:\.get\(\s*|\[\s*)["']([a-z_][a-z0-9_]*)["']""")
        for rel in _SCAN:
            path = os.path.join(_HERE, rel)
            if os.path.exists(path):
                ks |= set(pat.findall(open(path, encoding="utf-8").read()))
        try:
            from ..qa import ELEMENT_KEYS, COMMON_KEYS
            ks |= set(COMMON_KEYS)
            for v in ELEMENT_KEYS.values():
                ks |= set(v)
        except Exception:  # noqa: BLE001
            pass
        _KEYS = {k for k in ks if not k.startswith("_")} | {"_comment"}
    return _KEYS


def layer_types():
    from ..v14 import layers as LY
    return set(LY.DRAW) - {"subtitles"}


def _sugg(x, pool):
    m = difflib.get_close_matches(str(x), list(pool), n=1, cutoff=0.72)
    return f" — did you mean '{m[0]}'?" if m else ""


def lint(spec_path):
    """→ (errors, warnings)."""
    E, Wn = [], []
    base = os.path.dirname(os.path.abspath(spec_path))
    try:
        spec = json.load(open(spec_path, encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"JSON syntax error: {e}"], []

    def rel(p):
        return p if os.path.isabs(p) else os.path.join(base, p)
    src = spec.get("source") or {}
    if not src.get("src"):
        E.append("source.src missing")
    elif not os.path.exists(rel(src["src"])):
        E.append(f"source.src file not found: {src['src']}  (projects/*/media is git-ignored: "
                 f"`python3 -m mograph.pro doctor --fix` restores it)")
    if src.get("words") and not os.path.exists(rel(src["words"])):
        E.append(f"source.words file not found: {src['words']}")
    from ..motion.retime import OPS
    for i, o in enumerate(src.get("retime") or []):
        if o.get("op") not in OPS:
            E.append(f"source.retime[{i}].op '{o.get('op')}' unknown{_sugg(o.get('op'), OPS)}  ({', '.join(OPS)})")
    g = spec.get("grade") or {}
    from .color import LOOKS
    if g.get("look") and g["look"] not in LOOKS:
        E.append(f"grade.look '{g['look']}' unknown{_sugg(g['look'], LOOKS)}  ({', '.join(LOOKS)})")
    if g.get("lut") and not os.path.exists(rel(g["lut"])):
        E.append(f"grade.lut file not found: {g['lut']}")
    gm = g.get("match")
    if gm:
        ref = gm["ref"] if isinstance(gm, dict) else gm
        if not os.path.exists(rel(ref)):
            E.append(f"grade.match.ref not found: {ref}")
    mu = (spec.get("audio") or {}).get("music") or {}
    if isinstance(mu, dict) and mu.get("file") and not os.path.exists(rel(mu["file"])):
        E.append(f"audio.music.file not found: {mu['file']}")
    types, keys = layer_types(), known_keys()
    from .broll import TRANSITIONS, MODES
    try:
        from ..motion.saber import _P as SABER_P
    except Exception:  # noqa: BLE001
        SABER_P = None
    ids = set()
    for i, L in enumerate(spec.get("layers") or []):
        nm = f"layers[{i}]" + (f" ({L.get('id')})" if L.get("id") else "")
        ty = L.get("type")
        if ty not in types:
            E.append(f"{nm}: unknown type '{ty}'{_sugg(ty, types)}")
            continue
        if L.get("id"):
            if L["id"] in ids:
                Wn.append(f"{nm}: duplicate id '{L['id']}'")
            ids.add(L["id"])
        for k in L:
            if k not in keys:
                Wn.append(f"{nm}: unknown key '{k}' (the engine ignores it){_sugg(k, keys)}")
        if ty == "broll":
            if not L.get("src"):
                E.append(f"{nm}: broll needs 'src'")
            elif not os.path.exists(rel(L["src"])):
                E.append(f"{nm}: broll src not found: {L['src']}")
            if L.get("mode", "full") not in MODES:
                E.append(f"{nm}: broll mode '{L.get('mode')}' unknown ({'|'.join(MODES)})")
            for kk in ("in", "out_fx"):
                if L.get(kk) and L[kk] not in TRANSITIONS:
                    E.append(f"{nm}: {kk}='{L[kk]}' unknown{_sugg(L[kk], TRANSITIONS)}")
        if ty == "saber" and SABER_P is not None and L.get("preset") and L["preset"] not in SABER_P:
            E.append(f"{nm}: saber preset '{L['preset']}' unknown{_sugg(L['preset'], SABER_P)}")
        if ty == "object_3d":
            from .. import three3d as _T3
            if L.get("three_model") and not _T3.resolve_model_name(L["three_model"]):
                E.append(f"{nm}: three_model '{L['three_model']}' unknown{_sugg(L['three_model'], _T3.list_models())}")
            if L.get("three_script"):
                sp, fn = _T3.split_script_ref(L["three_script"])
                if not sp.endswith(".js"):
                    E.append(f"{nm}: three_script must be a .js file (optionally 'file.js:builderName'), got {sp}")
                elif not os.path.exists(rel(sp)) and not os.path.exists(os.path.join(_HERE, sp)):
                    E.append(f"{nm}: three_script file not found: {sp}")
                elif fn and not _T3._JS_IDENT.match(fn):
                    E.append(f"{nm}: three_script builder '{fn}' is not a valid JS function name")
            an = L.get("anim")
            an = an.get("type") if isinstance(an, dict) else an
            if an:
                from .. import three_anim as _TA
                need = {"hourglass_loop": "hourglass_3d", "laptop_open": "laptop_3d"}
                mdl = _T3.resolve_model_name(L.get("three_model") or L.get("src"))
                if an not in _TA.ANIMS:
                    E.append(f"{nm}: anim '{an}' unknown{_sugg(an, _TA.ANIMS)}  ({', '.join(_TA.ANIMS)})")
                elif an in need and mdl != need[an] and _T3.resolve_model_name(mdl) != need[an] and \
                        not (mdl and _T3.MODELS.get(mdl) is _T3.MODELS[need[an]]):
                    E.append(f"{nm}: anim '{an}' needs three_model '{need[an]}' (got {mdl})")
                elif L.get("allow_slab"):
                    E.append(f"{nm}: anim '{an}' needs the procedural 3D model — remove allow_slab")
            if L.get("flips_at") and an != "hourglass_loop":
                Wn.append(f"{nm}: flips_at only works with anim 'hourglass_loop'")
            if L.get("allow_slab") and (float(L.get("yaw_amp", 28)) > 25 or L.get("motion") == "spin_360"):
                Wn.append(f"{nm}: 2D slab extrusion with large yaw rotation looks flat past 45° — use three_model / three3d instead")
    if E:
        return E, Wn
    # ---- times (renderer load only: anchors, retime — no frame rendered)
    try:
        from ..v14.compositor import Renderer
        R = Renderer(spec_path)
    except KeyError as e:
        return E + [f"time anchor: {e}"], Wn
    except Exception as e:  # noqa: BLE001
        return E + [f"the renderer cannot load the spec: {type(e).__name__}: {e}"], Wn
    for L in R.layers:
        nm = L.get("id", L["type"])
        if L["_t1"] <= L["_t0"]:
            E.append(f"{nm}: out ({L['_t1']:.2f}) ≤ t ({L['_t0']:.2f}) — never visible")
        elif L["_t0"] >= R.duration:
            E.append(f"{nm}: starts at {L['_t0']:.2f}s, after the end ({R.duration:.2f}s)")
    hs = sorted([L for L in R.layers if L["type"] in HERO and L.get("pass", "front") == "front"], key=lambda L: L["_t0"])
    for a, b in zip(hs, hs[1:]):
        ov = min(a["_t1"], b["_t1"]) - b["_t0"]
        if a.get("group") and a.get("group") == b.get("group"):
            continue                                       # v21: parts of ONE composition (e.g. a dual comparison)
        if ov > 0.8:                                       # short hand-offs (exit while the next enters) are normal
            Wn.append(f"two FRONT hero layers overlap {ov:.2f}s: {a.get('id', a['type'])} & {b.get('id', b['type'])} "
                      f"at {b['_t0']:.2f}s (one idea at a time)")
    Wn.extend(spatial_overlaps(R))                         # v22: static spatial collision check (no pixels)
    if R.subs:
        hid = R.subs["_hide"]
        for L in R.layers:
            if L["type"] in LOWER and L.get("anchor", "") in ("chest", "belly", "neck") and L.get("pass", "front") == "front":
                t0, t1 = L["_t0"], min(L["_t1"], R.duration)
                cov = sum(max(0.0, min(t1, b) - max(t0, a)) for a, b in hid)
                if t1 - t0 > 0.5 and cov < 0.6 * (t1 - t0):
                    Wn.append(f"{L.get('id', L['type'])}: lower-third layer {t0:.2f}-{t1:.2f}s while subtitles are "
                              f"visible (add [{t0:.2f}, {t1:.2f}] to subtitles.hide)")
    return E, Wn


SPATIAL = {"kinetic_title", "disc", "glass_shape", "frame_box", "object_3d", "counter_badge", "shatter_text"}


def spatial_overlaps(R, step=0.25, min_dur=0.3, min_iou=0.18):
    """v22 STATIC SPATIAL CHECK: for every pair of placed layers that are on screen together, sample their real
    placement (layers.pos → anchors, offset, follow, layout guard) and bounding radius (layers.element_radius) every
    `step` s — no frame is rendered (≈ ms per sample).  Warn when two discs overlap by IoU ≥ min_iou for ≥ min_dur s.
    Members of one `group` (a designed composition) are exempt."""
    from ..v14 import layers as LY
    from ..v14.compositor import Frame
    Ls = [L for L in R.layers if L["type"] in SPATIAL]
    out, seen = [], set()
    if len(Ls) < 2:
        return out
    import math as _m
    t, cache = 0.0, {}
    while t < R.duration:
        live = [L for L in Ls if L["_t0"] + 0.15 <= t <= L["_t1"] - 0.15]
        if len(live) >= 2:
            F = Frame(R, t)
            F.M, F.zoom = R.matrix(t)
            hw = F.head_px()
            geo = []
            for L in live:
                try:
                    x, y, k = LY.pos(L, F)
                    r = float(LY.element_radius(L, F, hw)) * float(k or 1.0)
                except Exception:  # noqa: BLE001
                    continue
                if r > 4:
                    geo.append((L, x, y, r))
            for i in range(len(geo)):
                for j in range(i + 1, len(geo)):
                    a, b = geo[i], geo[j]
                    if a[0].get("group") and a[0].get("group") == b[0].get("group"):
                        continue
                    d = _m.hypot(a[1] - b[1], a[2] - b[2])
                    r1, r2 = a[3], b[3]
                    if d >= r1 + r2:
                        continue
                    if d <= abs(r1 - r2):
                        inter = _m.pi * min(r1, r2) ** 2
                    else:
                        c1 = _m.acos(max(-1, min(1, (d * d + r1 * r1 - r2 * r2) / (2 * d * r1))))
                        c2 = _m.acos(max(-1, min(1, (d * d + r2 * r2 - r1 * r1) / (2 * d * r2))))
                        inter = r1 * r1 * c1 + r2 * r2 * c2 - 0.5 * _m.sqrt(max(0.0, (-d + r1 + r2) * (d + r1 - r2) *
                                                                                   (d - r1 + r2) * (d + r1 + r2)))
                    iou = inter / (_m.pi * r1 * r1 + _m.pi * r2 * r2 - inter)
                    if iou >= min_iou:
                        key = (id(a[0]), id(b[0]))
                        c = cache.setdefault(key, [t, t, 0.0, a[0], b[0]])
                        c[1], c[2] = t, max(c[2], iou)
        t += step
    for (ka, kb), (t0, t1, iou, A, B) in cache.items():
        if t1 - t0 + step >= min_dur and (ka, kb) not in seen:
            seen.add((ka, kb))
            out.append(f"spatial collision {A.get('id', A['type'])} × {B.get('id', B['type'])} "
                       f"{t0:.2f}–{t1 + step:.2f}s (overlap IoU {iou:.2f}) — move one (offset/anchor) or give both "
                       f"the same \"group\" if it is one designed composition")
    return out


def run(spec_path):
    E, Wn = lint(spec_path)
    for e in E:
        print("✗", e)
    for w in Wn:
        print("⚠", w)
    print("lint clean ✓" if not (E or Wn) else f"lint: {len(E)} error(s), {len(Wn)} warning(s)")
    return 2 if E else (1 if Wn else 0)


def lint_all(root):
    bad = 0
    for p in sorted(glob.glob(os.path.join(root, "**", "edit_*.json"), recursive=True)):
        E, Wn = lint(p)
        print(f"{'✗' if E else ('⚠' if Wn else '✓')} {os.path.relpath(p, root)}  {len(E)} errors, {len(Wn)} warnings")
        for e in E:
            print("    ✗", e)
        bad += bool(E)
    return bad
