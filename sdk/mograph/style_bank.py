"""
mograph.style_bank — v21 EXPANDABLE STYLE & INSPIRATION BANK + ORIGINALITY GUARD.

    python3 -m mograph.pro style-bank --sync          scan sdk/style_bank/references/ → update catalog.json
    python3 -m mograph.pro style-bank --list          print every registered reference + its principles
    python3 -m mograph.pro style-bank --sheet [-o p]  visual overview sheet of all references
    python3 -m mograph.pro originality <edit.json>    ANTI-COPY / ANTI-REPETITION audit of a spec (exit 1 = fails)

THE LAW (user, v21 — قانون منع مطلق کپی و تکرار):
  References are STYLE STUDIES, never templates.  The engine and every agent must extract the PRINCIPLE (why it
  works: hierarchy, light, rhythm, semantic fit) and synthesise a NEW form for the new video.  It is FORBIDDEN to
  reproduce a reference frame, a reference's exact text, its exact widget/shape, its colours or its ordering, and
  FORBIDDEN to clone an existing project spec (same layer-type sequence / same texts / same geometry).

`originality_report(spec, base_dir)` enforces this mechanically:
  1. TEXT COPY        on-screen text that equals a reference's documented text (catalog "forbidden_texts")
  2. SPEC CLONE       layer-type sequence similarity vs every other edit_*.json in projects/ and the legacy archive
                      (difflib ratio ≥ 0.80 on the hero sequence = clone) and identical hero texts reused
  3. ROTATION         a fixed cycle (same 3+ hero types repeating in the same order) = mechanical template
  4. MONOTONY         one hero type > 50 % of the beats, or legacy showcase widgets (tracked_tag/disc/halo/frame_box)
                      used more than once
  5. PURE TYPO RATIO  container-free typography (kinetic_title / shatter_text / saber-text, grid off) must be
                      ≥ 55 % of hero beats (taste law #2: 60–70 %)
  6. REFERENCE PIXELS a style_bank reference image used as a source/sprite/b-roll inside a spec
"""
from __future__ import annotations

import difflib
import glob
import hashlib
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))           # sdk/mograph
SDK = os.path.dirname(HERE)
ROOT = os.path.dirname(SDK)
BANK = os.path.join(SDK, "style_bank")
REFS = os.path.join(BANK, "references")
CATALOG = os.path.join(BANK, "catalog.json")
MEDIA_EXT = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp4", ".mov", ".webm", ".mkv")

HERO_TYPES = {"kinetic_title", "halo", "disc", "glass_shape", "frame_box", "shatter_text", "orbit_ring", "aecomp",
              "morph_shape", "broll", "object_3d", "saber", "tracked_tag", "puppet_sticker", "character"}
ACCENT_TYPES = {"progress", "flash", "ambient_bokeh", "sweep_light", "strike", "god_rays", "counter_badge",
                "vegas_comet"}
LEGACY_SHOWCASE = {"tracked_tag", "disc", "halo", "frame_box"}
PURE_TYPO = {"kinetic_title", "shatter_text"}

# Hand-written semantic analysis of the 3 seed references (principles, NOT recipes).  New files dropped into
# references/ get a stub entry on --sync that the agent/user fills in (principle → when → synthesis).
SEED = {
    "ref_01_pure_glow_typo_and_dual_nodes": {
        "title": "Pure glow typography + symmetrical comparison nodes",
        "source": "user reference sheet 3.jpg (6 panels, one creator, studio desk)",
        "principles": [
            "Container-free hierarchy: a thin context line frames ONE heavy hero word; the hero is the only loud thing.",
            "Light is the container: the hero emits a soft coloured glow that spills onto the dark clothes behind it, "
            "so contrast comes from optics, not from a box.",
            "Colour = emotional temperature: cool violet for calm/teaching beats, red only when the meaning turns "
            "to a problem/warning.",
            "Two concepts compared = two equal nodes placed symmetrically left/right of the body, the shared word "
            "sits between them.",
            "A Latin script accent can contrast a Persian hero once (bilingual texture), never as decoration everywhere.",
        ],
        "when_to_use": ["teaching / explaining beats", "a single key word per sentence",
                        "explicit comparison of two options (A vs B, before vs after)"],
        "synthesis": [
            "Pick your OWN hero word from the transcript and your OWN palette from the footage (palette.py).",
            "Vary where the glow lives (behind shoulder / chest / beside the head) and its colour per section.",
            "For comparisons design a fresh dual composition: e.g. two light-pools, two morph_shape outlines, two "
            "kinetic_titles on a shared baseline, a split-tone grade — never two copied diamond pills.",
        ],
        "forbidden_texts": ["KBar", "آموزش ساده", "Tutorial", "مشکل حل بشه", "حرف آخر", "ادیت میزنی", "اگه نظم",
                            "تایم ادیتت", "تایم تدوینت", "پایین تر", "کامنت کنید"],
        "tags": ["pure_typography", "glow", "light_spill", "dual_nodes", "comparison", "bilingual_accent"],
    },
    "ref_02_state_flip_and_split_stage": {
        "title": "Two-state flip (cause → effect) + split stage product comparison",
        "source": "user reference sheet -1.jpg (6 panels)",
        "principles": [
            "A neutral state first, then the meaning flips it: the same centre label stays while its surrounding "
            "context changes (state A text → state B text), so the viewer reads a CHANGE, not two graphics.",
            "The light moves with the meaning: the glow travels from one side to the other at the flip.",
            "Above-the-head title + a thin luminous arc line separates title space from speaker space.",
            "Split background: the plate behind the speaker is replaced by two product/scene panels that slide in "
            "while the title stays world-locked.",
        ],
        "when_to_use": ["cause/effect, myth/fact, before/after", "naming two products or brands",
                        "a sentence that pivots on «اما / ولی / بیشتر از اینکه»"],
        "synthesis": [
            "Express the flip with YOUR form: text morph, colour temperature shift, shatter_text break, grade key, "
            "or a morph_shape — the circle-badge-with-arc-text is only one historical answer.",
            "Split stage: broll mode behind / split_* with your own media; change panel geometry and transition.",
        ],
        "forbidden_texts": ["بیشتر از اینکه", "روی کاهش وزن اثر بذارن", "روی سلامت دهان و دندونتون", "آمپول های لاغری",
                            "اسپارتینا", "SPARTINA"],
        "tags": ["state_flip", "cause_effect", "split_stage", "broll_behind", "arc_divider", "world_locked_title"],
    },
    "ref_03_multi_creator_diversity_matrix": {
        "title": "Multi-creator diversity matrix (10 panels, 4 creators)",
        "source": "user reference sheet 2.jpg",
        "principles": [
            "Every video has its OWN visual identity — four creators, four different systems; diversity is the rule.",
            "Habit/option comparison as two small object cards side by side with labels below (objects carry meaning).",
            "Loupe/zoom window behind the shoulder magnifies the detail being talked about.",
            "A live B-roll window on the chest shows the thing while it is named.",
            "A physical prop (tool) interacts with the typography — the object and the word share one gesture.",
            "Warning hook = environmental change (shattered glass over the whole plate, colour cast), not a sticker.",
            "Sometimes the best graphic is almost nothing: minimal box-free floating subtitle only.",
        ],
        "when_to_use": ["deciding the identity of a new video", "objects/habits named in speech",
                        "a warning or hook in the first second", "calm personal beats (minimal treatment)"],
        "synthesis": [
            "Start each video by choosing a DIFFERENT treatment family than the previous project (see originality).",
            "Objects: one clean 3D prop or cutout per idea; two ideas = two separate, spaced objects.",
            "Warnings: grade keys / dof / overlay textures generated fresh — not the reference's glass crack.",
        ],
        "forbidden_texts": ["این اشتباه رو نکن", "سیگار", "قهوه", "ماندگاری کامپوزیت", "سوهان ناخن", "آقا از من می",
                            "خشکی دهان", "سود نداریم", "بدون قند", "حتما الکلی"],
        "tags": ["diversity", "habit_cards", "loupe", "chest_broll", "prop_interaction", "env_warning",
                 "minimal_subtitle"],
    },
}


# ============================================================================================ catalog
def _sha(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _meta(path):
    info = {"bytes": os.path.getsize(path)}
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".mp4", ".mov", ".webm", ".mkv"):
            import cv2
            cap = cv2.VideoCapture(path)
            info.update(kind="video", w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        seconds=round(cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1.0, cap.get(cv2.CAP_PROP_FPS) or 30), 2))
            cap.release()
        else:
            from PIL import Image
            with Image.open(path) as im:
                info.update(kind="image", w=im.size[0], h=im.size[1])
    except Exception:  # noqa: BLE001
        info["kind"] = "unknown"
    return info


def load_catalog():
    if os.path.exists(CATALOG):
        with open(CATALOG, encoding="utf-8") as f:
            return json.load(f)
    return {"version": 21, "law": "", "references": []}


def sync(log=print, optimize=True, max_side=1400):
    """Register every file in references/ into catalog.json (keeps hand-written analysis, adds stubs for new files)."""
    os.makedirs(REFS, exist_ok=True)
    cat = load_catalog()
    old = {r["id"]: r for r in cat.get("references", [])}
    refs = []
    for p in sorted(glob.glob(os.path.join(REFS, "*"))):
        if not p.lower().endswith(MEDIA_EXT):
            continue
        if optimize and p.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            _optimize(p, max_side)
        rid = os.path.splitext(os.path.basename(p))[0]
        entry = dict(old.get(rid, {}))
        seed = SEED.get(rid)
        if seed:
            for k, v in seed.items():
                entry.setdefault(k, v)
        if not entry.get("principles"):
            entry.update(title=rid.replace("_", " "), source="added by user", principles=[
                "TODO: describe WHY it works (hierarchy, light, rhythm, semantic fit) — never describe how to copy it."],
                when_to_use=["TODO"], synthesis=["TODO: how to build a NEW variation from the principle"],
                forbidden_texts=[], tags=["unreviewed"])
        entry.update(id=rid, file=os.path.relpath(p, BANK), sha1=_sha(p), **_meta(p))
        refs.append(entry)
    cat = {"version": 21,
           "law": "STYLE STUDIES ONLY — extract principles, synthesise new forms. Copying any reference frame, text, "
                  "widget, colour scheme or ordering, or cloning another project's spec, is FORBIDDEN "
                  "(enforced by `python3 -m mograph.pro originality <edit.json>`).",
           "updated": time.strftime("%Y-%m-%d"), "references": refs}
    with open(CATALOG, "w", encoding="utf-8") as f:
        json.dump(cat, f, ensure_ascii=False, indent=1)
    tot = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fs in os.walk(BANK) for fn in fs) / 1e6
    log(f"[style-bank] {len(refs)} reference(s) registered → {os.path.relpath(CATALOG, ROOT)}  (bank {tot:.2f} MB)")
    if tot > 2.0:
        log("[style-bank] ⚠ bank > 2 MB — keep references compressed (repo < 100 MB rule)")
    return cat


def _optimize(p, max_side):
    """Down-scale / re-encode oversized stills so the bank stays small (< 2 MB total)."""
    from PIL import Image
    try:
        with Image.open(p) as im:
            big = max(im.size) > max_side
            heavy = os.path.getsize(p) > 450_000
            if not (big or heavy):
                return
            im = im.convert("RGB")
            if big:
                s = max_side / max(im.size)
                im = im.resize((int(im.size[0] * s), int(im.size[1] * s)), Image.LANCZOS)
            out = os.path.splitext(p)[0] + ".jpg"
            im.save(out, quality=84, optimize=True)
        if out != p:
            os.remove(p)
    except Exception:  # noqa: BLE001
        pass


def list_refs(log=print):
    cat = load_catalog()
    for r in cat.get("references", []):
        log(f"● {r['id']}  [{r.get('kind')} {r.get('w')}×{r.get('h')}]  — {r.get('title')}")
        for pr in r.get("principles", []):
            log(f"    · {pr}")
        log(f"    tags: {', '.join(r.get('tags', []))}")
    log(f"\nLAW: {cat.get('law')}")
    return cat


def sheet(out=None, log=print):
    """Overview contact sheet of every reference (thumbnail + id + tags)."""
    import cv2
    import numpy as np
    cat = load_catalog()
    refs = cat.get("references", [])
    if not refs:
        raise SystemExit("no references — drop files into sdk/style_bank/references and run --sync")
    tw, th = 520, 560
    cols = min(3, len(refs))
    rows = (len(refs) + cols - 1) // cols
    canvas = np.full((rows * (th + 70) + 20, cols * (tw + 20) + 20, 3), 18, np.uint8)
    for i, r in enumerate(refs):
        p = os.path.join(BANK, r["file"])
        img = cv2.imread(p)
        if img is None:
            cap = cv2.VideoCapture(p)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.4))
            img = cap.read()[1]
            cap.release()
        if img is None:
            continue
        s = min(tw / img.shape[1], th / img.shape[0])
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        cx, cy = 20 + (i % cols) * (tw + 20), 20 + (i // cols) * (th + 70)
        ox, oy = cx + (tw - img.shape[1]) // 2, cy + (th - img.shape[0]) // 2
        canvas[oy:oy + img.shape[0], ox:ox + img.shape[1]] = img
        cv2.putText(canvas, r["id"][:44], (cx, cy + th + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas, ", ".join(r.get("tags", []))[:64], (cx, cy + th + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (140, 200, 200), 1, cv2.LINE_AA)
    out = out or os.path.join(BANK, "style_bank_sheet.jpg")
    cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 82])
    log(f"[style-bank] sheet → {out}")
    return out


# ============================================================================================ originality guard
def _norm_txt(s):
    s = re.sub(r"[\*\^_=!~|،,.!?؟:;«»\"'()\[\]\u200c]", " ", str(s))
    return re.sub(r"\s+", " ", s).strip().lower()


def _layer_texts(L):
    out = []
    for k in ("text", "center_text", "top_text", "bottom_text", "title", "role", "hero", "kicker", "badge_text",
              "core_text"):
        v = L.get(k)
        if isinstance(v, str) and v.strip():
            out.append(v)
    return out


def hero_layers(spec):
    hs = []
    for L in spec.get("layers", []):
        t = L.get("type")
        if t in HERO_TYPES and not L.get("_comment_only"):
            if t == "object_3d" and str(L.get("id", "")).endswith("_obj"):
                continue                                 # props that accompany a hero are not beats
            hs.append(L)
    return hs


def _seq(spec):
    seq = []
    for L in hero_layers(spec):
        t = L["type"]
        if t == "kinetic_title":
            t += ":" + ("back" if L.get("pass") == "back" else "front")
        seq.append(t)
    return seq


def _is_pure_typo(L):
    t = L.get("type")
    if t in PURE_TYPO:
        st = L.get("stage")
        return not (isinstance(st, dict) and float(st.get("grid", 0.0)) > 0.0)
    if t == "saber" and L.get("core") == "text":
        return True
    if t == "aecomp" and (L.get("title_style") or L.get("lower_third")):
        return True
    return False


def _peer_specs(spec_path):
    pats = [os.path.join(ROOT, "projects", "*", "edit_*.json"), os.path.join(SDK, "tests", "legacy_maghale", "*.json")]
    me = os.path.abspath(spec_path) if spec_path else None
    out = []
    for pat in pats:
        for p in sorted(glob.glob(pat)):
            if me and os.path.abspath(p) == me:
                continue
            try:
                with open(p, encoding="utf-8") as f:
                    out.append((p, json.load(f)))
            except Exception:  # noqa: BLE001
                pass
    return out


def originality_report(spec, spec_path=None, clone_ratio=0.80, min_pure=0.55):
    """→ (errors, warnings, stats).  errors = copying/cloning (forbidden); warnings = repetition / taste drift."""
    E, W = [], []
    hs = hero_layers(spec)
    seq = _seq(spec)
    texts = [_norm_txt(x) for L in spec.get("layers", []) for x in _layer_texts(L)]
    # 1 text copy from references
    forb = [(r["id"], _norm_txt(ft)) for r in load_catalog().get("references", []) for ft in r.get("forbidden_texts", [])]
    for rid, ft in forb:
        if not ft:
            continue
        for tx in texts:
            if tx == ft or (len(ft) >= 4 and ft in tx and len(ft) >= 0.6 * len(tx)):
                E.append(f"TEXT COPY: on-screen «{tx}» reproduces reference text from {rid} — write your own copy")
    # 6 reference pixels used inside the spec
    blob = json.dumps(spec, ensure_ascii=False)
    if "style_bank/references" in blob or "style_bank\\references" in blob:
        E.append("REFERENCE PIXELS: a style_bank reference file is used as media — references are for study only")
    # 2 clone vs other projects / legacy archive
    hero_txt = {_norm_txt(x) for L in hs for x in _layer_texts(L) if len(_norm_txt(x)) > 2}
    worst = (0.0, None)
    for p, other in _peer_specs(spec_path):
        oseq = _seq(other)
        if len(seq) >= 3 and len(oseq) >= 3:
            r = difflib.SequenceMatcher(None, seq, oseq).ratio()
            if r > worst[0]:
                worst = (r, p)
            same_txt = hero_txt & {_norm_txt(x) for L in hero_layers(other) for x in _layer_texts(L)}
            if r >= clone_ratio and len(same_txt) >= max(2, len(hero_txt) // 2):
                E.append(f"SPEC CLONE: hero sequence {r:.0%} identical to {os.path.relpath(p, ROOT)} with the same "
                         f"texts {sorted(same_txt)[:3]} — design a new visual plan")
            elif r >= 0.92 and len(seq) >= 4:
                W.append(f"near-clone structure ({r:.0%}) of {os.path.relpath(p, ROOT)} — vary forms/order")
    # 3 mechanical rotation (period 3..6 repeating at least twice)
    for per in range(3, 7):
        if len(seq) >= 2 * per and len(set(seq[:per])) >= 3 and all(seq[i] == seq[i % per] for i in range(len(seq))):
            W.append(f"ROTATION: hero types repeat in a fixed cycle of {per} ({' → '.join(seq[:per])}) — "
                     f"let the meaning choose the form")
            break
    # 4 monotony / legacy widgets
    if len(seq) >= 4:
        from collections import Counter
        c = Counter(L["type"] for L in hs)
        t, n = c.most_common(1)[0]
        if t not in PURE_TYPO and n / len(hs) > 0.5:
            W.append(f"MONOTONY: {t} is {n}/{len(hs)} beats")
        for lt in LEGACY_SHOWCASE:
            if c.get(lt, 0) > 1 and not spec.get("meta", {}).get("legacy_showcase"):
                W.append(f"LEGACY WIDGET: {lt} used {c[lt]}× (max 1 per video, only when semantically justified)")
    # 5 pure typography ratio
    beats = [L for L in hs if L.get("type") not in ("broll", "object_3d", "character", "puppet_sticker")] or hs
    pure = sum(_is_pure_typo(L) for L in beats)
    ratio = pure / max(1, len(beats))
    if len(beats) >= 3 and ratio < min_pure:
        W.append(f"PURE TYPO RATIO {ratio:.0%} < {min_pure:.0%}: most beats should be container-free glow typography")
    grid_layers = [L.get("id", L["type"]) for L in spec.get("layers", [])
                   if isinstance(L.get("stage"), dict) and float(L["stage"].get("grid", 0)) > 0]
    if len(grid_layers) > 1:
        W.append(f"GRID REPEAT: coordinate grid enabled on {len(grid_layers)} layers ({', '.join(grid_layers[:4])})")
    stats = {"heroes": len(hs), "sequence": seq, "pure_typo_ratio": round(ratio, 2),
             "closest_peer": (os.path.relpath(worst[1], ROOT) if worst[1] else None), "closest_ratio": round(worst[0], 2)}
    return E, W, stats


def run_originality(spec_path, log=print):
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    E, W, S = originality_report(spec, spec_path)
    log(f"[originality] {os.path.basename(spec_path)}: {S['heroes']} hero beats · pure-typo {S['pure_typo_ratio']:.0%}"
        f" · closest peer {S['closest_peer']} ({S['closest_ratio']:.0%})")
    for e in E:
        log("✗ " + e)
    for w in W:
        log("⚠ " + w)
    log("originality ✓ (style study, not a copy)" if not E and not W else
        f"originality: {len(E)} error(s), {len(W)} warning(s)")
    return 2 if E else (1 if W else 0)


# ============================================================================================ selftest
def selftest():
    ok_all = True
    base = {"meta": {}, "layers": [
        {"type": "kinetic_title", "id": "a", "text": "ریتم|*روایت*"},
        {"type": "kinetic_title", "id": "b", "text": "*سکوت*", "pass": "back"},
        {"type": "morph_shape", "id": "c"},
        {"type": "kinetic_title", "id": "d", "text": "*مسیر* تازه"}]}
    E, W, S = originality_report(base)
    r1 = not E and S["pure_typo_ratio"] >= 0.7
    copy = {"meta": {}, "layers": [{"type": "kinetic_title", "text": "*KBar*"}, {"type": "kinetic_title", "text": "x"},
                                   {"type": "kinetic_title", "text": "y"}]}
    E2, _, _ = originality_report(copy)
    r2 = any("TEXT COPY" in e for e in E2)
    rot = {"meta": {}, "layers": [{"type": t, "text": str(i)} for i, t in
                                  enumerate(["halo", "disc", "frame_box"] * 3)]}
    _, W3, _ = originality_report(rot)
    r3 = any("ROTATION" in w for w in W3) and any("LEGACY" in w for w in W3) and any("PURE TYPO" in w for w in W3)
    for name, ok, msg in (("clean", r1, f"original spec passes, pure typo {S['pure_typo_ratio']:.0%}"),
                          ("text_copy", r2, "reference text on screen → error"),
                          ("rotation", r3, "fixed widget cycle / legacy widgets / low typo ratio → warnings")):
        ok_all &= ok
        print(f"  {'✓' if ok else '✗'} {name:<10} {msg}")
    print(f"style_bank selftest: {r1 + r2 + r3}/3 pass")
    return 0 if ok_all else 1
