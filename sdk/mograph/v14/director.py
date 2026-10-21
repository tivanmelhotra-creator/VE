"""
mograph.v14.director — AUTOMATIC BEAT-SHEET DIRECTOR (`auto-edit` brain).

Resolves Backlog #3 in `SKILL_TOOLBOX.md`:
Given `words.json` (word-level timestamps) and `media/raw.mp4`, automatically constructs a complete,
QA-clean `edit_v14.json` spec with:
  1. Automatic dead-pause excision (`gap > max_pause`) → `source.segments`,
  2. Semantic beat segmentation (`2.2s–3.8s` per beat) + Persian keyword salience scoring to pick ONE `*hero*` word per beat,
  3. Automatic depth-alternating layer choreography:
       Beat 0: `kinetic_title` (`pass: "back"`, `enter: "drop_bounce"`, `plane_in: {"rx": -70}`) + `tracked_tag`
       Beat 1: `orbit_ring` (`3D cylinder around head`, `ring_fit` <= 40% circumference)
       Beat 2: `frame_box` (`pass: "back"`, tilted wireframe + 45° diamond badge)
       Beat 3: `disc` (`chest` vinyl/CD with `flip_at` on the payoff word + red collar light spill)
       Beat 4: `halo` (`pass: "back"`, two-disc crimson halo above head with `peek: true`)
       Beat 5: `glass_shape` (`shape: "hept"`, nested gold outline + 7px gap + refractive frosted glass)
  4. Automatic `subtitles.hide` windows whenever a chest hero (`disc` or `glass_shape`) owns the lower third,
  5. Automatic camera punch-ins (`expo` / `jelly`) and impact `shakes` on cut boundaries and alert/hero hits!

CLI:
  PYTHONPATH=sdk python3 -m mograph.v14 auto-edit projects/maghale/words.json projects/maghale/media/raw.mp4 -o projects/maghale/edit_auto.json
"""
from __future__ import annotations

import json
import os
import re
import cv2
from .compositor import Timeline, load_words, norm_word

STOPWORDS_FA = {
    "و", "در", "به", "از", "که", "این", "را", "رو", "با", "است", "هست", "هستش",
    "برای", "اون", "آن", "یک", "یه", "تا", "هم", "بر", "یا", "شما", "ما", "من",
    "می", "رو", "باید", "وقتی", "اگر", "چون", "پس", "هر", "چیز", "کار"
}
CONTRAST_WORDS_FA = {"اما", "ولی", "بلکه", "برعکس", "نه", "اصلا", "هرگز", "دقیقا", "حتما"}
ORDINAL_WORDS_FA = {"اول", "اولین", "آخر", "آخرین", "ابتدا", "انتها", "شروع", "پایان", "مقدمه", "نتیجه"}
HIGH_IMPACT_FA = {"مقاله", "چکیده", "نگارش", "مقدمه", "شروع", "طلایی", "مهم", "کلیدی", "فرمول", "راز", "لاغری"}


def _score_word(w_norm: str, w_raw: str) -> float:
    if not w_norm or w_norm in STOPWORDS_FA or len(w_norm) <= 1:
        return 0.1
    sc = min(3.5, len(w_norm) * 0.42)
    if w_norm in HIGH_IMPACT_FA:
        sc += 5.0
    if w_norm in ORDINAL_WORDS_FA:
        sc += 4.2
    if w_norm in CONTRAST_WORDS_FA:
        sc += 3.8
    if any(ch.isdigit() or "\u06f0" <= ch <= "\u06f9" for ch in w_raw):
        sc += 4.5
    return sc


def build_segments_from_words(words: list[dict], clip_dur: float | None = None,
                              max_pause: float = 0.34, pad: float = 0.08) -> list[list[float]]:
    """Excise leading, trailing, and mid-sentence dead pauses (> max_pause) into clean EDL segments."""
    if not words:
        return [[0.0, clip_dur or 10.0]]
    segs = []
    cur_s = max(0.0, words[0]["t"] - pad)
    cur_e = words[0]["end"] + pad
    for w in words[1:]:
        gap = w["t"] - (cur_e - pad)
        if gap > max_pause:
            segs.append([round(cur_s, 2), round(cur_e, 2)])
            cur_s = max(cur_e, w["t"] - pad)
        cur_e = w["end"] + pad
    if clip_dur is not None:
        cur_e = min(clip_dur, cur_e)
    segs.append([round(cur_s, 2), round(cur_e, 2)])
    return segs


def auto_edit(words_path: str, video_path: str, out_json_path: str | None = None,
              theme: str = "doran", lofi_bpm: int = 92) -> dict:
    """Generate a complete, QA-clean v14 Studio Depth spec automatically from words.json + raw.mp4."""
    words_raw = load_words(words_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_dur = (n_fr / fps) if n_fr > 0 else None
    cap.release()

    segments = build_segments_from_words(words_raw, clip_dur=clip_dur, max_pause=0.34, pad=0.08)
    tl = Timeline(segments, words_raw)

    # Group timeline words into semantic beats of ~2.2s - 3.6s
    beats = []
    cur = []
    for w in tl.words:
        if cur and ((w["end"] - cur[0]["t"] > 2.8 and len(cur) >= 4) or w["t"] - cur[-1]["end"] > 0.28):
            beats.append(cur)
            cur = []
        cur.append(w)
        if w["w"].rstrip()[-1:] in "،,.!?؟" and (cur[-1]["end"] - cur[0]["t"] >= 1.8):
            beats.append(cur)
            cur = []
    if cur:
        if beats and (cur[-1]["end"] - cur[0]["t"] < 1.2):
            beats[-1].extend(cur)
        else:
            beats.append(cur)

    # Occurrence tracker so anchors use "w:word#2" when a word repeats
    occ_counts: dict[str, int] = {}

    def _anchor_for(w_obj: dict, prefix: str = "w", offset: float = 0.0) -> str:
        n = w_obj["n"]
        # Find which occurrence index this exact word object is in tl.words
        idx_occ = 1
        for tw in tl.words:
            if tw is w_obj:
                break
            if tw["n"] == n:
                idx_occ += 1
        base = f"{prefix}:{n}" + (f"#{idx_occ}" if idx_occ > 1 else "")
        if abs(offset) > 1e-3:
            base += f"{offset:+.2f}"
        return base

    layers = [{"type": "progress", "id": "prog", "t": 0, "h": 6, "color": "@accent"}]
    cam_keys = [[0.0, 1.08, 0.5, 0.42]]
    cam_shakes = []
    grade_keys = []
    sub_hide = []
    emph_set = set()

    cycle_types = ["kinetic_title", "orbit_ring", "frame_box", "disc", "halo", "glass_shape"]

    for bi, bw in enumerate(beats):
        t0_b = bw[0]["t"]
        t1_b = bw[-1]["end"]
        # Pick best hero word in this beat
        scored = sorted(bw, key=lambda x: _score_word(x["n"], x["w"]), reverse=True)
        hero_w = scored[0]
        emph_set.add(hero_w["n"])

        # Build a concise 2-to-3 word lockup around hero_w
        hi_idx = bw.index(hero_w)
        pre_w = bw[hi_idx - 1]["w"].strip("،,.!?؟") if hi_idx > 0 else ""
        post_w = bw[hi_idx + 1]["w"].strip("،,.!?؟") if hi_idx + 1 < len(bw) else ""
        hero_clean = hero_w["w"].strip("،,.!?؟")

        ltype = cycle_types[bi % len(cycle_types)]
        t_in = round(max(0.05, t0_b - 0.05), 2)
        t_out = round(min(tl.duration - 0.08, t1_b + 0.12), 2)
        h_anchor = _anchor_for(hero_w, "w")

        if ltype == "kinetic_title":
            txt = f"{pre_w}|*{hero_clean}*" if pre_w else f"*{hero_clean}*|{post_w}"
            layers.append({
                "type": "kinetic_title",
                "id": f"title_{bi}",
                "t": t_in,
                "out": t_out,
                "anchor": "head_top",
                "offset": [0, -0.62],
                "size": 78,
                "pass": "back",
                "text": txt.strip("|"),
                "enter": "drop_bounce",
                "exit": "pop_out",
                "idle": "float",
                "plane_in": {"rx": -70},
                "extrude": {"depth": 10, "light": 0.7},
                "hero_hit": h_anchor,
                "stagger": 0.03,
            })
            cam_keys.append([h_anchor, 1.15, 0.5, 0.41, "silk"])

        elif ltype == "orbit_ring":
            txt = f"*{hero_clean}* = {post_w or pre_w or 'بخش کلیدی'}"
            layers.append({
                "type": "orbit_ring",
                "id": f"ring_{bi}",
                "t": t_in,
                "out": t_out,
                "anchor": "head_top",
                "offset": [0, 0.15],
                "size": 78,
                "radius": 1.4,
                "tilt": -14,
                "spin": 48,
                "phase": 160,
                "text": txt,
                "styles": {"hero": {"scale": 1.45}},
                "extrude": {"depth": 7, "light": 0.6},
                "max_copies": 2,
                "sep": 1.0,
                "max_radius": 1.6,
            })
            cam_keys.append([h_anchor, 1.06, 0.5, 0.40, "expo"])

        elif ltype == "frame_box":
            txt = f"{pre_w}|*{hero_clean}* _{post_w}_".strip(" _") if pre_w else f"*{hero_clean}*|_{post_w}_".strip("|_")
            layers.append({
                "type": "frame_box",
                "id": f"box_{bi}",
                "t": t_in,
                "out": t_out,
                "anchor": "head_top",
                "offset": [0, -0.62],
                "size": 82,
                "w": 1.85,
                "h": 1.05,
                "clamp_x": [360, 720],
                "rot": -7,
                "pass": "back",
                "text": txt,
                "badge_text": str((bi % 4) + 1),
                "extrude": {"depth": 6},
            })
            grade_keys.append([t_in, {"dim": 0.50}])
            cam_keys.append([h_anchor, 1.18, 0.48, 0.41, "jelly"])
            cam_shakes.append([h_anchor, 0.25, 8])

        elif ltype == "disc":
            mid_t = round((t_in + t_out) * 0.52, 2)
            top_phr = " ".join(w["w"].strip("،,.!?؟") for w in bw[:max(2, len(bw)//2)][:4])
            bot_phr = " ".join(w["w"].strip("،,.!?؟") for w in bw[max(1, len(bw)//2):][:4])
            layers.append({
                "type": "disc",
                "id": f"disc_{bi}",
                "t": t_in,
                "out": t_out,
                "anchor": "chest",
                "offset": [0, 0.15],
                "radius": 0.92,
                "flip_at": mid_t,
                "neon": "@alert",
                "spill": 1.2,
                "center_text": f"*{hero_clean}*|{post_w or 'کجا؟'}",
                "top_text": top_phr or hero_clean,
                "bottom_text": bot_phr or hero_clean,
                "arc_size": 0.15,
                "follow": 0.6,
            })
            sub_hide.append([t_in, t_out])
            grade_keys.append([t_in, {"dim": 0.62, "blur": 11, "tint": "#1C0B0B"}])
            cam_keys.append([mid_t, 1.12, 0.48, 0.46, "expo"])
            cam_shakes.append([mid_t, 0.28, 7])

        elif ltype == "halo":
            layers.append({
                "type": "halo",
                "id": f"halo_{bi}",
                "t": t_in,
                "out": t_out,
                "pass": "back",
                "text": f"^{pre_w or 'نقطهٔ'}^|={hero_clean}=",
                "styles": {"hero_white": {"scale": 1.8, "glow": 0.45}, "tiny": {"scale": 0.75}},
                "color": "@alert",
                "peek": True,
                "peek_glyphs": 1,
                "text_mode": "center",
                "size": 94,
            })
            grade_keys.append([t_in, {"dim": 0.70, "blur": 12, "tint": "#160606", "desat": 0.65}])
            cam_keys.append([h_anchor, 1.22, 0.47, 0.41, "expo"])

        elif ltype == "glass_shape":
            layers.append({
                "type": "glass_shape",
                "id": f"hex_{bi}",
                "t": t_in,
                "out": t_out,
                "shape": "hept",
                "anchor": "chest",
                "offset": [0, 0.55],
                "w": 1.6,
                "h": 1.42,
                "size": 56,
                "text": f"^{pre_w or 'شروع از'}^|*{hero_clean}*",
                "styles": {"hero": {"scale": 1.7}, "tiny": {"scale": 0.7}},
                "style": "nested",
                "gap": 7,
            })
            sub_hide.append([t_in, t_out])
            grade_keys.append([t_in, {"dim": 0.52, "blur": 9, "tint": "#22180E", "desat": 0.5}])
            cam_keys.append([h_anchor, 1.08, 0.47, 0.45, "jelly"])

    spec = {
        "_about": "Auto-Directed by mograph.v14.director (v15 Craft-Omni Studio Depth)",
        "meta": {"w": 1080, "h": 1920, "fps": 30, "theme": theme, "palette": "auto"},
        "source": {
            "src": os.path.relpath(video_path, os.path.dirname(os.path.abspath(out_json_path or words_path))),
            "words": os.path.relpath(words_path, os.path.dirname(os.path.abspath(out_json_path or words_path))),
            "segments": segments,
        },
        "grade": {
            "bg": {"blur": 7, "dim": 0.42, "desat": 0.22, "tint": "#3A2414", "tint_amt": 0.24, "vignette_bg": 0.35},
            "fg": {"contrast": 1.07, "warm": 0.035, "sat": 1.04, "lift": 0.0},
            "keys": grade_keys,
        },
        "camera": {"keys": cam_keys, "shakes": cam_shakes},
        "post": {"bloom": 1.0, "spill": 1.35, "wrap_bg": 0.5, "wrap_glow": 1.8, "decontaminate": 0.65, "vignette": 0.3, "grain": 0.016, "fade": [0.0, 0.35]},
        "subtitles": {
            "anchor": "chest",
            "offset": [0, 0.95],
            "size": 44,
            "max_words": 4,
            "emph": sorted(emph_set),
            "hide": sub_hide,
        },
        "layers": layers,
        "audio": {
            "voice_gain": 1.0,
            "music": {"style": "lofi", "bpm": lofi_bpm, "key": "D", "scale": "minor", "gain": 0.16, "remix": True},
            "sfx_gain": 0.55,
            "lufs": -14,
            "denoise": "auto",
            "multiband": True,
            "vocal_booth_reverb": 0.055,
        },
    }
    if out_json_path:
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
    return spec
