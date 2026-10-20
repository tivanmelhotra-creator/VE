"""Builds templates/11..19 — one starter spec per reference video (reference_videos/ref01..ref09).
Run:  python3 tools/build_ref_templates.py     (then: python3 -m mograph qa templates/1x_*.json)
Every template uses only shipped media (examples/media/…) so it renders out of the box; swap "src" for the
user's own footage / generated images (keep the ids: trackers and behind-layers refer to them)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "templates")
HOST = "../examples/media/host.mp4"
HANDS = "../examples/media/hands.mp4"
GEN = "../examples/media/gen/"
NOHUD = {"header": {"show": False}, "dots": {"show": False}}


def save(name, spec):
    p = os.path.join(OUT, name)
    json.dump(spec, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(p)


# ---------------------------------------------------------------------------------------------------------------
# 11 — ref01: beauty / medical talking head, words BEHIND the speaker, photo cards, selective colour, VIBE look
# ---------------------------------------------------------------------------------------------------------------
save("11_ref01_talking_head_depth_beauty.json", {
    "_ref": "ref01 — talking head on a grey studio sweep; giant words behind the speaker (depth masking), "
            "octagon/rounded photo cards, 87% stat, selective colour (keep the orange), B&W + cyan-negative 'VIBE' "
            "look, halftone hook, logo watermark.",
    "meta": {"style": "promo", "palette": "crimson", "bg": "#7a7a78", "bloom": 0.0, "vignette": 0.25, "grain": 0.015,
             "duration": 12.0},
    "hud": {**NOHUD, "subs": {"show": False}},
    "subtitles": [{"t": 0.2, "end": 3.0, "text": "خانمایی که **مژه‌های** پُرتری دارن"},
                  {"t": 3.1, "end": 6.0, "text": "**۸۷ درصد** جذاب‌تر دیده میشن"},
                  {"t": 6.1, "end": 9.0, "text": "لب‌های درشت‌تر و **چهره** شاداب‌تر"},
                  {"t": 9.1, "end": 11.8, "text": "شما بگین **کدوم** چهره زیباتره؟"}],
    "chapters": [{"t": 0, "title": "hook", "elements": [
        {"type": "footage", "id": "host", "src": HOST, "x": 540, "y": 960, "w": 1080, "h": 1920, "t": 0,
         "grade": "natural", "frame": [[0, [0.5, 0.42, 1.0]], [3.0, [0.5, 0.40, 1.12], "out_expo"]],
         "punch": [3.0, 6.0, 9.0], "halftone": [{"t": 0, "dur": 0.5, "cell": 16, "color": "#e14c2c", "bg": "#f1e5dc"}],
         "selective": {"keep": "#e14c2c", "t": 6.0, "dur": 2.8, "tol": 14},
         "looks": [{"t": 9.0, "dur": 0.35, "look": "mono"}, {"t": 9.35, "dur": 0.35, "look": "invert_cyan"}],
         "flash": [3.0, 9.0], "trackers": {"face": {"kind": "face"}}},
        {"type": "text", "id": "behind1", "behind": "host", "text": "خانمایی که", "x": 540, "y": 640, "size": 170,
         "font": "title", "color": "#ffffff", "t": 0.3, "enter": "rise", "out": 2.9, "text_style": {"kind": "extrude_3d", "depth": 10, "extrude": "#3a1a12"}},
        {"type": "image_card", "id": "eye", "src": GEN + "hiker.jpg", "x": 300, "y": 1150, "w": 330, "h": 330,
         "shape": "octagon", "t": 1.0, "out": 5.8, "border": "#ffffff", "border_w": 5, "focus": [0.5, 0.35]},
        {"type": "big_number", "id": "stat", "value": 87, "unit": "٪", "x": 540, "y": 520, "size": 300, "t": 3.1,
         "out": 5.9, "color": "#ffffff", "roll": True, "shadow": {"blur": 18, "alpha": 0.5}},
        {"type": "text", "id": "stat_cap", "text": "**درصد** جذاب‌تر", "x": 540, "y": 760, "size": 72, "font": "title",
         "color": "#ffffff", "base_color": "#ffffff", "emph_color": "#ff5a36", "t": 3.3, "out": 5.9, "enter": "words", "shadow": {"blur": 16, "alpha": 0.7}},
        {"type": "text", "id": "lips", "text": "لب‌های **درشت‌تر**", "x": 300, "y": 420, "size": 76, "font": "title",
         "color": "#ffffff", "emph_color": "#e14c2c", "t": 6.1, "out": 8.9, "enter": "rise", "shadow": {"blur": 14, "alpha": 0.5}},
        {"type": "tracked_callout", "id": "face_call", "anchor": "host.face", "box": [800, 380], "text": "چهره شاداب",
         "sub": "استروژن بالا", "accent": "#e14c2c", "t": 6.6, "out": 8.9, "size": 44},
        {"type": "text", "id": "vibe", "text": "VIBE", "x": 540, "y": 1180, "size": 200, "font": "title",
         "color": "#ff7a1a", "t": 9.0, "enter": "slam", "out": 10.4, "glow": {"color": "#ff7a1a", "radius": 22}},
        {"type": "text", "id": "cta", "text": "شما بگین **کدوم** چهره زیباتره؟", "x": 540, "y": 420, "size": 70,
         "font": "title", "color": "#ffffff", "emph_color": "#e14c2c", "t": 10.4, "enter": "words",
         "shadow": {"blur": 16, "alpha": 0.5}},
        {"type": "handle_tag", "handle": "Dr.Name", "icon": False, "x": 540, "y": 1700, "size": 34, "t": 0.2,
         "color": "#e9fbff", "shadow": {"blur": 12, "alpha": 0.85}}]}]})

# ---------------------------------------------------------------------------------------------------------------
# 12 — ref02: editor showcase "Raw vs Edited" on teal paper, vector infographic inside the edited phone
# ---------------------------------------------------------------------------------------------------------------
edited_kids = [
    {"type": "footage", "id": "ed", "src": HOST, "x": 540, "y": 960, "w": 1080, "h": 1920, "t": 0, "out": 3.0,
     "grade": "vivid", "zooms": [{"t": 1.2, "dur": 0.5, "to": 1.15}]},
    {"type": "text", "id": "ed_word", "behind": "ed", "text": "ایمپلنت", "x": 540, "y": 560, "size": 210,
     "font": "title", "color": "#ffffff", "t": 0.4, "out": 3.0, "enter": "rise"},
    {"type": "backdrop", "id": "ed_paper", "kind": "studio", "color": "#efeee8", "color2": "#d8d6cc", "t": 3.0,
     "enter": "fade", "dur": 0.25},
    {"type": "icon", "id": "ed_skull", "name": "brain", "x": 540, "y": 900, "size": 520, "t": 3.2, "out": 6.4,
     "color": "#2d2e2d", "width": 5},
    {"type": "flash_pulse", "id": "ed_alert", "x": 300, "y": 600, "r": 160, "color": "#d63230", "t": 3.8},
    {"type": "text", "id": "ed_t1", "text": "که همه **ازش خبر ندارن!**", "x": 540, "y": 400, "size": 70,
     "font": "title", "color": "#2d2e2d", "base_color": "#2d2e2d", "emph_color": "#d63230", "t": 3.4, "out": 6.4, "enter": "words"},
    {"type": "disc", "id": "ed_24", "x": 540, "y": 760, "r": 150, "color": "#2d2e2d", "t": 6.5, "dur": 0.4},
    {"type": "counter", "id": "ed_24n", "from": 0, "to": 24, "x": 540, "y": 750, "size": 150, "color": "#ffffff",
     "t": 6.6, "dur": 0.8, "fa": True},
    {"type": "text", "id": "ed_24c", "text": "ساعت اول", "x": 540, "y": 1000, "size": 64, "font": "title",
     "color": "#2d2e2d", "t": 6.9}]
save("12_ref02_editor_showcase_raw_vs_edited.json", {
    "_ref": "ref02 — editor portfolio reel: teal paper, small 'Raw' phone + big 'Edited' phone, edited side = "
            "talking head with word behind the speaker, then a light infographic (alert pulse, 24h counter), 3D "
            "button 'ادیت ویدیو', handle, IG outro.",
    "meta": {"style": "swiss", "palette": "mono", "bg": "#9ab3b4", "bloom": 0, "grain": 0, "vignette": 0,
             "duration": 11.0},
    "hud": {**NOHUD, "subs": {"show": True, "y": 0.72, "size": 40, "box": {"color": "#ffffff"}}},
    "subtitles": [{"t": 0.3, "end": 2.9, "text": "مراقبت بعد از **ایمپلنت**"},
                  {"t": 3.2, "end": 6.3, "text": "یه نکته که **همه** ازش خبر ندارن"},
                  {"t": 6.5, "end": 9.4, "text": "تا **۲۴ ساعت** اول این کارو نکنید"}],
    "chapters": [{"t": 0, "title": "showcase", "elements": [
        {"type": "backdrop", "kind": "paper", "t": 0, "enter": "none"},
        {"type": "device_frame", "id": "raw", "x": 230, "y": 900, "w": 300, "h": 534, "label": "Raw",
         "label_color": "#1d2a2a", "t": 0, "children": [
            {"type": "footage", "id": "raw_clip", "src": HOST, "x": 540, "y": 960, "w": 1080, "h": 1920, "grade": "flat",
             "audio": True, "audio_gain": 0.0}]},
        {"type": "device_frame", "id": "edited", "x": 680, "y": 820, "w": 520, "h": 924, "label": "Edited",
         "label_color": "#1d2a2a", "t": 0.15, "fill": "#efeee8", "children": edited_kids},
        {"type": "badge_button", "text": "ادیت ویدیو", "x": 540, "y": 1470, "size": 54, "t": 0.4, "press_at": 2.8},
        {"type": "handle_tag", "handle": "yourhandle", "x": 540, "y": 1585, "size": 32, "color": "#2b3a3a", "t": 0.5},
        {"type": "social_outro", "handle": "@YOURHANDLE", "t": 9.6}]}]})

# ---------------------------------------------------------------------------------------------------------------
# 13 — ref03: storyboard sketch (auto_trace) vs finished motion graphic
# ---------------------------------------------------------------------------------------------------------------
save("13_ref03_storyboard_vs_motion.json", {
    "_ref": "ref03 — 'Storyboard' card (hand sketch drawn on with auto_trace) next to the 'Edited' phone with the "
            "finished motion design: notification badges, falling broken hearts, toggle, megaphone bubbles.",
    "meta": {"style": "swiss", "palette": "mono", "bg": "#9ab3b4", "bloom": 0, "grain": 0, "vignette": 0,
             "duration": 10.0},
    "hud": {**NOHUD, "subs": {"show": False}},
    "chapters": [{"t": 0, "title": "storyboard", "elements": [
        {"type": "backdrop", "kind": "paper", "t": 0, "enter": "none"},
        {"type": "device_frame", "id": "sb", "x": 250, "y": 860, "w": 330, "h": 560, "label": "Storyboard",
         "label_color": "#1d2a2a", "fill": "#ffffff", "tilt": -8, "t": 0, "children": [
            {"type": "auto_trace", "id": "sketch", "src": GEN + "elephant_line.jpg", "x": 540, "y": 900, "w": 980,
             "color": "#222222", "width": 7, "dur": 3.0, "t": 0.3}]},
        {"type": "device_frame", "id": "ed", "x": 690, "y": 800, "w": 500, "h": 890, "label": "Edited",
         "label_color": "#1d2a2a", "fill": "#d9d6c7", "t": 0.2, "children": [
            {"type": "backdrop", "kind": "studio", "color": "#e4e1d3", "color2": "#bdb8a6", "t": 0, "enter": "none"},
            {"type": "pill", "id": "n1", "text": "۱  خودمحوری", "x": 540, "y": 520, "size": 70, "filled": True,
             "t": 0.8, "color": "#b5323a", "out": 4.5},
            {"type": "sprite_rig", "id": "man", "src": GEN + "puppet_sheet.jpg", "x": 540, "y": 1150, "scale": 1.5,
             "idle": "talk", "t": 1.0, "out": 4.5},
            {"type": "pill", "id": "n2", "text": "۲  فقدان همدلی", "x": 540, "y": 520, "size": 70, "filled": True,
             "t": 4.6, "color": "#b5323a"},
            {"type": "emitter", "id": "hearts", "x": 540, "y": 300, "kind": "fall", "shape": "heart", "n": 14,
             "life": 3.0, "colors": ["#b5323a", "#d1495b"], "t": 4.8, "area": [900, 40], "size": 60},
            {"type": "emitter", "id": "insults", "x": 540, "y": 1250, "kind": "burst", "shape": "text:احمق|زشت|بازنده",
             "bubble": "#b5323a", "n": 6, "life": 2.2, "speed": 520, "spread": 70, "size": 110, "t": 7.0, "stagger": 0.15}]},
        {"type": "badge_button", "text": "موشن گرافی", "x": 540, "y": 1470, "size": 54, "t": 0.4, "press_at": 4.6},
        {"type": "handle_tag", "handle": "yourhandle", "x": 540, "y": 1585, "size": 32, "color": "#2b3a3a", "t": 0.5},
        {"type": "social_outro", "handle": "@YOURHANDLE", "t": 8.8}]}]})

# ---------------------------------------------------------------------------------------------------------------
# 14 — ref04: "editing process" — one clip, a segment timeline (Raw → Enhance audio → Color → Text → SFX)
# ---------------------------------------------------------------------------------------------------------------
save("14_ref04_editing_process_timeline.json", {
    "_ref": "ref04 — the same clip goes through the edit steps: raw (flat grade, noisy audio) → enhanced audio → "
            "colour correction (grade_keys) → text animation (pops) → sound effects; a segment timeline with a red "
            "playhead shows the step, IG outro.",
    "meta": {"style": "swiss", "palette": "mono", "bg": "#9ab3b4", "bloom": 0, "grain": 0, "vignette": 0,
             "duration": 15.0},
    "hud": {**NOHUD, "subs": {"show": False}},
    "audio": {"ambient": False},
    "chapters": [{"t": 0, "title": "process", "elements": [
        {"type": "backdrop", "kind": "paper", "t": 0, "enter": "none"},
        {"type": "device_frame", "id": "phone", "x": 470, "y": 760, "w": 560, "h": 996, "t": 0, "children": [
            {"type": "footage", "id": "clip", "src": HOST, "x": 540, "y": 960, "w": 1080, "h": 1920, "loop": True,
             "grade_keys": [[0, "flat"], [6.0, "vivid", 0.6]], "audio": True, "audio_enhance": False},
            {"type": "kinetic", "id": "k1", "mode": "slam", "text": "سه تا کاری که **هرگز** انجام نمیدم",
             "x": 540, "y": 1250, "size": 110, "t": 9.0, "out": 12.0, "color": "#ffffff", "accent": "#ffd23f", "shadow": {"blur": 18, "alpha": 0.6}},
            {"type": "icon_grid", "id": "nums", "names": ["check", "check", "check"], "rows": 1, "cols": 3,
             "x": 540, "y": 1000, "size": 120, "gap": 40, "t": 9.2, "out": 12.0}]},
        {"type": "pill", "id": "tag_vid", "text": "ویدیو", "x": 860, "y": 330, "size": 40, "t": 0.3, "color": "#ffffff", "filled": True},
        {"type": "pill", "id": "tag_edit", "text": "تدوین", "x": 900, "y": 450, "size": 40, "t": 0.5, "color": "#ffffff", "filled": True},
        {"type": "nle_timeline", "id": "steps", "mode": "segments", "x": 540, "y": 1560, "w": 1000, "label_size": 26,
         "t": 0, "items": [{"label": "Raw video", "t": 0, "text_color": "#1d2a2a"},
                           {"label": "Enhance audio", "t": 3.0, "text_color": "#1d2a2a"},
                           {"label": "Color correction", "t": 6.0, "text_color": "#1d2a2a"},
                           {"label": "Text animation", "t": 9.0, "text_color": "#1d2a2a"},
                           {"label": "Sound effect", "t": 12.0, "text_color": "#1d2a2a"}]},
        {"type": "handle_tag", "handle": "yourhandle", "x": 540, "y": 1700, "size": 32, "color": "#2b3a3a", "t": 0.5},
        {"type": "social_outro", "handle": "@YOURHANDLE", "t": 13.6}]}]})

# ---------------------------------------------------------------------------------------------------------------
# 15 — ref05: talking head inside a phone on dark tiles, clip timeline under an I-beam, text tracked to hands
# ---------------------------------------------------------------------------------------------------------------
save("15_ref05_tiles_phone_clip_timeline_tracking.json", {
    "_ref": "ref05 — dark diamond tiles, framed phone with a talking head; a coloured clip timeline (Raw/Color/"
            "Music/Text/Motion) slides under an I-beam as each edit layer is added; words follow the hands (hand "
            "tracker), a speed ramp on the gesture, a grid + kinetic words behind the speaker, B&W ending.",
    "meta": {"style": "promo", "palette": "silver", "bg": "#2b2b2d", "bloom": 0.15, "grain": 0.01, "vignette": 0.2,
             "duration": 6.0},
    "hud": {**NOHUD, "subs": {"show": True, "y": 0.62, "size": 44, "box": {"color": "#000000", "alpha": 0.9}}},
    "subtitles": [{"t": 0.2, "end": 2.6, "text": "و ادیت **حرفه‌ای** تر"}, {"t": 2.8, "end": 5.6, "text": "محصولتو **لوکس‌تر** نشون میده"}],
    "chapters": [{"t": 0, "title": "phone", "elements": [
        {"type": "backdrop", "kind": "tiles", "t": 0, "enter": "none"},
        {"type": "device_frame", "id": "phone", "x": 540, "y": 860, "w": 700, "h": 1240, "t": 0, "border": "#e6e6e6",
         "children": [
            {"type": "footage", "id": "clip", "src": HANDS, "x": 540, "y": 960, "w": 1080, "h": 1920,
             "trackers": {"hand": {"kind": "hand", "point": "palm"}},
             "grade_keys": [[0, "flat"], [1.2, "moody", 0.4]], "ramps": [{"t": 3.6, "dur": 0.25, "speed": 0.35, "hold": 0.6}],
             "looks": [{"t": 5.3, "dur": 0.7, "look": "mono"}]},
            {"type": "kinetic", "id": "behind_k", "behind": "clip", "mode": "stack", "text": "محصولتو **لوکس‌تر** نشون میده",
             "x": 540, "y": 520, "size": 130, "t": 2.8, "color": "#ffffff", "accent": "#ffd23f", "shadow": {"blur": 20, "alpha": 0.8}},
            {"type": "pill", "id": "hand_tag", "text": "لوکس", "x": 0, "y": 0, "follow": "clip.hand",
             "follow_offset": [0, -120], "follow_lag": 0.08, "t": 3.0, "out": 5.2, "filled": True, "size": 54, "color": "#ff2d55"}]},
        {"type": "handle_tag", "handle": "@YOURHANDLE", "x": 790, "y": 330, "size": 26, "t": 0.2},
        {"type": "nle_timeline", "id": "layers", "mode": "clips", "x": 540, "y": 1380, "w": 1080, "h": 110,
         "t": 0, "items": [{"label": "ویدیو خام", "sub": "Raw", "color": "#c9a7d9", "t": 0},
                           {"label": "اصلاح رنگ", "sub": "Color", "color": "#e48fd0", "t": 1.2},
                           {"label": "موزیک", "sub": "Music", "color": "#8ea1e6", "t": 2.2},
                           {"label": "متن", "sub": "Text", "color": "#8edb84", "t": 2.8},
                           {"label": "حرکت دوربین", "sub": "Motion", "color": "#e9dc7c", "t": 3.6}]},
        {"type": "text", "id": "read_caption", "text": "کپشن بخون", "x": 540, "y": 1600, "size": 40, "font": "handwritten",
         "color": "#dddddd", "t": 0.5}]}]})

# ---------------------------------------------------------------------------------------------------------------
# 16 — ref06: BEFORE/AFTER, neon blue — saber glow on the speaker, 2D illustration, brain with words, streaks
# ---------------------------------------------------------------------------------------------------------------
save("16_ref06_before_after_neon_saber.json", {
    "_ref": "ref06 — small BEFORE clip + big AFTER frame on purple smoke; AFTER: close-up speaker with neon blue "
            "words, blue edge glow (saber) around the speaker, a depth-plate illustration, a glowing brain with words "
            "wrapping it, blue light streaks + neon numbers.",
    "meta": {"style": "promo", "palette": "cyber", "bg": "#0d0b12", "bloom": 0.35, "grain": 0.012, "vignette": 0.3,
             "duration": 9.0},
    "hud": {**NOHUD, "subs": {"show": False}},
    "chapters": [{"t": 0, "title": "before_after", "elements": [
        {"type": "backdrop", "kind": "smoke", "t": 0, "enter": "none", "drift": 20},
        {"type": "text", "id": "lbl_after", "text": "AFTER", "x": 640, "y": 300, "size": 56, "font": "title", "t": 0},
        {"type": "text", "id": "lbl_before", "text": "BEFORE", "x": 175, "y": 760, "size": 30, "font": "ui", "t": 0, "color": "#cccccc"},
        {"type": "device_frame", "id": "before", "x": 175, "y": 960, "w": 290, "h": 516, "radius": 14, "border": None,
         "t": 0, "children": [{"type": "footage", "id": "before_clip", "src": HOST, "x": 540, "y": 960, "w": 1080, "h": 1920,
                               "grade": "flat"}]},
        {"type": "device_frame", "id": "after", "x": 680, "y": 980, "w": 600, "h": 1066, "radius": 22, "border": None,
         "t": 0, "fill": "#05060c", "children": [
            {"type": "footage", "id": "aclip", "src": HOST, "x": 540, "y": 960, "w": 1080, "h": 1920, "t": 0, "out": 3.0,
             "grade": "moody", "frame": [[0, [0.5, 0.38, 1.25]]],
             "edge_glow": {"t": 1.2, "dur": 1.8, "color": "#1e7bff", "width": 14, "core": "#cfe6ff", "flicker": 0.2}},
            {"type": "text", "id": "neon1", "text": "چون کلاً **آدم شبم**", "x": 540, "y": 1350, "size": 110, "font": "title",
             "color": "#ffffff", "emph_color": "#3f9bff", "t": 0.3, "out": 3.0, "enter": "words",
             "glow": {"color": "#1e7bff", "radius": 22, "strength": 1.4}},
            {"type": "depth_plate", "id": "night", "src": GEN + "hiker.jpg", "x": 540, "y": 960, "w": 1080, "h": 1920,
             "t": 3.0, "out": 5.6, "push": [[0, 1.0], [2.6, 1.12]], "enter": "fade", "dur": 0.2},
            {"type": "backdrop", "id": "streaks", "kind": "streaks", "t": 5.6, "enter": "fade", "dur": 0.2},
            {"type": "icon", "id": "brain", "name": "brain", "x": 540, "y": 900, "size": 460, "t": 5.7, "color": "#ff5a6a",
             "width": 6, "glow": {"color": "#ff2a4a", "radius": 26}},
            {"type": "orbit_trail", "id": "orbit", "x": 540, "y": 900, "r": 330, "period": 2.4, "color": "#3f9bff", "t": 5.8},
            {"type": "kinetic", "id": "k_brain", "mode": "stack", "text": "و مغزم شروع می‌کنه به **کار کردن**",
             "x": 540, "y": 1450, "size": 100, "t": 6.0, "glow": {"color": "#1e7bff", "radius": 18}}]},
        {"type": "handle_tag", "handle": "@YOURHANDLE", "x": 860, "y": 520, "size": 22, "t": 0.2, "color": "#ffffff", "shadow": {"blur": 10, "alpha": 0.7}}]}]})

# ---------------------------------------------------------------------------------------------------------------
# 17 — ref07: ENSAN noir line art — podiums, dashed leader lines, pulsing eye, chain pan (pure SDK lineart)
# ---------------------------------------------------------------------------------------------------------------
save("17_ref07_noir_lineart_comparison_story.json", {
    "_ref": "ref07 — @ENSAN.BESHAM noir line-art essay: white strokes on black, figures on podiums, bars that grow, "
            "dashed callouts, a pulsing 'AI eye', phone scroll, chain-of-envy camera pan; karaoke subtitles at the "
            "bottom. (Core lineart style — see templates 03/04 for more widgets.)",
    "meta": {"style": "lineart", "palette": "noir", "bloom": 0.45, "vignette": 0.35, "grain": 0.02, "duration": 13.0},
    "hud": {"header": {"show": False}, "dots": {"show": False}, "subs": {"show": True}, "handle": "@YOURHANDLE"},
    "subtitles": [{"t": 0.3, "end": 3.8, "text": "از بین این دو گزینه **فقط یکیش** رو می‌تونی"},
                  {"t": 4.0, "end": 7.6, "text": "توی دانشگاه **هاروارد** همین سؤال رو پرسیدن"},
                  {"t": 7.8, "end": 12.6, "text": "عدد معنی نداره؛ **فاصله** معنی داره"}],
    "chapters": [
        {"t": 0, "title": "choice", "elements": [
            {"type": "box", "id": "optA", "x": 300, "y": 800, "w": 400, "h": 760, "r": 18, "t": 0.2},
            {"type": "box", "id": "optB", "x": 780, "y": 800, "w": 400, "h": 760, "r": 18, "t": 0.4},
            {"type": "bars", "id": "barsA", "values": [50, 25, 25], "x": 300, "y": 980, "w": 320, "h": 380, "t": 0.8, "fmt": "{:.0f}"},
            {"type": "bars", "id": "barsB", "values": [100, 200, 200], "x": 780, "y": 980, "w": 320, "h": 380, "t": 1.4, "fmt": "{:.0f}"},
            {"type": "callout", "id": "c_you", "anchor": "barsA+-100,60", "elbow": [-60, -260], "shelf": -120, "label": "تو", "t": 2.2}]},
        {"t": 4.0, "title": "هاروارد", "transition": {"type": "pan", "dir": "down"}, "elements": [
            {"type": "icon", "id": "eye", "name": "eye", "x": 540, "y": 520, "size": 200, "t": 0.3, "idle": "pulse"},
            {"type": "glow_ring", "id": "eye_ring", "x": 540, "y": 520, "r": 120, "color": "#ffffff", "t": 0.5},
            {"type": "icon_grid", "id": "crowd", "names": "human", "rows": 3, "cols": 6, "x": 540, "y": 1050,
             "size": 90, "gap": 30, "t": 1.0, "highlight": [7], "highlight_at": 2.4}]},
        {"t": 7.8, "title": "فاصله", "transition": {"type": "pan", "dir": "down"}, "elements": [
            {"type": "box", "id": "pod1", "x": 380, "y": 1200, "w": 200, "h": 300, "t": 0.2},
            {"type": "box", "id": "pod2", "x": 700, "y": 1100, "w": 200, "h": 500, "t": 0.4},
            {"type": "icon", "id": "p1", "name": "human", "x": 380, "y": 960, "size": 160, "t": 0.6},
            {"type": "icon", "id": "p2", "name": "human", "x": 700, "y": 760, "size": 160, "t": 0.8},
            {"type": "line", "id": "gap", "points": [[540, 1050], [540, 850]], "arrow": "both", "dashed": [10, 8], "t": 1.6},
            {"type": "text", "id": "gap_l", "text": "فاصله", "x": 540, "y": 780, "size": 52, "t": 1.9}]}]})

# ---------------------------------------------------------------------------------------------------------------
# 18 — ref08: radial "viral content formula" on a light studio sweep, open-loop dark interludes, viewfinder end
# ---------------------------------------------------------------------------------------------------------------
nodes = [{"label": "هوک", "icon": "target", "t": 1.0}, {"label": "وعده", "icon": "handshake", "t": 3.0},
         {"label": "ارزش", "icon": "star", "t": 5.0}, {"label": "حلقه‌ی باز ۱", "icon": "question", "t": 7.0},
         {"label": "ارزش", "icon": "star", "t": 9.0}, {"label": "حلقه‌ی باز ۲", "icon": "question", "t": 11.0},
         {"label": "آخرین ارزش", "icon": "trophy", "t": 13.0}, {"label": "دعوت به اقدام", "icon": "megaphone", "t": 15.0}]
save("18_ref08_radial_viral_formula.json", {
    "_ref": "ref08 — light grey studio, a black octagon hub 'فرمول محتوای وایرال' with 8 nodes; the wheel rotates "
            "node by node, red progress arc, a big ghost step number top-right, redacted keyword reveals, two dark "
            "'open loop' interludes with a red dot orbiting, final zoom-out under a REC viewfinder.",
    "meta": {"style": "swiss", "palette": "red", "bg": "#efefec", "bloom": 0, "grain": 0, "vignette": 0.12,
             "duration": 19.0},
    "hud": {**NOHUD, "subs": {"show": False}},
    "chapters": [
        {"t": 0, "title": "formula", "camera": [[0, [540, 960, 0.6]], [1.0, [540, 1250, 1.0], "smooth"],
                                                 [16.6, [540, 1250, 1.0]], [17.4, [540, 960, 0.62], "smooth"]],
         "elements": [
            {"type": "backdrop", "kind": "studio", "t": 0, "enter": "none"},
            {"type": "radial_formula", "id": "wheel", "x": 540, "y": 1500, "r": 470, "hub_r": 250, "nodes": nodes,
             "hub_title": "محتوای وایرال", "hub_sub": "فرمولِ", "accent": "#ff4b2b", "t": 0.1, "label_size": 56},
            {"type": "redact_text", "id": "r1", "text": "اول، **هوک**:", "x": 600, "y": 620, "size": 70, "t": 1.0,
             "reveal_at": 0.5, "out": 2.9},
            {"type": "redact_text", "id": "r2", "text": "قراره چی یاد بگیرن", "x": 600, "y": 620, "size": 60,
             "t": 3.0, "reveal_at": 0.5, "out": 4.9},
            {"type": "big_number", "id": "ghost", "x": 860, "y": 420, "size": 260, "color": "#e2e2df", "t": 1.0, "decor": True,
             "tracks": {"value": [[1.0, 1], [3.0, 2], [5.0, 3], [7.0, 4], [9.0, 5], [11.0, 6], [13.0, 7], [15.0, 8]]},
             "fmt": "{:02.0f}", "roll": True},
            {"type": "viewfinder_hud", "id": "rec", "x": 540, "y": 960, "w": 980, "h": 1700, "t": 5.6, "out": 7.4,
             "label": "REC"}]},
        {"t": 7.4, "title": "open loop", "transition": {"type": "cut"}, "bg": "#0b0b0c", "elements": [
            {"type": "orbit_trail", "id": "loop1", "x": 540, "y": 900, "r": 330, "period": 2.2, "color": "#ff4b2b",
             "t": 0, "ring_alpha": 0.15},
            {"type": "text", "id": "loop_t", "text": "ولی به نکته‌ی **مهم‌تر**", "x": 540, "y": 900, "size": 80,
             "font": "title", "color": "#ffffff", "emph_color": "#ff4b2b", "t": 0.2, "enter": "words"}]},
        {"t": 8.8, "title": "back", "transition": {"type": "cut"}, "origin": [0, 0], "elements": [
            {"type": "backdrop", "kind": "studio", "t": 0, "enter": "none"},
            {"type": "radial_formula", "id": "wheel2", "x": 540, "y": 1500, "r": 470, "hub_r": 250,
             "nodes": [dict(n, t=n["t"] - 8.8) for n in nodes], "hub_title": "محتوای وایرال", "hub_sub": "فرمولِ",
             "accent": "#ff4b2b", "t": 0, "enter": "none", "label_size": 56}]}]})

# ---------------------------------------------------------------------------------------------------------------
# 19 — ref09: neon line-art storytelling (AI images → auto_trace with source colours + glow), puppet, depth
# ---------------------------------------------------------------------------------------------------------------
save("19_ref09_neon_lineart_story_genimages.json", {
    "_ref": "ref09 — black background, colourful neon line-art illustrations drawn on stroke by stroke (generated "
            "images → auto_trace color_mode source + glow), shaking icons under stress, fire/blood accents, a "
            "puppet character, light rays, big white Persian captions in the middle. Use `mograph prompts` to "
            "generate the illustration set for a new script.",
    "meta": {"style": "lineart", "palette": "noir", "bg": "#000000", "bloom": 0.5, "vignette": 0.3, "grain": 0.015,
             "duration": 12.0},
    "hud": {**NOHUD, "subs": {"show": True, "y": 0.5, "size": 64, "box": {"color": "#000000", "alpha": 0.88}}, "handle": "@YOURHANDLE"},
    "subtitles": [{"t": 0.3, "end": 3.6, "text": "مغز تو مثل **یک فیل** گیر کرده"},
                  {"t": 3.8, "end": 7.6, "text": "و هر چی بیشتر **شلاق** بزنی"},
                  {"t": 7.8, "end": 11.6, "text": "بیشتر **فرو** میره"}],
    "chapters": [
        {"t": 0, "title": "elephant", "elements": [
            {"type": "auto_trace", "id": "eleph", "src": GEN + "elephant_line.jpg", "x": 540, "y": 640, "w": 1000,
             "color_mode": "source", "width": 3.4, "dur": 3.0, "stagger": 0.5, "glow": {"radius": 16, "strength": 1.3},
             "gen": {"prompt": "neon line art elephant stuck in mud", "aspect": "1:1"}},
            {"type": "icon", "id": "battery", "name": "battery", "x": 250, "y": 1350, "size": 140, "t": 1.5,
             "shiver": {"amp": 5, "freq": 20}, "color": "#ff3b3b", "glow": {"color": "#ff3b3b", "radius": 14}},
            {"type": "icon", "id": "clock", "name": "clock", "x": 830, "y": 1350, "size": 140, "t": 1.8,
             "shiver": {"amp": 5, "freq": 20}, "color": "#2ad1ff", "glow": {"color": "#2ad1ff", "radius": 14}}]},
        {"t": 3.8, "title": "whip", "transition": {"type": "cut"}, "shake": [{"t": 1.2, "dur": 0.35, "amp": 22}],
         "flashes": [{"t": 1.2, "dur": 0.15, "color": "#ff2a2a", "amount": 0.35}], "elements": [
            {"type": "sprite_rig", "id": "rider", "src": GEN + "puppet_sheet.jpg", "x": 540, "y": 650, "scale": 0.9,
             "idle": "talk", "t": 0.1, "pose": {"arm_r": [[0, 0], [1.0, -120, "in"], [1.2, 40, "out_expo"], [2.0, 0]]}},
            {"type": "saber", "id": "whip", "points": [[640, 520], [820, 380], [930, 520], [880, 760]], "color": "#ff2a4a",
             "t": 1.0, "dur": 0.25, "width": 7},
            {"type": "flash_pulse", "id": "hit", "x": 880, "y": 760, "r": 200, "color": "#ff4444", "t": 1.2}]},
        {"t": 7.8, "title": "sink", "transition": {"type": "cut"}, "elements": [
            {"type": "light_rays", "id": "rays", "x": 540, "y": 400, "r": 900, "color": "#ffd58a", "t": 0, "alpha": 0.25},
            {"type": "depth_plate", "id": "scene", "src": GEN + "hiker.jpg", "x": 540, "y": 1250, "w": 900, "h": 900,
             "t": 0.2, "push": [[0, 1.0], [3.5, 1.15]], "pan": [[0, [0, 0]], [3.5, [0, 40]]]}]}]})
