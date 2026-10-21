"""
mograph.qa — automatic quality gates (run BEFORE spending time on a full render).

lint(scene)            static checks: unknown types, missing glyphs (▯), bad targets,
                       word anchors, timing beyond duration, HUD-zone intrusions.
overlap_report(scene)  samples the timeline and reports overlapping element bboxes
                       (text-on-text / text-on-icon collisions) + out-of-frame content
                       — computed from REAL vector geometry, not by eyeballing.
contact_sheet(scene, path, times=None, cols=5)   one JPG grid of key frames with timestamps,
                       sampled at every chapter "hero" moment (≈ end of each chapter).
probe(path)            ffprobe summary of the final mp4 (duration / fps / size / audio).

All return plain lists/dicts so the agent can read them and fix issues in the spec.
"""
from __future__ import annotations

import json
import math
import os
import subprocess

import cairo
import numpy as np

from .elements import REGISTRY, compute_state
from .render import frame_to_rgb, render_frame
from .text import fallback_chars, missing_glyphs, parse_markup, resolve_font

TEXT_TYPES = {"liquid_blob", "starburst", "prompt_box", "filmstrip", "iso_cube", "waffle_grid", "swiss_bars", "versus_table", "headline", "big_number", "index", "kinetic", "block", "text", "callout", "stamp", "pill", "strike", "counter", "quote_rig", "invoice", "laser_scan"}
_TEXT_KEYS = ("text", "label", "false", "true", "tab", "right", "left", "prefix", "suffix", "accent_label",
              "base_label", "right_title", "left_title", "unit", "caption")


COMMON_KEYS = {"type", "id", "t", "dur", "out", "out_dur", "enter", "exit", "ease", "x", "y", "color", "dim_at",
               "undim_at", "dim_color", "idle", "tracks", "layer", "space", "sfx", "sfx_offset", "sfx_gain",
               "allow_overlap", "opacity", "group", "font", "weight", "label", "label_size", "label_color",
               "label_font", "size", "rot", "note", "col", "row", "cues", "sound", "w", "h",
               "z", "pulse_amount", "vo_ring",
               # v9 shared typography / motion keys
               "text_style", "stroke_reveal", "stroke_reveal_at", "extrude_grow", "kashida", "kashida_at",
               "kashida_dur", "kashida_ease", "kashida_back", "kashida_word", "whip", "whip_dir", "whip_dur",
               "whip_dist", "blur", "blur_max", "emph_style", "prefix", "prefix_color", "prefix_scale",
               "prefix_font",
               # v11 shared keys (tracking / depth / glow / containers)
               "follow", "follow_offset", "follow_rot", "follow_rot0", "follow_scale", "follow_lag", "follow_axis",
               "behind", "parent", "glow", "shadow", "shiver", "children", "radius", "border", "border_w", "tilt",
               "text_color", "fill", "accent", "ink", "paper", "src", "font", "text", "sub", "sub_font", "width", "gen", "decor", "fx", "preset", "preset_dur", "preset_out", "preset_out_dur", "animators", "selectors"}
ELEMENT_KEYS = {
    "icon": {"name", "style", "fill", "width", "dashed", "detail", "label_gap", "morph_to", "morph_at", "morph_dur",
             "morph_ease", "twist"},
    "text": {"text", "align", "valign", "max_width", "line_height", "emph_color", "base_color", "underline",
             "fa_digits", "emph_dim_others"},
    "box": {"w", "h", "r", "fill", "double", "tab", "tab_w", "lines", "dashed", "width"},
    "line": {"points", "arrow", "closed", "dashed", "fillet", "width", "head"},
    "arrow": {"from", "to", "curve", "head", "label_offset", "label_align", "dashed", "width"},
    "callout": {"anchor", "elbow", "shelf"},
    "stamp": {"text", "angle", "w", "h"},
    "strike": {"false", "true"},
    "counter": {"from", "to", "value", "fmt", "prefix", "suffix", "fa", "punch", "roll"},
    "dot_grid": {"rows", "cols", "gap", "r", "highlight", "highlight_at"},
    "icon_grid": {"names", "name", "rows", "cols", "gap", "highlight", "highlight_at", "cross_at", "stagger", "width"},
    "bars": {"values", "labels", "w", "h", "highlight", "fmt", "suffix", "rtl", "value_size"},
    "ring": {"value", "from", "r", "text", "track_width", "width", "punch"},
    "timeline": {"w", "items", "rtl"},
    "pill": {"text", "filled", "w", "h"},
    "divider": {"h", "right", "left", "heading_offset", "line_color"},
    "image": {"src", "w", "radius"},
    "block": {"w", "h", "color", "side", "text", "text_color", "size", "pad", "align", "line_height", "col", "row", "rows"},
    "headline": {"text", "w", "size", "line_height", "align", "color", "accent", "stagger", "col", "row"},
    "rule": {"w", "weight", "color", "side", "col", "row"},
    "big_number": {"value", "from", "w", "size", "fmt", "fa", "align", "color", "unit", "caption", "caption_size",
                   "caption_color", "unit_font", "caption_font", "punch", "col", "row", "roll"},
    "index": {"items", "w", "gap", "size", "stagger", "color", "accent", "mute", "highlight", "highlight_at", "col", "row"},
    "disc": {"r", "color"},
    "kinetic": {"text", "mode", "size", "color", "accent", "mute", "rate", "sync", "max_lines", "max_width", "lead",
                "rows", "gap", "line_gap", "min_size", "max_size", "slab", "slab_pad", "sweep_dur", "align"},
    "waffle_grid": {"value", "total", "n", "side", "gap", "cell_r", "fill_at", "accent_label", "base_label", "accent",
                    "mute", "label_size", "label_font", "rule_weight", "build", "align", "rows"},
    "swiss_bars": {"items", "max", "orient", "bar_h", "gap", "stagger", "highlight", "accent", "paper", "label_size",
                   "value_size", "label_font", "value_font", "fmt", "suffix", "pad", "rows"},
    "versus_table": {"rows", "right_title", "left_title", "title_size", "head_h", "row_h", "pad", "accent", "paper",
                     "mute", "title_font", "rule_weight"},
    "waveform": {"bars", "mode", "lag", "floor", "fill", "cycles", "width"},
    "video": {"src", "w", "h", "fill", "clip_start", "speed", "loop", "blend", "grade", "radius", "border"},
    "path": {"d", "fill", "fillet", "dashed", "width"},
    "circle": {"r", "target", "number", "dashed", "start", "width", "fill"},
    "morph": {"from", "to", "chain", "at", "draw_in", "twist", "width", "detail"},
    "quote_rig": {"text", "source", "w", "highlights", "header", "line_height"},
    "flow_loop": {"top", "left", "right", "w", "h", "speed", "particles", "flips", "arrow_labels", "arrow_label_size"},
    "laser_scan": {"title", "clause", "w", "h", "lines", "scan_at", "scan_dur", "clause_row"},
    "invoice": {"title", "rows", "total", "stamp", "w", "row_h", "value_font", "stamp_font"},
    "strata": {"w", "surface", "layers", "particles", "drill_at", "drill_x", "label_top", "label_bottom", "end", "rise", "bend", "q_size"},
    "scale_rig": {"left", "right", "drops", "max_tilt", "w", "item_size"},
    "speech_bubble": {"text", "tail", "max_width"},
    # v9 promo / 3D / UI widgets
    "liquid_blob": {"r", "text", "a1", "a2", "w1", "w2", "drift", "drift_speed", "seed", "text_color", "invert_color",
                    "text_x", "text_y", "max_width", "glow"},
    "iso_cube": {"s", "rx", "ry", "rz", "wx", "wy", "wz", "shape", "h", "edges", "edge_color", "face_colors", "light",
                 "persp", "spin_in", "label", "label_size", "label_color"},
    "starburst": {"r", "points", "inner", "spin", "angle", "text", "text_color", "text_rot", "stroke", "stroke_w",
                  "rings"},
    "prompt_box": {"w", "h", "label", "label_size", "prompt", "placeholder", "type_at", "cps", "button", "accent",
                   "mute", "panel", "radius", "send_at"},
    "timeline_ui": {"w", "h", "title", "layers", "layer_colors", "strike_at", "scrub_speed", "panel", "mute"},
    "bezier_graph": {"w", "h", "from", "to", "flex_at", "flex_dur", "title", "label", "label_size", "label_color",
                     "accent", "panel"},
    "viewfinder_hud": {"w", "h", "tc_start", "fa", "label", "rec_color", "bracket", "width", "speed_lines", "lines",
                       "speed", "seed"},
    "filmstrip": {"w", "h", "angle", "speed", "frames", "word", "word_color", "word_delay", "film_color",
                  "hole_color", "frame_colors", "accent"},
    "sine_wave": {"w", "amp", "wavelength", "phase_speed", "width", "dir", "taper", "vo", "glow"},
    "talking_head": {"src", "w", "h", "bg", "rim", "rim_w", "rim_alpha", "behind", "pip_at", "pip_dur", "pip",
                     "punch", "matte", "matte_h", "key", "grade", "zoom", "speed", "clip_start", "loop",
                     "follow_cuts", "feather_bottom"},
    # v10 ui_physics widgets
    "ui_toggle": {"w", "h", "accent", "off_color", "click_at", "reset_at", "knob", "knob_r", "cursor"},
    "ui_slider": {"w", "track_h", "ticks", "accent", "track_bg", "tooltip_bg", "fa", "roll_at", "roll_dur",
                  "rewind_at", "rewind_dur", "max_value", "tick_offset", "knob", "knob_r", "tooltip",
                  "tooltip_offset", "hide_tooltip_after"},
    "ui_notification_stack": {"w", "card_h", "gap", "accent", "card_bg", "card_border", "cards", "stagger"},
    "ui_keyboard": {"scale", "key_bg", "key_border", "key_active", "hits", "row1", "row2", "return_label"},
    "ui_reply_pill": {"accent", "reply_bg", "reply_border", "placeholder", "type_at", "send_at", "hits",
                      "pill_w", "field_w", "h", "sent_text"},
    "physics_ball": {"r", "color", "waypoints"},
    # v11 Studio Pantheon
    "footage": {"src", "w", "h", "fit", "anchor_uv", "frame", "reframe", "reframe_at", "reframe_zoom", "punch",
                "punch_amt", "punch_every", "zooms", "beat_pump", "shake", "clip_start", "speed", "loop", "ramps",
                "remap", "beat_ramp", "freeze", "grade", "grade_keys", "selective", "hue_shift", "halftone", "flash",
                "looks", "matte", "matte_h", "key", "bg", "roto", "edge_glow", "trackers", "audio", "audio_gain",
                "audio_enhance", "follow_cuts", "edit", "segments", "enhance", "auto_grade", "dof_blur", "headroom", "bg_dim", "bg_tint", "xfade_ms", "skin", "auto_push"},
    "title_card": {"slab", "slab2", "slab_show", "icon_3d", "icon2_3d", "left_sprite", "right_sprite", "skew", "emph_color", "sub_color", "sub_size", "sub_font", "pad", "wipe_dur",
                   "rotate", "max_width", "word_gap", "size", "text", "sub"},
    "logo_chip": {"items", "gap", "max_w", "style", "radius", "bg", "size"},
    "route_arc": {"from", "to", "label_from", "label_to", "bend", "trail_color", "plane", "plane_size", "pin_color",
                  "pin_color2", "pin_r", "label_bg", "label_color", "width", "size", "ease"},
    "price_tag": {"value", "from", "prefix", "suffix", "label", "w", "h", "text_color", "swing", "count_dur", "fa",
                  "shape", "fmt", "rotate", "size"},
    "grade_scale": {"grades", "pick", "w", "h", "at", "title", "title_size", "gap", "check"},
    "profit_meter": {"buy", "sell", "unit", "w", "h", "colors", "labels", "fa"},
    "sticker": {"icon", "text_color", "rotate", "wobble", "edge", "size"},
    "cta_arrow": {"from", "to", "label", "label_color", "width", "curve", "head", "size"},
    "light_leak": {"color2", "dir", "alpha"},
    "shine_sweep": {"w", "h", "angle", "alpha", "width", "radius"},
    "progress_bar": {"h", "bg", "from", "to"},
    "vignette_pulse": {"amount", "at", "decay"},
    "planar_insert": {"src", "fit", "edge", "blend", "grade", "speed"},
    "saber": {"points", "d", "target", "pad", "core", "width", "haze", "travel", "seg", "period", "loop", "flicker",
              "additive", "r", "preset", "start", "end", "core_size", "core_softness", "core_start", "core_end",
              "reveal", "out_fx", "shape", "paths", "closed", "evolution", "glow_intensity", "glow_spread", "glow_bias",
              "halo_intensity", "flicker_intensity", "flicker_speed", "distortion", "distortion_amount",
              "distortion_mode", "noise_scale", "noise_speed", "noise_complexity", "noise_aspect", "wind_speed",
              "wind_dir", "core_distortion", "lightning", "glows", "ramp", "gamma", "brightness", "saturation",
              "alpha_boost", "composite", "text_mode", "quality", "seed", "spill", "spill_amount",
              "start_anchor", "end_anchor", "start_offset", "end_offset", "angle", "length"},
    "aecomp": {"gen", "comp", "lower_third", "title_style", "title", "role", "hero", "kicker", "seed", "c1", "c2", "c3",
               "stroke", "speed", "loop", "bg", "controls"},
    "character": {"rig", "actions", "lip_sync", "timing", "exaggeration", "look", "blink", "breathe", "outline", "seed"},
    "mouth": {"lip_sync", "visemes", "open", "shape"},
    "flash_pulse": {"r", "rings", "rays", "ray_rot", "screen"},
    "glow_ring": {"r", "rings", "period", "width", "spread"},
    "tracked_callout": {"anchor", "box", "sub", "box_style", "elbow", "dot", "lag", "box_follow", "anchor0",
                        "width", "border_alpha", "sub_color"},
    "light_rays": {"r", "n", "spin", "alpha", "width"},
    "orbit_trail": {"r", "ry", "period", "start_deg", "trail", "ring", "ring_color", "ring_alpha", "dir",
                    "speed_keys"},
    "emitter": {"kind", "shape", "n", "rate", "life", "speed", "spread", "gravity", "colors", "seed", "area",
                "total", "stagger", "angle", "bubble"},
    "device_frame": {"label_pos", "label_size", "label_color", "label_font", "vw", "vh", "glass"},
    "backdrop": {"kind", "color2", "seed", "drift", "alpha", "n"},
    "badge_button": {"edge", "lip", "press_at", "stroke"},
    "handle_tag": {"handle", "icon", "align", "gradient"},
    "social_outro": {"handle", "dim", "dim_color", "white_at", "handle_size"},
    "nle_timeline": {"items", "mode", "clip_w", "gap", "playhead_color", "bar_color", "bar_alpha", "sub_size",
                     "slide_dur", "rtl", "bar_h", "last_dur"},
    "image_card": {"shape", "kenburns", "kb_dur", "fit", "focus", "clip_start", "grade"},
    "radial_formula": {"nodes", "r", "hub_r", "node_r", "rotate", "rot_dur", "hub_title", "hub_sub", "step_fmt",
                       "show_labels", "label_gap"},
    "redact_text": {"bar_color", "reveal_at", "reveal_dur", "dir", "pad"},
    "auto_trace": {"mode", "invert", "thresh", "min_len", "simplify", "color_mode", "order", "stagger", "fill_image"},
    "depth_plate": {"layers", "split", "push", "pan", "strength", "dof", "bg_blur"},
    "sprite_rig": {"grid", "parts", "scale", "bg", "pose", "speed", "joints", "auto_joints"},
    "ui_film_world": {"loop_dur", "accent", "fa", "cards", "word"},
    # v13.5 Visual-DNA & Zero-Box elements
    "dna_stage": {"mode", "accent", "grid_step", "dof_blur", "headroom", "bg_dim", "bg_tint", "telemetry"},
    "dna_connector": {"from", "to", "device", "amp", "ticks"},
    "tactile_3d": {"sprite", "asset", "src", "bg_key", "size", "yaw_from", "pitch_from", "yaw", "pitch", "roll",
                   "idle_spin", "spin_amp", "pitch_amp", "shadow", "extrude_layers", "title", "title_size",
                   "sub", "sub_size", "sub_font", "label_offset_y", "accent", "text_color"},
    "hud_loading_bar": {"w", "h", "from", "to", "value", "max", "load_dur", "segments", "rtl", "fa",
                        "label", "label_size", "value_size", "accent", "text_color"},
    # v13.8 Creative Viral Reel Widgets
    "wireframe_badge": {"r", "shape", "angle", "accent", "text_color", "corner_sprite", "sprite",
                        "top", "sub", "title", "text", "bottom", "unit", "top_font", "font", "bot_font",
                        "top_size", "size", "bot_size"},
    "circular_dial": {"r", "accent", "center_text", "text", "top_text", "bottom_text", "flip_at",
                      "arc_size", "center_size"},
    "glass_hud_frame": {"w", "h", "angle", "accent", "size", "top_text", "right_text", "bottom_text"},
    "zigzag_path_3d": {"w", "h", "items", "accent", "walk_dur", "label_size"},
    "avatar_carousel_3d": {"count", "sprites"},
    "silhouette_scribble": {"r", "glasses"},
    # v13.9 @novadent 100/100 Master Widgets
    "novadent_split_dome": {"y_split", "panels_at", "left_en", "left_fa", "right_en", "right_fa",
                            "dome_y", "dome_r", "accent", "top_title", "title", "top_size", "size"},
    "novadent_cd_dial": {"r", "rotate_at", "rotate_dur", "red_color", "gold_color", "top_text",
                         "bottom_text", "arc_size", "center_top", "center_bot", "center_size", "center_sub_size"},
    "novadent_diamond_square": {"r", "angle", "accent", "corner_sprite", "top", "title", "bottom",
                                "top_size", "size", "bot_size"},
    "novadent_chair_glow_text": {"accent", "line1", "hero", "line3", "line4", "size1", "hero_size", "size3", "size4"},
    "novadent_edge_light_spill": {"r", "color"},
    "novadent_red_halo": {"r_outer", "r_inner", "darken_room", "top", "title", "bottom", "top_size", "size", "bot_size"},
    "novadent_glass_hex": {"r", "angle", "accent", "top", "title", "bottom", "top_size", "size", "bot_size"},
    "orbiting_title_3d": {"mode", "text", "font", "size", "color", "swoop_dur", "rx", "ry", "speed", "phase", "half"},
    # v14.4 3D engine (sam3d_bridge.py + sculpt3d.py)
    "sam3d_object": {"src", "scale", "yaw", "yaw_speed", "pitch", "roll"},
    "real3d": {"model", "w", "size", "yaw", "pitch", "roll", "motion", "yaw_speed", "yaw_amp", "pitch_amp", "speed", "shadow", "ssaa"},
    "sculpt3d": {"preset", "tree", "w", "size", "yaw", "yaw_speed", "pitch", "pitch_amp", "roll", "open_dur", "shadow", "ssaa"},
    "image_to_3d": {"src", "sprite", "w", "size", "yaw", "pitch", "roll", "motion", "yaw_speed", "speed", "yaw_amp", "pitch_amp", "depth", "specular", "shadow", "geometry", "views", "label", "three_model", "three_script", "explode", "allow_slab"},
    "three3d": {"three_model", "model", "preset", "three_script", "src", "w", "size", "yaw", "pitch", "roll", "motion", "yaw_speed", "yaw_amp", "pitch_amp", "speed", "explode", "shadow", "ssaa"},
}
for _t in ("headline", "kinetic", "block", "big_number"):
    ELEMENT_KEYS[_t] = ELEMENT_KEYS[_t] | set()


def _close_match(k, pool):
    import difflib
    m = difflib.get_close_matches(k, pool, n=1, cutoff=0.75)
    return m[0] if m else None



def _beat_times(scene):
    """Every visible beat: element starts + widget-internal events (highlights, drops, rows, scans,
    morph chain steps, value-track changes, flips) + chapter starts."""
    ts = [c["t"] for c in scene.chapters]
    for ch in scene.chapters:
        for el in ch["elements"]:
            ts.append(el["t"])
            if el.get("type") == "kinetic":
                from .typo import kinetic_words
                for _, wt, _, _ in kinetic_words(el, scene):
                    ts.append(float(wt))
            for k in ("scan_at", "drill_at", "morph_at", "stamp_at", "flip_at", "highlight_at",
                      "kashida_at", "stroke_reveal_at", "pip_at", "strike_at", "flex_at", "type_at", "send_at"):
                if isinstance(el.get(k), (int, float)):
                    ts.append(float(el[k]))
            if el.get("type") in ("talking_head", "prompt_box", "timeline_ui", "bezier_graph", "viewfinder_hud",
                                  "liquid_blob", "iso_cube", "starburst", "filmstrip", "sine_wave",
                                  "ui_toggle", "ui_slider", "ui_notification_stack", "ui_keyboard", "ui_reply_pill", "physics_ball", "ui_film_world", "waffle_grid", "versus_table", "swiss_bars"):
                t1 = float(el["out"]) if el.get("out") is not None else (scene.chapters[el["_chapter"]]["end"] or scene.duration)
                ts += [float(el["t"]) + k * 1.2 for k in range(int(max(0.0, t1 - float(el["t"])) / 1.2) + 1)]
            if isinstance(el.get("fill_at"), (int, float)):
                ts.append(float(el["fill_at"]))
            if el.get("type") == "footage":           # v11: edit events inside the clip are beats (local → abs)
                t0 = float(el["t"])
                for k in ("ramps", "zooms", "flash", "hue_shift", "halftone", "looks", "shake"):
                    for it in el.get(k) or []:
                        v = it if isinstance(it, (int, float)) else it.get("t") if isinstance(it, dict) else None
                        if isinstance(v, (int, float)):
                            ts.append(t0 + float(v))
                for k in ("frame", "grade_keys"):
                    ts += [t0 + float(kk[0]) for kk in el.get(k) or [] if isinstance(kk[0], (int, float))]
                if isinstance(el.get("punch"), list):
                    ts += [t0 + float(v) for v in el["punch"]]
                if el.get("trackers") or el.get("reframe"):      # live footage of a moving speaker is never dead
                    t1 = float(el["out"]) if el.get("out") is not None else (scene.chapters[el["_chapter"]]["end"] or scene.duration)
                    ts += [t0 + k * 2.0 for k in range(int(max(0.0, t1 - t0) / 2.0) + 1)]
            if el.get("type") in ("orbit_trail", "emitter", "glow_ring", "sprite_rig", "depth_plate", "light_rays",
                                  "nle_timeline", "radial_formula", "dna_stage", "dna_connector", "tactile_3d", "hud_loading_bar"):
                t1 = float(el["out"]) if el.get("out") is not None else (scene.chapters[el["_chapter"]]["end"] or scene.duration)
                ts += [float(el["t"]) + k * 1.5 for k in range(int(max(0.0, t1 - float(el["t"])) / 1.5) + 1)]
            if el.get("type") == "radial_formula":
                ts += [float(n["t"]) for n in el.get("nodes") or [] if isinstance(n.get("t"), (int, float))]
            if el.get("type") == "kinetic" and el.get("mode") == "highlight_sweep":
                from .typo import kinetic_words
                ts += [float(w[1]) for w in kinetic_words(el, scene) if w[3]]
            if el.get("type") == "waveform":           # an audio-reactive element is never "dead" while VO plays
                t1 = float(el["out"]) if el.get("out") is not None else (scene.chapters[el["_chapter"]]["end"] or scene.duration)
                ts += [float(el["t"]) + k * 1.5 for k in range(int((t1 - float(el["t"])) / 1.5) + 1)]
            for k in ("highlights", "drops", "rows", "events", "items"):
                for it in (el.get(k) if isinstance(el.get(k), list) else []):
                    if isinstance(it, dict) and isinstance(it.get("t"), (int, float)):
                        ts.append(float(it["t"]))
            for k in ("total", "stamp"):
                if isinstance(el.get(k), dict) and isinstance(el[k].get("t"), (int, float)):
                    ts.append(float(el[k]["t"]))
            ts += [float(x) for x in (el.get("_chain_t") or []) if isinstance(x, (int, float))]
            for f in el.get("flips") or []:
                if isinstance(f, (list, tuple)) and isinstance(f[0], (int, float)):
                    ts.append(float(f[0]))
            for kf in ((el.get("tracks") or {}).get("value") or []):
                if isinstance(kf[0], (int, float)):
                    ts.append(float(kf[0]))
    return ts


def lint(scene):
    issues = []

    def add(level, where, msg):
        issues.append({"level": level, "where": where, "msg": msg})

    W, H = scene.W, scene.H
    hud = scene.hud
    # v6: media / voice / omni
    for ch in scene.chapters:
        for el in ch["elements"]:
            if el.get("type") in ("video", "talking_head"):
                src = el.get("_src_abs") or el.get("src")
                if not src or not os.path.exists(src):
                    add("error", el["id"], f"video src not found: {el.get('src')}")
    vo = (scene.spec.get("audio") or {}).get("vo")
    if vo and not os.path.exists(os.path.join(scene.base_dir, vo)):
        add("error", "audio.vo", f"voice-over file not found: {vo} (run: python -m mograph voice spec.json)")
    if scene.spec.get("voice") and not vo:
        add("warn", "voice", "spec has a \"voice\" block but no audio.vo — run `mograph voice spec.json` and render the "
            "generated *.voiced.json")
    for jid in getattr(scene, "missing_plates", []):
        add("warn", f"omni:{jid}", "clip not generated yet → the vector version is shown there "
            "(run `mograph omni spec.json`, show the plan to the user, then --yes)")
    try:
        from .omni import resolve_jobs
        for j in resolve_jobs(scene):
            span = j["t1"] - j["t0"]
            if not 3.0 <= span <= 10.0:
                add("error", f"omni:{j['id']}", f"span {span:.1f}s — Omni clips must be 3..10 s")
            if any("\u0600" <= ch <= "\u06ff" for ch in j.get("prompt", "")):
                add("warn", f"omni:{j['id']}", "prompt contains Persian — write Omni prompts in English, keep text vector")
    except Exception as e:  # noqa: BLE001
        add("error", "omni", f"cannot resolve omni jobs: {e}")
    for ci, ch in enumerate(scene.chapters):
        hud = ch.get("_hud") or scene.hud            # v8: each chapter's own HUD
        safe_top = H * (hud["header"]["y"] + 0.05) if hud["header"].get("show", True) else 0
        safe_bot = H * (hud["dots"]["y"] - 0.025) if hud["dots"].get("show", True) else H
        from .text import use_roles as _ur
        _ur((ch.get("_world") or {}).get("fonts"))
        if ch.get("title"):
            m = missing_glyphs(ch["title"])
            if m:
                add("error", f"chapter {ci + 1} title", f"missing glyphs {sorted(m)}")
        for el in ch["elements"]:
            where = f"ch{ci + 1}:{el.get('id')}"
            if el.get("_bad"):
                add("error", where, el["_bad"])
                continue
            known = ELEMENT_KEYS.get(el["type"])
            if known is not None:
                pool = known | COMMON_KEYS
                for k in el:
                    if k.startswith("_") or k in pool:
                        continue
                    sug = _close_match(k, pool)
                    add("warn", where, f"unknown key '{k}' is ignored" + (f" — did you mean '{sug}'?" if sug else ""))
            for it in (el.get("rows") if isinstance(el.get("rows"), list) else []) + \
                    (el.get("items") if isinstance(el.get("items"), list) else []):
                for kk, vv in (it.items() if isinstance(it, dict) else [("item", it)]):
                    if isinstance(vv, str) and kk not in ("t",):
                        mm = missing_glyphs(parse_markup(vv)[0])
                        if mm:
                            add("error", where, f"missing glyphs in {kk} '{vv}': {sorted(mm)}")
            for k in _TEXT_KEYS:
                if isinstance(el.get(k), str):
                    m = missing_glyphs(parse_markup(el[k])[0])
                    if m:
                        add("error", where, f"missing glyphs in '{k}': {sorted(m)} (would render as ▯)")
                    fnt = resolve_font(el.get("font") if k == "text" else el.get("label_font", el.get("font")),
                                       el.get("weight"), "body")
                    fb = fallback_chars(parse_markup(el[k])[0], fnt) - set("→←↑↓✓✕")
                    if fb:
                        add("warn", where, f"font '{fnt}' lacks {sorted(fb)} in '{k}' → those chars fall back to "
                            f"another font (use a role with full coverage, e.g. title/body, or rephrase)")
            # v13.5 Strict Zero-Box Aesthetic Law Check
            if el.get("slab_show") is True and el["type"] in ("title_card", "logo_chip", "price_tag", "grade_scale", "sticker"):
                add("warn", where, "Zero-Box Law violation: 'slab_show: true' puts a flat 2D box behind the element — use 3D tactile objects, behind-speaker coordinate grids, or dof_blur instead!")
            if (scene.spec.get("meta") or {}).get("style") == "visual_dna" and el["type"] in ("box", "block", "pill"):
                add("warn", where, f"Zero-Box Law violation in visual_dna style: '{el['type']}' is a flat 2D rectangle — use 'tactile_3d', 'dna_connector', or 'hud_loading_bar' instead!")
            if el["type"] == "icon":
                from .icons import ICONS
                if el.get("name") not in ICONS:
                    add("error", where, f"unknown icon '{el.get('name')}' (see icons.find_icons)")
            if el["t"] > scene.duration:
                add("warn", where, f"starts at {el['t']:.2f}s after video end {scene.duration:.2f}s")
            if el.get("out") is not None and el["out"] < el["t"]:
                add("error", where, "out < t (exits before it enters)")
            if ch["end"] is not None and el["t"] > ch["end"] + 0.01:
                add("warn", where, f"enters at {el['t']:.2f}s, after its chapter ended ({ch['end']:.2f}s) — never seen")
            for key in ("anchor", "from", "to", "target"):
                v = el.get(key)
                if isinstance(v, str):
                    nm = v.split("+")[0]
                    if nm not in scene.by_id and "." in nm:          # v11: "<footage>.<tracker>"
                        fid, trk = nm.split(".", 1)
                        fe = scene.by_id.get(fid)
                        if fe is not None and trk in (fe.get("trackers") or {}):
                            continue
                    if nm not in scene.by_id:
                        add("error", where, f"{key} references unknown id '{nm}'")
            if "y" in el and isinstance(el["y"], (int, float)) and el.get("space") != "screen":
                if el["type"] in TEXT_TYPES | {"icon"} and (el["y"] < safe_top or el["y"] > safe_bot):
                    add("warn", where, f"y={el['y']} is inside the HUD zone (header/dots/subtitles)")
    # orphan callouts: raw [x,y] anchor, no out, while the element under the anchor exits
    for ci, ch in enumerate(scene.chapters):
        for el in ch["elements"]:
            if el.get("type") not in ("callout", "arrow", "circle") or el.get("out") is not None:
                continue
            anc = el.get("anchor") or el.get("target") or el.get("from")
            if not isinstance(anc, (list, tuple)):
                continue
            for other in ch["elements"]:
                if other is el or other.get("out") is None or other["type"] in ("callout", "arrow", "text"):
                    continue
                try:
                    from .elements import compute_state
                    st = compute_state(other, other["t"] + 1.5, scene)
                    surf = cairo.ImageSurface(cairo.FORMAT_A8, 2, 2)
                    bb = REGISTRY[other["type"]]["fn"](cairo.Context(surf), other, st)
                except Exception:
                    continue
                if bb and bb[0] - 30 <= anc[0] <= bb[2] + 30 and bb[1] - 30 <= anc[1] <= bb[3] + 30:
                    add("warn", f"ch{ci + 1}:{el['id']}",
                        f"points at '{other['id']}' which exits at {other['out']:.2f}s, but has no 'out' "
                        f"→ leader line left pointing at nothing (anchor to \"{other['id']}\" or add out)")
                    break
    # pacing: long stretches with no new visual event feel "dead"
    events = sorted(_beat_times(scene))
    for a, b in zip(events, events[1:] + [scene.duration]):
        if b - a > 3.5:
            add("warn", f"pacing {a:.1f}-{b:.1f}s", f"{b - a:.1f}s without any new visual event (add a beat / camera move)")
    for s in scene.subs:
        m = missing_glyphs(s["text"])
        if m:
            add("error", f"subtitle @{s['t']:.2f}", f"missing glyphs {sorted(m)}")
    prev = None
    for s in scene.subs:
        if prev and s["t"] < prev["end"] - 0.01:
            add("warn", f"subtitle @{s['t']:.2f}", "overlaps previous subtitle")
        prev = s
    return issues


_CHEAP_BBOX = {"footage", "depth_plate", "backdrop", "talking_head", "video"}


def _boxes_at(scene, t):
    out = []
    surf = cairo.ImageSurface(cairo.FORMAT_A8, 4, 4)
    ctx = cairo.Context(surf)
    for ch, ca in scene.visible_chapters(t):
        if ca < 0.5:
            continue
        ox, oy = ch["origin"]
        for el in ch["elements"]:
            if el.get("_bad"):
                continue
            s = compute_state(el, t, scene)
            # v5 strict: check anything clearly visible, INCLUDING enter/exit cross-fades
            # (v4 skipped them and missed the "۲۲۲ ۱دلار" collision).  Draw-in elements count once ≥ 35% drawn.
            if not s.ok() or s.alpha < 0.20:
                continue
            if s.enter in ("draw", "words", "wipe") and s.raw_p < 0.35:
                continue
            if el.get("parent") or el.get("behind"):
                continue        # v11: children live in a virtual screen / depth layers sit behind a speaker on purpose
            if el["type"] in _CHEAP_BBOX:                     # v11: heavy media → bbox from geometry, no decode
                x, y = float(el.get("x", scene.W / 2)), float(el.get("y", scene.H / 2))
                w, h = float(el.get("w", scene.W)), float(el.get("h", el.get("w", scene.H)))
                out.append((el, (x - w / 2, y - h / 2, x + w / 2, y + h / 2), ch))
                continue
            ctx.save()
            from . import text as _T
            _T.use_roles((ch.get("_world") or {}).get("fonts"))
            bb = REGISTRY[el["type"]]["fn"](ctx, el, s)
            ctx.restore()
            if bb:
                from .render import parallax, parallax_bbox
                pz = parallax(scene, el, t, ch)
                if pz is not None:
                    bb = parallax_bbox(bb, pz)
                if el.get("follow"):                       # v11: tracked elements live at the tracker position
                    from .render import _follow
                    fx = _follow(scene, el, t)
                    if fx:
                        bb = (bb[0] + fx[0], bb[1] + fx[1], bb[2] + fx[0], bb[3] + fx[1])
                out.append((el, bb, ch))
    return out


CONTAINERS = {"box", "divider", "dot_grid", "icon_grid", "timeline", "quote_rig", "laser_scan", "invoice", "strata",
              "scale_rig", "flow_loop"}


def _anchor_id(el):
    a = el.get("anchor") or el.get("target")
    if isinstance(a, dict):
        return a.get("id")
    return a.split("+")[0] if isinstance(a, str) else None


def _contains(a, b, pad=6):
    return a[0] - pad <= b[0] and a[1] - pad <= b[1] and a[2] + pad >= b[2] and a[3] + pad >= b[3]


def _inter(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)


# v9: full-bleed framing layers — designed to sit under / around content and to bleed past the frame
BACKDROP_TYPES = {"viewfinder_hud", "filmstrip", "talking_head", "liquid_blob", "starburst", "sine_wave", "ui_film_world", "ui_keyboard", "ui_notification_stack", "ui_slider", "ui_toggle", "ui_reply_pill", "physics_ball"}
BACKDROP_TYPES |= {"footage", "backdrop", "device_frame", "image_card", "depth_plate", "light_rays", "emitter",
                   "flash_pulse", "glow_ring", "orbit_trail", "saber", "social_outro", "planar_insert", "video"}
BLEED_TYPES_V11 = {"footage", "backdrop", "light_rays", "emitter", "depth_plate", "social_outro", "flash_pulse",
                   "nle_timeline", "orbit_trail"}
BLEED_TYPES = {"viewfinder_hud", "filmstrip", "talking_head", "ui_film_world", "ui_keyboard", "ui_notification_stack", "ui_slider", "ui_toggle", "ui_reply_pill", "physics_ball"}


def overlap_report(scene, step=0.25, min_ratio=0.12):
    """Sample every `step` seconds; report text collisions and out-of-frame elements."""
    seen = {}
    t = 0.0
    while t < scene.duration:
        boxes = _boxes_at(scene, t)
        cx, cy, z, _ = scene.camera.view(t)
        for i in range(len(boxes)):
            ei, bi, chi = boxes[i]
            # out of frame (convert chapter-local bbox → screen)
            ox, oy = chi["origin"]
            sx0 = (bi[0] + ox - cx) * z + scene.W / 2
            sx1 = (bi[2] + ox - cx) * z + scene.W / 2
            sy0 = (bi[1] + oy - cy) * z + scene.H / 2
            sy1 = (bi[3] + oy - cy) * z + scene.H / 2
            if ei["type"] not in BLEED_TYPES and ei["type"] not in BLEED_TYPES_V11 and not ei.get("parent") and (sx0 < -4 or sx1 > scene.W + 4 or sy0 < -4 or sy1 > scene.H + 4) and \
                    scene.chapter_index(t) == ei["_chapter"] and t - scene.chapters[ei["_chapter"]]["t"] > 1.0:
                k = ("frame", ei["id"])
                seen.setdefault(k, {"kind": "out_of_frame", "id": ei["id"], "t": round(t, 2),
                                    "screen_bbox": [round(v) for v in (sx0, sy0, sx1, sy1)]})
            for j in range(i + 1, len(boxes)):
                ej, bj, chj = boxes[j]
                if chi is not chj:
                    continue
                if ei["type"] not in TEXT_TYPES and ej["type"] not in TEXT_TYPES:
                    continue
                if _anchor_id(ej) == ei.get("id") or _anchor_id(ei) == ej.get("id"):
                    continue
                if _contains(bi, bj) and ei["type"] in CONTAINERS or _contains(bj, bi) and ej["type"] in CONTAINERS:
                    continue  # text placed INSIDE a box/divider on purpose
                if {ei["type"], ej["type"]} & ({"stamp"} | BACKDROP_TYPES):
                    continue  # stamps are meant to slam on top of things
                if ei.get("allow_overlap") or ej.get("allow_overlap"):
                    continue
                inter = _inter(bi, bj)
                if inter <= 0:
                    continue
                small = min((bi[2] - bi[0]) * (bi[3] - bi[1]), (bj[2] - bj[0]) * (bj[3] - bj[1])) or 1
                r = inter / small
                if r >= min_ratio:
                    k = ("ov", ei["id"], ej["id"])
                    if k not in seen:
                        seen[k] = {"kind": "overlap", "a": ei["id"], "b": ej["id"], "t": round(t, 2),
                                   "ratio": round(r, 2)}
        t += step
    return list(seen.values())


def hero_times(scene):
    ts = []
    for c in scene.chapters:
        end = c["end"] if c["end"] is not None else scene.duration
        ts.append(round(c["t"] + (end - c["t"]) * 0.5, 2))
        ts.append(round(max(c["t"], end - 0.35), 2))
    return ts


def contact_sheet(scene, path, times=None, cols=5, thumb_w=270):
    from PIL import Image, ImageDraw
    times = times or hero_times(scene)
    scale = thumb_w / scene.W
    thumbs = []
    for t in times:
        rgb = frame_to_rgb(render_frame(scene, t, scale=scale))
        im = Image.fromarray(rgb)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 70, 18], fill=(180, 0, 0))
        d.text((4, 3), f"{t:6.2f}s", fill=(255, 255, 255))
        thumbs.append(im)
    rows = math.ceil(len(thumbs) / cols)
    tw, th = thumbs[0].size
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * 6, rows * th + (rows + 1) * 6), (40, 40, 40))
    for i, im in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet.paste(im, (6 + c * (tw + 6), 6 + r * (th + 6)))
    sheet.save(path, quality=90)
    return path


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration,size:stream=codec_type,codec_name,width,height,r_frame_rate",
                          "-of", "json", path], capture_output=True, text=True).stdout
    return json.loads(out or "{}")


def text_audit(scene, step=None, min_ratio=3.0):
    """Render the timeline at a coarse step with text.AUDIT on → real problems only:
       clip      a glyph (dot, kasra, ی/ن tail …) is cut by a mask / slab / clip rect
       contrast  a fully-visible word has < min_ratio contrast against the pixels under it
    Returns [{"kind","tag","text","t",…}] (worst instance per (kind, tag, text))."""
    from . import text as T
    from .render import render_frame
    step = step or max(0.15, min(0.3, scene.duration / 80))
    worst = {}
    prev_clips = {}                      # (tag,text) -> (px, y, run)   a clip that does not change = at rest = bug
    t = 0.05
    while t < scene.duration:
        T.AUDIT = []
        try:
            render_frame(scene, t, scale=0.5, bloom=0.0)
        finally:
            rec, T.AUDIT = T.AUDIT, None
        cur_clips = {}
        for r in rec:
            if r["kind"] == "clip":
                if r["px"] < 1.5:
                    continue
                key = (r["tag"], r["text"])
                pv = prev_clips.get(key)
                run = pv[2] + 1 if pv and abs(pv[0] - r["px"]) < 1.0 and abs(pv[1] - r["y"]) < 1.0 else 1
                cur_clips[key] = (r["px"], r["y"], run)
                if run < 3:              # still moving (a reveal in progress) → intentional
                    continue
            if r["kind"] == "contrast":
                need = 1.6 if r.get("quiet") else min_ratio      # quiet = deliberately recessive (upcoming, muted)
                if r["ratio"] >= need:
                    continue
            k = (r["kind"], r["tag"], r["text"])
            bad = r.get("px", 0) if r["kind"] == "clip" else -r["ratio"]
            if k not in worst or bad > worst[k][0]:
                worst[k] = (bad, dict(r, t=round(t, 2)))
        prev_clips = cur_clips
        t += step
    return [v[1] for v in worst.values()]


def report(scene, overlaps=True, audit=True):
    """Human-readable QA summary; returns (text, n_errors)."""
    iss = lint(scene)
    ov = overlap_report(scene) if overlaps else []
    lines = []
    errs = sum(1 for i in iss if i["level"] == "error")
    for i in iss:
        lines.append(f"[{i['level'].upper():5}] {i['where']}: {i['msg']}")
    for o in ov:
        if o["kind"] == "overlap":
            lines.append(f"[WARN ] overlap {o['a']} × {o['b']} at {o['t']}s (ratio {o['ratio']})")
        else:
            lines.append(f"[WARN ] {o['id']} leaves the frame at {o['t']}s bbox={o['screen_bbox']}")
    if audit:
        for r in text_audit(scene):
            if r["kind"] == "clip":
                lines.append(f"[WARN ] glyph clipped: «{r['text']}» ({r['tag']}) at {r['t']}s — ink {r['px']*2:.0f}px outside "
                             f"its mask/slab (enlarge the box or use text.mask_rect)")
            else:
                lines.append(f"[WARN ] low contrast: «{r['text']}» ({r['tag']}) at {r['t']}s — {r['ratio']}:1 "
                             f"(need ≥ {'1.6' if r.get('quiet') else '3'}:1; use @ink/@accent or contrast.ensure)")
    if not lines:
        lines.append("QA clean ✓ (no lint errors, no overlaps, nothing out of frame, no clipped glyphs, contrast OK)")
    return "\n".join(lines), errs


# ---------------------------------------------------------------------------
# POST-RENDER REVIEW (on the real mp4 — catches what spec-level QA cannot)
# ---------------------------------------------------------------------------

def review_video(path, expect_duration=None, expect_fps=None, sample_step=0.5, sheet_path=None):
    """Decode the rendered mp4 and check: duration/fps, black stretches, frozen stretches
    (> 2.5s with no pixel change = 'dead' video), audio silence, clipping and integrated loudness.
    Returns (report_text, n_problems). Optionally writes a contact sheet of sampled frames."""
    import re
    import cv2
    import numpy as np
    probs, lines = 0, []
    meta = probe(path)
    fmt = meta.get("format", {})
    dur = float(fmt.get("duration", 0) or 0)
    vs = next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), {})
    has_audio = any(s.get("codec_type") == "audio" for s in meta.get("streams", []))
    num, den = (vs.get("r_frame_rate", "0/1").split("/") + ["1"])[:2]
    fps = float(num) / float(den or 1)
    lines.append(f"[info ] {vs.get('width')}x{vs.get('height')} @ {fps:g}fps, {dur:.2f}s, audio={'yes' if has_audio else 'NO'}")
    if expect_duration and abs(dur - expect_duration) > 0.25:
        probs += 1; lines.append(f"[ERROR] duration {dur:.2f}s ≠ expected {expect_duration:.2f}s")
    if expect_fps and abs(fps - expect_fps) > 0.01:
        probs += 1; lines.append(f"[ERROR] fps {fps} ≠ expected {expect_fps}")
    cap = cv2.VideoCapture(path)
    t, prev, frozen_since, black_since = 0.0, None, None, None
    thumbs = []
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(fr, (135, 240)), cv2.COLOR_BGR2GRAY).astype(np.int16)
        if sheet_path and abs((t / sample_step) % max(1, int(dur / sample_step / 20))) < 1e-6:
            thumbs.append((t, cv2.resize(fr, (180, 320))))
        if g.max() < 18:
            black_since = t if black_since is None else black_since
        else:
            if black_since is not None and t - black_since >= 0.5 and black_since > 0.2:
                probs += 1; lines.append(f"[WARN ] black frames {black_since:.1f}-{t:.1f}s")
            black_since = None
        if prev is not None:
            diff = np.abs(g - prev).mean()
            if diff < 0.15:
                frozen_since = t - sample_step if frozen_since is None else frozen_since
            else:
                if frozen_since is not None and t - frozen_since >= 2.5:
                    probs += 1; lines.append(f"[WARN ] frozen/static {frozen_since:.1f}-{t:.1f}s (no motion → feels dead)")
                frozen_since = None
        prev = g
        t += sample_step
    if frozen_since is not None and dur - frozen_since >= 2.5:
        lines.append(f"[info ] static tail {frozen_since:.1f}-{dur:.1f}s (ok if it's an end card)")
    cap.release()
    if has_audio:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af",
                              "ebur128=peak=true,silencedetect=n=-45dB:d=1.5", "-f", "null", "-"],
                             capture_output=True, text=True).stderr
        m = re.findall(r"I:\s+(-?[\d.]+) LUFS", out)
        pk = re.findall(r"Peak:\s+(-?[\d.]+) dBFS", out)
        if m:
            lufs = float(m[-1])
            ok = -16.5 <= lufs <= -11.5
            if not ok:
                probs += 1
            lines.append(f"[{'info ' if ok else 'WARN '}] loudness {lufs:.1f} LUFS (target -14)")
        if pk and float(pk[-1]) > -0.5:
            probs += 1; lines.append(f"[WARN ] true peak {pk[-1]} dBFS → clipping risk")
        for a_, b_ in re.findall(r"silence_start: ([\d.]+).*?silence_end: ([\d.]+)", out, re.S):
            lines.append(f"[info ] silence {float(a_):.1f}-{float(b_):.1f}s")
    else:
        probs += 1; lines.append("[ERROR] no audio stream")
    if sheet_path and thumbs:
        import math as _m
        cols = 8
        rows = _m.ceil(len(thumbs) / cols)
        sheet = np.full((rows * 330 + 10, cols * 190 + 10, 3), 40, np.uint8)
        for i, (tt, im) in enumerate(thumbs):
            r, c = divmod(i, cols)
            y0, x0 = 10 + r * 330, 10 + c * 190
            sheet[y0:y0 + 320, x0:x0 + 180] = im
            cv2.putText(sheet, f"{tt:.1f}s", (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.imwrite(sheet_path, sheet)
        lines.append(f"[info ] frame sheet → {sheet_path}")
    lines.append("REVIEW PASSED ✓" if probs == 0 else f"REVIEW: {probs} problem(s) — fix before delivering")
    return "\n".join(lines), probs
