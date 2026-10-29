"""
3-layer dual-track lip-sync renderer (still plates) + in-place lip-sync for continuous character video.

render(...)  Layer 0 background → Layer 1 character (BODY track: pose per frame · MOUTH track: viseme per frame ·
             auto blinks) → Layer 2 foreground props (auto occlusion line: hands above the desk top are in front,
             legs below it are behind).  Everything is rendered NATIVELY at the output size (plates are cover-fit and
             remastered ONCE, not every frame) and piped straight into libx264 + AAC (no mp4v round trip).

Mouth track drivers (first available wins):
  1. mouth_timeline   explicit per-frame list (viseme names, or grid-cell indices — v29.2 compatible)
  2. words            words.json / list → mograph.motion.lipsync.track (Persian + English G2P, Preston Blair,
                      closures ≥ 2 frames, anticipation lead) — shapes BLEND between keys (vector mouths)
  3. audio only       audio_viseme_track(): loudness → jaw opening, spectral centroid → round (O/U) vs spread (E)

Body track: body_timeline per-frame list or [[t_sec, pose_idx], …] keys; default = auto: rest pose in silence,
transition poses once (2 frames each), gesture pose while talking, back to rest at the end.

Per-pose cache: bg + clean pose + props are composited once per pose; each frame only touches the mouth / eye
boxes (≈ 40× less pixel work than full recompositing at 1080×1920).

API: render(...) → report · render_video_lipsync(video, out, ...) → report · audio_viseme_track(audio, fps) ·
     auto_body_timeline(n_poses, speech, fps) · viseme_sheet(pose, out) · contact_sheet(mp4, out)
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Sequence

import cv2
import numpy as np

from .. import lipsync as LS
from .anatomy import anatomy_summary, detect_face_anatomy
from .layers import character_alpha, occlusion_line, props_alpha, split_pose_sheet
from .media import FFmpegWriter, cover_fit, has_audio, load_image, read_audio_mono, video_info
from .mouths import VISEMES, GridMouths, VectorMouths, erase_base_mouth, place_mouth
from .remaster import enhance_lineart_hd

INK = (12, 10, 10)


# ------------------------------------------------------------------------------------------------ drivers
def audio_viseme_track(audio: str, fps: float = 24.0, sr: int = 16000, min_hold: int = 2) -> list[dict[str, Any]]:
    """audio-only mouth keys [{t, v, open}] — loudness → opening, spectral centroid → round vs spread."""
    x = read_audio_mono(audio, sr)
    hop = max(1, int(round(sr / fps)))
    n = max(1, len(x) // hop)
    fr = x[:n * hop].reshape(n, hop) if len(x) >= hop else np.pad(x, (0, hop - len(x)))[None]
    db = 20 * np.log10(np.sqrt(np.mean(fr ** 2, axis=1) + 1e-10) + 1e-9)
    hi, lo = np.percentile(db, 97), np.percentile(db, 30)
    op = np.clip((db - lo) / max(6.0, hi - lo), 0, 1)
    op = np.convolve(op, [0.25, 0.5, 0.25], "same")
    spec = np.abs(np.fft.rfft(fr * np.hanning(fr.shape[1])[None], axis=1))
    freqs = np.fft.rfftfreq(fr.shape[1], 1.0 / sr)
    band = (freqs > 150) & (freqs < 5000)
    cen = (spec[:, band] * freqs[band]).sum(1) / (spec[:, band].sum(1) + 1e-9)
    voiced = op > 0.12
    if voiced.sum() > 4:
        c_lo, c_hi = np.percentile(cen[voiced], 20), np.percentile(cen[voiced], 80)
    else:
        c_lo, c_hi = 300.0, 2500.0
    cn = np.clip((cen - c_lo) / max(1.0, c_hi - c_lo), 0, 1)
    vs = []
    for o, c in zip(op, cn):
        if o < 0.12:
            v = "rest"
        elif o < 0.35:
            v = "U" if c < 0.25 else "etc"
        elif o < 0.62:
            v = "O" if c < 0.3 else "E"
        else:
            v = "O" if c < 0.22 else "AI"
        vs.append(v)
    # short silences between voiced frames read as lip closures (MBP), not as a slack jaw
    i = 0
    while i < n:
        if vs[i] == "rest":
            j = i
            while j < n and vs[j] == "rest":
                j += 1
            if 0 < i and j < n and (j - i) <= max(2, int(0.12 * fps)):
                for k in range(i, j):
                    vs[k] = "MBP"
            i = j
        else:
            i += 1
    # minimum hold: runs shorter than min_hold frames merge into the previous shape (no 1-frame flicker)
    i = 0
    while i < n:
        j = i
        while j < n and vs[j] == vs[i]:
            j += 1
        if (j - i) < min_hold and i > 0:
            for k in range(i, j):
                vs[k] = vs[i - 1]
        i = j
    keys = []
    for f, v in enumerate(vs):
        if keys and keys[-1]["v"] == v:
            continue
        base = LS.MOUTHS[v]["h"]
        o = float(op[f])
        keys.append({"t": round(f / fps, 4), "v": v,
                     "open": round(base * (0.6 + 0.6 * o) if v not in ("rest", "MBP") else 0.0, 3)})
    return keys or [{"t": 0.0, "v": "rest", "open": 0.0}]


def speech_mask_from_keys(keys: list[dict[str, Any]], n: int, fps: float) -> np.ndarray:
    m = np.zeros(n, bool)
    for k, nxt in zip(keys, keys[1:] + [{"t": n / fps}]):
        if k["v"] not in ("rest",):
            m[int(k["t"] * fps):int(nxt["t"] * fps)] = True
    return m


def auto_body_timeline(n_poses: int, speech: np.ndarray, fps: float, lead_s: float = 0.2,
                       tail_s: float = 0.35, trans_frames: int = 2) -> list[int]:
    """rest (0) in silence → transition poses (1..n-2, each `trans_frames`) → gesture (n-1) while talking → back."""
    n = len(speech)
    tl = [0] * n
    if n_poses <= 1 or not speech.any():
        return tl
    gest = n_poses - 1
    trans = list(range(1, n_poses - 1))
    on = int(np.argmax(speech)) + int(round(lead_s * fps))
    off = n - 1 - int(np.argmax(speech[::-1])) + int(round(tail_s * fps))
    f = on
    for p in trans:
        for _ in range(trans_frames):
            if f < n:
                tl[f] = p
            f += 1
    for k in range(max(0, f), min(n, off)):
        tl[k] = gest
    back = len(trans) * trans_frames
    if off + back < n:
        f = off
        for p in reversed(trans):
            for _ in range(trans_frames):
                if f < n:
                    tl[f] = p
                f += 1
    else:
        for k in range(max(0, f), n):
            tl[k] = gest
    return tl


def _expand_timeline(tl, n: int, fps: float) -> list[int]:
    if tl is None:
        return None
    tl = list(tl)
    if tl and isinstance(tl[0], (list, tuple)):          # [[t, idx], …] keys
        keys = sorted((float(t), int(i)) for t, i in tl)
        out, j = [], 0
        for f in range(n):
            t = f / fps
            while j + 1 < len(keys) and keys[j + 1][0] <= t:
                j += 1
            out.append(keys[j][1] if keys[0][0] <= t else keys[0][1])
        return out
    out = [int(v) for v in tl[:n]]
    return out + [out[-1] if out else 0] * (n - len(out))


def blink_schedule(n: int, fps: float, seed: int = 7, every: tuple[float, float] = (2.4, 4.8),
                   dur_s: float = 0.16, start_s: float = 0.9) -> np.ndarray:
    """per-frame lid closure 0..1 (deterministic for a seed)."""
    rng = np.random.default_rng(seed)
    c = np.zeros(n, np.float32)
    t = start_s + rng.uniform(0, 1.2)
    d = max(3, int(round(dur_s * fps)))
    prof = np.sin(np.linspace(0, np.pi, d + 2)[1:-1]) ** 0.6
    while t * fps < n:
        f0 = int(t * fps)
        for k, p in enumerate(prof):
            if f0 + k < n:
                c[f0 + k] = max(c[f0 + k], float(p))
        t += rng.uniform(*every)
    return c


def apply_blink(frame: np.ndarray, info: dict[str, Any], c: float) -> None:
    """close the sclera eyes by `c` (0..1) in place: face colour from the top + an ink lid edge."""
    if c <= 0.02:
        return
    lw = max(1, int(round(info["stroke_px"] * 0.9)))
    for ef in info["eye_fills"]:
        if ef["kind"] != "sclera":
            continue
        x, y, w, h = ef["box"]
        m = ef["mask"] > 0
        roi = frame[y:y + h, x:x + w]
        if roi.shape[:2] != m.shape:
            continue
        lid = int(round(min(1.0, c) * (h - 1) * (0.62 if c >= 0.98 else 1.0)))
        cover = m.copy()
        cover[lid + 1:, :] = False
        roi[cover] = info["face_color"]
        rows = np.arange(h)[:, None]
        edge = m & (np.abs(rows - lid) <= lw // 2 + (1 if c >= 0.98 else 0))
        roi[edge] = INK


# ------------------------------------------------------------------------------------------------ poses
def _load_poses(poses, wh, sheet, remaster: bool):
    if isinstance(poses, str) and "," in poses:
        poses = [p.strip() for p in poses.split(",") if p.strip()]
    if isinstance(poses, (list, tuple)):
        raw = [load_image(p) for p in poses]
        from_sheet = False
    else:
        bgr, a = load_image(poses)
        if sheet:
            raw = [(p, None) for p in split_pose_sheet(bgr, sheet)]
            from_sheet = True
        else:
            raw, from_sheet = [(bgr, a)], False
    imgs, alphas = [], []
    for b, a in raw:
        b = cover_fit(b, wh)
        if remaster:
            b = enhance_lineart_hd(b)
        imgs.append(b)
        alphas.append(None if a is None else cover_fit(a, wh, interp=cv2.INTER_LINEAR))
    return imgs, alphas, from_sheet


def _align_to(ref: dict, img, alpha, info, wh):
    s = ref["inter_ocular"] / max(1.0, info["inter_ocular"])
    if "profile" in ref["view_mode"] or "profile" in info["view_mode"]:
        s = ref["head_scale"] / max(1.0, info["head_scale"])
    tx = ref["eyes_mid"][0] - info["eyes_mid"][0] * s
    ty = ref["eyes_mid"][1] - info["eyes_mid"][1] * s
    M = np.float32([[s, 0, tx], [0, s, ty]])
    img = cv2.warpAffine(img, M, wh, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if alpha is not None:
        alpha = cv2.warpAffine(alpha, M, wh, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return img, alpha


# ------------------------------------------------------------------------------------------------ render
def render(
    bg: str,
    poses,
    audio: str,
    out: str,
    fg: str | None = None,
    words=None,
    visemes: str | None = None,
    grid: tuple[int, int] = (4, 4),
    viseme_map: dict[str, int] | None = None,
    mouth_timeline: Sequence | None = None,
    body_timeline: Sequence | None = None,
    pose_sheet: int | str | None = None,
    align: str = "auto",
    ref: str | None = None,
    fps: float = 24.0,
    wh: tuple[int, int] = (1080, 1920),
    remaster: bool = True,
    occlusion: str | int = "auto",
    blink: bool = True,
    seed: int = 7,
    mouth_style: dict | None = None,
    crf: int = 18,
    report_path: str | None = "auto",
    contact: str | None = None,
) -> dict[str, Any]:
    """render a 3-layer lip-synced MP4. Returns a JSON-safe report (also written to `<out>.report.json`)."""
    t_start = time.time()
    wh = (int(wh[0]) // 2 * 2, int(wh[1]) // 2 * 2)
    W, H = wh
    # ---- plates
    bg_bgr, _ = load_image(bg)
    bg_bgr = cover_fit(bg_bgr, wh)
    if remaster:
        bg_bgr = enhance_lineart_hd(bg_bgr)
    fg_bgr = fg_a = None
    occ_y = None
    if fg:
        fg_bgr, fa = load_image(fg)
        fg_bgr = cover_fit(fg_bgr, wh)
        if remaster:
            fg_bgr = enhance_lineart_hd(fg_bgr)
        fg_a = cover_fit(fa, wh, interp=cv2.INTER_LINEAR) if fa is not None else props_alpha(fg_bgr)
        if occlusion == "auto":
            occ_y = occlusion_line(fg_bgr, fg_a)
        elif isinstance(occlusion, (int, float)) and not isinstance(occlusion, bool):
            occ_y = int(occlusion)
    # ---- poses (registered, erased, alpha)
    imgs, alphas, from_sheet = _load_poses(poses, wh, pose_sheet, remaster)
    infos = [detect_face_anatomy(im) for im in imgs]
    ref_info = infos[0]
    if ref:
        rb, _ = load_image(ref)
        ref_info = detect_face_anatomy(cover_fit(rb, wh))
    do_align = (align == "eyes") or (align == "auto" and (from_sheet or ref is not None) and len(imgs) > 0)
    if do_align:
        for k in range(len(imgs)):
            imgs[k], alphas[k] = _align_to(ref_info, imgs[k], alphas[k], infos[k], wh)
            infos[k] = detect_face_anatomy(imgs[k], hint=ref_info["eyes_mid"])
    clean = [erase_base_mouth(im, inf) for im, inf in zip(imgs, infos)]
    c_alpha = [a if a is not None else character_alpha(im) for a, im in zip(alphas, imgs)]

    # ---- mouth source
    if visemes:
        gb, _ = load_image(visemes)
        if remaster:
            gb = enhance_lineart_hd(gb)
        msrc = GridMouths(gb, rows=grid[0], cols=grid[1], mapping=viseme_map)
    else:
        msrc = VectorMouths(mouth_style)

    # ---- timing
    x = read_audio_mono(audio, 16000)
    n = max(1, int(np.ceil(len(x) / 16000.0 * fps)))
    driver = "timeline" if mouth_timeline is not None else ("words" if words else "audio")
    keys = None
    if driver == "words":
        keys = LS.track(words, audio=audio, fps=fps)
    elif driver == "audio":
        keys = audio_viseme_track(audio, fps)
    if mouth_timeline is not None:
        mt = list(mouth_timeline)[:n]
        mt += [mt[-1] if mt else "rest"] * (n - len(mt))
        speech = np.array([(v not in ("rest", 0)) for v in mt])
    else:
        speech = speech_mask_from_keys(keys, n, fps)
    bt = _expand_timeline(body_timeline, n, fps) or auto_body_timeline(len(imgs), speech, fps)
    bt = [int(np.clip(b, 0, len(imgs) - 1)) for b in bt]
    blinks = blink_schedule(n, fps, seed) if blink else np.zeros(n, np.float32)

    # ---- per-pose composite cache
    bgf = bg_bgr.astype(np.float32)
    cache = {}
    rows = np.arange(H, dtype=np.float32)[:, None]

    def pose_base(p):
        if p in cache:
            return cache[p]
        a = c_alpha[p][:, :, None]
        comp = clean[p].astype(np.float32) * a + bgf * (1 - a)
        front = None
        if fg_bgr is not None:
            if occlusion == "front":
                char_front = c_alpha[p]
            elif occ_y is not None:
                char_front = c_alpha[p] * (rows < occ_y)
            else:
                char_front = np.zeros_like(c_alpha[p])
            front = np.clip(fg_a * (1.0 - char_front), 0, 1)
            comp = fg_bgr.astype(np.float32) * front[:, :, None] + comp * (1 - front[:, :, None])
        hx, hy, hw, hh = infos[p]["head_box"]
        pad = int(0.25 * hw)
        box = (max(0, hx - pad), max(0, hy - pad), min(W, hx + hw + pad), min(H, hy + hh + pad))
        fr = None
        if front is not None and front[box[1]:box[3], box[0]:box[2]].max() > 0.01:
            fr = front[box[1]:box[3], box[0]:box[2], None].copy()
        cache[p] = (np.clip(comp + 0.5, 0, 255).astype(np.uint8), box, fr)
        return cache[p]

    rp = report_path
    if rp == "auto":
        rp = os.path.splitext(out)[0] + ".report.json"
    qa_fail, worst_clip, used = 0, 0.0, {}
    with FFmpegWriter(out, wh, fps, audio=audio, crf=crf) as wr:
        for f in range(n):
            p = bt[f]
            base, box, fr = pose_base(p)
            frame = base.copy()
            info = infos[p]
            if blinks[f] > 0:
                apply_blink(frame, info, float(blinks[f]))
            if mouth_timeline is not None:
                mv = mt[f]
                if isinstance(mv, (int, np.integer)) and isinstance(msrc, GridMouths):
                    sp = msrc.sprites[int(mv)] or msrc.get({"v": "rest"}, info)
                    vname = f"cell{int(mv)}"
                else:
                    vname = VISEMES[int(mv) % len(VISEMES)] if isinstance(mv, (int, np.integer)) else str(mv)
                    g = dict(LS.MOUTHS.get(vname, LS.MOUTHS["rest"]), v=vname)
                    sp = msrc.get(g, info)
            else:
                g = LS.shape_at(keys, f / fps)
                vname = g["v"]
                sp = msrc.get(g, info)
            used[vname] = used.get(vname, 0) + 1
            frame, qa = place_mouth(frame, info, sp, in_place=True)
            if not qa["ok"]:
                qa_fail += 1
            worst_clip = max(worst_clip, qa.get("clipped_frac", 0.0))
            if fr is not None:
                x0, y0, x1, y1 = box
                roi = frame[y0:y1, x0:x1].astype(np.float32)
                frame[y0:y1, x0:x1] = (fg_bgr[y0:y1, x0:x1] * fr + roi * (1 - fr) + 0.5).astype(np.uint8)
            wr.write(frame)
    report = {
        "out_mp4": out, "frames": n, "fps": fps, "duration_s": round(n / fps, 3), "resolution": f"{W}x{H}",
        "remaster": remaster, "driver": driver, "mouth_source": msrc.kind,
        "viseme_map": getattr(msrc, "mapping", None), "grid_failures": getattr(msrc, "failures", None),
        "visemes_used": used, "poses": [anatomy_summary(i) for i in infos], "aligned": bool(do_align),
        "body_timeline_counts": {str(k): bt.count(k) for k in sorted(set(bt))},
        "occlusion_y": occ_y, "blinks": int((np.diff(np.r_[0, (blinks > 0).astype(int)]) == 1).sum()),
        "qa_clean": qa_fail == 0, "qa_failures": qa_fail, "worst_clipped_frac": round(worst_clip, 3),
        "render_s": round(time.time() - t_start, 2),
    }
    if contact:
        report["contact_sheet"] = contact_sheet(out, contact)
    if rp:
        with open(rp, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1)
        report["report_path"] = rp
    return report


# ------------------------------------------------------------------------------------------------ video mode
def render_video_lipsync(
    video: str,
    out: str,
    audio: str | None = None,
    words=None,
    mouth_style: dict | None = None,
    smooth: float = 0.6,
    remaster: bool = False,
    wh: tuple[int, int] | None = None,
    crf: int = 18,
) -> dict[str, Any]:
    """lip-sync a character that MOVES (continuous body video): per-frame detection with temporal hint + EMA."""
    vi = video_info(video)
    fps = vi["fps"]
    audio = audio or (video if has_audio(video) else None)
    if audio is None:
        raise ValueError("no audio: pass `audio` or use a video with an audio track")
    keys = LS.track(words, audio=audio, fps=fps) if words else audio_viseme_track(audio, fps)
    out_wh = (int(wh[0]) // 2 * 2, int(wh[1]) // 2 * 2) if wh else (vi["w"] // 2 * 2, vi["h"] // 2 * 2)
    msrc = VectorMouths(mouth_style)
    cap = cv2.VideoCapture(video)
    ema, prev, n, missed, qa_fail = None, None, 0, 0, 0
    with FFmpegWriter(out, out_wh, fps, audio=audio, crf=crf) as wr:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != out_wh:
                frame = cover_fit(frame, out_wh)
            if remaster:
                frame = enhance_lineart_hd(frame)
            try:
                info = detect_face_anatomy(frame, hint=None if ema is None else ema["c"])
            except ValueError:
                info = None
            if info is None or (ema is not None and np.hypot(info["mouth_center"][0] - ema["c"][0],
                                                             info["mouth_center"][1] - ema["c"][1])
                                > 1.5 * ema["io"]):
                info = prev if info is None else info
                if info is None:
                    missed += 1
                    wr.write(frame)
                    n += 1
                    continue
                ema = None
            cur = {"c": np.array(info["mouth_center"]), "io": info["inter_ocular"], "a": info["eye_angle_deg"]}
            if ema is None:
                ema = cur
            else:
                ema = {"c": smooth * ema["c"] + (1 - smooth) * cur["c"],
                       "io": smooth * ema["io"] + (1 - smooth) * cur["io"],
                       "a": smooth * ema["a"] + (1 - smooth) * cur["a"]}
            sm = dict(info, mouth_center=tuple(ema["c"]), inter_ocular=float(ema["io"]), eye_angle_deg=float(ema["a"]))
            frame = erase_base_mouth(frame, info)
            g = LS.shape_at(keys, n / fps)
            frame, qa = place_mouth(frame, sm, msrc.get(g, sm), in_place=True)
            qa_fail += 0 if qa["ok"] else 1
            prev = info
            wr.write(frame)
            n += 1
    cap.release()
    return {"out_mp4": out, "frames": n, "fps": fps, "resolution": f"{out_wh[0]}x{out_wh[1]}",
            "missed_detections": missed, "qa_failures": qa_fail, "driver": "words" if words else "audio"}


# ------------------------------------------------------------------------------------------------ QC helpers
def viseme_sheet(pose: str, out: str, visemes: str | None = None, grid: tuple[int, int] = (4, 4),
                 remaster: bool = True) -> dict[str, Any]:
    """all 10 Preston Blair mouths on one pose (head crops in a 5×2 grid) — LOOK before rendering."""
    im, _ = load_image(pose)
    if remaster:
        im = enhance_lineart_hd(im)
    info = detect_face_anatomy(im)
    clean = erase_base_mouth(im, info)
    if visemes:
        gb, _ = load_image(visemes)
        msrc = GridMouths(gb, rows=grid[0], cols=grid[1])
    else:
        msrc = VectorMouths()
    hx, hy, hw, hh = info["head_box"]
    pad = int(0.12 * hw)
    tiles, fails = [], 0
    for v in VISEMES:
        fr, qa = place_mouth(clean, info, msrc.get(dict(LS.MOUTHS[v], v=v), info))
        fails += 0 if qa["ok"] else 1
        t = fr[max(0, hy - pad):hy + hh + pad, max(0, hx - pad):hx + hw + pad].copy()
        t = cv2.resize(t, (240, int(240 * t.shape[0] / max(1, t.shape[1]))))
        cv2.putText(t, v + ("" if qa["ok"] else " !QA"), (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2,
                    cv2.LINE_AA)
        tiles.append(t)
    h = min(t.shape[0] for t in tiles)
    tiles = [t[:h] for t in tiles]
    cv2.imwrite(out, np.vstack([np.hstack(tiles[:5]), np.hstack(tiles[5:])]))
    return {"out": out, "qa_failures": fails, "mouth_source": msrc.kind,
            "viseme_map": getattr(msrc, "mapping", None), "anatomy": anatomy_summary(info)}


def contact_sheet(mp4: str, out: str, n: int = 8, cols: int = 4, tile_w: int = 270) -> str:
    """n evenly spaced frames of a video tiled into one JPG (frame index stamped)."""
    cap = cv2.VideoCapture(mp4)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = np.linspace(0, total - 1, n).astype(int)
    tiles = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        t = cv2.resize(fr, (tile_w, int(tile_w * fr.shape[0] / fr.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.putText(t, f"#{i}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2, cv2.LINE_AA)
        tiles.append(t)
    cap.release()
    if not tiles:
        raise ValueError(f"no frames read from {mp4}")
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows_ = [np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(len(tiles) // cols)]
    cv2.imwrite(out, np.vstack(rows_))
    return out
