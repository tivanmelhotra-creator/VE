"""
mograph.pro.export — DELIVERY like a pro editor: captions, EDL, timeline report, chapters, thumbnails, reformat.

    python3 -m mograph.pro captions edit.json [--fmt srt|vtt|ass|all] [-o out/]   the on-screen subtitle phrases
        (exactly what the v14 subtitle engine shows, in ON-SCREEN time — cuts and retime included)
    python3 -m mograph.pro edl edit.json [-o edit.edl]        CMX3600 EDL of the kept source ranges + layer markers
        (Premiere / Resolve / FCP import → the same cut on the original camera file)
    python3 -m mograph.pro timeline edit.json                 the "timeline panel" as text
    python3 -m mograph.pro chapters edit.json                 YouTube chapters (≥ 10 s apart)
    python3 -m mograph.pro thumbnail edit.json [--t auto]     best frame (sharp + eyes open + hero layer on screen)
    python3 -m mograph.pro reformat in.mp4 --to 1:1|4:5|16:9|9:16 [--spec edit.json] [--mode crop|blur]
    python3 -m mograph.pro platform edit.json [--to instagram_reels|…]   length / aspect / loudness / safe zones
"""
from __future__ import annotations

import math
import os
import subprocess


# ============================================================================================== time formats
def ts_srt(t):
    ms_total = int(round(max(0.0, t) * 1000))
    h, r = divmod(ms_total, 3600000)
    m, r = divmod(r, 60000)
    s, ms = divmod(r, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ts_vtt(t):
    return ts_srt(t).replace(",", ".")


def ts_ass(t):
    cs = int(round(max(0.0, t) * 100))
    h, r = divmod(cs, 360000)
    m, r = divmod(r, 6000)
    s, c = divmod(r, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def tc(t, fps=30):
    """SMPTE timecode HH:MM:SS:FF (non-drop)."""
    r = int(round(fps))
    f = int(round(t * fps))
    s = f // r
    return f"{s // 3600:02d}:{(s // 60) % 60:02d}:{s % 60:02d}:{f % r:02d}"


# ============================================================================================== captions
def phrases(R):
    """[(t0, t1, text)] — the subtitle phrases as shown; falls back to groups of 4 words."""
    out = []
    if R.subs:
        for p in R.subs["_phr"]:
            txt = " ".join(w["w"].strip("*") for w in p["words"])
            out.append((p["t0"] + 0.06, p["t1"], txt))
        return out
    ws = R.tl.words
    for i in range(0, len(ws), 4):
        g = ws[i:i + 4]
        out.append((g[0]["t"], g[-1]["end"] + 0.1, " ".join(w["w"] for w in g)))
    return out


def _rtl(s):
    """RLE … PDF marks so Persian lines containing Latin / digits keep their order in every player."""
    return "\u202b" + s + "\u202c" if any("\u0600" <= ch <= "\u06ff" for ch in s) else s


def write_srt(ph, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, (a, b, s) in enumerate(ph, 1):
            f.write(f"{i}\n{ts_srt(a)} --> {ts_srt(b)}\n{_rtl(s)}\n\n")
    return path


def write_vtt(ph, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, (a, b, s) in enumerate(ph, 1):
            f.write(f"{i}\n{ts_vtt(a)} --> {ts_vtt(b)} align:center line:80%\n{_rtl(s)}\n\n")
    return path


def write_ass(ph, path, W=1080, H=1920, font="Vazirmatn", size=64):
    hdr = (f"[Script Info]\nScriptType: v4.00+\nPlayResX: {W}\nPlayResY: {H}\nWrapStyle: 0\n\n[V4+ Styles]\n"
           "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
           "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
           "MarginR, MarginV, Encoding\n"
           f"Style: Default,{font},{size},&H00FFFFFF,&H0000D7FF,&H00101010,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,"
           f"60,60,{int(H * 0.16)},1\n\n[Events]\n"
           "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(hdr)
        for a, b, s in ph:
            f.write(f"Dialogue: 0,{ts_ass(a)},{ts_ass(b)},Default,,0,0,0,,{s}\n")
    return path


# ============================================================================================== EDL
def write_edl(R, path, title=None, reel="AX"):
    """CMX3600: one event per kept source segment (record = edited time) + locator comments per layer / retime."""
    fps = R.sfps or 30.0
    title = title or os.path.splitext(os.path.basename(R.src))[0]
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    base = getattr(R.tl, "base", R.tl)
    for i, (a, b) in enumerate(base.seg):
        lines.append(f"{i + 1:03d}  {reel:<8} AA/V  C        {tc(a, fps)} {tc(b, fps)} "
                     f"{tc(base.cum[i], fps)} {tc(base.cum[i + 1], fps)}")
        lines.append(f"* FROM CLIP NAME: {os.path.basename(R.src)}")
    lines.append("")
    for L in sorted(R.layers, key=lambda L: L["_t0"]):
        lines.append(f"* LOC: {tc(L['_t0'], fps)} YELLOW  {L.get('id', L['type'])} ({L['type']})")
    if getattr(R, "retimed", False):
        for u0, u1, s0, s1, op in R.tl.rt.regions():
            lines.append(f"* RETIME: {tc(u0, fps)}-{tc(u1, fps)} speed {s0:.2f}..{s1:.2f} {(op or {}).get('op', '')}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


# ============================================================================================== timeline report
def timeline(R):
    base = getattr(R.tl, "base", R.tl)
    rows = [f"on screen {R.duration:.2f}s · edited {base.duration:.2f}s · source {R.sn / R.sfps:.2f}s · "
            f"{len(base.seg)} segments · {len(R.tl.cuts)} cuts · {len(R.layers)} layers",
            "── cuts (screen time): " + (", ".join(f"{c:.2f}" for c in R.tl.cuts) or "none")]
    if getattr(R, "retimed", False):
        for u0, u1, s0, s1, op in R.tl.rt.regions():
            rows.append(f"── retime {u0:6.2f}→{u1:6.2f}  speed {s0:.2f}..{s1:.2f}  {(op or {}).get('op', '')}")
    rows.append("── layers   start →   end  type           pass  id           content")
    for L in sorted(R.layers, key=lambda L: L["_t0"]):
        txt = str(L.get("text") or L.get("title") or L.get("hero") or L.get("src") or L.get("preset") or
                  L.get("gen") or "")[:40]
        rows.append(f"  {L['_t0']:6.2f} → {min(L['_t1'], R.duration):6.2f}  {L['type']:<14} {L.get('pass', 'front'):<5} "
                    f"{str(L.get('id', '')):<12} {txt}")
    if R.subs:
        rows.append(f"── subtitles: {len(R.subs['_phr'])} phrases, hidden: " +
                    (", ".join(f"{a:.2f}-{b:.2f}" for a, b in R.subs["_hide"]) or "never"))
    from ..v14.audio import sfx_events
    ev = sorted(sfx_events(R), key=lambda e: e[0])
    rows.append(f"── sfx: {len(ev)} events: " + ", ".join(f"{e[0]:.2f}:{e[1] if isinstance(e[1], str) else e[1][0]}"
                                                       for e in ev[:40]))
    rows.append("── camera keys: " + ", ".join(f"{k[0]:.2f}×{k[1]:.2f}" for k in R.cam.keys))
    g = R.spec.get("grade", {})
    if g.get("look") or g.get("lut"):
        rows.append(f"── grade: look={g.get('look')} lut={g.get('lut')} strength={g.get('lut_strength', 1.0)}")
    return "\n".join(rows)


# ============================================================================================== chapters
def chapters(R, min_gap=10.0):
    marks = [(0.0, "شروع")]
    for L in sorted(R.layers, key=lambda L: L["_t0"]):
        name = L.get("chapter") or (L.get("text") if L["type"] in ("kinetic_title", "halo", "shatter_text") else None) \
            or L.get("title") or L.get("hero")
        if not name:
            continue
        name = str(name).replace("|", " ")
        for ch in "*_^=!~":
            name = name.replace(ch, "")
        if L["_t0"] - marks[-1][0] >= min_gap:
            marks.append((L["_t0"], " ".join(name.split())[:60]))
    return "\n".join(f"{int(t // 60):02d}:{int(t % 60):02d} {n}" for t, n in marks)


# ============================================================================================== thumbnail
def best_frame_time(R, n=24):
    """sharpness (Laplacian var) × eye-region contrast (not blinking) × bonus when a hero layer is on screen."""
    import cv2
    best, bt = -1.0, R.duration * 0.3
    heroes = [(L["_t0"] + 0.6, L["_t1"]) for L in R.layers if L["type"] in ("kinetic_title", "halo", "aecomp")]
    for i in range(n):
        t = (i + 0.5) * R.duration / n
        st = R.tl.to_src(t)
        fr, _ = R.src_frame(st)
        g = cv2.cvtColor(cv2.resize(fr, (270, 480)), cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(g, cv2.CV_32F).var())
        try:
            u, v = R.tracks.anchor("face", st)
            ex, ey = int(u * 270), int(v * 480)
            eye = g[max(0, ey - 30):max(1, ey), max(0, ex - 40):ex + 40]
            eyes = float(eye.std()) if eye.size else 1.0
        except Exception:  # noqa: BLE001
            eyes = 1.0
        bonus = 1.25 if any(a <= t <= b for a, b in heroes) else 1.0
        sc = math.log1p(sharp) * (0.5 + eyes / 40.0) * bonus
        if sc > best:
            best, bt = sc, t
    return bt


# ============================================================================================== reformat
ASPECTS = {"9:16": (1080, 1920), "1:1": (1080, 1080), "4:5": (1080, 1350), "16:9": (1920, 1080), "4:3": (1440, 1080)}


def reformat(src, out, to="1:1", spec=None, mode="crop", log=print):
    """re-frame a finished master.  crop: face-centred (0.8 s smoothed, from the v14 pose cache when --spec is given,
    else centre).  blur: whole picture fits, the empty area is a blurred, darkened zoom of itself (no black bars)."""
    import cv2
    import numpy as np
    if to not in ASPECTS:
        raise KeyError(f"--to must be one of {', '.join(ASPECTS)}")
    W, H = ASPECTS[to]
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sw, sh = int(cap.get(3)), int(cap.get(4))
    centres = None
    if spec and mode == "crop":
        from ..v14.compositor import Renderer, Frame
        R = Renderer(spec)
        pts = []
        for i in range(n):
            F = Frame(R, min(i / fps, R.duration - 1e-3))
            F.M, F.zoom = R.matrix(F.t)
            x, y = F.anchor("face")
            pts.append((x / R.W * sw, y / R.H * sh))
        a = np.array(pts, np.float32)
        k = max(1, int(fps * 0.8))
        ker = np.ones(k) / k
        for c in (0, 1):
            a[:, c] = np.convolve(np.pad(a[:, c], (k // 2, k - k // 2 - 1), mode="edge"), ker, "valid")
        centres = a
    silent = os.path.splitext(out)[0] + "_v.mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r",
                          str(fps), "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt",
                          "yuv420p", silent], stdin=subprocess.PIPE)
    for i in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        if mode == "blur":
            k = min(W / sw, H / sh)
            fg = cv2.resize(fr, (int(sw * k), int(sh * k)), interpolation=cv2.INTER_AREA)
            kb = max(W / sw, H / sh)
            bg = cv2.resize(fr, (int(sw * kb) + 2, int(sh * kb) + 2))
            y0, x0 = (bg.shape[0] - H) // 2, (bg.shape[1] - W) // 2
            bg = (cv2.GaussianBlur(bg[y0:y0 + H, x0:x0 + W], (0, 0), 28).astype(np.float32) * 0.55).astype(np.uint8)
            oy, ox = (H - fg.shape[0]) // 2, (W - fg.shape[1]) // 2
            bg[oy:oy + fg.shape[0], ox:ox + fg.shape[1]] = fg
            img = bg
        else:
            k = max(W / sw, H / sh)
            cw, ch = W / k, H / k
            if centres is not None:
                fx, fy = centres[min(i, len(centres) - 1)]
                fy += ch * 0.12                                  # face in the upper third
            else:
                fx, fy = sw / 2, sh / 2
            x0 = int(min(max(fx - cw / 2, 0), sw - cw))
            y0 = int(min(max(fy - ch / 2, 0), sh - ch))
            img = cv2.resize(fr[y0:y0 + int(ch), x0:x0 + int(cw)], (W, H), interpolation=cv2.INTER_AREA)
        p.stdin.write(np.ascontiguousarray(img).tobytes())
        if log and i % 120 == 0:
            log(f"[reformat {to}] {i}/{n}")
    p.stdin.close()
    p.wait()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", silent, "-i", src, "-map", "0:v", "-map", "1:a?", "-c:v",
                    "copy", "-c:a", "copy", "-shortest", out])
    os.remove(silent)
    return out


# ============================================================================================== platforms
PLATFORMS = {
    "instagram_reels": {"aspect": "9:16", "max_s": 90, "lufs": -14, "safe": "keep top 220 px / bottom 380 px clear"},
    "youtube_shorts": {"aspect": "9:16", "max_s": 60, "lufs": -14, "safe": "bottom 300 px (title + buttons)"},
    "tiktok": {"aspect": "9:16", "max_s": 600, "lufs": -14, "safe": "right 140 px (buttons), bottom 420 px"},
    "youtube": {"aspect": "16:9", "max_s": None, "lufs": -14, "safe": "title-safe 90 %"},
    "instagram_feed": {"aspect": "4:5", "max_s": 60, "lufs": -14, "safe": "centre 1:1 for the grid crop"},
    "linkedin": {"aspect": "1:1", "max_s": 600, "lufs": -16, "safe": "burn captions in (autoplay is muted)"},
}


def platform_check(R, platform):
    P = PLATFORMS[platform]
    issues = []
    if P["max_s"] and R.duration > P["max_s"]:
        issues.append(f"too long: {R.duration:.1f}s > {P['max_s']}s")
    w, h = ASPECTS[P["aspect"]]
    if abs(R.W / R.H - w / h) > 0.02:
        issues.append(f"aspect {R.W}x{R.H} ≠ {P['aspect']} → `python3 -m mograph.pro reformat master.mp4 --to {P['aspect']}`")
    lufs = float(R.spec.get("audio", {}).get("lufs", -14))
    if abs(lufs - P["lufs"]) > 1.0:
        issues.append(f"loudness target {lufs} LUFS, platform normalises to {P['lufs']} (set audio.lufs)")
    return issues, P
