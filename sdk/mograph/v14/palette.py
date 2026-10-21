"""
mograph.v14.palette — AUTOMATIC colour design from the footage (no reference needed).

The reference's gold/red works because ITS background is warm-dark wood.  A different plate needs different
colours.  This module looks at the actual graded background + the speaker and designs a palette with real
colour-theory rules, so even a weak agent gets harmonious, readable elements:

  1  sample      background pixels (matte < 0.2) and speaker pixels (clothes / skin) over the whole clip
  2  measure     dominant background hue (circular mean, weighted by chroma), background luminance,
                 speaker clothing hues (to AVOID: an accent that matches her clothes disappears)
  3  design      accent  = complementary-ish to the background hue (≈ +150..210°), pushed away from
                           clothing hues, warm bias (warm accents read as "light"),
                 accent2 = split-complement (+30° from accent) for secondary emphasis,
                 alert   = strongest red that keeps ≥ 40° from the accent (fallback magenta),
                 ink     = warm or cool white matched to the background temperature,
                 glow    = accent lightened, grid/stage tint = background hue darkened.
  4  check       WCAG-style contrast of every text colour against the darkened stage (≥ 4.5 : 1);
                 lightness is raised until it passes.
  5  report      <media>.v14/palette.json   (+ swatch PNG with `python3 -m mograph.v14.palette media/raw.mp4`)

Spec: "meta": {"palette": "auto"}  (default)   |  "reference"  (old fixed gold/red)  |  {"accent": "#..", ...}
Use colours in layers as "@accent" "@accent2" "@alert" "@ink" "@glow" "@stage" — NEVER hard-code hex unless the
user asked for a brand colour.  Aliases kept for old specs: @gold → accent, @red → alert.
"""
from __future__ import annotations

import colorsys
import json
import math
import os

import cv2
import numpy as np


def _hex(rgb):
    r, g, b = [int(round(max(0, min(1, c)) * 255)) for c in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(rgb):
    r, g, b = [_lin(c) for c in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _hls(h, l, s):
    return colorsys.hls_to_rgb((h % 360) / 360.0, max(0, min(1, l)), max(0, min(1, s)))


def _hue_dist(a, b):
    d = abs((a - b) % 360)
    return min(d, 360 - d)


def _circ_mean(h, w):
    a = np.radians(h)
    x, y = (np.cos(a) * w).sum(), (np.sin(a) * w).sum()
    return float(np.degrees(math.atan2(y, x)) % 360), float(math.hypot(x, y) / (w.sum() + 1e-9))


def sample(video, matte_dir=None, n=12):
    cap = cv2.VideoCapture(video)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bg, fg = [], []
    for i in np.linspace(0, max(0, N - 1), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        fr = cv2.resize(fr, (180, int(180 * fr.shape[0] / fr.shape[1])), interpolation=cv2.INTER_AREA)
        m = None
        if matte_dir:
            p = os.path.join(matte_dir, f"{int(i):05d}.png")
            if os.path.exists(p):
                m = cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (fr.shape[1], fr.shape[0])) / 255.0
        if m is None:            # fallback: centre ellipse = speaker
            m = np.zeros(fr.shape[:2], np.float32)
            cv2.ellipse(m, (fr.shape[1] // 2, int(fr.shape[0] * .62)), (fr.shape[1] // 3, fr.shape[0] // 2), 0, 0, 360, 1, -1)
        rgb = fr[..., ::-1].reshape(-1, 3) / 255.0
        mm = m.reshape(-1)
        bg.append(rgb[mm < 0.2])
        fg.append(rgb[mm > 0.8])
    return np.concatenate(bg) if bg else np.zeros((1, 3)), np.concatenate(fg) if fg else np.zeros((1, 3))


def _hue_stats(px):
    hls = np.array([colorsys.rgb_to_hls(*p) for p in px[:: max(1, len(px) // 4000)]])
    h, l, s = hls[:, 0] * 360, hls[:, 1], hls[:, 2]
    chroma = s * (1 - np.abs(2 * l - 1))
    return h, l, s, chroma


def design(bg_px, fg_px, mode="dark_stage"):
    h, l, s, c = _hue_stats(bg_px)
    bg_h, bg_strength = _circ_mean(h, c + 1e-3)
    bg_l = float(np.median(l))
    bg_c = float(np.median(c))
    fh, fl, fs, fc = _hue_stats(fg_px)
    # clothing hues with real chroma (to avoid)
    cloth = fh[fc > 0.12]
    hist = np.histogram(cloth, bins=36, range=(0, 360))[0] if len(cloth) else np.zeros(36)
    avoid = [int(i) * 10 + 5 for i in np.argsort(hist)[::-1][:3] if hist[i] > 0.04 * max(1, len(cloth))]
    warm_bg = (bg_h < 70 or bg_h > 300)
    # candidate accents: complement ± 60°, scored
    # curated "cinematic" accent families (hue centre, preference bonus) — colour theory picks among THESE,
    # so a weak agent can never land on a cheap UI purple / acid green.
    FAMILIES = [(196, 6.0, "teal-cyan"), (38, 5.0, "amber-gold"), (14, 3.0, "coral"), (350, 2.0, "crimson"),
                (215, 3.0, "azure"), (160, 0.0, "mint"), (280, -4.0, "violet")]
    best, best_sc, fam = None, -1e9, None
    for hc, bonus, name in FAMILIES:
        d = _hue_dist(hc, bg_h)
        sc = bonus - abs(d - 180) * 0.12                             # contrast with the background hue
        sc -= sum(max(0, 45 - _hue_dist(hc, a_)) * 0.5 for a_ in avoid)   # not the colour of her clothes
        if d < 60:
            sc -= 25                                                 # same family as the plate = invisible
        if bg_strength < 0.08:                                       # neutral/grey plate → warm amber wins
            sc = bonus + (8 if name == "amber-gold" else 0)
        if sc > best_sc:
            best, best_sc, fam = hc, sc, name
    acc_h = best
    accent = _hls(acc_h, 0.62, 0.82)
    accent2 = _hls(acc_h + (-22 if acc_h > 180 else 18), 0.62, 0.72)   # analogous neighbour
    # alert: red family, ≥ 40° from accent
    alert_h = min([356, 350, 0, 6, 344, 338, 330], key=lambda x: (_hue_dist(x, acc_h) < 40,
                  sum(max(0, 18 - _hue_dist(x, a_)) for a_ in avoid) + _hue_dist(x, 356) * 0.3))
    alert = _hls(alert_h, 0.55, 0.95)
    ink = _hls(40, 0.95, 0.30)        # warm white always (cool white looks clinical on skin footage)
    stage = _hls(bg_h, 0.07, min(0.5, bg_c * 3 + 0.1))
    grid = _hls(acc_h, 0.75, 0.35)
    glow = _hls(acc_h - 8, 0.55, 1.0)
    # contrast guarantee vs the darkened stage
    def lift(rgb, target=4.5):
        hh, ll, ss = colorsys.rgb_to_hls(*rgb)
        while contrast(rgb, stage) < target and ll < 0.95:
            ll += 0.02
            rgb = colorsys.hls_to_rgb(hh, ll, ss)
        return rgb
    accent, accent2, alert, ink = lift(accent), lift(accent2), lift(alert, 3.6), lift(ink, 7)
    grad = [_hls(acc_h + 6, 0.74, 1.0), accent, _hls(acc_h - 10, 0.42, 1.0)]
    pal = {"accent": _hex(accent), "accent2": _hex(accent2), "alert": _hex(alert), "ink": _hex(ink),
           "glow": _hex(glow), "stage": _hex(stage), "grid": _hex(grid), "mute": _hex(_hls(bg_h, 0.72, 0.08)),
           "accent_grad": [_hex(c) for c in grad],
           "alert_grad": [_hex(_hls(alert_h + 4, 0.72, 1.0)), _hex(alert), _hex(_hls(alert_h - 6, 0.38, 1.0))]}
    pal["gold"], pal["red"], pal["amber"] = pal["accent"], pal["alert"], pal["accent2"]
    info = {"bg_hue": round(bg_h, 1), "bg_hue_strength": round(bg_strength, 3), "bg_lum": round(bg_l, 3),
            "bg_chroma": round(bg_c, 3), "avoid_hues": avoid, "accent_hue": int(acc_h), "accent_family": fam,
            "contrast_vs_stage": {k: round(contrast(_rgb(pal[k]), stage), 2) for k in ("accent", "alert", "ink")}}
    return pal, info


def auto_palette(video, log=print):
    cdir = video + ".v14"
    os.makedirs(cdir, exist_ok=True)
    out = os.path.join(cdir, "palette.json")
    if os.path.exists(out):
        with open(out) as f:
            return json.load(f)["palette"]
    mdir = os.path.join(cdir, "matte")
    bg, fg = sample(video, mdir if os.path.isdir(mdir) else None)
    pal, info = design(bg, fg)
    with open(out, "w") as f:
        json.dump({"palette": pal, "analysis": info}, f, indent=1)
    log(f"[palette] bg hue {info['bg_hue']}° (strength {info['bg_hue_strength']}) → accent {pal['accent']} "
        f"alert {pal['alert']} ink {pal['ink']}  contrast {info['contrast_vs_stage']}")
    return pal


def swatch(pal, path):
    keys = ["stage", "grid", "accent", "accent2", "glow", "alert", "ink", "mute"]
    im = np.zeros((120, 120 * len(keys), 3), np.uint8)
    for i, k in enumerate(keys):
        r, g, b = _rgb(pal[k])
        im[:, i * 120:(i + 1) * 120] = (int(b * 255), int(g * 255), int(r * 255))
        cv2.putText(im, k, (i * 120 + 6, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 1)
    cv2.imwrite(path, im)




# ====================================================================================================
# v15 (FilmCraft color_match.rs): Oklab 3-Range Perceptual Color Match with <= 8° Skin-Tone Guard
# ====================================================================================================
def bgr_to_oklab(bgr_u8: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 image to perceptual Oklab (L, a, b) float32 (Björn Ottosson 2020)."""
    f = bgr_u8.astype(np.float32) / 255.0
    lin = np.where(f <= 0.04045, f / 12.92, np.power((f + 0.055) / 1.055, 2.4))
    b, g, r = lin[..., 0], lin[..., 1], lin[..., 2]
    l = np.cbrt(np.maximum(0.0, 0.41222147 * r + 0.53633255 * g + 0.051445995 * b))
    m = np.cbrt(np.maximum(0.0, 0.21190350 * r + 0.68069950 * g + 0.107396960 * b))
    s = np.cbrt(np.maximum(0.0, 0.08830246 * r + 0.28171885 * g + 0.629978700 * b))
    L = 0.21045426 * l + 0.79361780 * m - 0.004072047 * s
    a = 1.97799850 * l - 2.42859220 * m + 0.450593700 * s
    b_o = 0.025904037 * l + 0.78277177 * m - 0.808675770 * s
    return np.stack([L, a, b_o], axis=-1)


def oklab_to_bgr(lab: np.ndarray) -> np.ndarray:
    """Convert Oklab (L, a, b) float32 image back to BGR uint8."""
    L, a, b_o = lab[..., 0], lab[..., 1], lab[..., 2]
    l_ = L + 0.39633778 * a + 0.21580376 * b_o
    m_ = L - 0.105561346 * a - 0.06385417 * b_o
    s_ = L - 0.08948418 * a - 1.29148550 * b_o
    l, m, s = l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_
    r = +4.07674170 * l - 3.30771160 * m + 0.23096994 * s
    g = -1.26843800 * l + 2.60975740 * m - 0.34131938 * s
    b = -0.0041960863 * l - 0.70341860 * m + 1.70761470 * s
    rgb = np.stack([b, g, r], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    srgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055)
    return np.clip(srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def match_reference_grade(src_bgr: np.ndarray, ref_bgr: np.ndarray,
                          person_alpha: np.ndarray | None = None,
                          max_skin_deg: float = 8.0) -> dict:
    """
    Ported from FilmCraft `crates/render/src/color_match.rs`:
    1. Downsamples both frames (<= 96px wide) and converts to Oklab.
    2. Separates into 3 tonal ranges: shadows smoothstep(0.55, 0.15, L), highlights smoothstep(0.45, 0.85, L), midtones.
    3. Skin-tone protection: pixels in Oklab hue 20°–75°, chroma 0.02–0.20, L 0.25–0.92 are down-weighted (x0.2)
       and the solved color cast is scaled so skin hue moves by at most `max_skin_deg` (8°).
    """
    s_sm = cv2.resize(src_bgr, (96, int(96 * src_bgr.shape[0] / max(1, src_bgr.shape[1]))), interpolation=cv2.INTER_AREA)
    r_sm = cv2.resize(ref_bgr, (96, int(96 * ref_bgr.shape[0] / max(1, ref_bgr.shape[1]))), interpolation=cv2.INTER_AREA)
    s_lab = bgr_to_oklab(s_sm)
    r_lab = bgr_to_oklab(r_sm)

    def _stats(lab, is_src=False):
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        ch = np.hypot(a, b)
        hue = np.degrees(np.arctan2(b, a)) % 360.0
        skin = (hue >= 20.0) & (hue <= 75.0) & (ch >= 0.02) & (ch <= 0.20) & (L >= 0.25) & (L <= 0.92)
        w_skin = np.where(skin, 0.2, 1.0) if is_src else np.ones_like(L)
        w_sh = _smoothstep(0.55, 0.15, L) * w_skin
        w_hi = _smoothstep(0.45, 0.85, L) * w_skin
        w_md = np.maximum(0.0, 1.0 - _smoothstep(0.55, 0.15, L) - _smoothstep(0.45, 0.85, L)) * w_skin
        out = {}
        for name, w in (("shadows", w_sh), ("midtones", w_md), ("highlights", w_hi)):
            ws = float(w.sum()) + 1e-6
            out[name] = np.array([float((L * w).sum() / ws), float((a * w).sum() / ws), float((b * w).sum() / ws)], np.float32)
        out["chroma"] = float(ch.mean()) + 1e-5
        out["skin_mask"] = skin
        out["skin_hue"] = float(np.degrees(math.atan2(float(b[skin].mean()), float(a[skin].mean()))) % 360.0) if np.any(skin) else None
        return out

    st_s = _stats(s_lab, is_src=True)
    st_r = _stats(r_lab, is_src=False)
    shifts = {k: (st_r[k] - st_s[k]) for k in ("shadows", "midtones", "highlights")}
    sat_ratio = float(np.clip(st_r["chroma"] / st_s["chroma"], 0.65, 1.45))

    # Skin-tone protection clamp: ensure mean skin hue shifts by at most max_skin_deg (8°)
    scale_cast = 1.0
    if st_s["skin_hue"] is not None and np.any(st_s["skin_mask"]):
        sk = st_s["skin_mask"]
        L_sk = s_lab[..., 0][sk]
        w_sh = _smoothstep(0.55, 0.15, L_sk)
        w_hi = _smoothstep(0.45, 0.85, L_sk)
        w_md = np.maximum(0.0, 1.0 - w_sh - w_hi)
        da = float((w_sh * shifts["shadows"][1] + w_md * shifts["midtones"][1] + w_hi * shifts["highlights"][1]).mean())
        db = float((w_sh * shifts["shadows"][2] + w_md * shifts["midtones"][2] + w_hi * shifts["highlights"][2]).mean())
        a_new = float(s_lab[..., 1][sk].mean()) + da
        b_new = float(s_lab[..., 2][sk].mean()) + db
        new_hue = float(np.degrees(math.atan2(b_new, a_new)) % 360.0)
        dh = _hue_dist(new_hue, st_s["skin_hue"])
        if dh > max_skin_deg:
            scale_cast = max_skin_deg / max(dh, 1e-4)
            for k in shifts:
                shifts[k][1:] *= scale_cast

    return {
        "shadows_dab": [round(float(x), 4) for x in shifts["shadows"]],
        "midtones_dab": [round(float(x), 4) for x in shifts["midtones"]],
        "highlights_dab": [round(float(x), 4) for x in shifts["highlights"]],
        "sat_scale": round(sat_ratio, 3),
        "skin_guard_scale": round(scale_cast, 3),
    }


if __name__ == "__main__":
    import sys
    v = sys.argv[1]
    p = os.path.join(v + ".v14", "palette.json")
    if os.path.exists(p):
        os.remove(p)
    pal = auto_palette(v)
    swatch(pal, os.path.join(v + ".v14", "palette.png"))
    print(json.dumps(pal, indent=1))
