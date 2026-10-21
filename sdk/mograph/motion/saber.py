"""
mograph.motion.saber — light / energy effects modelled on Video Copilot SABER (free AE plugin), rebuilt from its
public parameter set (read from the plugin binary: Preset · Glow Color/Intensity/Spread/Bias · Core Size/Start/End ·
Core Type Saber|Layer Masks|Text Layer · Start/End Size/Offset/Roundness · Halo · Core Softness · Flicker ·
Mask Randomization · Distortion (Smoke|Fluid|Energy, Distortion|Multiply, wind, noise speed/scale/bias/complexity/
aspect, Lock Noise to Saber, Core Distortion) · Glow 1–4 Intensity/Size · Gamma/Brightness/Saturation/Alpha Boost ·
Composite Transparent|Black|Add · Mask Core / Mask Glow / Use Text Alpha) + our own LIGHTNING bolt generator.

One call renders a frame:   img = render(params, t, W, H)            → BGRA premultiplied uint8 (W×H)
    params = {"preset": "neon", "core": "text", "text": "سلام", "color": "#ff2a6d", ...overrides}
Cores ("core"):
    "saber"  straight blade  "start": [x, y], "end": [x, y]  (px) + start_size / end_size / roundness (taper)
    "path"   polylines        "paths": [[[x,y],...], ...], "closed": true   (= AE layer masks / shape layers)
    "shape"  generated shape  "shape": "circle|rect|hex|star|heart|triangle|line", "x","y","w","h","rot"
    "text"   text layer       "text", "font", "size", "x", "y", "text_mode": "outline" (neon tube) | "fill"
    "alpha"  any layer alpha  (used by the fx stack: {"fx": [{"saber": {...}}]} glows the layer itself)
Animation:  "core_start"/"core_end" (0–100 %, trim along the path — the light-saber ignition), "evolution" (0–100 %,
            Saber "Mask Evolution": the lit segment travels around a closed path/shape),
            "reveal": {"t": 0, "dur": 0.6, "ease": "out_expo"}  → core_end 0→100 (paths) / RTL wipe (text, alpha)
            "out": {"t": 3.0, "dur": 0.4} retract;  every scalar also accepts a keyframe curve {"k": [[t, v], ...]}
Look:       "glow_intensity" "glow_spread" "glow_bias" (0 coloured → 1 white-hot) "core_size" "core_softness"
            "halo_intensity" "halo_size" "flicker_intensity" "flicker_speed"
            "distortion": "smoke|fluid|energy|none", "distortion_amount", "distortion_mode": "displace|multiply",
            "noise_scale" "noise_speed" "noise_complexity" "noise_aspect" "wind_speed" "wind_dir" (deg, -90 = up),
            "core_distortion" (0–1: the core itself wobbles), "lightning": {"bolts": 3, "rate": 14, "jitter": .22,
            "branches": 2, "width": 2.5}, "glows": [[i1, s1], [i2, s2], [i3, s3], [i4, s4]] (Saber Glow 1–4)
Composite:  "composite": "transparent" (alpha, default) | "black" | "add" (only for render_over),
            "gamma" "brightness" "saturation" "alpha_boost"
Speed: all heavy maths at ½ res (glow pyramid at ¼), separable gaussians → ~60–120 ms/frame for 1080×1920 on 2 vCPU.
"""
from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from .easing import ease as _ease

# =============================================================================================== presets
# compact preset table:  colour, intensity, spread, bias, core_size, softness, halo, flicker, distortion, amount,
#                         noise_scale, noise_speed, extras
_P = {
    "default":       ("#2a7dff", 1.00, 1.00, 0.35, 1.0, 0.25, 0.40, 0.08, "none", 0, 1.0, 1.0, {}),
    "arc_reactor":   ("#43d6ff", 1.25, 0.85, 0.55, 1.1, 0.35, 0.70, 0.10, "energy", 0.25, 0.7, 1.4, {}),
    "burning":       ("#ff6a1a", 1.15, 1.10, 0.30, 0.9, 0.40, 0.30, 0.20, "fluid", 0.80, 0.9, 1.6, {"ramp": [[0, "#200400"], [0.25, "#a81800"], [0.5, "#ff5a0a"], [0.75, "#ffb43a"], [0.92, "#fff0b0"], [1, "#ffffff"]], "wind_dir": -90, "wind_speed": 1.2}),
    "cellular":      ("#38ff8c", 1.00, 0.90, 0.30, 0.8, 0.30, 0.20, 0.05, "energy", 0.55, 0.5, 0.6, {"distortion_mode": "multiply"}),
    "core":          ("#ffffff", 1.40, 0.70, 0.80, 1.4, 0.15, 0.50, 0.05, "none", 0, 1.0, 1.0, {}),
    "electric":      ("#5aa8ff", 1.20, 0.80, 0.50, 0.6, 0.10, 0.30, 0.45, "energy", 0.70, 0.6, 3.0, {"lightning": {"bolts": 3}}),
    "energize":      ("#9a5bff", 1.10, 1.00, 0.45, 0.9, 0.30, 0.40, 0.25, "energy", 0.45, 0.8, 2.0, {}),
    "energy":        ("#2ee6ff", 1.15, 1.05, 0.45, 0.9, 0.30, 0.40, 0.15, "energy", 0.85, 0.9, 1.6, {}),
    "fire":          ("#ff5a14", 1.20, 1.20, 0.30, 0.9, 0.45, 0.35, 0.25, "smoke", 1.00, 0.8, 1.8, {"ramp": [[0, "#200400"], [0.25, "#a81800"], [0.5, "#ff5a0a"], [0.75, "#ffb43a"], [0.92, "#fff0b0"], [1, "#ffffff"]], "wind_dir": -90, "wind_speed": 1.6}),
    "firestorm":     ("#ff3c0a", 1.35, 1.40, 0.35, 1.0, 0.45, 0.40, 0.30, "fluid", 1.30, 0.7, 2.2, {"ramp": [[0, "#200400"], [0.25, "#a81800"], [0.5, "#ff5a0a"], [0.75, "#ffb43a"], [0.92, "#fff0b0"], [1, "#ffffff"]], "wind_dir": -90, "wind_speed": 2.2}),
    "fog":           ("#b9d4ff", 0.70, 1.80, 0.15, 0.7, 0.80, 0.10, 0.03, "smoke", 1.20, 1.6, 0.5, {"distortion_mode": "multiply"}),
    "fractal":       ("#ff4fd8", 1.05, 0.95, 0.35, 0.7, 0.25, 0.25, 0.10, "energy", 0.90, 0.6, 1.0, {"noise_complexity": 6}),
    "fuel":          ("#ffb627", 1.10, 1.00, 0.40, 0.9, 0.30, 0.30, 0.18, "fluid", 0.60, 0.8, 1.6, {"wind_dir": -90}),
    "fusion":        ("#ff7a1a", 1.30, 1.00, 0.60, 1.1, 0.30, 0.60, 0.15, "energy", 0.40, 0.8, 1.4, {}),
    "ghost":         ("#c9f5ff", 0.80, 1.40, 0.25, 0.7, 0.60, 0.20, 0.10, "smoke", 0.90, 1.2, 0.8, {"wind_dir": -90}),
    "ghostbuster":   ("#ff3d7a", 1.30, 0.90, 0.40, 0.7, 0.20, 0.30, 0.40, "energy", 0.85, 0.5, 2.6, {"lightning": {"bolts": 2, "branches": 3}}),
    "green_energy":  ("#27ff6a", 1.15, 1.05, 0.45, 0.9, 0.30, 0.40, 0.15, "energy", 0.45, 0.9, 1.6, {}),
    "hard_core":     ("#ff2a2a", 1.50, 0.60, 0.85, 1.5, 0.08, 0.60, 0.05, "none", 0, 1.0, 1.0, {}),
    "haze":          ("#ffd9a8", 0.75, 1.70, 0.20, 0.8, 0.70, 0.10, 0.05, "smoke", 0.80, 1.4, 0.4, {}),
    "heat":          ("#ff9a3c", 0.90, 1.20, 0.30, 0.8, 0.50, 0.20, 0.10, "fluid", 0.70, 0.7, 1.2, {"wind_dir": -90, "core_distortion": 0.4}),
    "hot":           ("#ff4a1f", 1.30, 1.00, 0.70, 1.1, 0.25, 0.50, 0.10, "none", 0, 1.0, 1.0, {}),
    "inferno":       ("#ff2a00", 1.45, 1.50, 0.35, 1.0, 0.50, 0.50, 0.30, "smoke", 1.40, 0.7, 2.4, {"ramp": [[0, "#200400"], [0.25, "#a81800"], [0.5, "#ff5a0a"], [0.75, "#ffb43a"], [0.92, "#fff0b0"], [1, "#ffffff"]], "wind_dir": -90, "wind_speed": 2.4}),
    "kryptonite":    ("#5bff3c", 1.20, 0.90, 0.50, 0.9, 0.25, 0.40, 0.20, "energy", 0.55, 0.6, 1.2, {}),
    "meteor":        ("#ff8a2a", 1.30, 1.10, 0.55, 1.0, 0.30, 0.40, 0.15, "fluid", 0.80, 0.8, 2.0, {"wind_dir": 180, "wind_speed": 2.0}),
    "myst":          ("#a98bff", 0.85, 1.50, 0.25, 0.7, 0.60, 0.20, 0.08, "smoke", 1.00, 1.3, 0.6, {}),
    "narrow":        ("#2a7dff", 1.00, 0.50, 0.40, 0.6, 0.15, 0.20, 0.05, "none", 0, 1.0, 1.0, {}),
    "narrow_bright": ("#4fc3ff", 1.50, 0.55, 0.70, 0.7, 0.12, 0.40, 0.05, "none", 0, 1.0, 1.0, {}),
    "nebula":        ("#d14bff", 1.00, 1.60, 0.30, 0.8, 0.60, 0.30, 0.08, "fluid", 1.10, 1.5, 0.5, {}),
    "neon":          ("#ff2a6d", 1.10, 0.75, 0.55, 0.8, 0.12, 0.50, 0.04, "none", 0, 1.0, 1.0, {}),
    "patronus":      ("#bfe9ff", 1.10, 1.30, 0.55, 0.9, 0.45, 0.60, 0.12, "smoke", 0.60, 1.0, 1.0, {}),
    "portal":        ("#ff8c1a", 1.30, 0.90, 0.55, 0.9, 0.20, 0.40, 0.35, "energy", 0.60, 0.4, 3.0, {"lightning": {"bolts": 4, "jitter": 0.12, "width": 1.6}, "wind_dir": 90}),
    "protocol":      ("#00ffd0", 1.05, 0.80, 0.45, 0.7, 0.15, 0.30, 0.10, "energy", 0.30, 0.8, 1.2, {"distortion_mode": "multiply"}),
    "proton":        ("#ff5ad1", 1.25, 1.00, 0.55, 0.9, 0.20, 0.40, 0.45, "energy", 0.90, 0.5, 3.0, {"lightning": {"bolts": 3, "branches": 2}}),
    "pulse":         ("#3c8cff", 1.20, 1.00, 0.50, 1.0, 0.25, 0.50, 0.55, "none", 0, 1.0, 1.0, {"flicker_speed": 2.0}),
    "replicant":     ("#ff2a2a", 1.10, 0.90, 0.40, 0.8, 0.20, 0.30, 0.20, "energy", 0.35, 0.7, 1.4, {}),
    "simple_orange": ("#ff8a1a", 1.00, 1.00, 0.40, 1.0, 0.25, 0.40, 0.05, "none", 0, 1.0, 1.0, {}),
    "soft_smoke":    ("#7fa6ff", 0.55, 1.50, 0.05, 0.8, 0.60, 0.20, 0.05, "smoke", 1.10, 1.2, 0.8, {"wind_dir": -90, "wind_speed": 0.8}),
    "solar":         ("#ffb31a", 1.35, 1.20, 0.60, 1.1, 0.30, 0.60, 0.10, "fluid", 0.50, 1.0, 1.2, {}),
    "solar_wave":    ("#ffc02a", 1.25, 1.30, 0.50, 1.0, 0.35, 0.50, 0.15, "fluid", 0.90, 1.2, 1.6, {"wind_dir": 0, "wind_speed": 1.6}),
    "starkiller":    ("#ff1f1f", 1.40, 0.90, 0.50, 1.1, 0.15, 0.50, 0.35, "energy", 0.50, 0.5, 2.4, {"core_distortion": 0.5}),
    "tesseract":     ("#3ab0ff", 1.30, 1.00, 0.60, 1.0, 0.25, 0.60, 0.25, "energy", 0.70, 0.6, 1.8, {"noise_complexity": 5}),
    "thin":          ("#ff2a2a", 1.20, 0.45, 0.60, 0.35, 0.08, 0.30, 0.03, "none", 0, 1.0, 1.0, {}),
    "torch":         ("#ff7a1a", 1.20, 1.10, 0.40, 1.0, 0.40, 0.40, 0.25, "smoke", 0.90, 0.8, 1.8, {"ramp": [[0, "#200400"], [0.25, "#a81800"], [0.5, "#ff5a0a"], [0.75, "#ffb43a"], [0.92, "#fff0b0"], [1, "#ffffff"]], "wind_dir": -90, "wind_speed": 1.4}),
    "tractor_beam":  ("#9ff0ff", 0.90, 1.60, 0.40, 1.6, 0.70, 0.40, 0.10, "smoke", 0.50, 1.6, 1.0, {"wind_dir": 90, "wind_speed": 1.0}),
    "velocity":      ("#4f7dff", 1.15, 1.00, 0.45, 0.9, 0.30, 0.30, 0.10, "fluid", 0.70, 0.8, 2.6, {"wind_dir": 180, "wind_speed": 3.0}),
    "wavelet":       ("#2affd5", 1.00, 1.00, 0.40, 0.8, 0.30, 0.30, 0.10, "fluid", 0.80, 1.0, 1.2, {"noise_aspect": 2.5}),
    "wide":          ("#2a7dff", 1.00, 1.80, 0.30, 1.6, 0.40, 0.50, 0.05, "none", 0, 1.0, 1.0, {}),
    "windy":         ("#9ad1ff", 0.95, 1.20, 0.35, 0.8, 0.40, 0.20, 0.10, "smoke", 1.00, 1.0, 1.6, {"wind_dir": 0, "wind_speed": 2.5}),
    "wormhole":      ("#7a4bff", 1.25, 1.10, 0.55, 1.0, 0.30, 0.50, 0.20, "energy", 0.80, 0.7, 1.4, {"noise_complexity": 5}),
    "yellow_smoke":  ("#ffd21f", 0.90, 1.40, 0.30, 0.8, 0.55, 0.20, 0.08, "smoke", 1.10, 1.1, 1.0, {"wind_dir": -90}),
    "zap":           ("#d7f0ff", 1.40, 0.70, 0.70, 0.5, 0.08, 0.30, 0.60, "energy", 0.95, 0.4, 4.0, {"lightning": {"bolts": 5, "branches": 3, "rate": 20}}),
}
PRESET_TAGS = {
    "neon": "clean neon tube (text / logos)", "electric": "arcing lightning", "zap": "violent white lightning",
    "fire": "rising flames", "inferno": "big fire", "soft_smoke": "soft smoky wisps", "portal": "Dr-Strange sparks",
    "default": "classic blue lightsaber", "starkiller": "unstable red saber", "hard_core": "red hot white core",
    "thin": "laser line", "tractor_beam": "wide soft beam", "energy": "energy plasma", "ghost": "ethereal ghost",
}
GLOSSARY_CAPS = {"light_beam": ["default", "thin", "narrow", "wide", "tractor_beam", "hard_core"],
                 "neon": ["neon", "narrow_bright", "protocol", "core"],
                 "lightning": ["electric", "zap", "proton", "ghostbuster", "portal"],
                 "smoke_noise": ["soft_smoke", "fog", "haze", "myst", "yellow_smoke", "fire", "inferno"],
                 "distortion": ["heat", "wavelet", "fluid", "velocity", "nebula"]}


def preset_names():
    return list(_P)


def preset(name):
    """preset → full parameter dict (names accept 'Arc Reactor', 'arc-reactor', 'arc_reactor')."""
    key = str(name or "default").strip().lower().replace(" ", "_").replace("-", "_")
    if key not in _P:
        raise KeyError(f"saber preset '{name}' unknown — python3 -m mograph.motion saber list")
    c, gi, sp, bias, cs, soft, halo, fl, dt, da, ns, nsp, ex = _P[key]
    d = {"color": c, "glow_intensity": gi, "glow_spread": sp, "glow_bias": bias, "core_size": cs,
         "core_softness": soft, "halo_intensity": halo, "halo_size": 1.0, "flicker_intensity": fl, "flicker_speed": 1.0,
         "distortion": dt, "distortion_amount": da, "distortion_mode": "displace", "noise_scale": ns,
         "noise_speed": nsp, "noise_complexity": 3, "noise_aspect": 1.0, "noise_bias": 0.0, "wind_speed": 0.0,
         "wind_dir": -90.0, "core_distortion": 0.0}
    d.update({k: (dict(v) if isinstance(v, dict) else v) for k, v in ex.items()})
    return d


DEFAULTS = {"core": "saber", "core_start": 0.0, "core_end": 100.0, "start_size": 1.0, "end_size": 1.0,
            "roundness": 1.0, "glows": [[1.0, 0.15], [0.9, 0.35], [0.6, 0.9], [0.35, 2.2]], "glow_pre_gamma": 1.0,
            "gamma": 1.0, "brightness": 1.0, "saturation": 1.0, "alpha_boost": 1.0, "composite": "transparent",
            "seed": 0, "scale": 1.0, "opacity": 1.0}


def params(spec):
    """spec (preset + overrides) → resolved parameter dict."""
    p = dict(DEFAULTS)
    p.update(preset(spec.get("preset", "default")))
    for k, v in spec.items():
        if k == "lightning" and isinstance(v, dict) and isinstance(p.get("lightning"), dict):
            p["lightning"] = dict(p["lightning"], **v)
        else:
            p[k] = v
    return p


def _val(v, t, default=0.0):
    if isinstance(v, dict) and "k" in v:
        from ..aefx.keyframes import evaluate
        ks = v["k"]
        if ks and len(ks[0]) == 2:                                   # [[t, v]] → smooth (easy-ease-ish) keys
            for i in range(len(ks) - 1):
                if ks[i][0] <= t <= ks[i + 1][0]:
                    p = (t - ks[i][0]) / max(1e-9, ks[i + 1][0] - ks[i][0])
                    g = _ease(v.get("ease", "ease_in_out"))(p)
                    a, b = ks[i][1], ks[i + 1][1]
                    return [x + (y - x) * g for x, y in zip(a, b)] if isinstance(a, list) else a + (b - a) * g
            return ks[0][1] if t < ks[0][0] else ks[-1][1]
        return evaluate(v, t, default)
    return default if v is None else v


NAMED = {"red": "#ff2a2a", "blue": "#2a7dff", "green": "#27ff6a", "yellow": "#ffd21f", "orange": "#ff7a1a",
         "purple": "#a43cff", "pink": "#ff3fb4", "white": "#ffffff", "gold": "#ffb627", "cyan": "#18e6e0",
         "magenta": "#ff1f8f", "black": "#000000"}


def _hex(c):
    if isinstance(c, (list, tuple)):                       # [r, g, b] 0..1 or 0..255
        v = np.array(c[:3], np.float32)
        v = v / 255.0 if v.max() > 1.0 else v
        return v[::-1].copy()
    c = str(c).strip()
    if not c.startswith("#"):
        from .glossary import COLOURS
        c = NAMED.get(c.lower(), COLOURS.get(c, c))
        if not c.startswith("#"):
            raise ValueError(f"saber colour '{c}': use #rrggbb, [r,g,b] or a name ({', '.join(NAMED)})")
    c = c.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return np.array([int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)], np.float32) / 255.0   # BGR


# =============================================================================================== noise
@lru_cache(maxsize=96)
def _grid(seed, o, k, gw, gh):
    rng = np.random.default_rng((seed * 7919 + o * 104729 + k * 1299709) & 0x7FFFFFFF)   # & → never negative
    return rng.standard_normal((gh, gw)).astype(np.float32)


def fbm(W, H, t, scale=1.0, speed=1.0, octaves=3, seed=0, aspect=1.0, wind=(0.0, 0.0), ridged=False, base_px=70.0):
    """animated fractal noise (-1..1-ish), W×H float32.  Time-continuous: each octave cross-fades between random
    lattices k and k+1 (smoothstep) while scrolling with the wind.  ridged=True → 1-|n| ridges (energy / plasma)."""
    out = np.zeros((H, W), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(int(max(1, octaves))):
        cell = base_px * scale / (2 ** o)                         # px per lattice cell
        cx, cy = max(2.0, cell * aspect), max(2.0, cell)
        gw, gh = int(W / cx) + 4, int(H / cy) + 4
        tt = t * speed * (0.6 + 0.25 * o) + o * 0.37
        k = int(math.floor(tt))
        f = tt - k
        f = f * f * (3 - 2 * f)
        ox = wind[0] * t * (1 + 0.3 * o) / cx
        oy = wind[1] * t * (1 + 0.3 * o) / cy
        ox, oy = ox % max(1, gw - 4), oy % max(1, gh - 4)
        M = np.array([[1 / cx, 0, ox + 1], [0, 1 / cy, oy + 1]], np.float32)
        layer = None
        for kk, w in ((k, 1 - f), (k + 1, f)):
            g = np.tile(_grid(seed, o, kk, gw, gh), (2, 2))
            up = cv2.warpAffine(g, M, (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_WRAP)
            layer = up * w if layer is None else layer + up * w
        layer /= math.sqrt((1 - f) ** 2 + f ** 2)                  # keep variance during the cross-fade
        if ridged:
            layer = 1.0 - np.abs(layer)
            layer = layer * layer * 2.0 - 1.0
        out += layer * amp
        tot += amp
        amp *= 0.5
    return out / (tot * 0.9)


def flicker(t, speed=1.0, seed=0):
    """smooth 1-D flicker value in [-1, 1] (sum of incommensurate sines + a little jitter)."""
    x = t * speed * 9.0 + seed * 13.1
    v = 0.55 * math.sin(x) + 0.3 * math.sin(x * 2.31 + 1.7) + 0.15 * math.sin(x * 5.73 + 0.4)
    j = math.sin(math.floor(x * 1.9) * 12.9898 + seed) * 43758.5453
    return float(np.clip(v + 0.25 * ((j - math.floor(j)) * 2 - 1), -1, 1))


# =============================================================================================== cores
def _poly_trim(pts, a, b, closed):
    from ..varstroke import polyline_slice
    pts = np.asarray(pts, np.float32)
    if len(pts) < 2 or b <= a:
        return None
    seg = np.vstack([pts, pts[:1]]) if closed else pts
    L = float(np.hypot(*np.diff(seg, axis=0).T).sum())
    if a <= 0.0005 and b >= 0.9995:
        return seg
    return polyline_slice(pts, a * L, b * L, closed=closed, step=2.0)


def shape_path(kind, x, y, w, h, rot=0.0, n=96):
    """closed polyline of a generated shape (px)."""
    if kind == "line":
        P = np.array([[-0.5, 0], [0.5, 0]], np.float32)
    elif kind == "circle":
        a = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
        P = np.stack([np.cos(a), np.sin(a)], 1) * 0.5
    elif kind in ("rect", "square"):
        P = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], np.float32)
    elif kind in ("hex", "triangle", "tri"):
        m = 6 if kind == "hex" else 3
        a = np.linspace(0, 2 * np.pi, m, endpoint=False) - np.pi / 2
        P = np.stack([np.cos(a), np.sin(a)], 1) * 0.5
    elif kind == "star":
        a = np.linspace(0, 2 * np.pi, 10, endpoint=False) - np.pi / 2
        r = np.where(np.arange(10) % 2 == 0, 0.5, 0.21)
        P = np.stack([np.cos(a) * r, np.sin(a) * r], 1)
    elif kind == "heart":
        s = np.linspace(0, 2 * np.pi, n, endpoint=False)
        P = np.stack([16 * np.sin(s) ** 3, -(13 * np.cos(s) - 5 * np.cos(2 * s) - 2 * np.cos(3 * s) - np.cos(4 * s))], 1) / 34
    else:
        raise KeyError(f"saber shape '{kind}' (circle rect hex triangle star heart line)")
    P = P * np.array([w, h], np.float32)
    if rot:
        c, s_ = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        P = P @ np.array([[c, s_], [-s_, c]], np.float32)
    return (P + np.array([x, y], np.float32)).astype(np.float32)


def _stroke_mask(polys, W, H, k, width, closed):
    m = np.zeros((H, W), np.uint8)
    for P in polys:
        if P is None or len(P) < 2:
            continue
        q = np.round(np.asarray(P) * k * 8).astype(np.int32)
        cv2.polylines(m, [q], False, 255, max(1, int(round(width))), cv2.LINE_AA, shift=3)
    return m.astype(np.float32) / 255.0


@lru_cache(maxsize=16)
def _text_mask(text, font, size, x, y, W, H, align):
    import cairo
    from ..text import draw_text, resolve_font
    s = cairo.ImageSurface(cairo.FORMAT_A8, W, H)
    c = cairo.Context(s)
    draw_text(c, text, x, y, size, "#ffffff", resolve_font(font, None, "title"), align=align, quiet=True)
    s.flush()
    a = np.ndarray((H, s.get_stride()), np.uint8, s.get_data())[:, :W].astype(np.float32) / 255.0
    return a


def core_mask(p, t, W, H, k=0.5, alpha=None):
    """core intensity map (0..1) at work scale k (W·k × H·k) — before softness / distortion."""
    w, h = max(2, int(W * k)), max(2, int(H * k))
    kind = p.get("core", "saber")
    sc = float(p.get("scale", 1.0))
    size = float(_val(p.get("core_size"), t, 1.0)) * 10.0 * sc * k          # core width (work px) at size 1 = 10 px
    a0 = float(_val(p.get("core_start"), t, 0.0)) / 100.0
    a1 = float(_val(p.get("core_end"), t, 100.0)) / 100.0
    rv, ro = p.get("reveal"), p.get("out")
    prog_in = prog_out = None
    if rv:
        prog_in = _ease(rv.get("ease", "out_expo"))((t - float(rv.get("t", 0.0))) / max(1e-3, float(rv.get("dur", 0.6))))
    if ro:
        prog_out = _ease(ro.get("ease", "in_cubic"))((t - float(ro.get("t", 0.0))) / max(1e-3, float(ro.get("dur", 0.4))))
    if kind in ("saber", "path", "shape"):
        if prog_in is not None:
            a1 = a0 + (a1 - a0) * prog_in
        if prog_out is not None:
            if p.get("out_mode", "retract") == "retract":
                a1 = a0 + (a1 - a0) * (1 - prog_out)
            else:
                a0 = a0 + (a1 - a0) * prog_out
    if kind == "saber":
        S = np.array(_val(p.get("start", [W * 0.5, H * 0.75]), t), np.float32)
        E = np.array(_val(p.get("end", [W * 0.5, H * 0.25]), t), np.float32)
        if a1 <= a0:
            return np.zeros((h, w), np.float32)
        P0, P1 = S + (E - S) * a0, S + (E - S) * a1
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = (P1 - P0) * k
        L2 = float(d @ d) + 1e-6
        u = ((xx - P0[0] * k) * d[0] + (yy - P0[1] * k) * d[1]) / L2
        uc = np.clip(u, 0, 1)
        dist = np.hypot(xx - (P0[0] * k + uc * d[0]), yy - (P0[1] * k + uc * d[1]))
        ss, es = float(p.get("start_size", 1.0)), float(p.get("end_size", 1.0))
        rad = 0.5 * size * (ss + (es - ss) * uc)
        rnd = float(p.get("roundness", 1.0))
        cap = np.where((u < 0) | (u > 1), 1.0 + (1 - rnd) * 3.0, 1.0)         # roundness 0 → blunt ends
        return np.clip(1.0 - (dist * cap - rad) / 1.2, 0, 1)
    if kind in ("path", "shape"):
        if kind == "shape":
            polys = [shape_path(p.get("shape", "circle"), float(p.get("x", W / 2)), float(p.get("y", H / 2)),
                                float(p.get("w", W * 0.5)), float(p.get("h", p.get("w", W * 0.5))),
                                float(_val(p.get("rot", 0.0), t)))]
            closed = p.get("shape", "circle") != "line"
        else:
            polys = [np.asarray(q, np.float32) for q in p.get("paths", [])]
            closed = bool(p.get("closed", True))
        off = float(_val(p.get("evolution", 0.0), t, 0.0)) / 100.0            # Mask Evolution: run along a loop (%)
        out = []
        for P in polys:
            if closed and abs(off) > 1e-6:
                seg = None
                b0, b1 = (a0 + off) % 1.0, (a0 + off) % 1.0 + (a1 - a0)
                if b1 <= 1.0:
                    seg = [_poly_trim(P, b0, b1, True)]
                else:
                    seg = [_poly_trim(P, b0, 1.0, True), _poly_trim(P, 0.0, b1 - 1.0, True)]
                out += seg
            else:
                out.append(_poly_trim(P, a0, a1, closed))
        return _stroke_mask(out, w, h, k, size, closed)
    if kind in ("text", "alpha"):
        if kind == "text":
            A = _text_mask(str(p.get("text", "SABER")), p.get("font"), float(p.get("size", 160)) * sc,
                           float(p.get("x", W / 2)), float(p.get("y", H / 2)), W, H, p.get("align", "center"))
        else:
            A = alpha if alpha is not None else np.zeros((H, W), np.float32)
        A = cv2.resize(A, (w, h), interpolation=cv2.INTER_AREA)
        if p.get("text_mode", "outline" if kind == "text" else "fill") == "outline":
            r = max(1, int(round(size * 0.5)))
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            A = np.clip(cv2.dilate(A, ker) - cv2.erode(A, ker), 0, 1)
        if prog_in is not None or prog_out is not None:                     # RTL wipe (Persian reads right → left)
            ys, xs = np.nonzero(A > 0.05)
            if len(xs):
                x0, x1 = float(xs.min()) - 4, float(xs.max()) + 4
                xx = np.arange(w, dtype=np.float32)[None, :]
                rtl = p.get("direction", "rtl") == "rtl"
                feather = max(4.0, (x1 - x0) * 0.06)
                m = np.ones((1, w), np.float32)
                if prog_in is not None:
                    edge = x1 - (x1 - x0 + feather) * prog_in if rtl else x0 + (x1 - x0 + feather) * prog_in
                    m *= np.clip((xx - edge) / feather + 0.5, 0, 1) if rtl else np.clip((edge - xx) / feather + 0.5, 0, 1)
                if prog_out is not None:
                    edge = x1 - (x1 - x0 + feather) * prog_out if rtl else x0 + (x1 - x0 + feather) * prog_out
                    m *= np.clip((edge - xx) / feather + 0.5, 0, 1) if rtl else np.clip((xx - edge) / feather + 0.5, 0, 1)
                A = A * m
        return A
    raise KeyError(f"saber core '{kind}' (saber path shape text alpha)")


def _lightning(mask, p, t, k):
    """branching bolts (midpoint displacement) between random points of the core, re-struck `rate` times / s."""
    cfg = p.get("lightning") or {}
    n = int(cfg.get("bolts", 3))
    if n <= 0:
        return mask
    h, w = mask.shape
    ys, xs = np.nonzero(mask > 0.4)
    if len(xs) < 4:
        return mask
    strike = int(math.floor(t * float(cfg.get("rate", 14))))
    rng = np.random.default_rng((int(p.get("seed", 0)) * 977 + strike) & 0x7FFFFFFF)
    bolt = np.zeros_like(mask)
    span = max(w, h) * float(cfg.get("reach", 0.35))
    jit = float(cfg.get("jitter", 0.22))
    width = max(1, int(round(float(cfg.get("width", 2.5)) * k * 2)))

    def bolt_pts(a, b, depth):
        pts = [a, b]
        disp = np.hypot(*(b - a)) * jit
        for _ in range(depth):
            new = [pts[0]]
            for i in range(len(pts) - 1):
                m = (pts[i] + pts[i + 1]) / 2
                d = pts[i + 1] - pts[i]
                nrm = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-6)
                new += [m + nrm * rng.normal() * disp, pts[i + 1]]
            pts = new
            disp *= 0.55
        return np.array(pts, np.float32)
    for _ in range(n):
        i = rng.integers(len(xs))
        a = np.array([xs[i], ys[i]], np.float32)
        if cfg.get("mode", "arc") == "arc":                         # arc between two core points
            j = rng.integers(len(xs))
            b = np.array([xs[j], ys[j]], np.float32)
            if np.hypot(*(b - a)) < span * 0.2:
                ang = rng.uniform(0, 2 * np.pi)
                b = a + np.array([math.cos(ang), math.sin(ang)]) * span * rng.uniform(0.3, 0.8)
        else:                                                       # outward discharge
            ang = rng.uniform(0, 2 * np.pi)
            b = a + np.array([math.cos(ang), math.sin(ang)]) * span * rng.uniform(0.3, 1.0)
        P = bolt_pts(a, b, 6)
        cv2.polylines(bolt, [np.round(P * 8).astype(np.int32)], False, 1.0, width, cv2.LINE_AA, shift=3)
        for _b in range(int(cfg.get("branches", 2))):
            s = P[rng.integers(len(P) // 4, max(len(P) // 4 + 1, 3 * len(P) // 4))]
            ang = rng.uniform(0, 2 * np.pi)
            e = s + np.array([math.cos(ang), math.sin(ang)]) * span * rng.uniform(0.1, 0.35)
            cv2.polylines(bolt, [np.round(bolt_pts(s, e, 4) * 8).astype(np.int32)], False, 0.7,
                          max(1, width - 1), cv2.LINE_AA, shift=3)
    return np.maximum(mask, bolt)


# =============================================================================================== render
def energy_map(spec, t, W, H, alpha=None, k=0.5):
    """(core C, glow G) float32 maps at work scale k, plus resolved params.  Exposed for tests / custom comps."""
    p = params(spec) if "_resolved" not in spec else spec
    C = core_mask(p, t, W, H, k, alpha)
    h, w = C.shape
    sc = float(p.get("scale", 1.0)) * k
    dist = p.get("distortion", "none")
    amt = float(_val(p.get("distortion_amount"), t, 0.0))
    seed = int(p.get("seed", 0))
    if p.get("lightning"):
        C = _lightning(C, p, t, k)
    soft = float(_val(p.get("core_softness"), t, 0.25))
    if soft > 0:
        C = np.maximum(C, cv2.GaussianBlur(C, (0, 0), max(0.4, soft * 6 * sc)) * 1.3).clip(0, 1)
    src = C
    if dist != "none" and amt > 0:
        q = 2                                                     # noise at ½ of work res (¼ of the frame)
        nw, nh = max(8, w // q), max(8, h // q)
        wd = math.radians(float(p.get("wind_dir", -90)))
        ws = float(p.get("wind_speed", 0.0)) * 120.0 * sc
        wind = (math.cos(wd) * ws / q, math.sin(wd) * ws / q)
        if p.get("lock_noise", False):
            wind = (0.0, 0.0)
        ns = float(p.get("noise_scale", 1.0)) * sc * 2 / q
        oc = int(p.get("noise_complexity", 3))
        spd = float(p.get("noise_speed", 1.0))
        asp = float(p.get("noise_aspect", 1.0))
        ridged = dist == "energy"
        n1 = fbm(nw, nh, t, ns, spd, oc, seed, asp, wind, ridged) + float(p.get("noise_bias", 0.0))
        if p.get("distortion_mode", "displace") == "multiply":
            mul = cv2.resize(np.clip(0.55 + 0.6 * n1, 0, 1.6), (w, h), interpolation=cv2.INTER_LINEAR)
            src = C * (1 - min(1.0, amt) + min(1.0, amt) * mul)
        else:
            n2 = fbm(nw, nh, t + 17.3, ns, spd, oc, seed + 101, asp, wind, ridged)
            if dist == "fluid":                                    # domain warp: noise of noise
                n2 = 0.6 * n2 + 0.4 * np.roll(n1, nh // 7, 0)
            mag = amt * 22.0 * sc
            DX, DY = cv2.resize(n1, (w, h)), cv2.resize(n2, (w, h))
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            src = cv2.remap(C, gx + DX * mag, gy + DY * mag, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            if dist in ("smoke", "fluid") and (abs(wind[0]) + abs(wind[1]) > 0 or dist == "smoke"):
                # plumes: the core is advected downwind in taps, each tap warped harder + fading (flame tongues)
                ux, uy = math.cos(wd), math.sin(wd)
                step = (5.0 + 7.0 * float(p.get("wind_speed", 1.0))) * sc * amt
                plume = np.zeros_like(C)
                taps = 14
                Cb = cv2.GaussianBlur(C, (0, 0), max(0.8, 2.5 * sc))
                for i in range(1, taps + 1):
                    f = i / taps
                    ox, oy = -ux * step * i, -uy * step * i
                    tap = cv2.remap(Cb if i > 2 else C, gx + ox + DX * mag * (1 + 2.5 * f), gy + oy + DY * mag * (1 + 2.5 * f),
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    plume = np.maximum(plume, tap * (1 - f) ** 1.8 * np.clip(0.7 + DY * 0.9 * f, 0, 1.3))
                src = np.maximum(src, cv2.GaussianBlur(plume, (0, 0), max(0.6, 1.5 * sc)))
            if ridged:
                fil = np.clip(cv2.resize(n1, (w, h)), 0, 1) ** 2
                halo_c = cv2.GaussianBlur(C, (0, 0), max(1.0, 6.0 * sc * (1 + amt)))
                src = np.maximum(src * np.clip(0.7 + 0.6 * fil, 0.3, 1.6), halo_c * fil * 2.2 * amt)
            cd = float(p.get("core_distortion", 0.0)) + (0.35 if dist in ("smoke", "fluid") else 0.0)
            if cd > 0:
                C = cv2.remap(C, gx + DX * mag * cd * 0.35, gy + DY * mag * cd * 0.35, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
    # ---- glow pyramid (Saber Glow 1–4) at ¼ frame res
    gi = float(_val(p.get("glow_intensity"), t, 1.0))
    sp = float(_val(p.get("glow_spread"), t, 1.0))
    fl = float(p.get("flicker_intensity", 0.0))
    if fl > 0:
        gi *= max(0.0, 1.0 + fl * flicker(t, float(p.get("flicker_speed", 1.0)), seed))
    pg = float(p.get("glow_pre_gamma", 1.0))
    s2 = cv2.resize(src, (max(4, w // 2), max(4, h // 2)), interpolation=cv2.INTER_AREA)
    if pg != 1.0:
        s2 = np.power(np.clip(s2, 0, None), pg)
    G = np.zeros_like(s2)
    for i, (ii, ss) in enumerate(p.get("glows", DEFAULTS["glows"])):
        sig = max(0.5, float(ss) * sp * 40.0 * sc / 2)
        G += cv2.GaussianBlur(s2, (0, 0), sig) * float(ii) * (1.0 + 0.6 * i)
    halo = float(p.get("halo_intensity", 0.0))
    if halo > 0:
        G += cv2.GaussianBlur(s2, (0, 0), max(1.0, 8.0 * float(p.get("halo_size", 1.0)) * sc)) * halo * 1.6
    G = cv2.resize(G, (w, h), interpolation=cv2.INTER_LINEAR) * gi
    if dist in ("smoke", "fluid") and amt > 0:
        C = np.maximum(C, np.clip(src - C, 0, 1) * 0.32)          # flame body is visible, not only its glow
    return C, G, p


def render(spec, t, W, H, alpha=None, k=0.5):
    """→ BGRA premultiplied uint8 (W×H).  composite 'black' → opaque on black."""
    C, G, p = energy_map(spec, t, W, H, alpha, k)
    col = _hex(_val(p.get("color"), t, "#2a7dff"))
    bias = float(_val(p.get("glow_bias"), t, 0.35))
    gi = max(0.2, float(p.get("glow_intensity", 1.0)))
    core_e = C * (1.6 + 2.5 * bias) * min(1.5, gi)
    sat_col = np.clip(col / (col.max() + 1e-6), 0, 1)
    E = G[..., None] * sat_col[None, None, :] * 2.2 + core_e[..., None] * (0.55 + 0.45 * sat_col[None, None, :])
    E = E + (G * bias * 0.6)[..., None]                           # bias: bright glow drifts to white
    img = 1.0 - np.exp(-E * float(p.get("brightness", 1.0)))       # filmic: white-hot core, saturated falloff
    g = float(p.get("gamma", 1.0))
    if g != 1.0:
        img = np.power(np.clip(img, 0, 1), 1.0 / g)
    s = float(p.get("saturation", 1.0))
    if s != 1.0:
        L = img.mean(axis=2, keepdims=True)
        img = np.clip(L + (img - L) * s, 0, 1)
    if p.get("ramp"):                                                # heat ramp: colour by energy (fire / plasma)
        stops = [(float(a), _hex(c)) for a, c in p["ramp"]]
        e = np.clip(1.0 - np.exp(-(G * 0.9 + core_e * 0.5)), 0, 1)
        xs = np.array([a for a, _ in stops], np.float32)
        ramp_img = np.stack([np.interp(e, xs, np.array([c[ch] for _, c in stops], np.float32)) for ch in range(3)], -1)
        img = ramp_img * np.clip(e * 1.6, 0, 1)[..., None]
    op = float(_val(p.get("opacity"), t, 1.0))
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR) * op
    out = np.empty((H, W, 4), np.uint8)
    out[..., :3] = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
    if p.get("composite") == "black":
        out[..., 3] = 255
    else:
        a = np.clip(img.max(axis=2) * float(p.get("alpha_boost", 1.0)), 0, 1)
        out[..., 3] = np.maximum(out[..., :3].max(axis=2), np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8))
    return out


def render_over(bg_bgr, spec, t, alpha=None, mode=None):
    """composite onto a BGR frame: 'add' (screen-like, default — light adds) or 'over' (premultiplied over)."""
    H, W = bg_bgr.shape[:2]
    fx = render(spec, t, W, H, alpha)
    mode = mode or spec.get("composite_mode", "add")
    if mode == "over":
        a = fx[..., 3:4].astype(np.uint16)
        return ((bg_bgr.astype(np.uint16) * (255 - a) + fx[..., :3].astype(np.uint16) * 255) // 255).astype(np.uint8)
    return cv2.add(bg_bgr, fx[..., :3])


def blit_cairo(ctx, fx):
    """paint a BGRA premultiplied array (cairo ARGB32 layout) onto a cairo context with ADD (light)."""
    import cairo
    H, W = fx.shape[:2]
    arr = np.ascontiguousarray(fx)
    s = cairo.ImageSurface.create_for_data(memoryview(arr), cairo.FORMAT_ARGB32, W, H)
    ctx.save()
    ctx.identity_matrix()
    ctx.set_operator(cairo.OPERATOR_ADD)
    ctx.set_source_surface(s, 0, 0)
    ctx.paint()
    ctx.restore()
    return arr
