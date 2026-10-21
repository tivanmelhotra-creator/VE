"""
mograph.pro.doctor — ONE command after any sandbox reset:    python3 -m mograph.pro doctor --fix

  1. python deps (cairo, harfbuzz, cv2, onnxruntime, mediapipe, numba, scipy, shapely…) → pip install
  2. big binaries kept out of git (effectcraft-cli, deep-filter, modnet.onnx) → mograph.fetch_assets
  3. fonts visible to fontconfig (~/.local/share/fonts)
  4. projects/maghale/media/raw.mp4 ← work/input/raw.mp4 (media/ is git-ignored)
  5. footage TRACKING CACHE (<video>.v14/: pose.json + matte/*.png — normally ~9 min of MODNet + MediaPipe):
     restored in seconds from the PACK committed next to the original (work/input/raw.mp4.v14pack/:
     pose.json + palette.json + matte.mkv = lossless FFV1 half-res gray video, a few MB).  No pack → prep runs.
  6. all 5 selftest suites (v14 · aefx · aecomp · motion · pro) → must all pass

Packing a new clip (after `python3 -m mograph.v14.track media/clip.mp4`):
    python3 -m mograph.pro pack-cache media/clip.mp4 [--out work/input/clip.mp4.v14pack]
"""
from __future__ import annotations

import glob
import importlib
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # sdk/mograph
SDK = os.path.dirname(HERE)
ROOT = os.path.dirname(SDK)
DEPS = {"cairo": "pycairo", "uharfbuzz": "uharfbuzz", "cv2": "opencv-python", "onnxruntime": "onnxruntime",
        "mediapipe": "mediapipe", "numba": "numba", "scipy": "scipy", "shapely": "shapely", "fontTools": "fonttools",
        "freetype": "freetype-py", "noisereduce": "noisereduce", "yaml": "pyyaml", "PIL": "pillow"}
# project video → committed pack location
PACKS = {os.path.join(ROOT, "projects", "maghale", "media", "raw.mp4"): os.path.join(ROOT, "work", "input", "raw.mp4.v14pack")}
ORIGINALS = {os.path.join(ROOT, "projects", "maghale", "media", "raw.mp4"): os.path.join(ROOT, "work", "input", "raw.mp4")}


def _ok(b):
    return "✓" if b else "✗"


# ============================================================================================== cache pack
def pack_cache(video, out=None, log=print):
    import cv2
    src = video + ".v14"
    md = os.path.join(src, "matte")
    n = int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_COUNT))
    files = [os.path.join(md, f"{i:05d}.png") for i in range(n)]
    miss = [f for f in files if not os.path.exists(f)]
    if not os.path.exists(os.path.join(src, "pose.json")) or miss:
        raise FileNotFoundError(f"tracking cache incomplete ({len(miss)} matte frames missing) — run "
                                f"`python3 -m mograph.v14.track {video}` first")
    dst = out or PACKS.get(os.path.abspath(video), video + ".v14pack")
    os.makedirs(dst, exist_ok=True)
    for f in ("pose.json", "palette.json"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    a = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    h, w = a.shape
    W2, H2 = (w // 2) // 2 * 2, (h // 2) // 2 * 2
    mkv = os.path.join(dst, "matte.mkv")
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W2}x{H2}",
                          "-r", "30", "-i", "-", "-c:v", "ffv1", "-level", "3", "-g", "1", "-pix_fmt", "gray", mkv],
                         stdin=subprocess.PIPE)
    for f in files:
        p.stdin.write(cv2.resize(cv2.imread(f, cv2.IMREAD_GRAYSCALE), (W2, H2), interpolation=cv2.INTER_AREA).tobytes())
    p.stdin.close()
    p.wait()
    with open(os.path.join(dst, "meta.txt"), "w") as f:
        f.write(f"frames {n}\nw {w}\nh {h}\n")
    sz = sum(os.path.getsize(os.path.join(dst, x)) for x in os.listdir(dst)) / 1e6
    log(f"[pack] {n} matte frames + pose → {dst} ({sz:.1f} MB)")
    return dst


def unpack_cache(video, pack=None, log=print):
    import cv2
    pk = pack or PACKS.get(os.path.abspath(video), video + ".v14pack")
    if not os.path.exists(os.path.join(pk, "matte.mkv")):
        return False
    meta = dict(l.split() for l in open(os.path.join(pk, "meta.txt")) if l.strip())
    n, w, h = int(meta["frames"]), int(meta["w"]), int(meta["h"])
    dst = video + ".v14"
    md = os.path.join(dst, "matte")
    os.makedirs(md, exist_ok=True)
    for f in ("pose.json", "palette.json"):
        if os.path.exists(os.path.join(pk, f)):
            shutil.copy2(os.path.join(pk, f), os.path.join(dst, f))
    cap = cv2.VideoCapture(os.path.join(pk, "matte.mkv"))
    i = 0
    while i < n:
        ok, fr = cap.read()
        if not ok:
            break
        g = fr[..., 0] if fr.ndim == 3 else fr
        cv2.imwrite(os.path.join(md, f"{i:05d}.png"), cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR),
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])
        i += 1
    log(f"[unpack] {i}/{n} matte frames + pose restored → {dst}")
    return i == n


def cache_complete(video):
    d = video + ".v14"
    if not os.path.exists(os.path.join(d, "pose.json")):
        return False
    import cv2
    n = int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_COUNT))
    return n > 0 and len(glob.glob(os.path.join(d, "matte", "*.png"))) >= n


# ============================================================================================== doctor
def doctor(fix=False, tests=True, log=print):
    t0 = time.time()
    bad = 0
    miss = []
    for mod, pipn in DEPS.items():
        try:
            importlib.import_module(mod)
        except Exception:  # noqa: BLE001
            miss.append(pipn)
    if miss and fix:
        log(f"… pip install {' '.join(miss)}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *miss])
        subprocess.run("sudo apt-get install -y -q libegl1 libgles2 >/dev/null 2>&1", shell=True)
        importlib.invalidate_caches()
        miss = [p for m, p in DEPS.items() if p in miss and importlib.util.find_spec(m) is None]
    log(f"{_ok(not miss)} python deps" + (f" — missing: {' '.join(miss)} (run with --fix)" if miss else ""))
    bad += bool(miss)
    try:
        from .. import fetch_assets as FA
        st = {k: os.path.exists(v["path"]) for k, v in FA.ASSETS.items()}
    except Exception:  # noqa: BLE001
        st = {}
    if not all(st.values()) and fix:
        subprocess.run([sys.executable, "-m", "mograph.fetch_assets"], cwd=SDK)
        st = {k: os.path.exists(v["path"]) for k, v in FA.ASSETS.items()}
    log(f"{_ok(all(st.values()))} big binaries  " + "  ".join(f"{k}:{_ok(v)}" for k, v in st.items()) +
        ("" if all(st.values()) else "  (optional: fallbacks are used)"))
    for b in (os.path.join(HERE, "bin", "deep-filter"), os.path.join(SDK, "bin", "effectcraft-cli")):
        if os.path.exists(b):
            os.chmod(b, 0o755)
    fd = os.path.expanduser("~/.local/share/fonts")
    if fix:
        os.makedirs(fd, exist_ok=True)
        for f in glob.glob(os.path.join(SDK, "fonts", "*.ttf")):
            if not os.path.exists(os.path.join(fd, os.path.basename(f))):
                shutil.copy2(f, fd)
    nf = len(glob.glob(os.path.join(fd, "*.ttf")))
    log(f"{_ok(nf >= 50)} fonts in fontconfig: {nf}")
    for v, orig in ORIGINALS.items():
        if not os.path.exists(v) and os.path.exists(orig) and fix:
            os.makedirs(os.path.dirname(v), exist_ok=True)
            shutil.copy2(orig, v)
        if not os.path.exists(v):
            log(f"✗ {os.path.relpath(v, ROOT)} missing (run with --fix)")
            bad += 1
            continue
        ok = cache_complete(v)
        how = "present"
        if not ok and fix:
            ok = unpack_cache(v, log=log)
            how = "restored from pack"
            if not ok:
                log(f"… no pack → running prep on {v} (~1 s/frame, do not run renders in parallel)")
                subprocess.run([sys.executable, "-m", "mograph.v14.track", v], cwd=SDK)
                ok, how = cache_complete(v), "rebuilt"
        log(f"{_ok(ok)} tracking cache {os.path.relpath(v, ROOT)} ({how if ok else 'missing — run with --fix'})")
        bad += not ok
    if tests:
        for m in ("v14", "aefx", "aecomp", "motion", "pro"):
            r = subprocess.run([sys.executable, "-m", f"mograph.{m}", "selftest"], cwd=SDK, capture_output=True, text=True)
            last = [l for l in (r.stdout + r.stderr).splitlines() if "selftest" in l and "pass" in l]
            good = r.returncode == 0 and bool(last)
            log(f"{_ok(good)} {m:<7} {last[-1].strip() if last else (r.stdout + r.stderr)[-300:]}")
            bad += not good
    log(f"doctor: {'ALL GOOD ✓' if not bad else f'{bad} problem(s)'}  ({time.time() - t0:.0f}s)")
    return bad
