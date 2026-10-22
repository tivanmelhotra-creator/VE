"""
mograph.models_hub — v22 OPTIONAL ML models (downloaded on first use, sha256-verified, NEVER in git).

The repo must stay < 100 MB, so every v22 model lives in  ~/.cache/mograph_models/  (override: MOGRAPH_MODELS).
Each feature has a pure-numpy/OpenCV fallback, so nothing breaks offline.

    python3 -m mograph.models_hub --list            # status of every model
    python3 -m mograph.models_hub rvm esrgan        # fetch the named ones
    python3 -m mograph.models_hub --all             # fetch all (except opt-in: isnet 178 MB, gyro voice)

| name          | file                              |   MB | licence      | used by                                  |
|---------------|-----------------------------------|------|--------------|------------------------------------------|
| rvm           | rvm_mobilenetv3_fp32.onnx         | 15.0 | GPL-3.0      | v14.track matte engine "rvm" (temporal)   |
| esrgan        | realesr-general-x4v3.onnx         |  4.9 | BSD-3-Clause | pro.enhance (tiled, mix)                  |
| u2netp        | u2netp.onnx                       |  4.6 | Apache-2.0   | pro.cutout (objects / products, default)  |
| isnet         | isnet-general-use.onnx            |178.6 | Apache-2.0   | pro.cutout quality="high" (opt-in only)   |
| silero_vad    | silero_vad.onnx (v5.1.2)          |  2.3 | MIT          | mograph.vad (speech onsets, words.json)   |
| piper_fa      | vits-piper-fa_IR-amir-medium-int8 | 20.7 | MODEL_CARD   | voice provider "local" (offline Persian)  |
| piper_fa_gyro | vits-piper-fa_IR-gyro-medium-int8 | 21.0 | MODEL_CARD   | voice provider "local", voice "gyro"      |
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request

CACHE = os.environ.get("MOGRAPH_MODELS") or os.path.expanduser("~/.cache/mograph_models")

MODELS = {
    "rvm": {"file": "rvm_mobilenetv3_fp32.onnx", "mb": 15.0, "licence": "GPL-3.0",
            "url": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx",
            "sha256": "88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828"},
    "esrgan": {"file": "realesr-general-x4v3.onnx", "mb": 4.9, "licence": "BSD-3-Clause",
               "url": "https://huggingface.co/CoderViking/realesr-general-x4v3-onnx/resolve/"
                      "c6a971706797c7502945a2b4c4274fce4900d4ab/realesr-general-x4v3.onnx",
               "sha256": "1940a93ee08283a0a7286183186357b1688fe9fa8ede74604b424586aaddf112"},
    "u2netp": {"file": "u2netp.onnx", "mb": 4.6, "licence": "Apache-2.0",
               "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
               "sha256": "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8"},
    "isnet": {"file": "isnet-general-use.onnx", "mb": 178.6, "licence": "Apache-2.0", "optin": True,
              "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
              "sha256": "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a"},
    "silero_vad": {"file": "silero_vad.onnx", "mb": 2.3, "licence": "MIT",
                   "url": "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx",
                   "sha256": "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f"},
    "piper_fa": {"file": "vits-piper-fa_IR-amir-medium-int8", "mb": 20.7, "licence": "see MODEL_CARD", "tar": True,
                 "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
                        "vits-piper-fa_IR-amir-medium-int8.tar.bz2",
                 "sha256": "b79dbf0b6b9629a36fd541bfda83817ec12e2697557d87b918638445971eac46"},
    "piper_fa_gyro": {"file": "vits-piper-fa_IR-gyro-medium-int8", "mb": 21.0, "licence": "see MODEL_CARD",
                      "tar": True, "optin": True,
                      "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
                             "vits-piper-fa_IR-gyro-medium-int8.tar.bz2",
                      "sha256": "99bd3b912261c1a87a3f7930555622b36174b3e459e6bc153ba5fc743b931d9d"},
}

_FAILED = set()


def path(name):
    return os.path.join(CACHE, MODELS[name]["file"])


def have(name):
    p = path(name)
    return os.path.isdir(p) if MODELS[name].get("tar") else (os.path.isfile(p) and os.path.getsize(p) > 0)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ensure(name, log=print):
    """local path of model `name` (downloads + verifies once).  Raises on failure."""
    m = MODELS[name]
    p = path(name)
    if have(name):
        return p
    if name in _FAILED:
        raise RuntimeError(f"{name}: download failed earlier in this process")
    os.makedirs(CACHE, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=CACHE, prefix=".dl_")      # on disk, never /tmp (tmpfs = RAM)
    try:
        log(f"[models] downloading {name} ({m['mb']} MB, {m['licence']}) …")
        dl = os.path.join(tmp, "dl")
        req = urllib.request.Request(m["url"], headers={"User-Agent": "mograph-sdk"})
        with urllib.request.urlopen(req, timeout=180) as r, open(dl, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        if _sha(dl) != m["sha256"]:
            raise RuntimeError(f"{name}: sha256 mismatch")
        if m.get("tar"):
            with tarfile.open(dl) as t:
                if sys.version_info >= (3, 12):
                    t.extractall(tmp, filter="data")
                else:
                    t.extractall(tmp)
            os.replace(os.path.join(tmp, m["file"]), p)
        else:
            os.replace(dl, p)
        log(f"[models] {name} OK → {p}")
        return p
    except Exception:
        _FAILED.add(name)
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def try_ensure(name, log=print):
    """like ensure() but None when offline / failed (callers then use their fallback)."""
    if os.environ.get("MOGRAPH_OFFLINE") and not have(name):
        return None
    try:
        return ensure(name, log=log)
    except Exception as e:  # noqa: BLE001
        log(f"[models] {name} unavailable ({e}) → fallback")
        return None


_SESS = {}


def session(name, threads=2, log=print):
    """cached onnxruntime session (CPU, ≤ 2 threads — the sandbox has 2 cores / 1 GB) or None."""
    if name in _SESS:
        return _SESS[name]
    p = try_ensure(name, log=log)
    s = None
    if p:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = int(os.environ.get("MOGRAPH_ORT_THREADS", threads))
        so.inter_op_num_threads = 1
        s = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
    _SESS[name] = s
    return s


def status():
    return [(n, have(n), m["mb"], m["licence"], bool(m.get("optin"))) for n, m in MODELS.items()]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or "--list" in argv:
        print(f"model cache: {CACHE}   (outside the repo — git size unaffected)")
        for n, ok, mb, lic, optin in status():
            print(f"  {'✓' if ok else '·'} {n:<14} {mb:6.1f} MB  {lic:<14} {'(opt-in)' if optin else ''}")
        return 0
    names = [n for n, m in MODELS.items() if not m.get("optin")] if "--all" in argv else argv
    bad = 0
    for n in names:
        if n not in MODELS:
            print(f"unknown model '{n}' ({', '.join(MODELS)})")
            bad += 1
            continue
        bad += try_ensure(n) is None
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
