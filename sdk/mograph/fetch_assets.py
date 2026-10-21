"""Large binaries kept OUT of git (repo must stay < 100 MB) — downloaded once on demand, sha256-verified.

    python3 -m mograph.fetch_assets          # fetch all (also run by tools/setup_env.sh)
    python3 -m mograph.fetch_assets --check  # show status only

Every file is byte-identical to the copy that used to be committed (sha256 below).
Archived deliverables of v14.2 (old SDK zip, final/showcase mp4) live in the GitHub Release
`v14.2-deliverables` — see `python3 -m mograph.fetch_assets --deliverables`.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

_OFFLINE_DETECTED = False

HERE = os.path.dirname(os.path.abspath(__file__))
SDK = os.path.dirname(HERE)
ROOT = os.path.dirname(SDK)

ASSETS = {
    "effectcraft-cli": {   # EffectCraft 0.11.1 headless (photoreal 3D, pbr3d.py) — 75 MB
        "path": os.path.join(SDK, "bin", "effectcraft-cli"),
        "url": "https://github.com/storytold/effectcraft/releases/download/v0.11.1/effectcraft-0.11.1-linux-x86_64.tar.gz",
        "member": "bin/effectcraft-cli",
        "sha256": "fe78d8df0221d06d5519e02f3c382c1dfcd3241217a0e46644b4598a215b8d61",
        "exec": True,
    },
    "deep-filter": {       # DeepFilterNet3 0.5.6 static binary, model embedded (v14 denoise) — 36 MB
        "path": os.path.join(HERE, "bin", "deep-filter"),
        "url": "https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/deep-filter-0.5.6-x86_64-unknown-linux-musl",
        "sha256": "70775e251eee44c0f2451a1e833326cf8bcbbe304d3e7cd12851e6fce72ef7da",
        "exec": True,
    },
    "modnet.onnx": {       # MODNet portrait matting (v14 track) — 25 MB, Apache-2.0
        "path": os.path.join(HERE, "models", "modnet.onnx"),
        "url": "https://huggingface.co/Xenova/modnet/resolve/main/onnx/model.onnx",
        "sha256": "07c308cf0fc7e6e8b2065a12ed7fc07e1de8febb7dc7839d7b7f15dd66584df9",
    },
}

MIRROR = "https://github.com/tivanmelhotra-creator/VE/releases/download/assets-mirror/"   # raw-file fallback
RELEASE = "https://github.com/tivanmelhotra-creator/VE/releases/download/v14.2-deliverables/"
DELIVERABLES = {   # optional archives, NOT needed by any code
    "mograph_v14_2_sdk.zip": "d811d40347e41a14bd20e648e3ee43e9aaf67c7a8cbb90e0b95e6dd6fafef1e6",
    "maghale_v14_2_final.mp4": "f5b966889d8e49b70c9f0a6976a079b91b1cacbc856f14df650d04d337c71dbd",
    "showcase_v14_2.mp4": "4bf7604ba2b1493dc7a7ccf21686df5d652bd3726985b57d5799426f7960aaa8",
}


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _get(url, dst):
    global _OFFLINE_DETECTED
    if _OFFLINE_DETECTED:
        raise RuntimeError("offline sandbox")
    # disk temp dir next to the target (NEVER /tmp: tmpfs = RAM in the sandbox)
    req = urllib.request.Request(url, headers={"User-Agent": "mograph-sdk"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)


def ok(name):
    a = ASSETS[name]
    return os.path.isfile(a["path"]) and os.path.getsize(a["path"]) > 0


def ensure(name, log=print, verify=True):
    """Return the local path of asset `name`, downloading + verifying it the first time."""
    a = ASSETS[name]
    p = a["path"]
    if ok(name):
        if a.get("exec") and not os.access(p, os.X_OK):
            os.chmod(p, 0o755)
        return p
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = tempfile.mkdtemp(dir=os.path.dirname(p), prefix=".dl_")
    try:
        log(f"[assets] downloading {name} (one time) …")
        dl = os.path.join(tmp, "dl")
        try:
            _get(a["url"], dl)
            mirror = False
        except Exception as e:
            _OFFLINE_DETECTED = True  # upstream gone -> mirror
            log(f"[assets] upstream failed ({e}) → mirror")
            _get(MIRROR + name, dl)
            mirror = True
        if a.get("member") and not mirror:
            with tarfile.open(dl) as t:
                m = next(x for x in t.getmembers() if x.name.endswith(a["member"]))
                with t.extractfile(m) as src, open(dl + ".x", "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
            dl += ".x"
        if verify and _sha(dl) != a["sha256"]:
            raise RuntimeError(f"[assets] sha256 mismatch for {name} ({a['url']})")
        os.replace(dl, p)
        if a.get("exec"):
            os.chmod(p, 0o755)
        log(f"[assets] {name} OK → {os.path.relpath(p, ROOT)}")
        return p
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def try_ensure(name, log=print):
    """Like ensure() but returns None instead of raising (offline → caller uses its fallback)."""
    try:
        return ensure(name, log=log)
    except Exception as e:  # noqa: BLE001
        log(f"[assets] {name} unavailable ({e}) → fallback")
        return None


def deliverables(dst=None):
    dst = dst or os.path.join(ROOT, "work", "deliver")
    os.makedirs(dst, exist_ok=True)
    for fn, sha in DELIVERABLES.items():
        p = os.path.join(dst, fn)
        if os.path.isfile(p) and _sha(p) == sha:
            continue
        print(f"[assets] {fn} …")
        subprocess.run(["curl", "-fL", "-o", p, RELEASE + fn], check=True)
        assert _sha(p) == sha, f"sha256 mismatch: {fn}"
    print(f"[assets] deliverables in {dst}")


if __name__ == "__main__":
    if "--deliverables" in sys.argv:
        deliverables()
    elif "--check" in sys.argv:
        for n, a in ASSETS.items():
            print(f"{'OK ' if ok(n) else '-- '} {n:16s} {os.path.relpath(a['path'], ROOT)}")
    else:
        for n in ASSETS:
            ensure(n)
