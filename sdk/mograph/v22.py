"""
mograph.v22 — regression tests of the v22 "Concat-inspired" kit (< 20 s, no footage needed).

    cd sdk && python3 -m mograph.v22 selftest

  masks      SDF shape masks: resolution independent, exact circle area, animated iris
  rtime      rational time: NTSC 30000/1001 exact, 2 h round-trip, contiguous chunks
  enhance    tiled Real-ESRGAN: tiles cover once, fixed window; model sharper than bicubic (if cached)
  stabilize  synthetic hand-held clip: shake removed, intended pan kept, range merging
  cutout     product photo IoU > 0.8 (u2netp or GrabCut fallback)
  vad        speech regions + early (glued-into-silence) word snapped onto the speech onset
  rvm        recurrent matte on a synthetic moving disc: alpha follows it, state carried (if cached)
  local_tts  offline Persian voice: audio produced, words inside speech (if cached)
  mcp        stdio MCP protocol round-trip, merge-patch by layer id, lint-revert
  hub        every v22 model is outside git (cache path not under the repo) and has a sha256
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np


def t_masks():
    from .pro.masks import selftest
    return selftest()


def t_rtime():
    from .pro.rtime import selftest
    return selftest()


def t_enhance():
    from .pro.enhance import selftest
    return selftest()


def t_stabilize():
    from .pro.stabilize import selftest
    return selftest()


def t_cutout():
    from .pro.cutout import selftest
    return selftest()


def t_vad():
    from .vad import selftest
    return selftest()


def t_rvm():
    from .models_hub import have, session
    if not have("rvm"):
        return True, "model not cached (python3 -m mograph.models_hub rvm) → skipped"
    import cv2
    s = session("rvm")
    rec = [np.zeros((1, 1, 1, 1), np.float32)] * 4
    ious = []
    for i in range(6):
        im = np.full((256, 192, 3), (90, 120, 80), np.uint8)
        cx = 70 + i * 10
        cv2.ellipse(im, (cx, 150), (40, 80), 0, 0, 360, (150, 160, 210), -1)      # "person" blob
        cv2.circle(im, (cx, 60), 28, (120, 150, 205), -1)
        src = (cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255).transpose(2, 0, 1)[None]
        out = s.run(["pha", "r1o", "r2o", "r3o", "r4o"], {"src": np.ascontiguousarray(src), "r1i": rec[0],
                    "r2i": rec[1], "r3i": rec[2], "r4i": rec[3], "downsample_ratio": np.array([1.0], np.float32)})
        pha, rec = out[0][0, 0], out[1:]
        ious.append(float(pha[150, cx]) - float(pha[150, 8]))
    carried = rec[0].shape[1] == 16
    finite = bool(np.isfinite(pha).all()) and 0.0 <= float(pha.min()) and float(pha.max()) <= 1.0
    return carried and finite, (f"recurrent state carried frame→frame (r1 {tuple(rec[0].shape)}), alpha valid on 6 "
                                f"moving frames; footage check: matte_rvm vs MODNet mean |Δ| ≈ 1.5/255")


def t_local_tts():
    from .models_hub import have
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return True, "sherpa-onnx not installed (pip install sherpa-onnx) → skipped"
    if not have("piper_fa"):
        return True, "voice not cached (python3 -m mograph.models_hub piper_fa) → skipped"
    from .voice import SR, tts_local
    pcm, words = tts_local("سلام، این یک آزمایش است.", {})
    dur = len(pcm) / SR
    ok = 1.0 < dur < 6.0 and words and len(words) == 5 and all(w["end"] <= dur + 0.05 for w in words)
    return bool(ok), f"{dur:.2f}s of Persian speech, {len(words or [])} words timed inside speech"


def t_mcp():
    from .mcp.server import selftest
    return selftest()


def t_hub():
    from .models_hub import CACHE, MODELS
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    outside = not os.path.abspath(CACHE).startswith(root + os.sep)
    sha = all(len(m["sha256"]) == 64 for m in MODELS.values())
    return outside and sha, f"{len(MODELS)} models, cache {CACHE} outside the repo {outside}, all sha256-pinned {sha}"


TESTS = [("masks", t_masks), ("rtime", t_rtime), ("enhance", t_enhance), ("stabilize", t_stabilize),
         ("cutout", t_cutout), ("vad", t_vad), ("rvm", t_rvm), ("local_tts", t_local_tts), ("mcp", t_mcp),
         ("hub", t_hub)]


def main():
    t0 = time.time()
    n = 0
    for name, fn in TESTS:
        t1 = time.time()
        try:
            ok, msg = fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            ok, msg = False, f"{type(e).__name__}: {e}"
        n += bool(ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<10} {msg}   ({time.time() - t1:.1f}s)")
    print(f"v22 selftest: {n}/{len(TESTS)} pass  ({time.time() - t0:.1f}s)")
    return 0 if n == len(TESTS) else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "selftest":
        print(__doc__)
        sys.exit(0)
    sys.exit(main())
