"""
mograph.pro.render — PARALLEL, RESUMABLE, FRAME-EXACT master render of a v14 edit (successor of render_chunks.sh).

    python3 -m mograph.pro render edit.json [-j 2] [--chunk 4] [--preview] [--crf 17] [--start a --end b]
        → out/<name>.mp4  + out/<name>.srt / .vtt captions + out/<name>_timeline.txt
  * chunks are cut on EXACT frame indices (no 1-frame gap / duplicate at the joins)
  * each chunk renders in its own process (-j workers; capped by free RAM, ~330 MB per worker)
  * finished chunks are kept (out/chunks/<name>/cAAAAAA_BBBBBB.mp4 + .ok with the spec hash) → re-running
    resumes after a crash / sandbox reset; editing the JSON invalidates stale chunks automatically
  * audio (voice EDL + denoise + SFX + music + loudnorm) is built once, in parallel with the video
  * --preview = half resolution, crf 23 (≈ 3–4× faster) for client review
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time

SDK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mem_avail_mb():
    try:
        for l in open("/proc/meminfo"):
            if l.startswith("MemAvailable"):
                return int(l.split()[1]) / 1024
    except Exception:  # noqa: BLE001
        pass
    return 2048


def _chunk_worker(spec, out, f0, f1, scale, crf):
    """render frames [f0, f1) of spec into out (runs in a subprocess)."""
    import cv2
    from ..v14.compositor import Renderer
    R = Renderer(spec)
    W, H = int(R.W * scale) // 2 * 2, int(R.H * scale) // 2 * 2
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r",
                          str(R.fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-threads",
                          "1", "-x264-params", "rc-lookahead=10:ref=2", "-pix_fmt", "yuv420p", out],
                         stdin=subprocess.PIPE)
    for i in range(f0, f1):
        img, _ = R.render(i / R.fps)
        if scale != 1.0:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        p.stdin.write(img.tobytes())
    p.stdin.close()
    return p.wait()


def _spec_hash(spec_path):
    return hashlib.sha1(open(spec_path, "rb").read()).hexdigest()[:12]


def render(spec_path, jobs=2, chunk=4.0, preview=False, crf=None, start=None, end=None, captions=True, log=print):
    from ..v14.compositor import Renderer
    spec_path = os.path.abspath(spec_path)
    base = os.path.dirname(spec_path)
    name = os.path.splitext(os.path.basename(spec_path))[0] + ("_preview" if preview else "")
    od = os.path.join(base, "out")
    cd = os.path.join(od, "chunks", name)
    os.makedirs(cd, exist_ok=True)
    R = Renderer(spec_path)
    fps = R.fps
    total = int(round(R.duration * fps))
    a = 0 if start is None else max(0, int(round(start * fps)))
    b = total if end is None else min(int(round(end * fps)), total)
    step = max(1, int(round(chunk * fps)))
    ranges = [(f, min(b, f + step)) for f in range(a, b, step)]
    sh = _spec_hash(spec_path)
    scale, q = (0.5, crf or 23) if preview else (1.0, crf or 17)
    mem = _mem_avail_mb()
    jobs = max(1, min(int(jobs), os.cpu_count() or 1, int(mem // 330) or 1, len(ranges)))
    log(f"[pro render] {name}: {b - a} frames, {len(ranges)} chunks, {jobs} worker(s)  (RAM free {mem:.0f} MB)")
    env = dict(os.environ, PYTHONPATH=SDK + os.pathsep + os.environ.get("PYTHONPATH", ""), OMP_NUM_THREADS="1",
               OPENCV_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    wav = os.path.join(od, f"{name}.wav")
    audio_p = subprocess.Popen([sys.executable, "-c",
                                "import sys; from mograph.v14.compositor import Renderer; from mograph.v14.audio import "
                                "build_audio; build_audio(Renderer(sys.argv[1]), sys.argv[2], float(sys.argv[3]), "
                                "float(sys.argv[4]))", spec_path, wav, str(a / fps), str(b / fps)],
                               cwd=SDK, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    todo = []
    for f0, f1 in ranges:
        c = os.path.join(cd, f"c{f0:06d}_{f1:06d}.mp4")
        okf = c + ".ok"
        if os.path.exists(okf) and os.path.exists(c) and open(okf).read().strip() == f"{sh} {scale} {q}":
            continue
        todo.append((c, f0, f1))
    log(f"[pro render] {len(ranges) - len(todo)} chunk(s) reused, {len(todo)} to render")
    t0 = time.time()
    running, done, need = [], 0, sum(f1 - f0 for _, f0, f1 in todo)
    while todo or running:
        while todo and len(running) < jobs:
            c, f0, f1 = todo.pop(0)
            tmp = c + ".part.mp4"
            p = subprocess.Popen([sys.executable, "-c",
                                  "import sys; from mograph.pro.render import _chunk_worker as w; "
                                  "sys.exit(w(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), "
                                  "float(sys.argv[5]), int(sys.argv[6])))",
                                  spec_path, tmp, str(f0), str(f1), str(scale), str(q)], env=env, cwd=SDK,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            running.append((p, c, tmp, f0, f1))
        time.sleep(0.5)
        for item in list(running):
            p, c, tmp, f0, f1 = item
            if p.poll() is None:
                continue
            running.remove(item)
            err = p.stderr.read().decode(errors="ignore")
            if p.returncode != 0 or not os.path.exists(tmp):
                for q_ in running:
                    q_[0].kill()
                raise RuntimeError(f"chunk {f0}-{f1} failed:\n{err[-1500:]}")
            os.replace(tmp, c)
            with open(c + ".ok", "w") as f:
                f.write(f"{sh} {scale} {q}")
            done += f1 - f0
            el = time.time() - t0
            eta = el / max(1, done) * (need - done)
            log(f"[pro render] {done}/{need} frames  {el / max(1, done):.2f} s/frame wall  ETA {eta:.0f}s")
    lst = os.path.join(cd, "list.txt")
    with open(lst, "w") as f:
        for f0, f1 in ranges:
            f.write(f"file 'c{f0:06d}_{f1:06d}.mp4'\n")
    silent = os.path.join(od, f"{name}_silent.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", silent],
                   check=True)
    audio_p.wait()
    out = os.path.join(od, f"{name}.mp4")
    if audio_p.returncode == 0 and os.path.exists(wav):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", silent, "-i", wav, "-c:v", "copy", "-c:a", "aac", "-b:a",
                        "192k", "-shortest", "-movflags", "+faststart", out], check=True)
        os.remove(silent)
    else:
        log("[pro render] ⚠ audio build failed → silent master\n" + audio_p.stderr.read().decode(errors="ignore")[-800:])
        os.replace(silent, out)
    if captions:
        from . import export as EX
        ph = [(max(0.0, x0 - a / fps), x1 - a / fps, s) for x0, x1, s in EX.phrases(R) if x1 > a / fps and x0 < b / fps]
        EX.write_srt(ph, os.path.join(od, f"{name}.srt"))
        EX.write_vtt(ph, os.path.join(od, f"{name}.vtt"))
        with open(os.path.join(od, f"{name}_timeline.txt"), "w", encoding="utf-8") as f:
            f.write(EX.timeline(R) + "\n")
    log(f"[pro render] DONE {out}  ({time.time() - t0:.0f}s)")
    return out
