"""Birthday v3 — the professional pass of the vector pipeline (v28).

What is new vs. birthday_vector.py:
  * physical spring easing (Remotion spring() formula) + anticipation on every action
  * secondary motion: hair / bun / collar lag and settle (springs driven by the puppet)
  * idle life: natural blink rhythm, breathing, faint wind on the hair
  * ink-pen tapered strokes, slightly lighter global ink weight
  * 3-beat camera: wide establishing -> push in on the two faces -> ease out to the title
  * film finish: vignette + soft grain;  parallel render on all cores

  PYTHONPATH=sdk python3 sdk/examples/vecanim/birthday_v3.py OUT_DIR [--preview] [--jobs N]
"""
import sys, os, time, json, subprocess, numpy as np, cv2
sys.path.insert(0, __file__.rsplit('/examples/', 1)[0])
from mograph.vecanim.trace import trace_panel
from mograph.vecanim.scene import Scene
from mograph.vecanim.rig import Rig
from mograph.vecanim import anim as A, audio, props

ROOT = __file__.rsplit('/sdk/', 1)[0]
args = [a for a in sys.argv[1:] if not a.startswith('--')]
OUT = args[0] if args else '/tmp/vecanim_v3'
JOBS = int(sys.argv[sys.argv.index('--jobs') + 1]) if '--jobs' in sys.argv else (os.cpu_count() or 2)
os.makedirs(OUT, exist_ok=True)
WORK = os.environ.get('VEC_WORK', '/home/user/vec_work'); os.makedirs(WORK, exist_ok=True)
FPS, T, OW, OH = 30, 12.0, 1920, 1080

sp = os.path.join(OUT, 'scene.json')
if not os.path.exists(sp):
    panel = cv2.imread(f'{ROOT}/work/vecanim/panel_rebuilt.png', 0).astype(np.float32) / 255
    sc, _ = trace_panel(panel); sc.save(sp)
scene = Scene.load(sp)
Scene.line_scale = 0.88

wav = os.path.join(OUT, 'music.wav')
audio.music_box_score(wav, T + 0.6, bpm=78, start=0.8)
x, sr = audio.read_wav(wav)
hits = audio.detect_transients(x, sr, 0.3, min_gap=0.15)
B = lambda t: A.snap(t, hits, 0.3)


def build():
    rig = Rig.load(f'{ROOT}/sdk/examples/vecanim/birthday_rig.json', scene)
    boy, girl = rig['boy'], rig['girl']
    # ================= performance =================
    boy.do('breathe', t=0, period=3.6, amount=1.3)
    girl.do('breathe', t=0.4, period=3.2, amount=1.1)
    boy.do('idle_blinks', t=2.4, seed=3, until=T)
    girl.do('idle_blinks', t=2.4, seed=8, until=T)
    boy.do('tilt_head', t=B(3.0), dur=1.3, angle=3.8)
    boy.do('smile', t=B(3.2), dur=0.9)
    boy.do('hand_on_shoulder', target=girl, t=B(4.2), dur=1.6)
    girl.do('turn_head', t=B(5.6), dur=1.4, from_=7, angle=-1.4)
    girl.do('look', t=5.4, dur=1.1, from_=(2.5, 3.0))
    girl.do('smile', t=B(6.3), dur=1.1, from_=-0.9, amount=0.0)
    boy.do('nod', t=B(7.4), dur=0.8, angle=1.6)
    girl.do('sway', t=7.5, amp=0.6, freq=0.28)
    # ===============================================
    rig.fx.append(props.flame((1345, 555, 1378, 602)))
    return rig


rig = build()
TITLE = [props.text_fill('Happy Birthday', 66, 1330, 335), props.text_fill('my little sister', 50, 1385, 400)]
TX0, TX1, T_TITLE = 1320, 1830, 8.6
base = rig.items_at(0.0)
NB = len(base)
starts = props.draw_on_schedule(base, np.array([0.0, 0.83 * scene.H]), t0=0.2, spread=2.4)
W, H = scene.W, scene.H
cam = props.Camera(W, H, OW, OH, [
    (0.0, W * 0.47, H * 0.46, 1.00),
    (2.9, W * 0.47, H * 0.46, 1.03, 'io'),
    (5.0, 840, 330, 1.62, 'spring_firm'),      # push in on the faces as the hand lands
    (7.9, 860, 335, 1.70, 'io'),               # slow creep while she turns and smiles
    (9.6, W * 0.55, H * 0.45, 1.08, 'io'),     # ease out to reveal the title
    (12.0, W * 0.55, H * 0.45, 1.11, 'io')], handheld=0.7)


def items_at(t):
    its = rig.items_at(t)
    if t < 3.2:
        props.apply_draw_on(its[:NB], starts, t)
    u = A.EASE['smooth']((t - T_TITLE) / 1.4) if t > T_TITLE else 0.0
    if u > 0:
        wx = TX0 + (TX1 - TX0) * u
        for f in TITLE:
            g = f.copy(); g.opacity = 0.93
            g.clip = [(TX0 - 10, 200), (wx, 200), (wx + 25, 450), (TX0 - 10, 450)]
            its.append(g)
    return its


def frame(t):
    M = cam(t)
    a = scene.render(out_wh=(OW, OH), M=M, items=items_at(t))
    if t >= 2.0:
        fl = 1 + 0.07 * np.sin(t * 23) + 0.05 * np.sin(t * 37 + 1)
        p = M @ np.array([1360, 588, 1.0])
        g = np.zeros((OH // 4, OW // 4), np.float32)
        r = max(4, int(15 * M[0, 0] / (OW / W)))
        cv2.circle(g, (int(p[0] / 4), int(p[1] / 4)), r, 1, -1)
        a = a + cv2.resize(cv2.GaussianBlur(g, (0, 0), r * 0.7), (OW, OH)) * 0.10 * fl * A.EASE['smooth'](t - 2)
    fade = min(1.0, t / 0.4, (T - t) / 0.6)
    a = props.finish(a, t) * fade
    return np.clip(a * 0.94 * 255 + 9 * fade, 0, 255).astype(np.uint8)


def render_range(job):
    """Worker: frames [f0, f1) -> raw gray file.  Secondary springs are integrated from the
    chunk's start minus a 1.5 s pre-roll so every chunk sees the same spring state."""
    f0, f1, path = job
    global rig, base
    rig = build()
    pre = max(0, f0 - int(1.5 * FPS))
    for f in range(pre, f0):
        rig.items_at(f / FPS)
    enc = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
                            '-s', f'{OW}x{OH}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                            '-crf', '17', '-preset', 'slow', '-tune', 'animation', path], stdin=subprocess.PIPE)
    for f in range(f0, f1):
        enc.stdin.write(frame(f / FPS).tobytes())
    enc.stdin.close(); enc.wait()
    return path


if __name__ == '__main__':
    print('hits', np.round(hits, 2))
    if '--preview' in sys.argv:
        for t in [1.6, 3.6, 4.9, 5.9, 7.2, 9.0, 11.0]:
            for f in range(max(0, int((t - 1.5) * FPS)), int(t * FPS)):
                rig.items_at(f / FPS)
            cv2.imwrite(os.path.join(OUT, f'pv_{t:04.1f}.png'), frame(t))
        print('preview ok'); sys.exit()
    from multiprocessing import Pool
    n = int(T * FPS)
    cuts = np.linspace(0, n, JOBS + 1).astype(int)
    jobs = [(int(a), int(b), os.path.join(WORK, f'chunk{i}.mp4')) for i, (a, b) in enumerate(zip(cuts, cuts[1:]))]
    t0 = time.time()
    with Pool(JOBS) as pool:
        parts = pool.map(render_range, jobs)
    rt = time.time() - t0
    out = os.path.join(OUT, 'birthday_v3.mp4')
    lst = os.path.join(WORK, 'list.txt')
    open(lst, 'w').write(''.join(f"file '{p}'\n" for p in parts))
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-i', wav,
                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', out], check=True)
    for p in parts:
        os.remove(p)
    info = dict(frames=n, jobs=JOBS, render_s=round(rt, 1), ms_per_frame=round(rt / n * 1000 * JOBS, 1))
    json.dump(info, open(os.path.join(OUT, 'timing.json'), 'w'))
    print('ok', out, info)
