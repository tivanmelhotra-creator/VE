"""Birthday panel rebuilt with the VECTOR pipeline (v27 vecanim) — no video model, no pixel warps.

  trace (once) -> rig JSON (once) -> the acting block below -> MP4 (+music) + SVG + Lottie

  PYTHONPATH=sdk python3 sdk/examples/vecanim/birthday_vector.py OUT_DIR [--preview] [--no-lottie]
"""
import sys, os, time, json, subprocess, numpy as np, cv2
sys.path.insert(0, __file__.rsplit('/examples/', 1)[0])
from mograph.vecanim.trace import trace_panel
from mograph.vecanim.scene import Scene
from mograph.vecanim.rig import Rig
from mograph.vecanim import anim as A, audio, lottie, props

ROOT = __file__.rsplit('/sdk/', 1)[0]
args = [a for a in sys.argv[1:] if not a.startswith('--')]
OUT = args[0] if args else '/tmp/vecanim_out'
os.makedirs(OUT, exist_ok=True)
FPS, T, OW, OH = 30, 9.0, 1920, 1080
timing = {}

# 1. trace once (cached) ------------------------------------------------------------
t0 = time.time()
sp = os.path.join(OUT, 'scene.json')
if not os.path.exists(sp):
    panel = cv2.imread(f'{ROOT}/work/vecanim/panel_rebuilt.png', 0).astype(np.float32) / 255
    sc, _ = trace_panel(panel); sc.save(sp); sc.to_svg(os.path.join(OUT, 'scene.svg'))
scene = Scene.load(sp)
timing['trace_s'] = round(time.time() - t0, 2)

# 2. music + hits ------------------------------------------------------------------
wav = os.path.join(OUT, 'music.wav')
audio.music_box_score(wav, T + 0.5)
x, sr = audio.read_wav(wav)
hits = audio.detect_transients(x, sr, 0.3, min_gap=0.15)
B = lambda t: A.snap(t, hits, 0.25)

# 3. rig once ----------------------------------------------------------------------
t0 = time.time()
rig = Rig.load(f'{ROOT}/sdk/examples/vecanim/birthday_rig.json', scene)
boy, girl = rig['boy'], rig['girl']
timing['rig_s'] = round(time.time() - t0, 2)

# ============ the performance: this block is all a new shot costs ============
boy.do('breathe', t=0, period=3.4, amount=1.2)
girl.do('breathe', t=0.5, period=3.1, amount=1.0)
boy.do('tilt_head', t=B(2.7), dur=1.2, angle=3.5)
boy.do('smile', t=B(2.7), dur=0.8)
boy.do('hand_on_shoulder', target=girl, t=B(3.45), dur=1.5)
girl.do('turn_head', t=B(4.87), dur=1.2, from_=7, angle=-1.2)
girl.do('look', t=4.6, dur=1.0, from_=(2.5, 3.0))
girl.do('smile', t=B(5.41), dur=1.0, from_=-0.9, amount=0.0)
girl.do('blink', t=B(7.01))
boy.do('blink', t=B(7.73))
girl.do('sway', t=6.5, amp=0.5, freq=0.3)
# =============================================================================

rig.fx.append(props.flame((1345, 555, 1378, 602)))
TITLE = [props.text_fill('Happy Birthday', 66, 1330, 335), props.text_fill('my little sister', 50, 1385, 400)]
TX0, TX1 = 1320, 1830
base = rig.items_at(0.0)
NB = len(base)
starts = props.draw_on_schedule(base, np.array([0.0, 0.83 * scene.H]))
camera = props.push_camera(scene.W, scene.H, OW, OH, T)


def items_at(t, title_clip=True):
    its = rig.items_at(t)
    if t < 2.9:
        props.apply_draw_on(its[:NB], starts, t)
    u = A.EASE['smooth']((t - 6.2) / 1.3) if t > 6.2 else 0.0
    for f in TITLE:
        g = f.copy(); g.opacity = 0.92 if u > 0 else 0.0
        if title_clip and u > 0:
            wx = TX0 + (TX1 - TX0) * u
            g.clip = [(TX0 - 10, 200), (wx, 200), (wx + 25, 450), (TX0 - 10, 450)]
        its.append(g)
    return its


def frame(t, its=None):
    M = camera(t)
    a = scene.render(out_wh=(OW, OH), M=M, items=its if its is not None else items_at(t))
    if t >= 2.0:                                   # candle glow
        fl = 1 + 0.07 * np.sin(t * 23) + 0.05 * np.sin(t * 37 + 1)
        p = M @ np.array([1360, 588, 1.0])
        g = np.zeros((OH // 4, OW // 4), np.float32)
        cv2.circle(g, (int(p[0] / 4), int(p[1] / 4)), 15, 1, -1)
        a = np.clip(a + cv2.resize(cv2.GaussianBlur(g, (0, 0), 10), (OW, OH)) * 0.10 * fl * A.EASE['smooth'](t - 2), 0, 1)
    return np.clip(a * 0.94 * 255 + 9, 0, 255).astype(np.uint8)


if __name__ == '__main__':
    print('hits', np.round(hits, 2))
    if '--preview' in sys.argv:
        for t in [1.2, 3.0, 4.0, 4.6, 5.6, 7.05, 8.5]:
            cv2.imwrite(os.path.join(OUT, f'pv_{t:.2f}.png'), frame(t))
        print('preview ok', timing); sys.exit()
    n = int(T * FPS)
    silent = os.path.join(OUT, 'silent.mp4')
    p = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{OW}x{OH}',
                          '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', silent],
                         stdin=subprocess.PIPE)
    tp = tr = 0.0; t0 = time.time()
    for f in range(n):
        a0 = time.time(); its = items_at(f / FPS); tp += time.time() - a0
        a0 = time.time(); p.stdin.write(frame(f / FPS, its).tobytes()); tr += time.time() - a0
    p.stdin.close(); p.wait()
    timing.update(render_s=round(time.time() - t0, 1), pose_ms=round(tp / n * 1000, 1), draw_ms=round(tr / n * 1000, 1))
    out = os.path.join(OUT, 'birthday_vector.mp4')
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', silent, '-i', wav, '-c:v', 'copy', '-c:a', 'aac',
                    '-b:a', '160k', '-shortest', out], check=True)
    os.remove(silent)
    Scene(scene.W, scene.H, scene.bg, items_at(8.5, False)).to_svg(os.path.join(OUT, 'final_pose.svg'))
    if '--no-lottie' not in sys.argv:
        t0 = time.time()
        frames = list(range(0, n, 3)) + [n - 1]
        per = [items_at(f / FPS, False) for f in frames]
        hand_ref = next(fi[NB:] for fi in per[::-1] if any(i.group.startswith('hand') for i in fi[NB:]))
        hand_ref = [i for i in hand_ref if i.group.startswith('hand')]
        W = lottie.LottieWriter(scene.W, scene.H, FPS, n, bg=scene.bg, name='birthday_vector')
        for nm in ('static', 'boy', 'girl'):
            idx = [i for i, it in enumerate(base) if (it.part or 'static') == nm]
            W.add_layer(nm, [[fi[i] for i in idx] for fi in per], frames)
        hands = []
        for fi in per:
            h = [i for i in fi[NB:] if i.group.startswith('hand')]
            if not h:
                h = [r.copy() for r in hand_ref]
                for r in h:
                    r.opacity = 0.0
            for r in h:
                r.clip = None
            hands.append(h)
        line = girl.L['shoulder_far_line']
        clip = [(a, b - 4) for a, b in line] + [(line[-1][0], line[-1][1] + 150), (line[0][0], line[0][1] + 150)]
        W.add_layer('hand', hands, frames, mask=clip)
        W.add_layer('title', [[i for i in fi if i.group == 'title'] for fi in per], frames)
        lp = os.path.join(OUT, 'birthday_vector.lottie.json')
        doc = W.save(lp, frames, [camera(f / FPS) for f in frames], out_wh=(OW, OH))
        timing['lottie_s'] = round(time.time() - t0, 1)
        timing['lottie'] = lottie.validate(doc); timing['lottie_kb'] = os.path.getsize(lp) // 1024
    json.dump(timing, open(os.path.join(OUT, 'timing.json'), 'w'), indent=1)
    print('ok', out, timing)
