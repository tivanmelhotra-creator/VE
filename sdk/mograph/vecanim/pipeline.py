"""One-command pipeline: line-art panel + shot JSON -> MP4 (+music) + SVG + Lottie.

    python3 -m mograph.vecanim make PANEL.png --shot duo_affection -o OUT [--preview] [--lottie]

Stages (each cached by content hash under OUT/cache, so re-runs only redo what changed):
  1. load_panel   — grey 0..1, auto-inverted to "light lines on dark paper", normalised
  2. trace        — VectorCraft-port tracer -> scene.json
  3. auto_rig     — characters / landmarks / pins / secondary -> rig.json (+ rig_debug.png).
                    A hand-edited rig.json is never overwritten (use --rerig to regenerate).
  4. shot         — template or JSON: actions on actor aliases (left/right/back/front/charN),
                    times in seconds or "beat:N", snapped to the music transients
  5. render       — N worker processes, each re-rigs + pre-rolls springs, encodes an x264
                    chunk; ffmpeg concat + music; contact sheet, final-pose SVG, timing.json
  6. lottie       — optional AE/lottie-web export, validated
"""
import os, sys, json, time, hashlib, subprocess, numpy as np, cv2
from .trace import trace_panel
from .scene import Scene
from .rig import Rig
from .autorig import auto_rig, _ink_mask
from . import anim as A, audio, props

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = os.path.join(HERE, 'templates')
WORK = os.environ.get('VEC_WORK', '/home/user/vec_work')


def log(*a):
    print('[vecanim]', *a, flush=True)


# ------------------------------------------------------------------ 1-3 prepare
def load_panel(path, max_side=2400):
    g = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if g is None:
        raise FileNotFoundError(path)
    if g.ndim == 3:
        if g.shape[2] == 4:                                   # transparent PNG -> on white
            a = g[..., 3:4].astype(np.float32) / 255
            g = (g[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    a = g.astype(np.float32) / 255
    s = max_side / max(a.shape)
    if s < 1:
        a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if np.median(a) > 0.5:                                    # dark ink on white paper
        a = 1 - a
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.7)
    return np.clip((a - lo) / max(hi - lo, 1e-3), 0, 1).astype(np.float32)


def _hash(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
    return h.hexdigest()[:12]


def prepare(panel_path, out, rerig=False):
    """-> dict(panel=..., scene=path, rig=path, cache=dir).  Cached by panel content."""
    raw = open(panel_path, 'rb').read()
    cache = os.path.join(out, 'cache', _hash(raw))
    os.makedirs(cache, exist_ok=True)
    np_path = os.path.join(cache, 'panel.npy')
    if not os.path.exists(np_path):
        np.save(np_path, load_panel(panel_path))
    panel = np.load(np_path)
    sp = os.path.join(cache, 'scene.json')
    t0 = time.time()
    if not os.path.exists(sp):
        sc, _ = trace_panel(panel, verbose=False)
        sc.save(sp)
        log(f'trace: {len(sc.items)} items in {time.time() - t0:.1f}s')
    else:
        log('trace: cached')
    rp = os.path.join(cache, 'rig.json')
    if rerig or not os.path.exists(rp):
        t0 = time.time()
        rig = auto_rig(Scene.load(sp), panel, debug=os.path.join(cache, 'rig_debug.png'))
        json.dump(rig, open(rp, 'w'), indent=1)
        log(f"auto_rig: {len(rig['characters'])} characters in {time.time() - t0:.1f}s -> {rp}")
    else:
        log('rig: cached (edit rig.json freely; --rerig regenerates)')
    return dict(panel=np_path, scene=sp, rig=rp, cache=cache)


# ------------------------------------------------------------------ 4 shot
def load_shot(name_or_path):
    p = name_or_path
    if not os.path.exists(p):
        p = os.path.join(TEMPLATES, name_or_path.replace('.json', '') + '.json')
    shot = json.load(open(p))
    shot['_path'] = os.path.abspath(p)
    return shot


def templates():
    return sorted(f[:-5] for f in os.listdir(TEMPLATES) if f.endswith('.json'))


def actor_aliases(rig):
    ch = list(rig.chars.values())
    al = {c.name: c for c in ch}
    by_x = sorted(ch, key=lambda c: c.L['neck'][0])
    by_z = sorted(ch, key=lambda c: c.z)
    al.update(left=by_x[0], right=by_x[-1], back=by_z[0], front=by_z[-1], A=by_x[0], B=by_x[-1] if len(ch) > 1 else by_x[0])
    al['main'] = max(ch, key=lambda c: c.spec.get('face_w', 0))
    return al


def detect_flame(scene, rig):
    """Candle flame = a small cluster of static items (fills or strokes, taller than wide)
    sitting on top of a thin vertical static stroke (wick / candle stick)."""
    W, H = scene.W, scene.H
    boxes = []
    for it in rig.static:
        P = it.points()
        if not len(P):
            continue
        boxes.append((P.min(0), P.max(0), it))
    small = [(lo, hi) for lo, hi, it in boxes if max(hi - lo) < 0.03 * W and (hi - lo)[1] >= 0.7 * (hi - lo)[0]]
    # cluster small boxes
    clusters = []
    for lo, hi in small:
        for c in clusters:
            if lo[0] < c[1][0] + 8 and hi[0] > c[0][0] - 8 and lo[1] < c[1][1] + 8 and hi[1] > c[0][1] - 8:
                c[0] = np.minimum(c[0], lo); c[1] = np.maximum(c[1], hi); break
        else:
            clusters.append([lo.copy(), hi.copy()])
    best = None
    for lo, hi in clusters:
        w, h = hi - lo
        if not (8 < h < 0.1 * H and 1.1 < h / max(w, 1) < 4.5):
            continue
        cx = 0.5 * (lo[0] + hi[0])
        for slo, shi, it in boxes:                       # stick directly below?
            sw, sh = shi - slo
            if it.kind == 'stroke' and sh > 2.0 * max(sw, 3) and sh > 0.5 * h and abs(0.5 * (slo[0] + shi[0]) - cx) < max(w, 10) \
                    and -6 < slo[1] - hi[1] < 30:
                if best is None or h > best[0]:
                    best = (h, (float(lo[0] - 4), float(lo[1] - 6), float(hi[0] + 4), float(hi[1] + 2)))
    return best[1] if best else None


def title_anchor(scene, rig, lines, sizes, font='Dancing Script'):
    """Emptiest region big enough for the title (prefers upper half, away from faces)."""
    ink = _ink_mask(scene).astype(np.float32)
    fills = [props.text_fill(t, s, 0, 0, font) for t, s in zip(lines, sizes)]
    ws = [np.ptp(f.points()[:, 0]) for f in fills]
    tw, th = max(ws) + 40, sum(sizes) * 1.25 + 30
    H, W = ink.shape
    tw, th = min(tw, W - 4), min(th, H - 4)
    ii = cv2.integral(ink)
    best = None
    ys, xs = np.nonzero(ink > 0)
    cx, cy = (xs.mean(), ys.mean()) if len(xs) else (W / 2, H / 2)
    m = 0.04 * W
    for y in np.linspace(m * 0.6, max(m * 0.6, H - th - m * 0.6), 24):
        for x in np.linspace(m, max(m, W - tw - m), 40):
            x0, y0, x1, y1 = int(x), int(y), int(x + tw), int(y + th)
            s = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
            d = np.hypot((x + tw / 2 - cx) / W, (y + th / 2 - cy) / H)
            cost = s / (tw * th) + 0.08 * (y / H) + 0.05 * d
            if best is None or cost < best[0]:
                best = (cost, x0, y0)
    _, x0, y0 = best
    out, y = [], y0 + sizes[0]
    for t, s, w in zip(lines, sizes, ws):
        out.append(props.text_fill(t, s, x0 + 20 + (max(ws) - w) / 2, y, font))
        y += s * 1.25
    return out, (x0, y0, x0 + tw, y0 + th)


def make_camera(spec, scene, rig, T, OW, OH, title_box, focus_t):
    W, H = scene.W, scene.H
    if spec not in (None, 'auto') and isinstance(spec, dict) and 'keys' in spec:
        keys = []
        for k in spec['keys']:
            if k[1] == 'faces':
                cx, cy, z = _faces_view(rig, W, H, k[2] if len(k) > 2 else 1.6, OW, OH)
                keys.append((k[0], cx, cy, z) + tuple(k[3:]))
            else:
                keys.append(tuple(k))
        return props.Camera(W, H, OW, OH, keys, handheld=spec.get('handheld', 0.7))
    ink = _ink_mask(scene)
    ys, xs = np.nonzero(ink)
    ib = (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else (0, 0, W, H)
    wide = _fit(ib, W, H, OW, OH, pad=1.06, zmin=0.9, zmax=1.3)
    fx, fy, fz = _faces_view(rig, W, H, OW=OW, OH=OH)
    if title_box is not None:
        ub = (min(ib[0], title_box[0]), min(ib[1], title_box[1]), max(ib[2], title_box[2]), max(ib[3], title_box[3]))
        ox, oy, oz = _fit(ub, W, H, OW, OH, pad=1.08, zmin=0.9, zmax=1.3)
    else:
        ox, oy, oz = wide
    fit = oz
    f0 = max(0.25 * T, focus_t - 1.4)
    keys = [(0.0,) + tuple(wide), (f0, wide[0], wide[1], wide[2] * 1.03, 'io'),
            (min(0.65 * T, f0 + 2.1), fx, fy, fz, 'spring_firm'),
            (0.67 * T, fx + 15, fy + 4, fz * 1.05, 'io'),
            (0.8 * T, ox, oy, fit, 'io'), (T, ox, oy, fit * 1.03, 'io')]
    keys = sorted(keys, key=lambda k: k[0])
    return props.Camera(W, H, OW, OH, keys, handheld=0.7)


def _fit(box, W, H, OW, OH, pad=1.12, zmin=0.9, zmax=1.8):
    """Camera (cx, cy, zoom) that contains box=(x0,y0,x1,y1); zoom 1 = full panel width."""
    x0, y0, x1, y1 = box
    bw, bh = (x1 - x0) * pad, (y1 - y0) * pad
    z = min(W / max(bw, 1), W * OH / OW / max(bh, 1))
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1), float(np.clip(z, zmin, zmax))


def _faces_view(rig, W, H, zmax=1.5, OW=1920, OH=1080):
    pts = []
    for c in rig.chars.values():
        n = np.array(c.L['neck'], float); fw = c.spec.get('face_w', 100)
        pts += [n + [-1.6 * fw, -2.1 * fw], n + [1.6 * fw, 1.0 * fw]]
    pts = np.array(pts)
    lo, hi = pts.min(0), pts.max(0)
    return _fit((lo[0], lo[1], hi[0], hi[1]), W, H, OW, OH, pad=1.05, zmin=1.1, zmax=zmax)


# ------------------------------------------------------------------ runtime
class ShotRuntime:
    """Everything needed to evaluate frame t — deterministic, so workers can rebuild it."""

    def __init__(self, prep, shot, out):
        self.prep, self.shot, self.out = prep, shot, out
        self.T = float(shot.get('duration', 10))
        self.FPS = int(shot.get('fps', 30))
        self.OW, self.OH = shot.get('size', [1920, 1080])
        self.scene = Scene.load(prep['scene'])
        Scene.line_scale = shot.get('line_scale', 0.88)
        self.panel = np.load(prep['panel'])
        # music + beats
        m = shot.get('music', {'type': 'music_box'})
        self.wav = os.path.join(out, 'music.wav')
        if m.get('file'):
            subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', m['file'], '-ac', '1', '-ar', '44100', self.wav], check=True)
        elif not (os.path.exists(self.wav) and os.path.exists(self.wav + '.json')
                  and json.load(open(self.wav + '.json')) == dict(m, dur=self.T)):
            audio.music_box_score(self.wav, self.T + 0.6, bpm=m.get('bpm', 80), start=m.get('start', 0.8))
            json.dump(dict(m, dur=self.T), open(self.wav + '.json', 'w'))
        x, sr = audio.read_wav(self.wav)
        self.hits = audio.detect_transients(x, sr, m.get('sensitivity', 0.3), min_gap=0.15)
        self.build()
        base = self.rig.items_at(0.0)
        self.NB = len(base)
        self.base_parts = [getattr(it, 'part', None) or 'static' for it in base]
        d = shot.get('draw_on', {'until': 3.2, 'spread': 2.4})
        self.draw_until = d.get('until', 3.2) if d else 0
        if d:
            self.starts = props.draw_on_schedule(base, np.array([0.0, 0.83 * self.scene.H]), t0=0.2, spread=d.get('spread', 2.4))
        tl = shot.get('title')
        self.title, self.title_box = [], None
        if tl and tl.get('lines'):
            sizes = tl.get('sizes', [66, 50, 44][:len(tl['lines'])])
            self.title, self.title_box = title_anchor(self.scene, self.rig, tl['lines'], sizes, tl.get('font', 'Dancing Script'))
            self.t_title = self.time(tl.get('at', 0.7 * self.T))
        focus = [self.time(a['t']) for a in shot['actions'] if a['g'] in ('hand_on_shoulder', 'turn_head', 'tilt_head')]
        self.cam = make_camera(shot.get('camera', 'auto'), self.scene, self.rig, self.T, self.OW, self.OH,
                               self.title_box, (min(focus) + 1.0) if focus else 0.4 * self.T)

    def time(self, t):
        if isinstance(t, str) and t.startswith('beat:'):
            i = int(t[5:])
            return float(self.hits[min(i, len(self.hits) - 1)]) if len(self.hits) else 0.0
        t = float(t)
        sn = self.shot.get('snap', 0.3)
        return A.snap(t, self.hits, sn) if sn else t

    def build(self):
        self.rig = Rig.load(self.prep['rig'], self.scene)
        al = actor_aliases(self.rig)
        self.alias = al
        for a in self.shot['actions']:
            who = [al[w] for w in (al if a['who'] == 'all' else [a['who']])] if a['who'] != 'all' else list(self.rig.chars.values())
            kw = {k: v for k, v in a.items() if k not in ('t', 'who', 'g', 'snap')}
            if 'target' in kw:
                kw['target'] = al[kw['target']]
            for k in ('from_',):
                if isinstance(kw.get(k), list):
                    kw[k] = tuple(kw[k])
            if 'until' in kw and kw['until'] == 'end':
                kw['until'] = self.T
            t = float(a['t']) if a.get('snap') is False else self.time(a['t'])
            for ch in who:
                ch.do(a['g'], t=t, **kw)
        fl = self.shot.get('flame', 'auto')
        self.flame_box = detect_flame(self.scene, self.rig) if fl == 'auto' else (tuple(fl) if fl else None)
        if self.flame_box:
            self.rig.fx.append(props.flame(self.flame_box))

    def items_at(self, t, keep_hidden=False):
        its = self.rig.items_at(t)
        if t < self.draw_until:
            props.apply_draw_on(its[:self.NB], self.starts, t)
        if self.title:
            x0, y0, x1, y1 = self.title_box
            for li, f in enumerate(self.title):
                t0 = self.t_title + 0.55 * li
                u = A.EASE['io']((t - t0) / 1.6) if t > t0 else 0.0
                if u <= 0 and not keep_hidden:
                    continue
                g = f.copy()
                g.opacity = 0.93 * min(1.0, u * 1.8)
                rise = 6 * (1 - A.EASE['spring_soft'](min(1.0, u * 1.2)))
                g.set_points(g.points() + [0, rise])
                if not keep_hidden:
                    wx = x0 + (x1 - x0 + 60) * u
                    g.clip = [(x0 - 10, y0 - 40), (wx, y0 - 40), (wx + 60, y1 + 40), (x0 - 10, y1 + 40)]
                its.append(g)
        return its

    def frame(self, t):
        M = self.cam(t)
        OW, OH, T = self.OW, self.OH, self.T
        a = self.scene.render(out_wh=(OW, OH), M=M, items=self.items_at(t))
        if self.flame_box and t >= 2.0:
            x0, y0, x1, y1 = self.flame_box
            fl = 1 + 0.07 * np.sin(t * 23) + 0.05 * np.sin(t * 37 + 1)
            p = M @ np.array([0.5 * (x0 + x1), y1 - 0.3 * (y1 - y0), 1.0])
            g = np.zeros((OH // 4, OW // 4), np.float32)
            r = max(4, int(0.45 * (x1 - x0) * M[0, 0] / 4 * 1.6))
            cv2.circle(g, (int(p[0] / 4), int(p[1] / 4)), r, 1, -1)
            a = a + cv2.resize(cv2.GaussianBlur(g, (0, 0), r * 0.7), (OW, OH)) * 0.10 * fl * A.EASE['smooth'](t - 2)
        fade = min(1.0, t / 0.4, (T - t) / 0.6)
        a = props.finish(a, t) * fade
        return np.clip(a * 0.94 * 255 + 9 * fade, 0, 255).astype(np.uint8)

    def preroll(self, f0):
        for f in range(max(0, f0 - int(1.5 * self.FPS)), f0):
            self.rig.items_at(f / self.FPS)


# ------------------------------------------------------------------ 5 render
def worker(job_path, f0, f1, path):
    job = json.load(open(job_path))
    rt = ShotRuntime(job['prep'], load_shot(job['shot']), job['out'])
    rt.preroll(f0)
    enc = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
                            '-s', f'{rt.OW}x{rt.OH}', '-r', str(rt.FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                            '-crf', '17', '-preset', 'veryfast', '-tune', 'animation', '-threads', '1',
                            '-x264-params', 'rc-lookahead=10', path], stdin=subprocess.PIPE)
    for f in range(f0, f1):
        enc.stdin.write(rt.frame(f / rt.FPS).tobytes())
    enc.stdin.close(); enc.wait()


def sheet(rt, path, times=None, cols=4):
    times = times or list(np.linspace(0.9, rt.T - 0.7, 8))
    tiles = []
    last = -1
    for t in times:
        f = int(t * rt.FPS)
        for g in range(max(last + 1, f - int(1.5 * rt.FPS)), f):
            rt.rig.items_at(g / rt.FPS)
        last = f
        im = cv2.resize(rt.frame(t), (480, 270), interpolation=cv2.INTER_AREA)
        cv2.putText(im, f'{t:4.1f}s', (8, 22), 0, 0.6, 255, 1)
        tiles.append(im)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.vstack(rows))
    return times


def export_lottie(rt, path, step=3):
    from . import lottie
    n = int(rt.T * rt.FPS)
    frames = list(range(0, n, step)) + [n - 1]
    rt.build()
    per = [rt.items_at(f / rt.FPS, keep_hidden=True) for f in frames]
    W = lottie.LottieWriter(rt.scene.W, rt.scene.H, rt.FPS, n, bg=rt.scene.bg, name='vecanim')
    for nm in ['static'] + sorted(set(rt.base_parts) - {'static'}, key=lambda p: rt.rig.chars[p].z if p in rt.rig.chars else 0):
        idx = [i for i, p in enumerate(rt.base_parts) if p == nm]
        W.add_layer(nm, [[fi[i] for i in idx] for fi in per], frames)
    # overlays (hands): fixed topology = the last seen set, hidden (opacity 0) before it appears
    ov = [[i for i in fi[rt.NB:] if i.group.startswith('hand')] for fi in per]
    ref = next((o for o in ov[::-1] if o), None)
    if ref:
        clip = ref[0].clip
        seq = []
        for o in ov:
            h = [r.copy() for r in (o if len(o) == len(ref) else ref)]
            for r in h:
                if len(o) != len(ref):
                    r.opacity = 0.0
                r.clip = None
            seq.append(h)
        W.add_layer('hand', seq, frames, mask=clip)
    if rt.title:
        W.add_layer('title', [[i for i in fi if i.group == 'title'] for fi in per], frames)
    doc = W.save(path, frames, [rt.cam(f / rt.FPS) for f in frames], out_wh=(rt.OW, rt.OH))
    return lottie.validate(doc)


def _sub(args, env):
    r = subprocess.run([sys.executable, '-m', 'mograph.vecanim'] + args, env=env)
    if r.returncode:
        raise RuntimeError('stage failed: ' + ' '.join(args))


def post(job_path, what):
    """Separate process: contact sheet / final-pose SVG / Lottie (keeps the parent light)."""
    job = json.load(open(job_path))
    out, name = job['out'], job['name']
    rt = ShotRuntime(job['prep'], load_shot(job['shot']), out)
    res = {}
    if 'sheet' in what:
        t1 = time.time(); sheet(rt, os.path.join(out, 'contact_sheet.png')); res['sheet_s'] = round(time.time() - t1, 1)
    if 'svg' in what:
        rt.build()
        tp = rt.t_title + 1.5 if rt.title else 0.8 * rt.T
        for f in range(max(0, int((tp - 1.5) * rt.FPS)), int(tp * rt.FPS)):
            rt.rig.items_at(f / rt.FPS)
        Scene(rt.scene.W, rt.scene.H, rt.scene.bg, rt.items_at(tp)).to_svg(os.path.join(out, 'final_pose.svg'))
    if 'lottie' in what:
        t1 = time.time()
        res['lottie'] = export_lottie(rt, os.path.join(out, f'{name}.lottie.json'))
        res['lottie_s'] = round(time.time() - t1, 1)
    res['actors'] = {k: v.name for k, v in rt.alias.items() if k in ('left', 'right', 'back', 'front')}
    res['beats'] = np.round(rt.hits, 2).tolist()
    res['flame'] = rt.flame_box
    res['title_box'] = [float(v) for v in rt.title_box] if rt.title_box else None
    json.dump(res, open(os.path.join(WORK, f'post_{what}.json'), 'w'))


def render(panel, shot, out, jobs=None, preview=False, rerig=False, lottie_out=False):
    """Orchestrator.  Holds no scene in memory: every heavy stage runs in its own process
    (2-core / 1 GB sandboxes: two x264 workers + two renderers barely fit)."""
    T0 = time.time()
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True); os.makedirs(WORK, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(HERE)) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    timing = {}
    _sub(['rig', os.path.abspath(panel), '-o', out] + (['--rerig'] if rerig else []), env)
    cache = os.path.join(out, 'cache', _hash(open(panel, 'rb').read()))
    prep = dict(panel=os.path.join(cache, 'panel.npy'), scene=os.path.join(cache, 'scene.json'),
                rig=os.path.join(cache, 'rig.json'), cache=cache)
    timing['prepare_s'] = round(time.time() - T0, 1)
    shot_d = load_shot(shot)
    name = os.path.splitext(os.path.basename(shot_d['_path']))[0]
    job_path = os.path.join(WORK, 'job.json')
    json.dump(dict(prep=prep, shot=shot_d['_path'], out=out, name=name), open(job_path, 'w'))
    _sub(['_post', job_path, 'sheet'], env)
    info = json.load(open(os.path.join(WORK, 'post_sheet.json')))
    timing['sheet_s'] = info['sheet_s']
    log('actors', info['actors'], '| beats', info['beats'][:12], '| flame', info['flame'], '| title', info['title_box'])
    if preview:
        timing['total_s'] = round(time.time() - T0, 1)
        json.dump(timing, open(os.path.join(out, 'timing.json'), 'w'), indent=1)
        log('preview ->', os.path.join(out, 'contact_sheet.png'), timing)
        return timing
    jobs = jobs or os.cpu_count() or 2
    T, FPS = float(shot_d.get('duration', 10)), int(shot_d.get('fps', 30))
    n = int(T * FPS)
    cuts = np.linspace(0, n, jobs + 1).astype(int)
    t1 = time.time()
    procs, parts = [], []
    for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
        p = os.path.join(WORK, f'chunk{i}.mp4'); parts.append(p)
        procs.append(subprocess.Popen([sys.executable, '-m', 'mograph.vecanim', '_worker', job_path, str(a), str(b), p], env=env))
    if any(pr.wait() for pr in procs):
        raise RuntimeError('a render worker failed')
    timing['render_s'] = round(time.time() - t1, 1)
    mp4 = os.path.join(out, f'{name}.mp4')
    lst = os.path.join(WORK, 'list.txt')
    open(lst, 'w').write(''.join(f"file '{p}'\n" for p in parts))
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-i', os.path.join(out, 'music.wav'),
                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', mp4], check=True)
    for p in parts:
        os.remove(p)
    what = 'svg+lottie' if lottie_out else 'svg'
    _sub(['_post', job_path, what], env)
    info = json.load(open(os.path.join(WORK, f'post_{what}.json')))
    for k in ('lottie', 'lottie_s'):
        if k in info:
            timing[k] = info[k]
    timing.update(frames=n, jobs=jobs, total_s=round(time.time() - T0, 1), mp4=mp4)
    json.dump(timing, open(os.path.join(out, 'timing.json'), 'w'), indent=1)
    log('done', timing)
    return timing
