"""Vector animation -> Lottie (bodymovin JSON 5.x): After Effects (Bodymovin / LottieFiles
importers), web (lottie-web), iOS/Android.

Layout follows EffectCraft's `crates/lottie/src/export.rs` (Apache-2.0): one shape layer per
character / prop; every item is a group (`gr`) of path shapes (`sh`) + fill (`fl`, even-odd
`r: 2` so holes stay holes) or stroke (`st`, round caps/joins); animation = path keyframes
(`ks.a = 1`) sampled from the deformer stack; draw-on = Trim Paths (`tm`); fades = group
opacity; clip = layer mask; camera = parent null with position/scale keys.
Items that never move more than `still_px` are written once (static) -> small files.
"""
import json, numpy as np
from .bezier import points_to_chain


def _shape(P, closed, dec=1):
    ch = points_to_chain(P, closed)
    if len(ch) == 0:
        return {'c': bool(closed), 'v': [], 'i': [], 'o': []}
    v = [ch[0][0]] + [c[3] for c in ch]
    if closed:
        v = v[:-1]
    n = len(v)
    i = [np.zeros(2) for _ in range(n)]
    o = [np.zeros(2) for _ in range(n)]
    for k, c in enumerate(ch):
        o[k] = c[1] - c[0]
        i[(k + 1) % n if closed else k + 1] = c[2] - c[3]
    r = lambda a: [round(float(a[0]), dec), round(float(a[1]), dec)]
    return {'c': bool(closed), 'v': [r(p) for p in v], 'i': [r(p) for p in i], 'o': [r(p) for p in o]}


def _val(v):
    return {'a': 0, 'k': v}


def _anim(keys):
    """keys: [(frame, value)] -> animated property (linear between samples, consecutive
    duplicates dropped)."""
    ks = []
    for j, (f, v) in enumerate(keys):
        if 0 < j < len(keys) - 1 and keys[j - 1][1] == v and keys[j + 1][1] == v:
            continue
        ks.append((f, v))
    if len(ks) == 1 or all(k[1] == ks[0][1] for k in ks):
        v = ks[0][1]                       # constant: shapes are stored unwrapped
        return {'a': 0, 'k': v[0] if isinstance(v, list) and len(v) == 1 else v}
    out = []
    for f, v in ks:
        out.append({'t': int(f), 's': v if isinstance(v, list) else [v],
                    'i': {'x': [0.833], 'y': [0.833]}, 'o': {'x': [0.167], 'y': [0.167]}})
    return {'a': 1, 'k': out}


def _grey(v):
    v = round(float(np.clip(v, 0, 1)), 4); return [v, v, v, 1]


def _poly_shape(pts):
    pts = [[round(float(x), 1), round(float(y), 1)] for x, y in pts]
    z = [[0, 0] for _ in pts]
    return {'c': True, 'v': pts, 'i': z, 'o': z}


def _tr(op):
    return {'ty': 'tr', 'p': _val([0, 0]), 'a': _val([0, 0]), 's': _val([100, 100]), 'r': _val(0), 'o': op,
            'sk': _val(0), 'sa': _val(0), 'nm': 'Transform'}


class LottieWriter:
    def __init__(self, W, H, fps, n_frames, bg=0.0, name='vecanim'):
        self.W, self.H, self.fps, self.n, self.bg, self.name = W, H, fps, n_frames, bg, name
        self.layers = []

    def add_layer(self, name, frames_items, frames, still_px=0.25, mask=None):
        """frames_items: one item list per sampled frame, same order/topology in every frame
        (true for the rig: deformation moves points, never adds them).  Back-to-front order."""
        groups = []
        for j, it0 in enumerate(frames_items[0]):
            seq = [fi[j] for fi in frames_items]
            P = np.array([s.points() for s in seq])
            moving = P.size and np.abs(P - P[0]).max() > still_px
            shapes = []
            if it0.kind == 'fill':
                o = 0
                for p in it0.paths:
                    k = len(p)
                    ks = _anim([(f, [_shape(s.points()[o:o + k], True)]) for f, s in zip(frames, seq)]) if moving \
                        else _val(_shape(p, True))
                    shapes.append({'ty': 'sh', 'ks': ks, 'nm': 'path'}); o += k
                shapes.append({'ty': 'fl', 'c': _val(_grey(it0.tone)), 'o': _val(100), 'r': 2, 'nm': 'fill'})
            else:
                ks = _anim([(f, [_shape(s.P, s.closed)]) for f, s in zip(frames, seq)]) if moving \
                    else _val(_shape(it0.P, it0.closed))
                shapes.append({'ty': 'sh', 'ks': ks, 'nm': 'path'})
                rev = [round(100 * s.reveal, 2) for s in seq]
                if min(rev) < 99.9:
                    shapes.append({'ty': 'tm', 's': _val(0), 'e': _anim([(f, [r]) for f, r in zip(frames, rev)]),
                                   'o': _val(0), 'm': 1, 'nm': 'draw-on'})
                shapes.append({'ty': 'st', 'c': _val(_grey(it0.color)), 'o': _val(100), 'w': _val(round(it0.width, 2)),
                               'lc': 2, 'lj': 2, 'ml': 4, 'nm': 'stroke'})
            ops = [round(100 * s.opacity, 1) for s in seq]
            shapes.append(_tr(_val(100) if min(ops) >= 99.9 else _anim([(f, [o_]) for f, o_ in zip(frames, ops)])))
            groups.append({'ty': 'gr', 'it': shapes, 'nm': f'{it0.group}_{j}', 'np': len(shapes)})
        layer = {'ddd': 0, 'ty': 4, 'nm': name, 'sr': 1, 'ao': 0, 'ip': 0, 'op': self.n, 'st': 0, 'bm': 0,
                 'ks': {'o': _val(100), 'r': _val(0), 'p': _val([0, 0, 0]), 'a': _val([0, 0, 0]), 's': _val([100, 100, 100])},
                 'shapes': groups[::-1]}                    # Lottie draws the first shape on top
        if mask is not None:
            layer['hasMask'] = True
            layer['masksProperties'] = [{'inv': False, 'mode': 'a', 'pt': _val(_poly_shape(mask)), 'o': _val(100),
                                         'x': _val(0), 'nm': 'clip'}]
        self.layers.append(layer)
        return layer

    def save(self, path, cam_frames=None, cam_mats=None, out_wh=None):
        W, H = out_wh or (self.W, self.H)
        layers, ind = [], 1
        cam = None
        if cam_frames is not None:
            cam = {'ddd': 0, 'ind': 1, 'ty': 3, 'nm': 'camera', 'sr': 1, 'ao': 0, 'ip': 0, 'op': self.n, 'st': 0, 'bm': 0,
                   'ks': {'o': _val(0), 'r': _val(0), 'a': _val([0, 0, 0]),
                          'p': _anim([(f, [round(float(M[0][2]), 2), round(float(M[1][2]), 2), 0]) for f, M in zip(cam_frames, cam_mats)]),
                          's': _anim([(f, [round(float(M[0][0]) * 100, 3), round(float(M[0][0]) * 100, 3), 100]) for f, M in zip(cam_frames, cam_mats)])}}
            ind = 2
        for L in self.layers[::-1]:
            L = dict(L); L['ind'] = ind; ind += 1
            if cam is not None:
                L['parent'] = 1
            layers.append(L)
        if cam is not None:
            layers.append(cam)
        g = int(round(self.bg * 255))
        layers.append({'ddd': 0, 'ind': ind, 'ty': 1, 'nm': 'paper', 'sr': 1, 'ao': 0, 'ip': 0, 'op': self.n, 'st': 0,
                       'bm': 0, 'sc': '#%02x%02x%02x' % (g, g, g), 'sw': W, 'sh': H,
                       'ks': {'o': _val(100), 'r': _val(0), 'p': _val([W / 2, H / 2, 0]), 'a': _val([W / 2, H / 2, 0]),
                              's': _val([100, 100, 100])}})
        doc = {'v': '5.7.4', 'fr': self.fps, 'ip': 0, 'op': self.n, 'w': W, 'h': H, 'nm': self.name, 'ddd': 0,
               'assets': [], 'layers': layers, 'markers': []}
        with open(path, 'w') as f:
            json.dump(doc, f, separators=(',', ':'))
        return doc


def validate(doc_or_path):
    """Structural check (fields every Lottie player needs) + counts."""
    d = json.load(open(doc_or_path)) if isinstance(doc_or_path, str) else doc_or_path
    for k in ('v', 'fr', 'ip', 'op', 'w', 'h', 'layers'):
        assert k in d, k
    cnt = dict(paths=0, animated_paths=0, keyframes=0)

    def walk(items):
        for s in items:
            if s['ty'] == 'gr':
                walk(s['it'])
            elif s['ty'] == 'sh':
                cnt['paths'] += 1
                if s['ks']['a']:
                    cnt['animated_paths'] += 1
                    for k in s['ks']['k']:
                        sh = k['s'][0]; cnt['keyframes'] += 1
                        assert len(sh['v']) == len(sh['i']) == len(sh['o'])
                else:
                    sh = s['ks']['k']; assert len(sh['v']) == len(sh['i']) == len(sh['o'])
    for L in d['layers']:
        assert 'ks' in L and 'ty' in L and 'ind' in L
        if L['ty'] == 4:
            walk(L['shapes'])
    return dict(layers=len(d['layers']), frames=d['op'], fps=d['fr'], **cnt)
