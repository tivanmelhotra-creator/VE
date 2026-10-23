"""Characters, deformer stack and the gesture API for vector line-art scenes.

    scene = Scene.load('scene.json')                 # traced once (trace.py)
    rig   = Rig.load('rigs/birthday.json', scene)     # characters, pins, landmarks (once)
    boy, girl = rig['boy'], rig['girl']
    boy.do('tilt_head', t=2.6, dur=1.2, angle=3.5)
    boy.do('hand_on_shoulder', target=girl, t=3.4)
    girl.do('blink', t=6.9)
    frame = rig.render(t, M=camera)                  # vector render (cairo)

Deformer stack per character and frame (all on Bézier control points, never pixels):
  1. local deformers in rest space (smile, blink, eye look, squash) — MLS-rigid / squash on the
     points inside a small region, displacement faded to zero at the region border;
  2. global ARAP puppet (arap.Puppet, prefactored) driven by the pins the gestures move;
     every control point rides its rest triangle (barycentric), local offsets are added.
Each character also gets a *silhouette* fill (paper colour) drawn under its items, so a
character in front hides what moves behind it (an arm reaching behind her back).
"""
import json, numpy as np, cv2
from .arap import Mesh, Pin, Puppet
from .scene import Fill, Stroke
from . import anim as A


def rot(P, piv, ang):
    c, s = np.cos(ang), np.sin(ang)
    d = np.asarray(P, float) - piv
    return np.c_[piv[0] + c * d[:, 0] - s * d[:, 1], piv[1] + s * d[:, 0] + c * d[:, 1]]


def _fade(d, r):
    u = np.clip(1 - d / r, 0, 1)
    return u * u * (3 - 2 * u)


def local_warp(P, center, radius, src, dst, alpha=1.6):
    """MLS-rigid on the points within `radius` of `center`; src/dst control points (rest space).
    The border ring is pinned and the displacement fades to 0 at the radius."""
    P = np.asarray(P, float)
    d = np.linalg.norm(P - center, axis=1)
    idx = np.nonzero(d < radius)[0]
    if len(idx) == 0:
        return np.zeros_like(P)
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ring = np.c_[center[0] + radius * np.cos(ang), center[1] + radius * np.sin(ang)]
    p = np.vstack([dst, ring]).astype(float)          # inverse map not needed: points are vectors
    q = np.vstack([src, ring]).astype(float)
    p, q = q, p                                        # forward: from src (rest) to dst
    v = P[idx]
    w = 1.0 / (((v[:, None] - p[None]) ** 2).sum(-1) ** alpha + 1e-6)
    ws = w.sum(1, keepdims=True)
    ps = (w[..., None] * p[None]).sum(1) / ws
    qs = (w[..., None] * q[None]).sum(1) / ws
    ph = p[None] - ps[:, None]; qh = q[None] - qs[:, None]
    phc = ph[..., 0] + 1j * ph[..., 1]; qhc = qh[..., 0] + 1j * qh[..., 1]
    vpc = (v - ps)[:, 0] + 1j * (v - ps)[:, 1]
    fr = (w * qhc * np.conj(phc)).sum(1) * vpc
    n = np.abs(fr)
    fr = np.where(n > 1e-9, fr / np.maximum(n, 1e-9) * np.abs(vpc), 0)
    out = np.c_[fr.real, fr.imag] + qs
    D = np.zeros_like(P)
    D[idx] = (out - v) * _fade(d[idx], radius)[:, None]
    return D


def squash(P, center, rx, ry, amount, anchor_y):
    """Blink-type deformer: points in the ellipse move toward the line y = anchor_y."""
    P = np.asarray(P, float)
    e = np.sqrt(((P[:, 0] - center[0]) / rx) ** 2 + ((P[:, 1] - center[1]) / ry) ** 2)
    w = _fade(e, 1.0) * amount
    D = np.zeros_like(P)
    D[:, 1] = w * (anchor_y - P[:, 1])
    return D


def shift(P, center, radius, delta):
    P = np.asarray(P, float)
    w = _fade(np.linalg.norm(P - center, axis=1), radius)
    return w[:, None] * np.asarray(delta, float)[None]


# ============================================================================ character
class Character:
    def __init__(self, name, scene, spec, z=0):
        self.name, self.scene, self.spec, self.z = name, scene, spec, z
        self.poly = np.array(spec['polygon'], float)
        self.L = {k: np.array(v, float) for k, v in spec.get('landmarks', {}).items()}
        self.groups = spec.get('pin_groups', {})
        H, W = scene.H, scene.W
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [self.poly.astype(np.int32)], 1)
        self.mask = mask
        self.items = []
        self.claim()
        self.mesh = Mesh.from_mask(mask, edge=spec.get('mesh_edge', 20), expansion=6)
        pins = [Pin(nm, xy) for nm, xy in spec['pins'].items()]
        pins += [Pin(nm, xy, kind='starch', extent=e, amount=a) for nm, (xy, e, a) in spec.get('starch', {}).items()]
        self.puppet = Puppet(self.mesh, pins, refine=2)
        self.rest_pins = {nm: np.array(xy, float) for nm, xy in spec['pins'].items()}
        # embed every control point once
        self.P0 = np.vstack([it.points() for it in self.items]) if self.items else np.zeros((0, 2))
        self.sizes = [len(it.points()) for it in self.items]
        self.emb = self.mesh.embed(self.P0)
        self.sil_P0 = self.art_outline() if spec.get('silhouette', True) else np.zeros((0, 2))
        self.sil_emb = self.mesh.embed(self.sil_P0) if len(self.sil_P0) else None
        self.silhouette = Fill([self.sil_P0], scene.bg, group='silhouette', part=name)
        self.actions = []
        self.props = []

    def art_outline(self, close=9):
        """Silhouette of the character's own art (strokes + fills, closed, holes filled), as a
        smooth closed Bezier path — hides whatever moves behind this character."""
        from .bezier import fit_closed, chain_to_points, Style
        sub = self.scene.copy(); sub.items = [i.copy() for i in self.items]
        for i in sub.items:
            if i.kind == 'stroke':
                i.color = 1.0
            else:
                i.tone = 1.0
        m = (sub.render(bg=0.0) > 0.25).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close + 1,) * 2)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k) & self.mask
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)[:, 0].astype(float)
        segs = fit_closed(c, Style(step=1.0, sigma=2.0, corner_deg=75, corner_win=4, tol=1.0, line_tol=0.6))
        return chain_to_points(segs, True)

    def claim(self):
        """Take the scene items whose control points lie mostly inside the polygon."""
        H, W = self.mask.shape
        for it in self.scene.items:
            if getattr(it, 'part', None):
                continue
            P = it.points()
            x = np.clip(P[:, 0].round().astype(int), 0, W - 1); y = np.clip(P[:, 1].round().astype(int), 0, H - 1)
            if self.mask[y, x].mean() > 0.6:
                it.part = self.name
                self.items.append(it)

    # ---------------------------------------------------------------- API
    def do(self, gesture, t=0.0, dur=None, **kw):
        from .gestures import GESTURES
        g = GESTURES[gesture]
        self.actions.append(g(self, t, dur, **kw))
        return self

    # ---------------------------------------------------------------- per frame
    def pose(self, t):
        st = State(self)
        for a in self.actions:
            a.apply(st, t)
        return st

    def frame_items(self, t, st=None):
        st = st or self.pose(t)
        P = self.P0
        D = np.zeros_like(P)
        for f in st.local:
            D += f(P)
        targets = {}
        for nm, p0 in self.rest_pins.items():
            p = p0.copy()
            for piv, ang, names in st.rotations:
                if nm in names:
                    p = rot(p[None], piv, ang)[0]
            p = p + st.offsets.get(nm, 0)
            targets[nm] = p
        moved = any(np.abs(targets[n] - self.rest_pins[n]).max() > 0.02 for n in targets)
        if moved:
            Vd = self.puppet.solve(targets)
            Q = self.mesh.carry(Vd, self.emb)
            S = self.mesh.carry(Vd, self.sil_emb) if self.sil_emb is not None else self.sil_P0
        else:
            Q, S = P.copy(), self.sil_P0
        Q = Q + D
        out = []
        if len(S):
            sil = self.silhouette.copy(); sil.paths = [S]
            out.append(sil)
        o = 0
        for it, n in zip(self.items, self.sizes):
            c = it.copy(); c.set_points(Q[o:o + n]); o += n
            out.append(c)
        return out, st


class State:
    def __init__(self, ch):
        self.ch = ch
        self.rotations = []      # (pivot, angle, pin names)
        self.offsets = {}        # pin -> dxdy
        self.local = []          # functions P -> displacement (rest space)
        self.overlay = []        # (z, items) drawn after characters (props like a hand)

    def add_offset(self, pin, d):
        self.offsets[pin] = self.offsets.get(pin, 0) + np.asarray(d, float)


# ============================================================================ rig
class Rig:
    def __init__(self, scene, chars, extras=None):
        self.scene, self.chars = scene, chars
        self.extras = extras or {}
        self.static = [it for it in scene.items if not getattr(it, 'part', None)]
        self.fx = []             # functions (t, items) -> items for static props (candle)
        self.post = []

    def __getitem__(self, k):
        return self.chars[k]

    @staticmethod
    def load(path, scene):
        spec = json.load(open(path))
        chars = {}
        for z, (nm, cs) in enumerate(sorted(spec['characters'].items(), key=lambda kv: -kv[1].get('z', 0))):
            chars[nm] = Character(nm, scene, cs, z=cs.get('z', z))
        return Rig(scene, chars, spec.get('extras'))

    def items_at(self, t):
        items = []
        bg_fills = [i for i in self.static if i.kind == 'fill']
        bg_strokes = [i for i in self.static if i.kind == 'stroke']
        st_items = [i.copy() for i in bg_fills + bg_strokes]      # per-frame copies (draw-on mutates)
        for f in self.fx:
            st_items = f(t, st_items)
        items += st_items
        overlays = []
        for ch in sorted(self.chars.values(), key=lambda c: c.z):
            its, st = ch.frame_items(t)
            fills = [i for i in its if i.kind == 'fill']
            strokes = [i for i in its if i.kind == 'stroke']
            items += fills + strokes
            overlays += st.overlay
        for z, its in sorted(overlays, key=lambda x: x[0]):
            items += its
        return items

    def render(self, t, out_wh=None, M=None, ss=1, items=None):
        items = items if items is not None else self.items_at(t)
        return self.scene.render(out_wh=out_wh, M=M, ss=ss, items=items)
