"""Vector scene: fills (compound closed Bézier paths, even-odd) + strokes (open/closed Bézier
centre lines with per-anchor width).  Renders with cairo (anti-aliased, any scale), exports
SVG and JSON.  Every item's control points can be read / replaced as one flat array, which is
what the puppet deformer moves (vector deformation = moving control points, never pixels).
"""
import json, numpy as np, cairo, cv2
from .bezier import points_to_chain, bez


class Fill:
    kind = 'fill'

    def __init__(self, paths, tone, group='fill', part=None):
        self.paths = [np.asarray(p, float) for p in paths]
        self.tone, self.group, self.part = float(tone), group, part
        self.opacity, self.clip = 1.0, None

    def points(self):
        return np.vstack(self.paths)

    def set_points(self, P):
        o = 0
        for i, p in enumerate(self.paths):
            self.paths[i] = P[o:o + len(p)]; o += len(p)

    def copy(self):
        f = Fill([p.copy() for p in self.paths], self.tone, self.group, self.part)
        f.opacity, f.clip = self.opacity, self.clip
        return f

    def to_json(self):
        return dict(kind='fill', tone=round(self.tone, 4), group=self.group, part=self.part,
                    paths=[np.round(p, 2).tolist() for p in self.paths])


class Stroke:
    kind = 'stroke'

    def __init__(self, P, closed, width, aw=None, color=1.0, group='line', part=None):
        self.P = np.asarray(P, float)
        self.closed, self.width = bool(closed), float(width)
        na = len(self.P) // 3 + (0 if closed else 1)
        self.aw = np.asarray(aw if aw is not None else np.full(na, width), float)
        self.color, self.group, self.part = float(color), group, part
        self.reveal = 1.0          # draw-on (0..1 of length)
        self.opacity, self.clip = 1.0, None
        self.taper = 1.0           # 0 = blunt, 1 = ink-pen taper at both open ends

    def points(self):
        return self.P

    def set_points(self, P):
        self.P = P

    def copy(self):
        s = Stroke(self.P.copy(), self.closed, self.width, self.aw.copy(), self.color, self.group, self.part)
        s.reveal, s.opacity, s.clip, s.taper = self.reveal, self.opacity, self.clip, self.taper
        return s

    def chain(self):
        return points_to_chain(self.P, self.closed)

    def to_json(self):
        return dict(kind='stroke', closed=self.closed, width=round(self.width, 3), color=round(self.color, 4),
                    group=self.group, part=self.part, pts=np.round(self.P, 2).tolist(),
                    aw=np.round(self.aw, 2).tolist())


_TCACHE = {}


def _tbasis(k):
    B = _TCACHE.get(k)
    if B is None:
        t = np.arange(k) / k
        s_ = 1 - t
        B = np.stack([s_ ** 3, 3 * t * s_ * s_, 3 * t * t * s_, t ** 3], 1)   # (k,4)
        _TCACHE[k] = B = (B, t)
    return B


def _sample_stroke(st, spacing=1.6):
    """Dense centre-line points + interpolated width (vectorised; one sample count per
    segment bucketed to powers of two so the Bernstein basis is cached)."""
    C = st.chain()
    if len(C) == 0:
        return None, None
    aw = st.aw
    L = np.abs(np.diff(C, axis=1)).sum((1, 2))                 # cheap length bound
    ks = np.maximum(2, 2 ** np.ceil(np.log2(np.maximum(L / spacing, 1)))).astype(int)
    pts, ws = [], []
    na = len(aw)
    for i, c in enumerate(C):
        Bm, t = _tbasis(int(ks[i]))
        pts.append(Bm @ c)
        w0, w1 = aw[i % na], aw[(i + 1) % na]
        ws.append(w0 + (w1 - w0) * t)
    pts.append(C[-1][3][None]); ws.append([aw[len(C) % na]])
    return np.vstack(pts), np.concatenate(ws)


def _poly(c, P):
    """Append a polyline to the current cairo path in one call."""
    c.move_to(float(P[0, 0]), float(P[0, 1]))
    for x, y in P[1:].tolist():
        c.line_to(x, y)


class Scene:
    line_scale = 1.0          # global ink weight (render-time)

    def __init__(self, W, H, bg, items):
        self.W, self.H, self.bg, self.items = int(W), int(H), float(bg), list(items)

    def n_points(self):
        return int(sum(len(i.points()) for i in self.items))

    def copy(self):
        return Scene(self.W, self.H, self.bg, [i.copy() for i in self.items])

    # ------------------------------------------------------------------ io
    def to_json(self):
        return dict(W=self.W, H=self.H, bg=self.bg, items=[i.to_json() for i in self.items])

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_json(), f, separators=(',', ':'))

    @staticmethod
    def load(path):
        d = json.load(open(path)) if isinstance(path, str) else path
        items = []
        for it in d['items']:
            if it['kind'] == 'fill':
                items.append(Fill(it['paths'], it['tone'], it.get('group', 'fill'), it.get('part')))
            else:
                items.append(Stroke(it['pts'], it['closed'], it['width'], it.get('aw'), it['color'],
                                    it.get('group', 'line'), it.get('part')))
        return Scene(d['W'], d['H'], d['bg'], items)

    # ------------------------------------------------------------------ render
    def render(self, scale=1.0, out_wh=None, M=None, ss=1, variable_width=True, items=None, bg=None):
        """Render to float32 grey (0..1).  M: 2x3 affine (scene → output px), applied in cairo
        so lines stay vector-sharp at any zoom."""
        if out_wh is None:
            out_wh = (int(self.W * scale), int(self.H * scale))
        ow, oh = out_wh
        surf = cairo.ImageSurface(cairo.FORMAT_RGB24, ow * ss, oh * ss)
        c = cairo.Context(surf)
        g = self.bg if bg is None else bg
        c.set_source_rgb(g, g, g); c.paint()
        c.scale(ss, ss)
        if M is not None:
            c.transform(cairo.Matrix(M[0][0], M[1][0], M[0][1], M[1][1], M[0][2], M[1][2]))
        else:
            c.scale(scale, scale)
        c.set_line_cap(cairo.LINE_CAP_ROUND); c.set_line_join(cairo.LINE_JOIN_ROUND)
        for it in (items if items is not None else self.items):
            if it.opacity <= 0.003:
                continue
            if it.clip is not None or it.opacity < 0.999:
                c.save()
                if it.clip is not None:
                    c.new_path(); c.move_to(*it.clip[0])
                    for q in it.clip[1:]:
                        c.line_to(*q)
                    c.close_path(); c.clip()
                if it.opacity < 0.999:
                    c.push_group()
            self._item(c, it, variable_width)
            if it.clip is not None or it.opacity < 0.999:
                if it.opacity < 0.999:
                    c.pop_group_to_source(); c.paint_with_alpha(it.opacity)
                c.restore()
        surf.flush()
        a = np.frombuffer(surf.get_data(), np.uint8).reshape(oh * ss, surf.get_stride() // 4, 4)[:, :ow * ss, 1]
        a = a.astype(np.float32) / 255
        if ss > 1:
            a = cv2.resize(a, (ow, oh), interpolation=cv2.INTER_AREA)
        return a

    @classmethod
    def _item(cls, c, it, variable_width):
        if True:
            if it.kind == 'fill':
                c.new_path()
                for p in it.paths:
                    ch = points_to_chain(p, True)
                    if len(ch) == 0:
                        continue
                    c.move_to(*ch[0][0])
                    for q in ch:
                        c.curve_to(*q[1], *q[2], *q[3])
                    c.close_path()
                c.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
                c.set_source_rgb(it.tone, it.tone, it.tone); c.fill()
            else:
                cls._stroke(c, it, variable_width)

    @staticmethod
    def _stroke(c, st, variable_width):
        c.set_source_rgb(st.color, st.color, st.color)
        if st.reveal <= 0:
            return
        if not variable_width or st.reveal < 1:
            pts, ws = _sample_stroke(st)
            if pts is None:
                return
            if st.reveal < 1:
                d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
                k = max(2, int(np.searchsorted(d, d[-1] * st.reveal)))
                pts, ws = pts[:k], ws[:k]
            if not variable_width:
                c.set_line_width(st.width)
                c.move_to(*pts[0])
                for p in pts[1:]:
                    c.line_to(*p)
                c.stroke()
                return
        else:
            pts, ws = _sample_stroke(st)
            if pts is None:
                return
        ws = ws * Scene.line_scale
        if not st.closed and st.taper > 0 and len(pts) > 3:
            # ink-pen taper: width grows from ~18 % at each end over min(30 % length, 4 widths)
            dd = np.diff(pts, axis=0)
            d = np.r_[0, np.cumsum(np.sqrt((dd * dd).sum(1)))]
            full = d[-1] if st.reveal >= 1 else d[-1] / max(st.reveal, 1e-3)
            Lt = max(1e-3, min(0.3 * full, 4.0 * float(np.median(ws))))
            e = np.minimum(d, (full - d) if st.reveal >= 1 else np.inf)
            u = np.clip(e / Lt, 0, 1)
            prof = 0.18 + 0.82 * np.sin(u * np.pi / 2) ** 0.8
            ws = ws * (1 - st.taper + st.taper * prof)
        # variable width: outline polygon from offset curves + round caps
        if len(pts) < 2:
            return
        t = np.empty_like(pts); t[1:-1] = pts[2:] - pts[:-2]; t[0] = pts[1] - pts[0]; t[-1] = pts[-1] - pts[-2]
        n = np.sqrt((t * t).sum(1, keepdims=True)); t = t / np.maximum(n, 1e-9)
        nr = np.c_[-t[:, 1], t[:, 0]] * (ws[:, None] / 2)
        left, right = pts + nr, pts - nr
        c.new_path()
        _poly(c, left)
        if not st.closed:
            a0 = np.arctan2(nr[-1, 1], nr[-1, 0])
            c.arc_negative(pts[-1, 0], pts[-1, 1], ws[-1] / 2, a0, a0 - np.pi)
        for x, y in right[::-1].tolist():
            c.line_to(x, y)
        if not st.closed:
            a0 = np.arctan2(-nr[0, 1], -nr[0, 0])
            c.arc_negative(pts[0, 0], pts[0, 1], ws[0] / 2, a0, a0 - np.pi)
        c.close_path()
        c.set_fill_rule(cairo.FILL_RULE_WINDING)
        c.fill()

    # ------------------------------------------------------------------ svg
    def to_svg(self, path=None, group_by='part'):
        def g255(v):
            v = int(round(np.clip(v, 0, 1) * 255)); return f'#{v:02x}{v:02x}{v:02x}'

        def d_of(P, closed):
            ch = points_to_chain(P, closed)
            if len(ch) == 0:
                return ''
            s = [f'M{ch[0][0][0]:.2f} {ch[0][0][1]:.2f}']
            for q in ch:
                s.append(f'C{q[1][0]:.2f} {q[1][1]:.2f} {q[2][0]:.2f} {q[2][1]:.2f} {q[3][0]:.2f} {q[3][1]:.2f}')
            if closed:
                s.append('Z')
            return ''.join(s)
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.W} {self.H}" '
               f'width="{self.W}" height="{self.H}">',
               f'<rect width="100%" height="100%" fill="{g255(self.bg)}"/>']
        groups = {}
        for it in self.items:
            key = getattr(it, group_by) or it.group
            groups.setdefault(key, []).append(it)
        for key, its in groups.items():
            out.append(f'<g id="{key}" stroke-linecap="round" stroke-linejoin="round">')
            for it in its:
                if it.kind == 'fill':
                    d = ''.join(d_of(p, True) for p in it.paths)
                    out.append(f'<path d="{d}" fill="{g255(it.tone)}" fill-rule="evenodd" data-kind="{it.group}"/>')
                else:
                    out.append(f'<path d="{d_of(it.P, it.closed)}" fill="none" stroke="{g255(it.color)}" '
                               f'stroke-width="{it.width:.2f}" data-kind="{it.group}"/>')
            out.append('</g>')
        out.append('</svg>')
        s = '\n'.join(out)
        if path:
            open(path, 'w').write(s)
        return s
