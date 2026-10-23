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

    def points(self):
        return np.vstack(self.paths)

    def set_points(self, P):
        o = 0
        for i, p in enumerate(self.paths):
            self.paths[i] = P[o:o + len(p)]; o += len(p)

    def copy(self):
        return Fill([p.copy() for p in self.paths], self.tone, self.group, self.part)

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

    def points(self):
        return self.P

    def set_points(self, P):
        self.P = P

    def copy(self):
        s = Stroke(self.P.copy(), self.closed, self.width, self.aw.copy(), self.color, self.group, self.part)
        s.reveal = self.reveal
        return s

    def chain(self):
        return points_to_chain(self.P, self.closed)

    def to_json(self):
        return dict(kind='stroke', closed=self.closed, width=round(self.width, 3), color=round(self.color, 4),
                    group=self.group, part=self.part, pts=np.round(self.P, 2).tolist(),
                    aw=np.round(self.aw, 2).tolist())


def _sample_stroke(st, spacing=1.2):
    """Dense centre-line points + interpolated width."""
    C = st.chain()
    if len(C) == 0:
        return None, None
    aw = st.aw
    pts, ws = [], []
    for i, c in enumerate(C):
        L = np.linalg.norm(np.diff(c, axis=0), axis=1).sum()
        k = max(2, int(L / spacing) + 1)
        t = np.linspace(0, 1, k, endpoint=False)
        pts.append(bez(c, t))
        w0, w1 = aw[i % len(aw)], aw[(i + 1) % len(aw)]
        ws.append(w0 + (w1 - w0) * t)
    pts.append(C[-1][3][None]); ws.append([aw[len(C) % len(aw)]])
    return np.vstack(pts), np.concatenate(ws)


class Scene:
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
                self._stroke(c, it, variable_width)
        surf.flush()
        a = np.frombuffer(surf.get_data(), np.uint8).reshape(oh * ss, surf.get_stride() // 4, 4)[:, :ow * ss, 1]
        a = a.astype(np.float32) / 255
        if ss > 1:
            a = cv2.resize(a, (ow, oh), interpolation=cv2.INTER_AREA)
        return a

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
        # variable width: outline polygon from offset curves + round caps
        if len(pts) < 2:
            return
        t = np.gradient(pts, axis=0)
        n = np.linalg.norm(t, axis=1, keepdims=True); t = t / np.maximum(n, 1e-9)
        nr = np.c_[-t[:, 1], t[:, 0]] * (ws[:, None] / 2)
        left, right = pts + nr, pts - nr
        c.new_path()
        c.move_to(*left[0])
        for p in left[1:]:
            c.line_to(*p)
        if not st.closed:
            a0 = np.arctan2(nr[-1, 1], nr[-1, 0])
            c.arc_negative(pts[-1, 0], pts[-1, 1], ws[-1] / 2, a0, a0 - np.pi)
        for p in right[::-1]:
            c.line_to(*p)
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
