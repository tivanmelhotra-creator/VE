"""Automatic rig for line-art characters: image -> rig JSON with no manual coordinates.

Works on the traced Scene + the source panel (grey 0..1).  Pipeline:
  1. characters  — the big light fills are faces/necks (skin); each skin blob with a
     face-shaped upper part is a character.  The body = everything below/around it, split
     between characters along the column midway between their faces (Voronoi on x, weighted
     by face size), bounded by the drawing's ink mask.
  2. head frame  — face = the upper part of the skin blob above its narrowest row (the neck);
     neck pivot = centre of that narrowest row; head radius from the face width.
  3. face features — inside the face: dark blobs (eyes = the two largest in the upper 60 %,
     else one eye + mirrored), mouth = darkest elongated blob in the lower part (else a point
     at 78 % of the face height); facing direction from the nose side (skin centroid vs. jaw).
  4. shoulders  — from the ink mask: widest row of the silhouette 0.5–1.2 head heights below
     the neck; the near-side shoulder line = upper contour of the silhouette there.
  5. pins       — head ring (8 pins around the face/hair), chest, shoulders, hips/base,
     pin groups; secondary regions (hair mass beside/behind the face, collar).
  6. optional MediaPipe pose as a *cross-check* (never required).

Output: the same rig JSON format as `examples/vecanim/birthday_rig.json`, plus a debug sheet.
"""
import json, numpy as np, cv2
from .scene import Scene


def _cc(mask, min_area):
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [(lab == k, st[k], cen[k]) for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= min_area]


def _ink_mask(scene, close=9):
    sub = scene.copy()
    for i in sub.items:
        if i.kind == 'stroke':
            i.color = 1.0
        else:
            i.tone = 1.0
    m = (sub.render(bg=0.0) > 0.25).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close + 1,) * 2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    # fill holes
    ff = m.copy(); h, w = m.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    return m | (1 - ff)


def _skin_mask(scene, tone_min=0.6):
    sub = Scene(scene.W, scene.H, 0.0, [i.copy() for i in scene.items if i.kind == 'fill' and i.tone >= tone_min])
    for i in sub.items:
        i.tone = 1.0
    return (sub.render(bg=0.0) > 0.5).astype(np.uint8)


def _face_from_skin(blob, stats):
    """Split a skin blob into face + neck at the narrowest row in its middle third."""
    x, y, w, h = stats[:4]
    rows = blob[y:y + h].sum(1).astype(float)
    sm = np.convolve(rows, np.ones(9) / 9, 'same')
    a, b = int(h * 0.35), int(h * 0.8)
    if b - a < 5:
        return None
    # face band = widest rows in the upper half; neck = narrowest after it
    top = int(np.argmax(sm[:max(a, 6)])) if a > 6 else 0
    neck_rel = a + int(np.argmin(sm[a:b]))
    if sm[neck_rel] > 0.85 * sm[:neck_rel].max():
        neck_rel = int(h * 0.62)            # no clear neck: assume one
    ny = y + neck_rel
    xs = np.nonzero(blob[ny])[0]
    if len(xs) == 0:
        return None
    face = blob.copy(); face[ny:] = 0
    fy, fx = np.nonzero(face)
    if len(fx) < 50:
        return None
    return dict(neck=np.array([xs.mean(), ny], float), neck_w=float(xs.ptp() + 1),
                face_box=(int(fx.min()), int(fy.min()), int(fx.max()), int(fy.max())), face=face,
                face_c=np.array([fx.mean(), fy.mean()]), top=y)


def _features(panel, f):
    """Eyes and mouth inside the face (dark blobs enclosed by skin)."""
    x0, y0, x1, y1 = f['face_box']
    fw, fh = x1 - x0 + 1, y1 - y0 + 1
    face = f['face'][y0:y1 + 1, x0:x1 + 1].astype(np.uint8)
    hull = np.zeros_like(face)
    cs, _ = cv2.findContours(face, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cs:
        cv2.drawContours(hull, [cv2.convexHull(max(cs, key=cv2.contourArea))], -1, 1, -1)
    inner = cv2.erode(hull, np.ones((3, 3), np.uint8), iterations=2)
    dark = ((panel[y0:y1 + 1, x0:x1 + 1] < 0.42) & (inner > 0)).astype(np.uint8)
    blobs = []
    for m, st, c in _cc(dark, max(6, fw * fh // 2500)):
        bx, by, bw, bh, ar = st
        blobs.append(dict(c=np.array([c[0] + x0, c[1] + y0]), w=bw, h=bh, area=ar, ry=(c[1]) / fh))
    eyes = sorted([b for b in blobs if 0.15 < b['ry'] < 0.62 and b['w'] < fw * 0.45], key=lambda b: -b['area'])[:2]
    # facing: skin centroid vs. hull centre -> where the nose/profile points
    hx = np.nonzero(hull)[1].mean() + x0
    facing = 1.0 if f['face_c'][0] >= hx else -1.0
    if len(eyes) == 2 and abs(eyes[0]['c'][1] - eyes[1]['c'][1]) < fh * 0.2:
        eyes = sorted(eyes, key=lambda b: b['c'][0])
    elif len(eyes) >= 1:
        e = eyes[0]
        eyes = [e]                       # profile / 3-4 view: one visible eye
    else:
        eyes = [dict(c=np.array([x0 + fw * (0.5 + 0.18 * facing), y0 + fh * 0.42]), w=fw * 0.15, h=fh * 0.08, area=0)]
    mouths = [b for b in blobs if b['ry'] > 0.58 and b['w'] > b['h'] * 1.3 and b not in eyes]
    if mouths:
        m = max(mouths, key=lambda b: b['w'] * (1 - abs(b['ry'] - 0.78)))
        mc, mw = m['c'], max(m['w'], fw * 0.12)
    else:
        ex = np.mean([e['c'][0] for e in eyes])
        mc, mw = np.array([ex + facing * fw * 0.04, y0 + fh * 0.78]), fw * 0.18
    mouth_front = mc + np.array([facing * mw * 0.5, 0])
    mouth_back = mc - np.array([facing * mw * 0.5, 0])
    es = [float(np.median([e['w'] for e in eyes])), float(np.median([e['h'] for e in eyes]))]
    return dict(eyes=[e['c'] for e in eyes], eye_size=[max(8, es[0] * 0.9), max(5, es[1] * 1.6)],
                mouth_front=mouth_front, mouth_back=mouth_back, facing=facing)


def _body_polygon(ink, faces, k, W, H):
    """Region of character k: ink silhouette restricted to its Voronoi column band."""
    xs = [f['neck'][0] for f in faces]
    order = np.argsort(xs)
    pos = list(order).index(k)
    lo = 0 if pos == 0 else int((xs[order[pos - 1]] + xs[k]) / 2)
    hi = W if pos == len(xs) - 1 else int((xs[order[pos + 1]] + xs[k]) / 2)
    m = np.zeros_like(ink); m[:, lo:hi] = ink[:, lo:hi]
    # keep the component touching the face
    n, lab = cv2.connectedComponents(m, connectivity=8)
    fx, fy = faces[k]['face_c'].astype(int)
    if n > 1:
        lid = lab[fy, fx] or np.bincount(lab[faces[k]['face'] > 0].ravel(), minlength=n)[1:].argmax() + 1
        m = (lab == lid).astype(np.uint8)
    m = cv2.dilate(m, np.ones((9, 9), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    poly = cv2.approxPolyDP(c, 6, True)[:, 0]
    return poly, m, (lo, hi)


def _shoulders(region, neck, head_h):
    ys = np.arange(int(neck[1] + 0.25 * head_h), min(region.shape[0] - 1, int(neck[1] + 1.4 * head_h)))
    if len(ys) == 0:
        return None
    widths = []
    for y in ys:
        xs = np.nonzero(region[y])[0]
        widths.append((xs.ptp() if len(xs) else 0, y, xs.min() if len(xs) else 0, xs.max() if len(xs) else 0))
    wmax = max(w[0] for w in widths)
    # shoulder row: first row reaching 85 % of the max width
    row = next(w for w in widths if w[0] >= 0.85 * wmax)
    _, y, xl, xr = row
    return dict(y=y, left=np.array([xl, y], float), right=np.array([xr, y], float))


def _upper_contour(region, x0, x1, n=5):
    out = []
    for x in np.linspace(x0, x1, n):
        col = np.nonzero(region[:, int(np.clip(x, 0, region.shape[1] - 1))])[0]
        if len(col):
            out.append([float(x), float(col.min())])
    return out


def auto_rig(scene, panel, min_face_area=None, debug=None):
    W, H = scene.W, scene.H
    skin = _skin_mask(scene)
    ink = _ink_mask(scene)
    min_face_area = min_face_area or int(W * H * 0.003)
    faces = []
    for blob, st, c in _cc(skin, min_face_area):
        f = _face_from_skin(blob, st)
        if f is None:
            continue
        fx0, fy0, fx1, fy1 = f['face_box']
        if (fy1 - fy0) < 0.6 * (fx1 - fx0) * 0.6:     # not face-shaped (a hand, a cake)
            continue
        faces.append(f)
    if not faces:
        raise RuntimeError('autorig: no face-like skin region found')
    faces.sort(key=lambda f: f['neck'][0])
    chars = {}
    dbg = []
    for k, f in enumerate(faces):
        x0, y0, x1, y1 = f['face_box']
        fw, fh = x1 - x0 + 1, y1 - y0 + 1
        head_h = fh * 1.45
        feat = _features(panel, f)
        poly, region, band = _body_polygon(ink, faces, k, W, H)
        sh = _shoulders(region, f['neck'], head_h)
        neck = f['neck']
        # head pins: ring around the face incl. hair (ink above/around the face)
        hc = np.array([(x0 + x1) / 2, y0 + fh * 0.45])
        R = np.array([fw * 0.85, fh * 0.75])
        hair_top = np.nonzero(region[:, int(hc[0])])[0]
        top_y = float(hair_top.min()) if len(hair_top) else y0 - fh * 0.3
        ring = []
        for a in np.linspace(-np.pi, np.pi, 8, endpoint=False):
            p = hc + R * np.array([np.cos(a), np.sin(a)])
            if np.sin(a) < -0.3:
                p[1] = min(p[1], top_y + 8)
            if p[1] > neck[1] - 4:
                p[1] = neck[1] - 4
            ring.append(p)
        head = {f'h{i}': p for i, p in enumerate(ring)}
        head['hc'] = hc
        body = {}
        if sh is not None:
            sy = sh['y']
            body.update(b0=np.array([neck[0] - fw * 0.45, sy + 10]), b1=np.array([neck[0] + fw * 0.45, sy + 10]),
                        b2=np.array([neck[0], sy + fh * 0.5]))
            rest_y = min(H - 10, sy + head_h * 1.6)
            row = np.nonzero(region[int(min(H - 1, rest_y))])[0]
            if len(row):
                body.update(b3=np.array([row.min() + 15, rest_y]), b4=np.array([row.max() - 15, rest_y]),
                            b5=np.array([row.mean(), min(H - 12, rest_y + head_h)]))
            # shoulder chain on the facing side (the arm that reaches)
            side = sh['right'] if feat['facing'] > 0 else sh['left']
            inward = -feat['facing']
            body.update(s0=side + [inward * 12, 6], s1=side + [inward * 8, head_h * 0.9], s2=side + [inward * 10, head_h * 1.7])
            far = sh['right'] if feat['facing'] < 0 else sh['left']
            far_side = sh['left'] if feat['facing'] < 0 else sh['right']
        pins = {**head, **{k_: v for k_, v in body.items()}}
        pins = {k_: [round(float(v[0]), 1), round(float(min(H - 2, max(2, v[1])))), ] for k_, v in pins.items()}
        # shoulder landmarks for hand_on_shoulder: the shoulder AWAY from the viewer-facing side
        lm = dict(neck=neck, mouth_back=feat['mouth_back'], mouth_front=feat['mouth_front'])
        for i, e in enumerate(feat['eyes']):
            lm['eye_l' if (i == 0 and len(feat['eyes']) == 2) else 'eye_r'] = e
        if sh is not None:
            s_far = sh['right'] if neck[0] > W / 2 or feat['facing'] < 0 else sh['left']
            s_far = sh['right'] if feat['facing'] < 0 else sh['left']
            # contour along the top of that shoulder
            dx = 1 if s_far[0] > neck[0] else -1
            line = _upper_contour(region, s_far[0] - dx * fw * 0.55, s_far[0] + dx * fw * 0.05, 5)
            if dx < 0:
                line = line[::-1]
            lm['shoulder_far'] = np.array([s_far[0] - dx * fw * 0.25, sh['y'] - 2])
            lm['shoulder_far_line'] = line
        name = f'char{k}'
        smile_dir = [0.15 * feat['facing'], -1]
        secondary = [
            dict(name='hair_side', center=[float(hc[0] - feat['facing'] * fw * 0.85), float(neck[1] - fh * 0.1)],
                 radius=float(fw * 0.9), root=[float(hc[0]), float(y0)], stiffness=32, damping=5.5, gain=1.2, wind=0.7, seed=k * 3 + 1),
            dict(name='hair_top', center=[float(hc[0]), float(top_y + fh * 0.15)], radius=float(fw * 0.7),
                 root=[float(hc[0]), float(y0 + fh * 0.3)], stiffness=42, damping=6, gain=0.9, wind=0.4, seed=k * 3 + 2)]
        chars[name] = dict(
            z=k, silhouette=True, mesh_edge=int(np.clip(fw / 6, 14, 28)), reach_side='r' if feat['facing'] > 0 else 'l',
            polygon=[[int(a), int(b)] for a, b in poly],
            landmarks={k_: (np.round(v, 1).tolist() if not isinstance(v, list) else v) for k_, v in lm.items()},
            eye_size=[round(feat['eye_size'][0], 1), round(feat['eye_size'][1], 1)], smile_dir=smile_dir,
            facing=feat['facing'], pins=pins,
            pin_groups=dict(head=[p for p in pins if p.startswith('h')], chest=[p for p in ('b0', 'b1', 'b2') if p in pins],
                            shoulder_r=[p for p in ('s0', 's1', 's2') if p in pins],
                            shoulder_l=[p for p in ('s0', 's1', 's2') if p in pins]),
            secondary=secondary)
        dbg.append((f, feat, sh, poly, pins, lm))
    # z-order: the character drawn lower/overlapping in front -> larger neck y = in front;
    # ties: the one whose polygon overlaps the other's face more is behind
    order = sorted(chars, key=lambda n: chars[n]['landmarks']['neck'][1])
    for z, n in enumerate(order):
        chars[n]['z'] = z
    chars[order[0]]['silhouette'] = len(order) == 1
    rig = dict(_doc='auto-generated by mograph.vecanim.autorig', characters=chars)
    if debug:
        _debug_sheet(panel, chars, debug)
    return rig


def _debug_sheet(panel, chars, path):
    im = cv2.cvtColor((np.clip(panel, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cols = [(0, 200, 255), (255, 120, 0), (0, 255, 0), (255, 0, 255)]
    for k, (n, c) in enumerate(chars.items()):
        col = cols[k % len(cols)]
        cv2.polylines(im, [np.array(c['polygon'], np.int32)], True, col, 2)
        for p in c['pins'].values():
            cv2.circle(im, (int(p[0]), int(p[1])), 5, col, -1)
        for key, v in c['landmarks'].items():
            if key == 'shoulder_far_line':
                cv2.polylines(im, [np.array(v, np.int32)], False, (0, 0, 255), 2)
                continue
            cv2.drawMarker(im, (int(v[0]), int(v[1])), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
            cv2.putText(im, key, (int(v[0]) + 6, int(v[1]) - 6), 0, 0.45, (0, 0, 255), 1)
        cv2.putText(im, f"{n} z={c['z']} facing={'R' if c['facing'] > 0 else 'L'}", (c['polygon'][0][0], 24 + 22 * k), 0, 0.7, col, 2)
    cv2.imwrite(path, im)
