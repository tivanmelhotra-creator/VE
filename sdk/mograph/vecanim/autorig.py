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
    """Face = skin above the jaw.  Jaw = first row under the widest face rows where the width
    drops below 72 % of the face width (the neck is narrower than the face)."""
    x, y, w, h = stats[:4]
    rows = blob[y:y + h].sum(1).astype(float)
    sm = np.convolve(rows, np.ones(5) / 5, 'same')
    top_half = sm[:max(6, int(h * 0.7))]
    wmax_i = int(np.argmax(top_half)); wmax = top_half[wmax_i]
    jaw_rel = None
    for r in range(wmax_i, int(h * 0.92)):
        if sm[r] < 0.72 * wmax:
            jaw_rel = r; break
    if jaw_rel is None or jaw_rel < 12:
        return None
    jaw = y + jaw_rel
    face = blob.copy(); face[jaw:] = 0
    fy, fx = np.nonzero(face)
    if len(fx) < 80:
        return None
    nx = np.nonzero(blob[min(jaw + 6, blob.shape[0] - 1)])[0]
    nx = nx.mean() if len(nx) else fx.mean()
    return dict(jaw=float(jaw), neck_x=float(nx), face_box=(int(fx.min()), int(y), int(fx.max()), int(jaw)),
                face=face, face_c=np.array([fx.mean(), fy.mean()]), top=y)


def _holes(scene, box, min_area=12, dedupe=6.0):
    """Enclosed holes (inner sub-paths) of light fills inside a box: eyes, brows, mouth."""
    x0, y0, x1, y1 = box
    out = []
    for it in scene.items:
        if it.kind != 'fill' or len(it.paths) < 2:
            continue
        for P in it.paths[1:]:
            q = P[0::3].astype(np.float32)
            a = abs(cv2.contourArea(q))
            c = q.mean(0)
            if a < min_area or not (x0 <= c[0] <= x1 and y0 <= c[1] <= y1):
                continue
            bx0, by0 = q.min(0); bx1, by1 = q.max(0)
            out.append(dict(c=c.astype(float), w=float(bx1 - bx0), h=float(by1 - by0), area=float(a), tone=it.tone))
    # one hole per place (shade + light layers both carry it): keep the larger
    uniq = []
    for hcur in sorted(out, key=lambda d: -d['area']):
        if all(np.linalg.norm(hcur['c'] - u['c']) > dedupe for u in uniq):
            uniq.append(hcur)
    return uniq


def _dark_blobs(panel, f, thr=0.42):
    """Dark blobs inside the convex hull of the face (eyes that open into the hair are
    notches of the skin shape, not holes)."""
    x0, y0, x1, y1 = f['face_box']
    face = f['face'][y0:y1 + 1, x0:x1 + 1].astype(np.uint8)
    cs, _ = cv2.findContours(face, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return []
    hull = np.zeros_like(face)
    cv2.drawContours(hull, [cv2.convexHull(max(cs, key=cv2.contourArea))], -1, 1, -1)
    hull = cv2.erode(hull, np.ones((5, 5), np.uint8))
    dark = ((panel[y0:y1 + 1, x0:x1 + 1] < thr) & (hull > 0)).astype(np.uint8)
    out = []
    for m, st, c in _cc(dark, 12):
        bx, by, bw, bh, ar = st
        out.append(dict(c=np.array([c[0] + x0, c[1] + y0], float), w=float(bw), h=float(bh), area=float(ar)))
    return out


def _features(scene, panel, f):
    x0, y0, x1, y1 = f['face_box']
    fw, fh = x1 - x0 + 1, y1 - y0 + 1
    cand = _holes(scene, (x0, y0, x1, y1), dedupe=0.07 * fw) + _dark_blobs(panel, f)
    hs = []
    for hcur in sorted(cand, key=lambda d: -d['area']):          # merge duplicates across sources
        if all(np.linalg.norm(hcur['c'] - u['c']) > 0.07 * fw for u in hs):
            hs.append(hcur)
    for hh in hs:
        hh['ry'] = (hh['c'][1] - y0) / fh
    upper = [hh for hh in hs if 0.12 < hh['ry'] < 0.7 and hh['w'] < fw * 0.5 and hh['w'] >= 0.8 * hh['h']
             and hh['area'] >= fw * fh * 0.0015]
    eyes = []
    for hh in sorted(upper, key=lambda d: -d['area']):
        # brow sits above the eye: prefer the blob just below it
        below = [e for e in upper if e is not hh and abs(e['c'][0] - hh['c'][0]) < max(hh['w'], e['w']) * 0.7
                 and fh * 0.05 < e['c'][1] - hh['c'][1] < fh * 0.22]
        c_ = max(below, key=lambda e: e['area']) if below else hh
        if all(np.linalg.norm(c_['c'] - e['c']) > fw * 0.18 for e in eyes):
            eyes.append(c_)
        if len(eyes) == 2:
            break
    if len(eyes) == 2 and abs(eyes[0]['c'][1] - eyes[1]['c'][1]) > fh * 0.18:
        eyes = eyes[:1]
    fcx = (x0 + x1) / 2
    if eyes:
        ex = np.mean([e['c'][0] for e in eyes])
        facing = 1.0 if ex >= fcx else -1.0
        # a single eye far to one side = 3/4 or profile
    else:
        facing = 1.0 if f['face_c'][0] >= fcx else -1.0
        eyes = [dict(c=np.array([fcx + facing * fw * 0.2, y0 + fh * 0.45]), w=fw * 0.12, h=fh * 0.06)]
    eyes = sorted(eyes, key=lambda e: e['c'][0])
    ey = np.mean([e['c'][1] for e in eyes]); ex = np.mean([e['c'][0] for e in eyes])
    # mouth: elongated dark shape below the eyes, near them in x (profile: toward the facing side)
    mx_ref = ex + (0.0 if len(eyes) == 2 else -facing * 0.05 * fw)
    mouths = [hh for hh in hs if ey + fh * 0.2 < hh['c'][1] < ey + fh * 0.58 and abs(hh['c'][0] - mx_ref) < 0.3 * fw
              and hh['w'] > hh['h'] * 1.2]
    if mouths:
        m = min(mouths, key=lambda d: abs(d['c'][1] - (ey + 0.4 * fh)) - 0.3 * d['w'])
        mc, mw = m['c'], float(np.clip(m['w'], fw * 0.15, fw * 0.35))
    else:
        mc, mw = np.array([mx_ref, ey + fh * 0.42]), fw * 0.2
    es = [float(np.median([e['w'] for e in eyes])), float(np.median([e['h'] for e in eyes]))]
    return dict(eyes=[e['c'] for e in eyes], eye_size=[float(np.clip(es[0] * 0.75, 8, 30)), float(np.clip(es[1] * 1.3, 5, 16))],
                mouth_front=mc + np.array([facing * mw * 0.5, 0]), mouth_back=mc - np.array([facing * mw * 0.5, 0]),
                facing=facing, fw=fw, fh=fh)


def _body_region(ink, faces, k, W, H):
    xs = [f['neck_x'] for f in faces]
    order = list(np.argsort(xs)); pos = order.index(k)
    lo = 0 if pos == 0 else int((xs[order[pos - 1]] + xs[k]) / 2)
    hi = W if pos == len(xs) - 1 else int((xs[order[pos + 1]] + xs[k]) / 2)
    fw = faces[k]['face_box'][2] - faces[k]['face_box'][0]
    lo = max(lo, int(xs[k] - 2.6 * fw)); hi = min(hi, int(xs[k] + 2.6 * fw))
    m = np.zeros_like(ink); m[:, lo:hi] = ink[:, lo:hi]
    n, lab = cv2.connectedComponents(m, connectivity=8)
    if n > 1:
        ids = np.bincount(lab[faces[k]['face'] > 0].ravel(), minlength=n); ids[0] = 0
        m = (lab == int(ids.argmax())).astype(np.uint8) if ids.max() > 0 else m
    m = cv2.dilate(m, np.ones((9, 9), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    poly = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 6, True)[:, 0]
    return poly, m


def _snap_to_ink(panel, pts, reach):
    """Move each point vertically to the brightest line pixel within +-reach (outline snap)."""
    out = []
    H = panel.shape[0]
    for x, y in pts:
        xi = int(np.clip(x, 0, panel.shape[1] - 1))
        ys = np.arange(int(max(0, y - reach)), int(min(H - 1, y + reach)))
        if len(ys) == 0:
            out.append([x, y]); continue
        col = panel[ys, xi] - 0.002 * np.abs(ys - y)
        out.append([float(x), float(ys[int(np.argmax(col))])])
    return out


def auto_rig(scene, panel, min_face_area=None, debug=None):
    W, H = scene.W, scene.H
    skin = _skin_mask(scene)
    ink = _ink_mask(scene)
    min_face_area = min_face_area or int(W * H * 0.0015)
    faces = []
    for blob, st, c in _cc(skin, min_face_area):
        f = _face_from_skin(blob, st)
        if f is None:
            continue
        if not _holes(scene, f['face_box']):        # a face has features (eyes / mouth holes)
            continue
        faces.append(f)
    if not faces:
        raise RuntimeError('autorig: no face-like skin region found')
    faces.sort(key=lambda f: f['neck_x'])
    chars = {}
    for k, f in enumerate(faces):
        x0, y0, x1, y1 = f['face_box']
        feat = _features(scene, panel, f)
        fw, fh, fc = feat['fw'], feat['fh'], feat['facing']
        poly, region = _body_region(ink, faces, k, W, H)
        nx, jaw = f['neck_x'], f['jaw']
        neck = np.array([(nx + (x0 + x1) / 2) / 2, jaw + 0.3 * fh])      # pivot between neck and face centre
        hc = np.array([(x0 + x1) / 2, y0 + fh * 0.4])
        col = np.nonzero(region[:, int(np.clip(hc[0], 0, W - 1))])[0]
        top_y = float(col.min()) if len(col) else y0 - 0.3 * fh
        R = np.array([fw * 0.75, max(fh * 0.7, (hc[1] - top_y) * 0.85)])
        pins = {}
        for i, a in enumerate(np.linspace(-np.pi, np.pi, 8, endpoint=False)):
            p = hc + R * np.array([np.cos(a), np.sin(a)])
            p[1] = min(p[1], jaw - 3)
            pins[f'h{i}'] = p
        pins['hc'] = hc
        sy = jaw + 0.55 * fh                           # shoulder line (proportions)
        sw = 1.25 * fw                                # half shoulder span
        pins.update(b0=np.array([nx - 0.5 * fw, sy + 0.2 * fh]), b1=np.array([nx + 0.5 * fw, sy + 0.2 * fh]),
                    b2=np.array([nx, sy + 0.9 * fh]),
                    b3=np.array([nx - 0.9 * fw, min(H - 8, sy + 2.6 * fh)]), b4=np.array([nx + 0.9 * fw, min(H - 8, sy + 2.6 * fh)]),
                    b5=np.array([nx, min(H - 8, sy + 3.6 * fh)]))
        reach = fc                                    # the arm on the facing side reaches
        sx = nx + reach * sw
        pins.update(s0=np.array([sx - reach * 0.15 * fw, sy + 0.1 * fh]), s1=np.array([sx, sy + 1.1 * fh]),
                    s2=np.array([sx - reach * 0.05 * fw, sy + 2.0 * fh]))
        pins = {n: [round(float(np.clip(v[0], 2, W - 2)), 1), round(float(np.clip(v[1], 2, H - 2)), 1)] for n, v in pins.items()}
        lm = dict(neck=neck, mouth_back=feat['mouth_back'], mouth_front=feat['mouth_front'])
        if len(feat['eyes']) == 2:
            lm['eye_l'], lm['eye_r'] = feat['eyes']
        else:
            lm['eye_r'] = feat['eyes'][0]
        for side, sg in (('l', -1), ('r', 1)):
            line = [[nx + sg * fw * u, jaw + fh * v] for u, v in ((0.76, 0.43), (1.0, 0.5), (1.24, 0.58), (1.45, 0.68), (1.65, 0.81))]
            line = _snap_to_ink(panel, line, 0.18 * fh)
            ys = np.array([p[1] for p in line]); ys = np.convolve(np.pad(ys, 1, mode='edge'), np.ones(3) / 3, 'valid')
            line = [[round(p[0], 1), round(float(y_), 1)] for p, y_ in zip(line, ys)]
            if sg < 0:
                line = line[::-1]
            lm[f'shoulder_{side}_line'] = line
            lm[f'shoulder_{side}'] = [round(nx + sg * 1.2 * fw, 1), round(float(np.interp(nx + sg * 1.2 * fw, *zip(*sorted(line)))) + 0.0, 1)]
        name = f'char{k}'
        chars[name] = dict(
            z=k, silhouette=True, mesh_edge=int(np.clip(fw / 6, 14, 28)), reach_side='r' if fc > 0 else 'l',
            polygon=[[int(a), int(b)] for a, b in poly],
            landmarks={n: (np.round(np.asarray(v, float), 1).tolist() if not (isinstance(v, list) and v and isinstance(v[0], list)) else v)
                       for n, v in lm.items()},
            eye_size=[round(x, 1) for x in feat['eye_size']], smile_dir=[round(0.15 * fc, 2), -1], facing=fc, pins=pins,
            pin_groups=dict(head=[p for p in pins if p.startswith('h')], chest=['b0', 'b1', 'b2'],
                            **{f"shoulder_{'r' if fc > 0 else 'l'}": ['s0', 's1', 's2']}),
            secondary=[
                dict(name='hair_back', center=[float(hc[0] - fc * fw * 0.8), float(jaw)], radius=float(fw * 0.85),
                     root=[float(hc[0]), float(y0)], stiffness=32, damping=5.5, gain=1.2, wind=0.7, seed=3 * k + 1),
                dict(name='hair_top', center=[float(hc[0]), float(top_y + 0.2 * fh)], radius=float(fw * 0.65),
                     root=[float(hc[0]), float(y0 + 0.3 * fh)], stiffness=42, damping=6, gain=0.9, wind=0.4, seed=3 * k + 2)])
    order = sorted(chars, key=lambda n: chars[n]['landmarks']['neck'][1])     # lower neck = nearer = in front
    for z, n in enumerate(order):
        chars[n]['z'] = z
    if len(order) > 1:
        chars[order[0]]['silhouette'] = False
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
            if key.endswith('_line'):
                cv2.polylines(im, [np.array(v, np.int32)], False, (0, 0, 255), 2)
                continue
            cv2.drawMarker(im, (int(v[0]), int(v[1])), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
            cv2.putText(im, key, (int(v[0]) + 6, int(v[1]) - 6), 0, 0.45, (0, 0, 255), 1)
        cv2.putText(im, f"{n} z={c['z']} facing={'R' if c['facing'] > 0 else 'L'}", (c['polygon'][0][0], 24 + 22 * k), 0, 0.7, col, 2)
    cv2.imwrite(path, im)
