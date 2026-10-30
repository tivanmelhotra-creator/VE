"""Measure face features on an ORIGINAL (with-face) x4 character plate → kit dict (screen L/R eyes, brows, nose,
mouth).  Output is reviewed visually (overlay) before it is frozen into shapes.py."""
import json, sys
import cv2, numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


def order_skel(mask):
    sk = skeletonize(mask > 0)
    ys, xs = np.where(sk)
    if len(xs) < 3:
        return np.c_[xs, ys].astype(float)
    pts = set(zip(xs.tolist(), ys.tolist()))
    def nb(p):
        x, y = p
        return [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx or dy) and (x + dx, y + dy) in pts]
    ends = [p for p in pts if len(nb(p)) == 1] or [min(pts)]
    start = min(ends, key=lambda p: p[0])
    path, seen, cur = [start], {start}, start
    while True:
        n = [q for q in nb(cur) if q not in seen]
        if not n:
            break
        cur = min(n, key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
        path.append(cur); seen.add(cur)
    return np.array(path, float)


def resample(p, n):
    p = np.asarray(p, float)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    s = np.linspace(0, d[-1], n)
    return np.c_[np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])]


def measure(path):
    im = cv2.imread(path)
    sm = cv2.medianBlur(im, 3)
    g = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 235)).astype(np.uint8)
    white[:150] = 0
    white[900:] = 0
    n, l, st, ce = cv2.connectedComponentsWithStats(white, connectivity=4)
    cand = sorted([i for i in range(1, n) if st[i, 4] > 1500 and st[i, 2] < 300], key=lambda i: -st[i, 4])[:2]
    cand = sorted(cand, key=lambda i: st[i, 0])
    dark = (g < 110).astype(np.uint8)
    out = {"eyes": {}, "brows": {}}
    for side, i in zip(("L", "R"), cand):
        m = (l == i).astype(np.uint8)
        m = cv2.dilate(m, np.ones((5, 5), np.uint8))
        filled = ndi.binary_fill_holes(m).astype(np.uint8)
        cs, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
        x, y, w, h = cv2.boundingRect(filled)
        # pupil = dark pixels enclosed by the sclera
        pm = dark & filled & (cv2.erode(filled, np.ones((7, 7), np.uint8)))
        np_, lp, sp, cp = cv2.connectedComponentsWithStats(pm, connectivity=8)
        j = 1 + int(np.argmax(sp[1:, 4])) if np_ > 1 else None
        if j:
            ys, xs = np.where(lp == j)
            pupil = (float(xs.mean()), float(ys.mean()))
            pr = (float(np.ptp(xs) / 2 + 1), float(np.ptp(ys) / 2 + 1))
        else:
            pupil, pr = (x + w / 2, y + h / 2), (w * 0.15, h * 0.25)
        # lid thickness above the sclera at inner/outer quarter
        def run_above(xq):
            col = dark[:, int(xq)]
            top = int(np.where(filled[:, int(xq)])[0].min())
            k = 0
            while top - 1 - k > 0 and col[top - 1 - k]:
                k += 1
            return float(k)
        a, b = run_above(x + 0.2 * w), run_above(x + 0.8 * w)
        out["eyes"][side] = {"outline": resample(np.vstack([c, c[:1]]), 16)[:-1].round(1).tolist(),
                             "pupil": [round(pupil[0], 1), round(pupil[1], 1)], "pupil_r": [round(pr[0], 1), round(pr[1], 1)],
                             "lid_run": [a, b], "bbox": [x, y, w, h]}
    # skin + hair colours
    eL, eR = out["eyes"]["L"]["bbox"], out["eyes"]["R"]["bbox"]
    cx = int((eL[0] + eL[2] / 2 + eR[0] + eR[2] / 2) / 2)
    ybot = max(eL[1] + eL[3], eR[1] + eR[3])
    skin = np.median(sm[ybot + 40:ybot + 50, cx - 10:cx + 10].reshape(-1, 3), 0)
    hair = np.array([202, 233, 252.])
    hm = (np.linalg.norm(sm.astype(float) - hair, axis=2) < 18).astype(np.uint8)
    for side, e in (("L", eL), ("R", eR)):
        x, y, w, h = e
        box = np.zeros_like(hm); box[max(0, y - int(0.9 * h)):y - 2, x - int(0.15 * w):x + int(1.15 * w)] = 1
        nb_, lb, sb, _ = cv2.connectedComponentsWithStats(hm & box, connectivity=4)
        best = None
        for k in range(1, nb_):
            if sb[k, 4] > 150 and sb[k, 2] > 0.4 * w and sb[k, 3] < 0.5 * h:
                if best is None or sb[k, 4] > sb[best, 4]:
                    best = k
        if best is None:
            continue
        mk = lb == best
        xs = np.arange(sb[best, 0], sb[best, 0] + sb[best, 2])
        pts = []
        for xx in xs:
            yy = np.where(mk[:, xx])[0]
            if len(yy):
                pts.append((xx, yy.mean(), np.ptp(yy) + 1))
        pts = np.array(pts, float)
        line = resample(pts[:, :2], 7)
        if side == "R":
            line = line[::-1]           # outer→inner: R brow outer end is on the right
        out["brows"][side] = {"line": line.round(1).tolist(), "th": round(float(np.percentile(pts[:, 2], 75)) + 6, 1)}
    # nose + mouth: dark strokes below the eyes, between them
    zone = np.zeros(dark.shape, bool)
    zone[ybot - 30:ybot + 260, eL[0] + int(0.3 * eL[2]):eR[0] + eR[2]] = True
    zone &= ~(cv2.dilate(((l == cand[0]) | (l == cand[1])).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0)
    nd, ld, sd, cd = cv2.connectedComponentsWithStats(((dark > 0) & zone).astype(np.uint8), connectivity=8)
    comps = [k for k in range(1, nd) if sd[k, 4] > 60]
    comps.sort(key=lambda k: cd[k][1])
    mouth = max(comps, key=lambda k: sd[k, 2] * (cd[k][1] > ybot + 60)) if comps else None
    noses = [k for k in comps if k != mouth and cd[k][1] < cd[mouth][1] - 20]
    if noses:
        nk = max(noses, key=lambda k: sd[k, 4])
        sk = order_skel(ld == nk)
        out["nose"] = {"line": resample(sk, 7).round(1).tolist()}
    if mouth:
        sk = order_skel(ld == mouth)
        sk = sk[np.argsort(sk[:, 0])]
        out["mouth"] = {"left": sk[0].round(1).tolist(), "right": sk[-1].round(1).tolist(),
                        "line": resample(sk, 9).round(1).tolist()}
    out["skin_bgr"] = skin.round().tolist()
    return out, im


def overlay(im, k, path):
    v = im.copy()
    for s, e in k["eyes"].items():
        cv2.polylines(v, [np.array(e["outline"], np.int32)], True, (255, 0, 0), 2)
        cv2.ellipse(v, (int(e["pupil"][0]), int(e["pupil"][1])), (int(e["pupil_r"][0]), int(e["pupil_r"][1])), 0, 0, 360, (0, 255, 0), 2)
    for s, b in k["brows"].items():
        p = np.array(b["line"], np.int32); cv2.polylines(v, [p], False, (0, 0, 255), 3); cv2.circle(v, tuple(p[0]), 6, (0, 0, 255), -1)
    if "nose" in k:
        cv2.polylines(v, [np.array(k["nose"]["line"], np.int32)], False, (255, 0, 255), 3)
    if "mouth" in k:
        cv2.polylines(v, [np.array(k["mouth"]["line"], np.int32)], False, (0, 200, 255), 3)
    cv2.imwrite(path, v)


if __name__ == "__main__":
    res = {}
    W = "/home/user/work_tf/"
    tiles = []
    for name in ("gf", "g34", "bf", "b34"):
        k, im = measure(W + name + "_orig.png")
        res[name] = k
        overlay(im, k, W + "out/ov_" + name + ".png")
        tiles.append(cv2.resize(cv2.imread(W + "out/ov_" + name + ".png")[250:800, 200:860], None, fx=0.6, fy=0.6))
    json.dump(res, open(W + "out/measured.json", "w"), indent=1)
    cv2.imwrite(W + "out/ov_all.jpg", np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    for n, k in res.items():
        print(n, {s: (e["bbox"], e["lid_run"]) for s, e in k["eyes"].items()}, "brows", list(k["brows"]), "nose" in k, "mouth" in k)
