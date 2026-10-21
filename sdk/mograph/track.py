"""
mograph.track — v11 offline motion tracking (Motion Tracking / Point Track / Planar Track / body landmarks).

Every tracker runs ONCE per (video, settings), on a downscaled copy of the clip (≤ 540 px wide), and is cached as
JSON next to the video: <video>.track/<kind>_<hash>.json.  Coordinates are stored NORMALISED (u, v in 0..1 of the
source frame) so the same track works at any placement / zoom of the footage.

Kinds (all declared inside a `footage` element:  "trackers": {"name": {...}}):
  point    {"kind":"point", "at":[u,v], "frame_t": 2.0, "win": 0.06}
           Lucas–Kanade feature cloud around a point, forward + backward from frame_t, forward-backward error
           check, re-seeding — "text stuck to a hand / a product".  Output: x, y, scale, rot.
  box      {"kind":"box", "box":[u0,v0,u1,v1], "frame_t": 0}         median-flow box (centre + scale + rotation)
  planar   {"kind":"planar", "quad":[[u,v]x4 TL,TR,BR,BL], "frame_t": 0}
           Mocha-style planar track: features inside the quad → RANSAC homography chained to the reference
           frame (drift-corrected).  Output: 4 corners per frame (screen replacement / box art swap).
  hand     {"kind":"hand", "point":"index_tip|thumb_tip|middle_tip|wrist|palm|pinch", "hand":"any|left|right"}
           MediaPipe hand landmarker (bundled .task, CPU).  Gaps are interpolated.
  pose     {"kind":"pose", "point":"nose|left_shoulder|right_shoulder|left_wrist|right_wrist|chest|head_top|..."}
  face     {"kind":"face"}   face centre + size (Haar, very fast) — auto-reframe / follow the head.
  color    {"kind":"color", "at":[u,v] | "hsv":[h,s,v], "tol":14}  centroid of a coloured object.

Common keys: "step" (analyse every Nth frame, interpolate the rest; default 1, hand/pose 2), "smooth" (One-Euro
min_cutoff, lower = smoother, default 1.2), "start"/"end" (source seconds to analyse, default whole clip).

API:  get_track(video_path, spec) -> Track (sample(src_t) -> dict(x,y,s,rot,conf[,quad]))
CLI:  python3 -m mograph track spec.json   → computes every tracker of the spec + a debug contact sheet.
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(_HERE, "models")
WORK_W = 540
VERSION = 3

HAND_POINTS = {"wrist": 0, "thumb_tip": 4, "index_tip": 8, "middle_tip": 12, "ring_tip": 16, "pinky_tip": 20,
               "index_mcp": 5, "middle_mcp": 9}
POSE_POINTS = {"nose": 0, "left_eye": 2, "right_eye": 5, "left_ear": 7, "right_ear": 8, "mouth_left": 9,
               "mouth_right": 10, "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
               "left_wrist": 15, "right_wrist": 16, "left_index": 19, "right_index": 20, "left_hip": 23,
               "right_hip": 24, "left_knee": 25, "right_knee": 26}


# ---------------------------------------------------------------------------
# result object
# ---------------------------------------------------------------------------
class Track:
    """Per-frame samples (normalised).  sample(src_t) interpolates linearly; outside the analysed range the
    nearest valid sample is held."""

    def __init__(self, data):
        self.data = data
        self.fps = float(data["fps"])
        self.kind = data["kind"]
        self.x = np.asarray(data["x"], np.float32)
        self.y = np.asarray(data["y"], np.float32)
        self.s = np.asarray(data.get("s") or [1.0] * len(self.x), np.float32)
        self.r = np.asarray(data.get("r") or [0.0] * len(self.x), np.float32)
        self.c = np.asarray(data.get("c") or [1.0] * len(self.x), np.float32)
        self.q = np.asarray(data["q"], np.float32) if data.get("q") else None   # (n, 4, 2)
        self.aspect = float(data.get("aspect", 1.0))      # source h / w (for rotation-correct offsets)

    def __len__(self):
        return len(self.x)

    def sample(self, src_t):
        n = len(self.x)
        if n == 0:
            return {"x": 0.5, "y": 0.5, "s": 1.0, "rot": 0.0, "conf": 0.0}
        f = src_t * self.fps
        i = int(math.floor(f))
        k = f - i
        i0, i1 = max(0, min(n - 1, i)), max(0, min(n - 1, i + 1))
        lerp = lambda a: float(a[i0] * (1 - k) + a[i1] * k)
        out = {"x": lerp(self.x), "y": lerp(self.y), "s": lerp(self.s), "rot": lerp(self.r), "conf": lerp(self.c)}
        if self.q is not None:
            out["quad"] = (self.q[i0] * (1 - k) + self.q[i1] * k).tolist()
        return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _key(path, spec):
    st = os.stat(path)
    blob = json.dumps({"v": VERSION, "spec": {k: v for k, v in spec.items() if not k.startswith("_")},
                       "size": st.st_size, "mtime": int(st.st_mtime)}, sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()[:10]


def cache_path(path, spec):
    d = path + ".track"
    return os.path.join(d, f"{spec.get('kind', 'point')}_{_key(path, spec)}.json")


def _read_frames(path, start=0.0, end=None, step=1, work_w=WORK_W):
    """Yield (frame_index, small_bgr) for the analysed range."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sc = min(1.0, work_w / max(1, w))
    f0 = int(round(start * fps)); f1 = n if end is None else min(n, int(round(end * fps)) + 1)
    if f0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    i = f0
    while i < f1:
        ok, fr = cap.read()
        if not ok:
            break
        if (i - f0) % step == 0:
            yield i, (cv2.resize(fr, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else fr)
        i += 1
    cap.release()


def video_info(path):
    cap = cv2.VideoCapture(path)
    info = {"fps": cap.get(cv2.CAP_PROP_FPS) or 30.0, "n": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return info


class OneEuro:
    """One-Euro filter (Casiez 2012): smooth at rest, responsive when moving."""

    def __init__(self, fps, min_cutoff=1.2, beta=0.02, d_cutoff=1.0):
        self.fps, self.mc, self.beta, self.dc = fps, min_cutoff, beta, d_cutoff
        self.x = self.dx = None

    @staticmethod
    def _a(cut, fps):
        tau = 1.0 / (2 * math.pi * cut)
        return 1.0 / (1.0 + tau * fps)

    def __call__(self, v):
        if self.x is None:
            self.x, self.dx = v, 0.0
            return v
        dx = (v - self.x) * self.fps
        self.dx += self._a(self.dc, self.fps) * (dx - self.dx)
        cut = self.mc + self.beta * abs(self.dx)
        self.x += self._a(cut, self.fps) * (v - self.x)
        return self.x


def _fill(arr, valid):
    """Linear interpolation over invalid samples (hold at the ends)."""
    arr = np.asarray(arr, np.float64)
    valid = np.asarray(valid, bool)
    if valid.sum() == 0:
        return arr
    idx = np.arange(len(arr))
    return np.interp(idx, idx[valid], arr[valid])


def _smooth(arr, fps, mc, beta=0.02, both=True):
    if mc is None or mc <= 0:
        return np.asarray(arr)
    f = OneEuro(fps, mc, beta)
    a = np.array([f(float(v)) for v in arr])
    if both:                         # zero-lag: forward + backward pass averaged
        g = OneEuro(fps, mc, beta)
        b = np.array([g(float(v)) for v in arr[::-1]])[::-1]
        a = (a + b) / 2
    return a


def _finish(info, kind, xs, ys, valid, spec, ss=None, rs=None, quads=None, conf=None):
    """Dense per-frame arrays (frame 0..n-1) from sparse samples → fill gaps, smooth, normalise."""
    n = info["n"]
    full = lambda a, d: np.full(n, d, np.float64) if a is None else a
    X, Y = full(None, np.nan), full(None, np.nan)
    S, R, C = full(None, np.nan), full(None, np.nan), np.zeros(n)
    V = np.zeros(n, bool)
    Q = np.full((n, 4, 2), np.nan) if quads is not None else None
    for i, x in xs.items():
        if 0 <= i < n:
            X[i], Y[i] = x, ys[i]
            V[i] = valid.get(i, True)
            S[i] = ss.get(i, 1.0) if ss else 1.0
            R[i] = rs.get(i, 0.0) if rs else 0.0
            C[i] = conf.get(i, 1.0 if V[i] else 0.0) if conf else (1.0 if V[i] else 0.0)
            if Q is not None and i in quads:
                Q[i] = quads[i]
    if V.sum() == 0:
        V[:] = True
        X[:] = 0.5; Y[:] = 0.5; S[:] = 1; R[:] = 0
    mc = float(spec.get("smooth", 1.2))
    fps = info["fps"]
    X = _smooth(_fill(X, V), fps, mc); Y = _smooth(_fill(Y, V), fps, mc)
    S = _smooth(_fill(S, V), fps, mc * 0.6); R = _smooth(np.unwrap(_fill(R, V)), fps, mc * 0.6)
    C = _fill(np.where(V, np.maximum(C, 0.05), np.nan), V)
    out = {"kind": kind, "fps": fps, "n": n, "aspect": info["h"] / max(1, info["w"]),
           "x": np.round(X, 5).tolist(), "y": np.round(Y, 5).tolist(), "s": np.round(S, 4).tolist(),
           "r": np.round(R, 4).tolist(), "c": np.round(C, 3).tolist(), "valid_ratio": round(float(V.mean()), 3)}
    if Q is not None:
        Vq = ~np.isnan(Q[:, 0, 0])
        if Vq.sum():
            for a in range(4):
                for b in range(2):
                    Q[:, a, b] = _smooth(_fill(Q[:, a, b], Vq), fps, mc * 1.5)
        out["q"] = np.round(Q, 5).tolist()
    return out


def _uv(spec, key, default):
    v = spec.get(key, default)
    return float(v[0]), float(v[1])


# ---------------------------------------------------------------------------
# feature trackers (point / box / planar) — bidirectional from a reference frame
# ---------------------------------------------------------------------------
_LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def _grab(path, spec, info):
    frames = {}
    for i, fr in _read_frames(path, float(spec.get("start", 0.0)), spec.get("end"), 1):
        frames[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    return frames


def _seed(gray, poly, max_pts=80):
    m = np.zeros_like(gray)
    cv2.fillPoly(m, [np.int32(poly)], 255)
    p = cv2.goodFeaturesToTrack(gray, max_pts, 0.01, 4, mask=m, blockSize=5)
    return p if p is not None else np.empty((0, 1, 2), np.float32)


def _lk_step(g0, g1, p0):
    if len(p0) < 3:
        return p0, p0, np.zeros(len(p0), bool)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **_LK)
    pb, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **_LK)
    fb = np.linalg.norm((p0 - pb).reshape(-1, 2), axis=1)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
    return p0, p1, ok


def track_features(path, spec, info, mode):
    frames = _grab(path, spec, info)
    if not frames:
        raise RuntimeError(f"no frames read from {path}")
    keys = sorted(frames)
    g = frames[keys[0]]
    H, W = g.shape
    fr = int(round(float(spec.get("frame_t", 0.0)) * info["fps"]))
    fr = min(max(fr, keys[0]), keys[-1])
    if mode == "point":
        u, v = _uv(spec, "at", (0.5, 0.5))
        r = float(spec.get("win", 0.06)) * W
        poly = np.array([[u * W - r, v * H - r], [u * W + r, v * H - r], [u * W + r, v * H + r], [u * W - r, v * H + r]])
    elif mode == "box":
        b = spec.get("box", [0.4, 0.4, 0.6, 0.6])
        poly = np.array([[b[0] * W, b[1] * H], [b[2] * W, b[1] * H], [b[2] * W, b[3] * H], [b[0] * W, b[3] * H]])
    else:
        q = spec.get("quad") or [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]]
        poly = np.array([[a * W, b * H] for a, b in q])
    ref_c = poly.mean(axis=0)
    xs, ys, ss, rs, valid, quads = {}, {}, {}, {}, {}, {}

    def run(order):
        cur = poly.astype(np.float64).copy()
        Hacc = np.eye(3)
        p = _seed(frames[order[0]], cur)
        prev = order[0]
        for idx in order[1:]:
            g0, g1 = frames[prev], frames[idx]
            p0, p1, ok = _lk_step(g0, g1, p)
            good = ok.sum()
            if good >= (4 if mode == "planar" else 3):
                a, b = p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)
                if mode == "planar":
                    Hm, inl = cv2.findHomography(a, b, cv2.RANSAC, 2.5)
                    if Hm is None:
                        Hm = np.eye(3)
                    Hacc = Hm @ Hacc
                    cur = cv2.perspectiveTransform(poly.reshape(-1, 1, 2).astype(np.float64), Hacc).reshape(-1, 2)
                else:
                    M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                    if M is None:
                        d = np.median(b - a, axis=0)
                        M = np.array([[1, 0, d[0]], [0, 1, d[1]]], np.float64)
                    cur = (np.c_[cur, np.ones(4)] @ M.T)
                valid[idx] = True
                keep = p1[ok]
            else:
                valid[idx] = False
                keep = p1[ok] if good else p
            c = cur.mean(axis=0)
            xs[idx], ys[idx] = c[0] / W, c[1] / H
            e = cur[1] - cur[0]
            e0 = poly[1] - poly[0]
            ss[idx] = float(np.linalg.norm(e) / max(1e-6, np.linalg.norm(e0)))
            rs[idx] = float(math.atan2(e[1], e[0]) - math.atan2(e0[1], e0[0]))
            if mode == "planar":
                quads[idx] = [[pt[0] / W, pt[1] / H] for pt in cur]
            if len(keep) < 25:                  # re-seed inside the current region
                p = _seed(g1, cur)
            else:
                p = keep.reshape(-1, 1, 2)
            prev = idx

    xs[fr], ys[fr] = ref_c[0] / W, ref_c[1] / H
    ss[fr], rs[fr], valid[fr] = 1.0, 0.0, True
    if mode == "planar":
        quads[fr] = [[pt[0] / W, pt[1] / H] for pt in poly]
    fwd = [k for k in keys if k >= fr]
    bwd = [k for k in keys if k <= fr][::-1]
    if len(fwd) > 1:
        run(fwd)
    if len(bwd) > 1:
        run(bwd)
    return _finish(info, mode, xs, ys, valid, spec, ss, rs, quads if mode == "planar" else None)


# ---------------------------------------------------------------------------
# landmark trackers (mediapipe) / face / colour
# ---------------------------------------------------------------------------
def _mp_detector(kind, num=2):
    import mediapipe as mp  # noqa: F401
    from mediapipe.tasks.python import BaseOptions, vision
    if kind == "hand":
        opt = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(MODELS, "hand_landmarker.task"),
                                     delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_hands=num, min_hand_detection_confidence=0.4,
            min_tracking_confidence=0.4)
        return vision.HandLandmarker.create_from_options(opt)
    opt = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODELS, "pose_landmarker_lite.task"),
                                 delegate=BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    return vision.PoseLandmarker.create_from_options(opt)


def _hand_point(lms, name):
    if name == "palm":
        ids = [0, 5, 9, 13, 17]
        return sum(lms[i].x for i in ids) / 5, sum(lms[i].y for i in ids) / 5
    if name == "pinch":
        return (lms[4].x + lms[8].x) / 2, (lms[4].y + lms[8].y) / 2
    i = HAND_POINTS.get(name, 8)
    return lms[i].x, lms[i].y


def _pose_point(lms, name):
    if name == "chest":
        return (lms[11].x + lms[12].x) / 2, (lms[11].y + lms[12].y) * 0.5 + 0.03
    if name == "head_top":
        sh = abs(lms[11].x - lms[12].x)
        return lms[0].x, lms[0].y - sh * 0.55
    if name == "shoulders":
        return (lms[11].x + lms[12].x) / 2, (lms[11].y + lms[12].y) / 2
    i = POSE_POINTS.get(name, 0)
    return lms[i].x, lms[i].y


def track_landmarks(path, spec, info, kind):
    import mediapipe as mp
    step = int(spec.get("step", 2))
    det = _mp_detector(kind)
    xs, ys, ss, rs, valid, conf = {}, {}, {}, {}, {}, {}
    want = spec.get("hand", "any")
    name = spec.get("point", "index_tip" if kind == "hand" else "nose")
    last = None
    try:
        for i, fr in _read_frames(path, float(spec.get("start", 0.0)), spec.get("end"), step):
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            ts = int(i * 1000 / info["fps"])
            r = det.detect_for_video(img, ts)
            pick = None
            if kind == "hand":
                cands = []
                for j, lms in enumerate(r.hand_landmarks or []):
                    lab = r.handedness[j][0].category_name.lower() if r.handedness else "any"
                    # mediapipe labels assume a mirrored selfie image; flip so "right" = the subject's right hand
                    lab = {"left": "right", "right": "left"}.get(lab, lab)
                    if want in ("any", lab):
                        cands.append((lms, r.handedness[j][0].score if r.handedness else 1.0))
                if cands:
                    if last is not None and len(cands) > 1:     # temporal consistency: nearest to last position
                        cands.sort(key=lambda c: (_hand_point(c[0], name)[0] - last[0]) ** 2 +
                                   (_hand_point(c[0], name)[1] - last[1]) ** 2)
                    lms, sc = cands[0]
                    pick = _hand_point(lms, name)
                    w = math.hypot(lms[5].x - lms[17].x, lms[5].y - lms[17].y)
                    ss[i] = w / 0.12
                    rs[i] = math.atan2(lms[9].y - lms[0].y, lms[9].x - lms[0].x) + math.pi / 2
                    conf[i] = float(sc)
            else:
                if r.pose_landmarks:
                    lms = r.pose_landmarks[0]
                    pick = _pose_point(lms, name)
                    ss[i] = math.hypot(lms[11].x - lms[12].x, lms[11].y - lms[12].y) / 0.25
                    rs[i] = math.atan2(lms[11].y - lms[12].y, lms[11].x - lms[12].x)
                    conf[i] = float(getattr(lms[0], "visibility", 1.0) or 1.0)
            if pick is not None:
                xs[i], ys[i] = pick
                valid[i] = True
                last = pick
            else:
                xs[i], ys[i] = (last if last else (0.5, 0.5))
                valid[i] = False
    finally:
        try:
            det.close()
        except Exception:  # noqa: BLE001
            pass
    return _finish(info, kind, xs, ys, valid, spec, ss, rs, conf=conf)


_FACE = None


def track_face(path, spec, info):
    global _FACE
    if _FACE is None:
        _FACE = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    step = int(spec.get("step", 2))
    xs, ys, ss, valid = {}, {}, {}, {}
    last = None
    for i, fr in _read_frames(path, float(spec.get("start", 0.0)), spec.get("end"), step, work_w=360):
        g = cv2.equalizeHist(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        H, W = g.shape
        faces = _FACE.detectMultiScale(g, 1.15, 5, minSize=(int(W * 0.08), int(W * 0.08)))
        if len(faces):
            if last is not None:
                faces = sorted(faces, key=lambda f: (f[0] + f[2] / 2 - last[0] * W) ** 2 + (f[1] + f[3] / 2 - last[1] * H) ** 2)
            else:
                faces = sorted(faces, key=lambda f: -f[2] * f[3])
            x, y, w, h = faces[0]
            xs[i], ys[i] = (x + w / 2) / W, (y + h / 2) / H
            ss[i] = w / W / 0.25
            valid[i] = True
            last = (xs[i], ys[i])
        else:
            xs[i], ys[i] = last if last else (0.5, 0.35)
            valid[i] = False
    return _finish(info, "face", xs, ys, valid, spec, ss)


def track_color(path, spec, info):
    step = int(spec.get("step", 1))
    tol = float(spec.get("tol", 14))
    hsv0 = spec.get("hsv")
    xs, ys, ss, valid = {}, {}, {}, {}
    last = None
    a0 = None
    for i, fr in _read_frames(path, float(spec.get("start", 0.0)), spec.get("end"), step, work_w=320):
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        H, W = hsv.shape[:2]
        if hsv0 is None:
            u, v = _uv(spec, "at", (0.5, 0.5))
            patch = hsv[max(0, int(v * H) - 3):int(v * H) + 4, max(0, int(u * W) - 3):int(u * W) + 4]
            hsv0 = [float(np.median(patch[..., k])) for k in range(3)]
        m = hue_mask(hsv, hsv0[0], tol, smin=max(40, hsv0[1] * 0.45), vmin=max(30, hsv0[2] * 0.35))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, cents = cv2.connectedComponentsWithStats(m)
        if n > 1:
            j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            if last is not None and n > 2:
                big = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] > stats[j, cv2.CC_STAT_AREA] * 0.3]
                j = min(big, key=lambda k: (cents[k][0] / W - last[0]) ** 2 + (cents[k][1] / H - last[1]) ** 2)
            area = stats[j, cv2.CC_STAT_AREA]
            xs[i], ys[i] = cents[j][0] / W, cents[j][1] / H
            a0 = a0 or area
            ss[i] = math.sqrt(area / max(1, a0))
            valid[i] = area > 12
            last = (xs[i], ys[i])
        else:
            xs[i], ys[i] = last if last else (0.5, 0.5)
            valid[i] = False
    return _finish(info, "color", xs, ys, valid, spec, ss)


def hue_mask(hsv, h0, tol, smin=40, vmin=30):
    """uint8 mask of pixels whose hue is within ±tol (OpenCV 0..180 hue, wraps around red)."""
    h = hsv[..., 0].astype(np.int16)
    d = np.abs(h - int(round(h0)))
    d = np.minimum(d, 180 - d)
    return ((d <= tol) & (hsv[..., 1] >= smin) & (hsv[..., 2] >= vmin)).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
_MEM = {}


def get_track(path, spec, log=None):
    spec = dict(spec)
    kind = spec.setdefault("kind", "point")
    cp = cache_path(path, spec)
    if cp in _MEM:
        return _MEM[cp]
    if os.path.exists(cp):
        try:
            tr = Track(json.load(open(cp)))
            _MEM[cp] = tr
            return tr
        except Exception:  # noqa: BLE001
            pass
    info = video_info(path)
    if log:
        log(f"[track] {kind} on {os.path.basename(path)} ({info['n']} frames) …")
    if kind in ("point", "box", "planar"):
        data = track_features(path, spec, info, kind)
    elif kind in ("hand", "pose"):
        data = track_landmarks(path, spec, info, kind)
    elif kind == "face":
        data = track_face(path, spec, info)
    elif kind == "color":
        data = track_color(path, spec, info)
    else:
        raise ValueError(f"unknown tracker kind '{kind}'")
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    json.dump(data, open(cp, "w"))
    tr = Track(data)
    _MEM[cp] = tr
    if log:
        log(f"[track]   → {cp}  (valid {data.get('valid_ratio', 1):.0%})")
    return tr


def debug_sheet(path, tracks, out, n=8, thumb_w=270):
    """Contact sheet of the clip with every track drawn (cross + label, planar quads) — eyeball the result."""
    info = video_info(path)
    cap = cv2.VideoCapture(path)
    tiles = []
    for k in range(n):
        f = int((k + 0.5) * info["n"] / n)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            continue
        fr = cv2.resize(fr, (thumb_w, int(thumb_w * info["h"] / info["w"])))
        H, W = fr.shape[:2]
        for j, (name, tr) in enumerate(tracks.items()):
            s = tr.sample(f / info["fps"])
            col = [(0, 255, 255), (255, 80, 255), (80, 255, 80), (255, 160, 0)][j % 4]
            if s.get("quad"):
                cv2.polylines(fr, [np.int32([[a * W, b * H] for a, b in s["quad"]])], True, col, 2)
            x, y = int(s["x"] * W), int(s["y"] * H)
            cv2.drawMarker(fr, (x, y), col, cv2.MARKER_CROSS, 16, 2)
            cv2.putText(fr, name, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        cv2.putText(fr, f"{f / info['fps']:.1f}s", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(fr)
    cap.release()
    if not tiles:
        return None
    cols = 4
    rows = [np.hstack(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols])))
            for i in range(0, len(tiles), cols)]
    cv2.imwrite(out, np.vstack(rows))
    return out
