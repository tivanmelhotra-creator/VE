"""
mograph.v14.track — offline, ZERO-LAG speaker analysis (computed once per clip, cached next to the video).

v13.9 audit: live one-euro tracking is causal (text trails the speaker) and the 256 px selfie matte has halo edges
("behind the speaker" looks pasted).  v14 analyses the whole clip OFFLINE:
  matte  MODNet portrait matting (models/modnet.onnx) → hair-accurate soft alpha, temporal median of 3 frames
         cached as PNG in <video>.v14/matte/   (fallback: bundled selfie segmenter)
  pose   MediaPipe pose every frame → CENTRED Gaussian smoothing (no lag) → anchors:
         head_top face nose chin neck chest belly shoulder_l shoulder_r hand_l hand_r hands head_c
         + head_width / shoulder_width (u units) so layers scale with the speaker.
CLI:  python3 -m mograph.v14.track media/raw.mp4 [--engine modnet|rvm]     (run once; resumable)

v22 matte engine "rvm" (Robust Video Matting, MobileNetV3, 15 MB, idea from Concat concat-vision/segment.rs):
  a RECURRENT model — the hidden state r1..r4 is carried from frame t-1 to t, so the alpha is temporally coherent
  by construction (no 3-frame median, no hair/hijab edge flicker on fast moves) and it runs ≈ 0.18 s/frame on
  2 CPU cores vs ≈ 1.5 s for MODNet (≈ 8× faster prep).  Cached separately in <video>.v14/matte_rvm/ so the
  committed MODNet pack keeps working.  Spec: "source": {"matte_engine": "rvm"}.  Model: models_hub "rvm".
"""
from __future__ import annotations

import json
import math
import os
import sys

import cv2
import numpy as np

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def cache_dir(path):
    d = path + ".v14"
    os.makedirs(os.path.join(d, "matte"), exist_ok=True)
    return d


_MOD = None


def _modnet():
    global _MOD
    if _MOD is None:
        p = os.path.join(MODELS, "modnet.onnx")
        if not os.path.exists(p):          # model not in git → fetch once (sha256-verified)
            try:
                sys.path.insert(0, os.path.dirname(os.path.dirname(MODELS)))
                from mograph import fetch_assets
                fetch_assets.try_ensure("modnet.onnx")
            except Exception:  # noqa: BLE001
                pass
        if os.path.exists(p):
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.log_severity_level = 3
            _MOD = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
        else:
            _MOD = False
    return _MOD


def matte_frame(bgr, ref=512):
    s = _modnet()
    h, w = bgr.shape[:2]
    if not s:
        from ..matte import matte_onnx
        sm_b = cv2.resize(bgr, (270, 480), interpolation=cv2.INTER_AREA)
        m_s = matte_onnx(sm_b)
        return cv2.resize(m_s, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    sc = ref / min(h, w)
    nh, nw = max(32, int(h * sc) // 32 * 32), max(32, int(w * sc) // 32 * 32)
    x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
    m = s.run(None, {"input": ((x - 127.5) / 127.5).transpose(2, 0, 1)[None]})[0][0, 0]
    return np.clip(cv2.resize(m, (w, h), interpolation=cv2.INTER_CUBIC), 0, 1).astype(np.float32)


RVM_DOWNSAMPLE = 0.25          # RVM internal working scale for 1080p+ (author recommendation for HD/4K)


def _rvm_downsample(h, w):
    return float(min(1.0, max(0.125, 512.0 / min(h, w) * RVM_DOWNSAMPLE * 2)))


def prep_matte_rvm(path, log=print, reset_every=0):
    """RVM over the whole clip, sequential (the recurrent state needs order).  Resumable: on restart the state is
    warmed up on the 15 frames before the first missing one."""
    from ..models_hub import session
    s = session("rvm")
    if s is None:
        log("[v14 matte rvm] model unavailable → MODNet", flush=True)
        return prep_matte(path, log=log, engine="modnet")
    d = cache_dir(path)
    md = os.path.join(d, "matte_rvm")
    os.makedirs(md, exist_ok=True)
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = next((i for i in range(n) if not os.path.exists(os.path.join(md, f"{i:05d}.png"))), n)
    if first >= n:
        log(f"[v14 matte rvm] ready ({n} frames, 0 new)", flush=True)
        return
    start = max(0, first - 15)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    rec = [np.zeros((1, 1, 1, 1), np.float32)] * 4
    i, new = start, 0
    while i < n:
        ok, fr = cap.read()
        if not ok:
            break
        h, w = fr.shape[:2]
        src = np.ascontiguousarray((cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
        out = s.run(["pha", "r1o", "r2o", "r3o", "r4o"], {"src": src, "r1i": rec[0], "r2i": rec[1], "r3i": rec[2],
                                                          "r4i": rec[3], "downsample_ratio": np.array(
                                                              [_rvm_downsample(h, w)], np.float32)})
        pha, rec = out[0][0, 0], out[1:]
        if reset_every and i % reset_every == 0:
            rec = [np.zeros((1, 1, 1, 1), np.float32)] * 4
        f = os.path.join(md, f"{i:05d}.png")
        if i >= first and not os.path.exists(f):
            cv2.imwrite(f, (np.clip(pha, 0, 1) * 255 + 0.5).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 1])
            new += 1
            if new % 50 == 0:
                log(f"[v14 matte rvm] {i + 1}/{n}", flush=True)
        i += 1
    log(f"[v14 matte rvm] ready ({i} frames, {new} new)", flush=True)


def prep_matte(path, log=print, engine="modnet"):
    if engine == "rvm":
        return prep_matte_rvm(path, log=log)
    d = cache_dir(path)
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i = new = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        f = os.path.join(d, "matte", f"{i:05d}.png")
        if not os.path.exists(f):
            cv2.imwrite(f, (matte_frame(fr) * 255).astype(np.uint8))
            new += 1
            if new % 50 == 0:
                log(f"[v14 matte] {i + 1}/{n}", flush=True)
        i += 1
    log(f"[v14 matte] ready ({i} frames, {new} new)", flush=True)


# v22: silhouette calibration (measured on maghale vs MediaPipe: hijab/hair silhouette at the eye line ≈ 1.81 ×
# ear-to-ear width; body silhouette 40 px below the shoulder line ≈ 1.49 × shoulder width).  The Haar box only gives
# the face position; widths and the shoulder line come from the person matte → no dependence on Haar box padding.
SIL_HEAD, SIL_SHOULDER = 1.81, 1.49


def _silhouette_geometry(a, eye_y, cx):
    """a = person alpha (h×w float, any size), eye_y / cx in px of a → (head_w, shoulder_w, shoulder_y, top_y) in
    NORMALISED units, or None when the silhouette is unusable."""
    h, w = a.shape
    m = a > 0.5
    rows = np.where(m.any(1))[0]
    if len(rows) < h * 0.2:
        return None

    def span(y):
        y = int(min(h - 1, max(0, y)))
        xs = np.where(m[y])[0]
        if not len(xs):
            return 0, cx, cx
        # the run that contains cx (ignore other blobs)
        left = xs[xs <= cx]
        right = xs[xs >= cx]
        if not len(left) or not len(right):
            return 0, cx, cx
        return int(right.max() - left.min()), int(left.min()), int(right.max())

    head_sil, _, _ = span(eye_y)
    if head_sil < w * 0.08:
        return None
    sh_y = None
    for y in range(int(eye_y), h):                         # shoulder line: silhouette ≥ 1.85 × head (measured 1.89)
        if span(y)[0] >= 1.85 * head_sil:
            sh_y = y
            break
    if sh_y is None:
        return None
    body = span(min(h - 1, sh_y + 0.02 * h))[0]
    head_w = head_sil / SIL_HEAD
    sw = max(body / SIL_SHOULDER, head_w * 1.9)
    return head_w / w, sw / w, sh_y / h, rows.min() / h


def _prep_pose_offline_fallback(path, out, log=print, work_w=360):
    """no MediaPipe: Haar face box (position) + person-matte silhouette (scale, shoulder line).  v22 calibrated."""
    from ..matte import matte_onnx
    fc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(3)), int(cap.get(4))
    work_h = int(H * work_w / W)
    asp = H / W
    frames, i = [], 0
    last_box = [0.35, 0.30, 0.30, 0.18]  # normalized [x, y, w, h]
    geo = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % 3 == 0:
            sm = cv2.resize(fr, (work_w, work_h), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
            faces = fc.detectMultiScale(gray, 1.1, 5, minSize=(int(work_w*0.18), int(work_w*0.18)))
            valid = [b for b in faces if 0.15*work_w <= b[0] <= 0.65*work_w and 0.12*work_h <= b[1] <= 0.60*work_h]
            if valid:
                bx, by, bw, bh = max(valid, key=lambda b: b[2]*b[3])
                last_box = [bx / work_w, by / work_h, bw / work_w, bh / work_h]
            fx, fy, fw, fh = last_box
            try:
                g = _silhouette_geometry(matte_onnx(sm), (fy + fh * 0.40) * work_h, (fx + fw * 0.5) * work_w)
            except Exception:  # noqa: BLE001
                g = None
            if g is not None:
                geo = g if geo is None else tuple(0.6 * a + 0.4 * b for a, b in zip(geo, g))
        fx, fy, fw, fh = last_box
        cx = fx + fw * 0.5
        eye_y = fy + fh * 0.40
        if geo is not None:
            hw, sw, sh_y, _top = geo
        else:                                   # no usable matte → calibrated clamps (report v22 §1)
            hw = float(np.clip(fw * 0.56, 0.22, 0.26))
            sw = float(np.clip(hw * 2.35, 0.52, 0.60))
            sh_y = eye_y + hw * 1.35 / asp
        pts = [[0.5, 0.5] for _ in range(33)]
        pts[0]  = [round(cx, 4), round(fy + fh * 0.56, 4)]                 # nose
        pts[2]  = [round(cx + hw * 0.30, 4), round(eye_y, 4)]              # eye_l (subject left = image right)
        pts[5]  = [round(cx - hw * 0.30, 4), round(eye_y, 4)]              # eye_r
        pts[7]  = [round(cx + hw * 0.50, 4), round(fy + fh * 0.48, 4)]     # ear_l
        pts[8]  = [round(cx - hw * 0.50, 4), round(fy + fh * 0.48, 4)]     # ear_r
        pts[11] = [round(cx + sw * 0.50, 4), round(sh_y, 4)]               # sh_l
        pts[12] = [round(cx - sw * 0.50, 4), round(sh_y, 4)]               # sh_r
        hy = round(min(0.92, sh_y + sw * 0.95 / asp), 4)
        pts[15] = [round(cx + sw * 0.35, 4), hy]                           # hand_l
        pts[16] = [round(cx - sw * 0.35, 4), hy]                           # hand_r
        frames.append(pts)
        i += 1
    cap.release()
    with open(out, "w") as f:
        json.dump({"fps": fps, "w": W, "h": H, "frames": frames, "source": "haar+silhouette"}, f)
    log(f"[v14 pose offline fallback] ready ({i} frames, silhouette-calibrated {geo is not None})", flush=True)
    return out


def prep_pose(path, log=print, work_w=540):
    out = os.path.join(cache_dir(path), "pose.json")
    if os.path.exists(out):
        return out
    try:
        import mediapipe as mp
    except ImportError:
        return _prep_pose_offline_fallback(path, out, log=log)
    from mediapipe.tasks.python import BaseOptions, vision
    det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODELS, "pose_landmarker_lite.task")),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(3)), int(cap.get(4))
    frames, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        sm = cv2.resize(fr, (work_w, int(H * work_w / W)), interpolation=cv2.INTER_AREA)
        r = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(sm, cv2.COLOR_BGR2RGB)),
                                 int(i * 1000 / fps))
        frames.append([[round(l.x, 4), round(l.y, 4)] for l in r.pose_landmarks[0]] if r.pose_landmarks else None)
        i += 1
    with open(out, "w") as f:
        json.dump({"fps": fps, "w": W, "h": H, "frames": frames}, f)
    log(f"[v14 pose] ready ({i} frames)", flush=True)
    return out


def _fill_nan(a):
    a = a.copy()
    idx = np.arange(len(a))
    for c in range(a.shape[1]):
        ok = ~np.isnan(a[:, c])
        if ok.sum() == 0:
            a[:, c] = 0.5
        elif ok.sum() < len(a):
            a[~ok, c] = np.interp(idx[~ok], idx[ok], a[ok, c])
    return a


def _smooth(a, sigma):
    """Centred (zero-lag) Gaussian smoothing along time."""
    r = max(1, int(sigma * 3))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(a, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, c], k, mode="valid") for c in range(a.shape[1])], 1)


def matte_dir(path, engine=None):
    """cache dir of the chosen matte engine; engine None/"modnet" = classic, "rvm" = v22, "auto" = rvm if complete."""
    if engine == "auto":
        rd = os.path.join(path + ".v14", "matte_rvm")
        engine = "rvm" if os.path.isdir(rd) and len(os.listdir(rd)) > 0 else "modnet"
    return os.path.join(path + ".v14", "matte_rvm" if engine == "rvm" else "matte"), engine or "modnet"


def median3(a, b, c):
    """exact per-pixel median of three uint8 images: a + b + c − min − max  (SIMD in OpenCV, no sort)."""
    lo = cv2.min(cv2.min(a, b), c)
    hi = cv2.max(cv2.max(a, b), c)
    s = a.astype(np.int16) + b + c
    return (s - lo - hi).astype(np.uint8)


class Tracks:
    def __init__(self, path, sigma=2.2, engine=None):
        self.path = path
        self.mdir, self.engine = matte_dir(path, engine)
        with open(prep_pose(path)) as f:
            data = json.load(f)
        self.fps, self.w, self.h = data["fps"], data["w"], data["h"]
        n = self.n = len(data["frames"])
        P = np.full((n, 33, 2), np.nan, np.float32)
        for i, r in enumerate(data["frames"]):
            if r:
                P[i] = np.array(r, np.float32)
        L = _smooth(_fill_nan(P.reshape(n, -1)), sigma).reshape(n, 33, 2)
        asp = self.h / self.w
        sh_l, sh_r, nose = L[:, 11], L[:, 12], L[:, 0]
        eye_l, eye_r, ear_l, ear_r = L[:, 2], L[:, 5], L[:, 7], L[:, 8]
        sw = np.abs(sh_l[:, 0] - sh_r[:, 0])
        head_w = np.maximum(np.abs(ear_l[:, 0] - ear_r[:, 0]), sw * 0.38)
        head_c = np.stack([(ear_l[:, 0] + ear_r[:, 0]) / 4 + nose[:, 0] / 2,
                           (eye_l[:, 1] + eye_r[:, 1]) / 2 - head_w * 0.05 / asp], 1)
        chin = np.stack([nose[:, 0], nose[:, 1] + head_w * 0.55 / asp], 1)
        shoulders = (sh_l + sh_r) / 2
        A = {"head_top": np.stack([head_c[:, 0], np.maximum(0.14, head_c[:, 1] - head_w * 0.95 / asp)], 1),
             "head_c": head_c, "face": (eye_l + eye_r) / 4 + nose / 2, "nose": nose, "chin": chin,
             "neck": shoulders * 0.6 + chin * 0.4, "shoulders": shoulders,
             "chest": np.stack([shoulders[:, 0], shoulders[:, 1] + sw * 0.42 / asp], 1),
             "belly": np.stack([shoulders[:, 0], shoulders[:, 1] + sw * 0.95 / asp], 1),
             "shoulder_l": sh_l, "shoulder_r": sh_r, "hand_l": L[:, 15], "hand_r": L[:, 16],
             "hands": (L[:, 15] + L[:, 16]) / 2}
        self.A = {k: _smooth(v.astype(np.float32), 1.0) for k, v in A.items()}
        # head roll (deg) from the eye line, and shoulder roll — layers can inherit the speaker's tilt
        roll = np.degrees(np.arctan2((eye_l[:, 1] - eye_r[:, 1]) * asp, eye_l[:, 0] - eye_r[:, 0]))
        sroll = np.degrees(np.arctan2((sh_l[:, 1] - sh_r[:, 1]) * asp, sh_l[:, 0] - sh_r[:, 0]))
        self.roll = np.clip(_smooth(roll[:, None].astype(np.float32), 3.0)[:, 0], -12.0, 12.0)
        self.sroll = np.clip(_smooth(sroll[:, None].astype(np.float32), 4.0)[:, 0], -10.0, 10.0)
        self.head_w = _smooth(np.maximum(head_w, sw * 0.40)[:, None], 12.0)[:, 0]   # ears hide under hair/hijab → stabilise
        self.sw = _smooth(sw[:, None], 12.0)[:, 0]
        self._mc = {}
        self._rmc = {}            # v22: raw uint8 matte frames (each PNG decoded once per sequential render)

    def _i(self, st):
        f = max(0.0, st * self.fps)
        i0 = min(self.n - 1, int(f))
        return i0, min(self.n - 1, i0 + 1), f - int(f)

    def anchor(self, name, st):
        a = self.A.get(name, self.A["face"])
        i0, i1, fr = self._i(st)
        return (float(a[i0, 0] * (1 - fr) + a[i1, 0] * fr), float(a[i0, 1] * (1 - fr) + a[i1, 1] * fr))

    def head_roll(self, st):
        i0, i1, fr = self._i(st)
        return float(self.roll[i0] * (1 - fr) + self.roll[i1] * fr)

    def shoulder_roll(self, st):
        i0, i1, fr = self._i(st)
        return float(self.sroll[i0] * (1 - fr) + self.sroll[i1] * fr)

    def head_width(self, st):
        i0, i1, fr = self._i(st)
        return float(self.head_w[i0] * (1 - fr) + self.head_w[i1] * fr)

    def shoulder_width(self, st):
        i0, i1, fr = self._i(st)
        return float(self.sw[i0] * (1 - fr) + self.sw[i1] * fr)

    def _raw(self, d, j):
        a = self._rmc.get((d, j))
        if a is None:
            a = cv2.imread(os.path.join(d, f"{j:05d}.png"), cv2.IMREAD_GRAYSCALE)
            if a is None:
                cap = cv2.VideoCapture(self.path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, j)
                a = (matte_frame(cap.read()[1]) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(d, f"{j:05d}.png"), a)
            if len(self._rmc) >= 5:
                self._rmc.pop(next(iter(self._rmc)))
            self._rmc[(d, j)] = a
        return a

    def matte(self, idx):
        idx = max(0, min(self.n - 1, int(idx)))
        if idx in self._mc:
            return self._mc[idx]
        d = self.mdir
        if self.engine == "rvm":                  # recurrent model: already temporally coherent → no median
            a = cv2.imread(os.path.join(d, f"{idx:05d}.png"), cv2.IMREAD_GRAYSCALE)
            if a is None:
                prep_matte_rvm(self.path)
                a = cv2.imread(os.path.join(d, f"{idx:05d}.png"), cv2.IMREAD_GRAYSCALE)
            if a is not None:
                m = a.astype(np.float32) / 255.0
                if len(self._mc) > 6:
                    self._mc.clear()
                self._mc[idx] = m
                return m
            d = os.path.join(self.path + ".v14", "matte")
        ims = [self._raw(d, max(0, min(self.n - 1, j))) for j in (idx - 1, idx, idx + 1)]
        m = median3(*ims).astype(np.float32) * (1.0 / 255.0)
        if len(self._mc) > 6:
            self._mc.clear()
        self._mc[idx] = m
        return m


if __name__ == "__main__":
    _a = [x for x in sys.argv[1:] if not x.startswith("--")]
    _eng = sys.argv[sys.argv.index("--engine") + 1] if "--engine" in sys.argv else "modnet"
    _a = [x for x in _a if x != _eng]
    prep_pose(_a[0])
    prep_matte(_a[0], engine=_eng)
