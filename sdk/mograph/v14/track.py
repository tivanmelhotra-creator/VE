"""
mograph.v14.track — offline, ZERO-LAG speaker analysis (computed once per clip, cached next to the video).

v13.9 audit: live one-euro tracking is causal (text trails the speaker) and the 256 px selfie matte has halo edges
("behind the speaker" looks pasted).  v14 analyses the whole clip OFFLINE:
  matte  MODNet portrait matting (models/modnet.onnx) → hair-accurate soft alpha, temporal median of 3 frames
         cached as PNG in <video>.v14/matte/   (fallback: bundled selfie segmenter)
  pose   MediaPipe pose every frame → CENTRED Gaussian smoothing (no lag) → anchors:
         head_top face nose chin neck chest belly shoulder_l shoulder_r hand_l hand_r hands head_c
         + head_width / shoulder_width (u units) so layers scale with the speaker.
CLI:  python3 -m mograph.v14.track media/raw.mp4      (run once; resumable)
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


def prep_matte(path, log=print):
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


def _prep_pose_offline_fallback(path, out, log=print, work_w=360):
    from ..matte import matte_onnx
    fc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(3)), int(cap.get(4))
    work_h = int(H * work_w / W)
    frames, i = [], 0
    last_box = [0.35, 0.30, 0.30, 0.18]  # normalized [x, y, w, h]
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
        cx = fx + fw * 0.5
        cy = fy + fh * 0.5
        pts = [[0.5, 0.5] for _ in range(33)]
        pts[0]  = [round(cx, 4), round(fy + fh * 0.56, 4)]              # nose
        pts[2]  = [round(cx + fw * 0.18, 4), round(fy + fh * 0.40, 4)]  # eye_l (subject left = image right)
        pts[5]  = [round(cx - fw * 0.18, 4), round(fy + fh * 0.40, 4)]  # eye_r (subject right = image left)
        pts[7]  = [round(cx + fw * 0.34, 4), round(fy + fh * 0.48, 4)]  # ear_l
        pts[8]  = [round(cx - fw * 0.34, 4), round(fy + fh * 0.48, 4)]  # ear_r
        pts[11] = [round(cx + fw * 0.78, 4), round(fy + fh * 1.32, 4)]  # sh_l
        pts[12] = [round(cx - fw * 0.78, 4), round(fy + fh * 1.32, 4)]  # sh_r
        pts[15] = [round(cx + fw * 0.55, 4), round(min(0.90, fy + fh * 2.35), 4)] # hand_l
        pts[16] = [round(cx - fw * 0.55, 4), round(min(0.90, fy + fh * 2.35), 4)] # hand_r
        frames.append(pts)
        i += 1
    cap.release()
    with open(out, "w") as f:
        json.dump({"fps": fps, "w": W, "h": H, "frames": frames}, f)
    log(f"[v14 pose offline fallback] ready ({i} frames)", flush=True)
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


class Tracks:
    def __init__(self, path, sigma=2.2):
        self.path = path
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

    def matte(self, idx):
        idx = max(0, min(self.n - 1, int(idx)))
        if idx in self._mc:
            return self._mc[idx]
        d = os.path.join(self.path + ".v14", "matte")
        ims = []
        for j in (idx - 1, idx, idx + 1):
            j = max(0, min(self.n - 1, j))
            a = cv2.imread(os.path.join(d, f"{j:05d}.png"), cv2.IMREAD_GRAYSCALE)
            if a is None:
                cap = cv2.VideoCapture(self.path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, j)
                a = (matte_frame(cap.read()[1]) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(d, f"{j:05d}.png"), a)
            ims.append(a)
        m = np.median(np.stack(ims), axis=0).astype(np.float32) / 255.0
        if len(self._mc) > 6:
            self._mc.clear()
        self._mc[idx] = m
        return m


if __name__ == "__main__":
    prep_pose(sys.argv[1])
    prep_matte(sys.argv[1])
