"""
mograph.motion.retime — speed changes on the EDITED timeline (Premiere "Time Remapping" / AE "Time Remap").

Ops (all times in EDITED seconds = after the EDL cuts, i.e. what `python3 -m mograph.v14 words` prints, or
"w:word" anchors; `dur` = edited seconds the op covers):
    {"op": "slow_motion",  "t": 3.2, "dur": 1.0, "speed": 0.35, "ramp": 0.25}     # 1 s of action → 2.9 s on screen
    {"op": "fast_motion",  "t": 5.0, "dur": 4.0, "speed": 4.0,  "ramp": 0.15}     # timelapse: "speed": 8+
    {"op": "speed_ramp",   "t": 2.0, "dur": 0.6, "from": 1.0, "to": 0.25, "ease": "in_out_sine",
                           "hold": 0.8, "back": 0.5}                            # 1× → ¼× (hold 0.8 s) → back to 1×
    {"op": "speed_ramp",   "t": 2.0, "keys": [[0, 1.0], [0.5, 0.2], [1.4, 0.2], [1.8, 3.0], [2.2, 1.0]]}
                                                                               # keys = [edited offset, speed]
    {"op": "freeze_frame", "t": "w:مقاله", "dur": 0.8, "flash": true}           # hold 0.8 s ON SCREEN (consumes 0)
    {"op": "reverse",      "t": 6.0, "dur": 1.2, "speed": 1.5, "mode": "rewind"}   # rewind: play, rewind, replay
                           mode "replace" = the span plays backwards once;  "boomerang" = forward → back
Global:  "speed": 1.1 (whole edit), "interpolate": "nearest" | "blend" | "flow" (optical flow, default for v14),
         "audio": "auto" (mute voice under |speed-1|>0.15 with 30 ms fades + riser/hit sfx) | "varispeed" (tape) |
         "stretch" (pitch-preserving atempo on constant parts) | "mute"

Speed ≠ 1 regions are built from analytic pieces (constant) and numerically integrated pieces (ramps, 1 ms
steps, monotone tables) → out→edited is exact and edited→out (word anchors) inverts the monotone parts.
"""
from __future__ import annotations

import bisect
import math

import numpy as np

from .easing import ease as _ease

OPS = ("slow_motion", "fast_motion", "speed_ramp", "freeze_frame", "reverse", "frame_interpolation")


class _Piece:
    """one output-time interval [u0, u1) mapped onto edited time.  kind: lin | tab | hold."""
    __slots__ = ("u0", "u1", "kind", "e0", "s", "tu", "te", "op")

    def __init__(self, u0, u1, kind, e0=0.0, s=1.0, tu=None, te=None, op=None):
        self.u0, self.u1, self.kind, self.e0, self.s, self.tu, self.te, self.op = u0, u1, kind, e0, s, tu, te, op

    def e(self, u):
        if self.kind == "hold":
            return self.e0
        if self.kind == "lin":
            return self.e0 + self.s * (u - self.u0)
        return float(np.interp(u, self.tu, self.te))

    def speed(self, u):
        if self.kind == "hold":
            return 0.0
        if self.kind == "lin":
            return self.s
        i = min(len(self.tu) - 2, max(0, int(np.searchsorted(self.tu, u)) - 1))
        du = self.tu[i + 1] - self.tu[i]
        return float((self.te[i + 1] - self.te[i]) / du) if du > 0 else 0.0


def _ramp_table(e0, length, s_of, sign=1.0, step=1e-3):
    """integrate du = de / s(e) over an edited span → monotone (u, e) tables (u relative)."""
    n = max(2, int(math.ceil(length / step)) + 1)
    es = np.linspace(0.0, length, n)
    sp = np.maximum(1e-3, np.array([s_of(x) for x in es]))
    du = np.diff(es) * 0.5 * (1 / sp[1:] + 1 / sp[:-1])
    us = np.concatenate([[0.0], np.cumsum(du)])
    return us, (e0 + sign * es) if sign > 0 else (e0 - es)


class Retime:
    """out time (on screen) ↔ edited time.  Build with Retime(ops, duration_edited, resolve, speed=1.0)."""

    def __init__(self, ops, edited_duration, resolve=None, speed=1.0, cfg=None):
        self.cfg = dict(cfg or {})
        self.E = float(edited_duration)
        self.base = float(speed or 1.0)
        res = resolve or (lambda v, d=None: float(v) if v is not None else d)
        ops = [dict(o) for o in (ops or []) if o.get("op") != "frame_interpolation"]
        for o in ops:
            if o.get("op") not in OPS:
                raise ValueError(f"retime op '{o.get('op')}' unknown — one of {OPS}")
            o["_t"] = float(res(o.get("t", 0.0)))
        ops.sort(key=lambda o: o["_t"])
        self.ops = ops
        self.pieces: list[_Piece] = []
        self.events = []                      # (out_time, kind) for sfx / flashes
        u, e = 0.0, 0.0
        for o in ops:
            t = min(max(0.0, o["_t"]), self.E)
            if t < e - 1e-6:
                raise ValueError(f"retime ops overlap: '{o['op']}' starts at edited {t:.2f}s but the previous op covers "
                                 f"edited time up to {e:.2f}s (a ramp covers dur+hold+back) — start it at "
                                 f"\"e:{e:.2f}\" or later, or shorten the previous op")
            if t > e:
                u = self._lin(u, e, t, self.base)
                e = t
            u, e = self._op(o, u, e)
        if e < self.E:
            u = self._lin(u, e, self.E, self.base)
        self.duration = u
        self._u0 = [p.u0 for p in self.pieces]

    # ------------------------------------------------------------------ builders
    def _lin(self, u, e0, e1, s, op=None, sign=1.0):
        d = (e1 - e0) / s
        if d > 1e-9:
            self.pieces.append(_Piece(u, u + d, "lin", e0 if sign > 0 else e1, s * sign, op=op))
        return u + max(0.0, d)

    def _tab(self, u, e0, length, s_of, op=None, sign=1.0):
        tu, te = _ramp_table(e0, length, s_of, sign)
        if tu[-1] > 1e-9:
            self.pieces.append(_Piece(u, u + tu[-1], "tab", tu=tu + u, te=te, op=op))
        return u + float(tu[-1])

    def _op(self, o, u, e):
        kind = o["op"]
        b = self.base
        if kind == "freeze_frame":
            d = float(o.get("dur", 0.8))
            self.events.append((u, "freeze"))
            self.pieces.append(_Piece(u, u + d, "hold", e, op=o))
            return u + d, e
        dur = min(float(o.get("dur", 1.0)), self.E - e)
        if dur <= 0:
            return u, e
        if kind in ("slow_motion", "fast_motion"):
            s = float(o.get("speed", 0.4 if kind == "slow_motion" else 4.0)) * b
            r = min(float(o.get("ramp", 0.2 if kind == "slow_motion" else 0.12)), dur / 2.5)
            g = _ease(o.get("ease", "in_out_sine"))
            self.events.append((u, "slow" if s < b else "fast"))
            if r > 1e-3:
                u = self._tab(u, e, r, lambda x: b + (s - b) * g(x / r), o)
                u = self._lin(u, e + r, e + dur - r, s, o)
                u = self._tab(u, e + dur - r, r, lambda x: s + (b - s) * g(x / r), o)
            else:
                u = self._lin(u, e, e + dur, s, o)
            return u, e + dur
        if kind == "speed_ramp":
            keys = o.get("keys")
            if not keys:
                a, z = float(o.get("from", 1.0)), float(o.get("to", 0.3))
                hold, back = float(o.get("hold", 0.0)), o.get("back")
                keys = [[0.0, a], [dur, z]]
                if hold > 0:
                    keys.append([dur + hold, z])
                if back:
                    keys.append([keys[-1][0] + float(back), float(o.get("back_to", a))])
            keys = [[float(k[0]), float(k[1]) * b] + list(k[2:]) for k in keys]
            span = min(keys[-1][0], self.E - e)
            g_default = o.get("ease", "in_out_sine")
            eases = [_ease(k[2] if len(k) > 2 else g_default) for k in keys]

            def s_of(x):
                if x <= keys[0][0]:
                    return keys[0][1]
                for i in range(1, len(keys)):
                    if x <= keys[i][0]:
                        k0, k1 = keys[i - 1], keys[i]
                        p = (x - k0[0]) / max(1e-9, k1[0] - k0[0])
                        return k0[1] + (k1[1] - k0[1]) * eases[i - 1](p)
                return keys[-1][1]
            self.events.append((u, "slow" if min(k[1] for k in keys) < b else "fast"))
            u = self._tab(u, e, span, s_of, o)
            return u, e + span
        if kind == "reverse":
            s = float(o.get("speed", 1.0)) * b
            mode = o.get("mode", "rewind")
            self.events.append((u, "rewind"))
            if mode == "replace":
                u = self._lin(u, e, e + dur, s, o, sign=-1.0)
            elif mode == "boomerang":
                u = self._lin(u, e, e + dur, b, o)
                u = self._lin(u, e, e + dur, s, o, sign=-1.0)
            else:                                                     # rewind: play → rewind → play again
                u = self._lin(u, e, e + dur, b, o)
                u = self._lin(u, e, e + dur, s * float(o.get("rewind_speed", 2.0)), o, sign=-1.0)
                u = self._lin(u, e, e + dur, b, o)
            return u, e + dur
        raise ValueError(kind)

    # ------------------------------------------------------------------ queries
    def piece(self, u):
        if not self.pieces:
            return None
        i = max(0, bisect.bisect_right(self._u0, u) - 1)
        return self.pieces[i]

    def edited(self, u):
        """on-screen time → edited time."""
        if not self.pieces:
            return u
        u = min(max(0.0, u), self.duration - 1e-6)
        return self.piece(u).e(u)

    def speed(self, u):
        p = self.piece(min(max(0.0, u), self.duration - 1e-6))
        return 1.0 if p is None else p.speed(u)

    def out(self, e):
        """edited time → FIRST on-screen time showing it (forward pieces), for word anchors / cuts."""
        for p in self.pieces:
            if p.kind == "hold":
                continue
            if p.kind == "lin" and p.s > 0:
                e0, e1 = p.e0, p.e0 + p.s * (p.u1 - p.u0)
                if e0 - 1e-6 <= e <= e1 + 1e-6:
                    return p.u0 + (e - e0) / p.s
            elif p.kind == "tab" and p.te[-1] >= p.te[0]:
                if p.te[0] - 1e-6 <= e <= p.te[-1] + 1e-6:
                    return float(np.interp(e, p.te, p.tu))
        return self.duration if e >= self.E else 0.0

    def regions(self, thr=0.15):
        """[(u0, u1, speed_min, speed_max, op)] where |speed - 1| > thr (or holds / reverse)."""
        out = []
        for p in self.pieces:
            if p.kind == "hold":
                out.append((p.u0, p.u1, 0.0, 0.0, p.op))
            elif p.kind == "lin":
                if abs(p.s - 1.0) > thr:
                    out.append((p.u0, p.u1, p.s, p.s, p.op))
            else:
                sp = np.diff(p.te) / np.maximum(1e-9, np.diff(p.tu))
                if np.any(np.abs(sp - 1.0) > thr):
                    out.append((p.u0, p.u1, float(sp.min()), float(sp.max()), p.op))
        return out

    def summary(self):
        return {"edited": round(self.E, 3), "out": round(self.duration, 3), "pieces": len(self.pieces),
                "regions": [[round(a, 2), round(b, 2), round(s0, 2), round(s1, 2), (o or {}).get("op")]
                            for a, b, s0, s1, o in self.regions()]}

    # ------------------------------------------------------------------ audio
    def remap_audio(self, x, sr, mode="auto", fade=0.03):
        """edited-time mono/stereo signal → on-screen signal (len = duration·sr)."""
        n = int(round(self.duration * sr))
        u = np.arange(n) / sr
        e = np.empty(n)
        for p in self.pieces:
            i0, i1 = int(round(p.u0 * sr)), min(n, int(round(p.u1 * sr)))
            if i1 <= i0:
                continue
            uu = u[i0:i1]
            if p.kind == "hold":
                e[i0:i1] = p.e0
            elif p.kind == "lin":
                e[i0:i1] = p.e0 + p.s * (uu - p.u0)
            else:
                e[i0:i1] = np.interp(uu, p.tu, p.te)
        idx = np.clip(e * sr, 0, len(x) - 1)
        if x.ndim == 1:
            y = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
        else:
            y = np.stack([np.interp(idx, np.arange(len(x)), x[:, c]) for c in range(x.shape[1])], 1).astype(np.float32)
        if mode == "varispeed":
            for a, b, s0, s1, o in self.regions(0.0):
                if s0 == 0.0 and s1 == 0.0:
                    y[int(a * sr):int(b * sr)] = 0.0            # a frozen frame is silent, not a buzz
            return y
        if mode == "stretch":
            y = self._stretch(x, sr, y)
        gain = np.ones(n, np.float32)
        if mode in ("auto", "mute"):
            for a, b, s0, s1, o in self.regions(0.15 if mode == "auto" else 0.0):
                i0, i1 = int(a * sr), int(b * sr)
                gain[i0:i1] = 0.0
            k = max(1, int(fade * sr))
            gain = np.convolve(gain, np.ones(k) / k, "same").astype(np.float32)
        return y * (gain if y.ndim == 1 else gain[:, None])

    def _stretch(self, x, sr, y):
        """pitch-preserving: constant-speed pieces through ffmpeg atempo (ramps / holds keep varispeed)."""
        import subprocess
        import tempfile
        import wave
        mono = x if x.ndim == 1 else x.mean(1)
        for p in self.pieces:
            if p.kind != "lin" or abs(abs(p.s) - 1.0) < 0.02:
                continue
            e0 = p.e0 if p.s > 0 else p.e0 + p.s * (p.u1 - p.u0)
            e1 = e0 + abs(p.s) * (p.u1 - p.u0)
            seg = mono[int(e0 * sr):int(e1 * sr)]
            if p.s < 0:
                seg = seg[::-1]
            if len(seg) < sr // 20:
                continue
            tempo, chain = abs(p.s), []
            while tempo < 0.5:
                chain.append("atempo=0.5")
                tempo /= 0.5
            while tempo > 2.0:
                chain.append("atempo=2.0")
                tempo /= 2.0
            chain.append(f"atempo={tempo:.5f}")
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                with wave.open(f.name, "wb") as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
                    wf.writeframes((np.clip(seg, -1, 1) * 32767).astype(np.int16).tobytes())
                raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f.name, "-af", ",".join(chain), "-f", "f32le",
                                      "-ac", "1", "-ar", str(sr), "-"], capture_output=True).stdout
            st = np.frombuffer(raw, np.float32)
            i0, i1 = int(p.u0 * sr), int(p.u1 * sr)
            m = min(i1 - i0, len(st))
            tgt = y[i0:i0 + m] if y.ndim == 1 else y[i0:i0 + m, 0]
            if y.ndim == 1:
                y[i0:i0 + m] = st[:m]
            else:
                y[i0:i0 + m] = st[:m, None]
            del tgt
        return y


# =============================================================================================== frames
class FlowInterp:
    """in-between frames for slow motion: nearest | blend | flow (bidirectional DIS optical flow, occlusion-aware
    blend; flow pair cached so a 0.25× shot computes ONE flow per source pair).  Work width 480 px (≈ 25 ms)."""

    def __init__(self, mode="flow", work_w=480):
        self.mode, self.work_w = mode, work_w
        self._key, self._flow = None, None
        self._dis = None

    def _flows(self, a, b, key):
        import cv2
        if key is not None and self._key == key:
            return self._flow
        H, W = a.shape[:2]
        ww = min(self.work_w, W)
        wh = max(16, int(round(H * ww / W)))
        ga = cv2.cvtColor(cv2.resize(a, (ww, wh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(cv2.resize(b, (ww, wh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if self._dis is None:
            self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        f01 = self._dis.calc(ga, gb, None)
        f10 = self._dis.calc(gb, ga, None)
        fb = np.hypot(f01[..., 0] + cv2.remap(f10[..., 0], *_grid_plus(f01), cv2.INTER_LINEAR),
                      f01[..., 1] + cv2.remap(f10[..., 1], *_grid_plus(f01), cv2.INTER_LINEAR))
        agree = np.exp(-fb / 1.5).astype(np.float32)
        sx, sy = W / ww, H / wh
        up = lambda f: cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR) * np.array([sx, sy], np.float32)  # noqa
        self._key = key
        self._flow = (up(f01), up(f10), cv2.resize(agree, (W, H), interpolation=cv2.INTER_LINEAR))
        return self._flow

    def frame(self, a, b, alpha, key=None):
        """frame between a (alpha 0) and b (alpha 1)."""
        import cv2
        if alpha <= 0.02 or self.mode == "nearest":
            return a if alpha < 0.5 or self.mode != "nearest" else b
        if alpha >= 0.98:
            return b
        if self.mode == "blend":
            return cv2.addWeighted(a, 1 - alpha, b, alpha, 0)
        f01, f10, agree = self._flows(a, b, key)
        H, W = a.shape[:2]
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        wa = cv2.remap(a, gx - alpha * f01[..., 0], gy - alpha * f01[..., 1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
        wb = cv2.remap(b, gx - (1 - alpha) * f10[..., 0], gy - (1 - alpha) * f10[..., 1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
        flow = cv2.addWeighted(wa, 1 - alpha, wb, alpha, 0)
        if agree.min() > 0.97:
            return flow
        cross = cv2.addWeighted(a, 1 - alpha, b, alpha, 0)
        k = agree[..., None]
        return np.clip(flow * k + cross * (1 - k) + 0.5, 0, 255).astype(np.uint8)

    def matte(self, ma, mb, alpha):
        return ma * (1 - alpha) + mb * alpha


def _grid_plus(f):
    H, W = f.shape[:2]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return gx + f[..., 0], gy + f[..., 1]


# =============================================================================================== standalone CLI
def retime_file(src, out, ops, speed=1.0, interpolate="flow", audio="auto", fps=None, log=print):
    """retime any video file (no v14 spec needed).  Returns the Retime summary."""
    import os
    import subprocess
    import cv2
    cap = cv2.VideoCapture(src)
    sfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(3)), int(cap.get(4))
    fps = float(fps or sfps)
    E = n_src / sfps
    rt = Retime(ops, E, speed=speed)
    FI = FlowInterp(interpolate)
    cache = {}

    def get(i):
        i = max(0, min(n_src - 1, i))
        if i not in cache:
            if len(cache) > 6:
                cache.pop(next(iter(cache)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            cache[i] = fr if ok else np.zeros((H, W, 3), np.uint8)
        return cache[i]

    silent = os.path.splitext(out)[0] + "_v.mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                          "-pix_fmt", "yuv420p", silent], stdin=subprocess.PIPE)
    N = int(round(rt.duration * fps))
    for k in range(N):
        u = k / fps
        e = rt.edited(u)
        x = e * sfps
        i = int(math.floor(x))
        a = x - i
        sp = abs(rt.speed(u))
        if sp > 1.6 and interpolate != "nearest":            # fast motion: shutter frame blend (≤ 6 samples)
            m = int(min(6, max(2, round(sp))))
            acc = np.zeros((H, W, 3), np.float32)
            for j in range(m):
                acc += get(int(round(rt.edited(u + j / (m * fps)) * sfps)))
            fr = (acc / m + 0.5).astype(np.uint8)
        else:
            fr = FI.frame(get(i), get(i + 1), a, key=i)
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
        if log and k % 60 == 0:
            log(f"[retime] {k}/{N}")
    p.stdin.close()
    p.wait()
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-ac", "2", "-ar", "48000", "-f", "f32le", "-"],
                         capture_output=True).stdout
    if raw and audio != "none":
        import wave
        x = np.frombuffer(raw, np.float32).reshape(-1, 2)
        y = rt.remap_audio(x, 48000, audio)
        wav = os.path.splitext(out)[0] + "_a.wav"
        with wave.open(wav, "wb") as wf:
            wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(48000)
            wf.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", silent, "-i", wav, "-c:v", "copy", "-c:a", "aac",
                        "-shortest", out])
        os.remove(wav)
        os.remove(silent)
    else:
        os.replace(silent, out)
    return rt.summary()


# =============================================================================================== v14 bridge
class RetimedTimeline:
    """drop-in for v14 compositor.Timeline: OUTPUT time = on screen after retime.  Word anchors ("w:…"), cuts,
    durations, to_src() all live in output time, so every layer / subtitle / sfx stays locked to the speech.
    `.base` is the plain EDL timeline (edited time), `.rt` the Retime."""

    def __init__(self, base, ops, speed=1.0, cfg=None):
        self.base = base
        self.rt = Retime(ops, base.duration, lambda v, d=None: base.resolve(v, d), speed, cfg)
        self.seg, self.cum = base.seg, base.cum                    # EDL (edited time) for the voice track
        self.duration = self.rt.duration
        self.cuts = [self.rt.out(c) for c in base.cuts]
        self.words = []
        for w in base.words:
            ot, oe = self.rt.out(w["t"]), self.rt.out(w["end"])
            self.words.append(dict(w, t=ot, end=max(ot + 0.05, oe)))

    def to_src(self, t):
        return self.base.to_src(self.rt.edited(t))

    def to_out(self, s):
        e = self.base.to_out(s)
        return None if e is None else self.rt.out(e)

    def speed(self, t):
        return self.rt.speed(t)

    def word(self, key, end=False):
        from ..v14.compositor import norm_word
        occ = 1
        if "#" in key:
            key, o = key.rsplit("#", 1)
            occ = int(o)
        n = norm_word(key)
        hits = [w for w in self.words if w["n"] == n] or [w for w in self.words if w["n"].startswith(n)]
        if not hits:
            raise KeyError(f"word anchor '{key}' not found in transcript")
        w = hits[min(len(hits), occ) - 1]
        return w["end"] if end else w["t"]

    def resolve(self, v, default=None):
        import re
        if v is None:
            return default
        if isinstance(v, (int, float)):
            return float(v)
        v = str(v).strip()
        m = re.match(r"^(we?):(.+?)([+-]\d*\.?\d+)?$", v)
        if m:
            return self.word(m.group(2), end=(m.group(1) == "we")) + (float(m.group(3)) if m.group(3) else 0.0)
        m = re.match(r"^e:(\d*\.?\d+)$", v)                         # "e:3.2" = EDITED time 3.2 s (before retime)
        if m:
            return self.rt.out(float(m.group(1)))
        return float(v)


def install_v14(R):
    """called by the v14 Renderer when spec.source has retime / speed (see compositor.Renderer.__init__)."""
    src = R.spec["source"]
    ops = list(src.get("retime") or [])
    sp = float(src.get("speed", 1.0))
    if not ops and abs(sp - 1.0) < 1e-6:
        return None
    R.tl = RetimedTimeline(R.tl, ops, sp)
    R.duration = float(R.spec.get("meta", {}).get("duration", R.tl.duration))
    R.interp = FlowInterp(src.get("interpolate", "flow")) if src.get("interpolate", "flow") != "nearest" else None
    R._rt_cache = {}
    return R.tl


def v14_frame(R, t, st):
    """(frame, matte or None, idx) with in-between synthesis for slow motion and shutter blend for fast motion."""
    import cv2
    sp = abs(R.tl.speed(t))
    x = st * R.sfps
    i = int(math.floor(x))
    a = x - i

    def get(j):
        j = max(0, min(R.sn - 1, j))
        c = R._rt_cache
        if j not in c:
            if len(c) > 5:
                c.pop(next(iter(c)))
            fr, _ = R.src_frame(j / R.sfps)
            c[j] = fr.copy()
        return c[j]
    if R.interp is not None and sp < 0.95 and 0.04 < a < 0.96 and i + 1 < R.sn:
        fr = R.interp.frame(get(i), get(i + 1), a, key=i)
        A = R.tracks.matte(i) * (1 - a) + R.tracks.matte(i + 1) * a
        return fr, A.astype(np.float32), i if a < 0.5 else i + 1
    if R.interp is not None and sp > 1.6:
        m = int(min(5, max(2, round(sp))))
        acc = None
        for j in range(m):
            k = int(round(R.tl.to_src(t + j / (m * R.fps)) * R.sfps))
            f = get(k).astype(np.float32)
            acc = f if acc is None else acc + f
        idx = int(round(x))
        return (acc / m + 0.5).astype(np.uint8), None, idx
    idx = max(0, min(R.sn - 1, int(round(x))))
    return get(idx), None, idx


def v14_events(R):
    """retime → sound events (whoosh into slow motion, hit on freeze, rewind tape)."""
    if not isinstance(R.tl, RetimedTimeline):
        return []
    snd = {"slow": ("whoosh", 0.8), "fast": ("whoosh", 0.6), "freeze": ("hit", 0.9), "rewind": ("scratch", 0.8)}
    return [(u, snd[k][0], snd[k][1], 0.0) for u, k in R.tl.rt.events if k in snd]
