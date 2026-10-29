"""
Media I/O for the layered lip-sync engine: image loading (RGBA aware), registered cover-fit, audio decode, and a
single-pass ffmpeg writer (raw BGR frames piped straight into libx264 + AAC mux — no lossy mp4v intermediate).

API: load_image(path) → (bgr, alpha|None) · cover_fit(img, (w, h)) · read_audio_mono(path, sr) · has_audio(path) ·
     FFmpegWriter(out, (w, h), fps, audio=None, crf=18)  (context manager, .write(bgr))
"""
from __future__ import annotations

import os
import shutil
import subprocess

import cv2
import numpy as np


def require_ffmpeg() -> None:
    """raise a clear error if ffmpeg / ffprobe are missing."""
    for b in ("ffmpeg", "ffprobe"):
        if shutil.which(b) is None:
            raise RuntimeError(f"`{b}` not found on PATH — install ffmpeg (apt-get install ffmpeg)")


def load_image(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """→ (bgr uint8, alpha float32 0..1 or None). Raises FileNotFoundError with the path (cv2.imread returns None)."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"image not found: {path}")
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise ValueError(f"could not decode image: {path}")
    if im.ndim == 2:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), None
    if im.shape[2] == 4:
        a = im[:, :, 3].astype(np.float32) / 255.0
        if a.min() > 0.999:
            return np.ascontiguousarray(im[:, :, :3]), None
        return np.ascontiguousarray(im[:, :, :3]), a
    if im.dtype != np.uint8:
        im = cv2.convertScaleAbs(im, alpha=255.0 / max(1, im.max()))
    return im, None


def cover_params(src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> tuple[float, int, int]:
    """scale + crop offsets that make src COVER dst (centre crop). Same params for every layer = registration."""
    sw, sh = src_wh
    dw, dh = dst_wh
    s = max(dw / sw, dh / sh)
    return s, int(round((sw * s - dw) / 2)), int(round((sh * s - dh) / 2))


def cover_fit(img: np.ndarray, dst_wh: tuple[int, int], interp: int | None = None) -> np.ndarray:
    """resize to cover dst_wh (aspect preserved) then centre-crop. Works for uint8 images and float masks."""
    h, w = img.shape[:2]
    if (w, h) == tuple(dst_wh):
        return img.copy()
    s, ox, oy = cover_params((w, h), dst_wh)
    if interp is None:
        interp = cv2.INTER_LANCZOS4 if s > 1 else cv2.INTER_AREA
    nw, nh = max(dst_wh[0], int(round(w * s))), max(dst_wh[1], int(round(h * s)))
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    return np.ascontiguousarray(r[oy:oy + dst_wh[1], ox:ox + dst_wh[0]])


def read_audio_mono(path: str, sr: int = 16000) -> np.ndarray:
    """decode any audio / video file to mono float32 at `sr`. Raises on missing file / no audio stream."""
    require_ffmpeg()
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"audio not found: {path}")
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                       capture_output=True)
    x = np.frombuffer(p.stdout, np.float32)
    if p.returncode != 0 or x.size == 0:
        raise ValueError(f"no decodable audio in {path}: {p.stderr.decode(errors='ignore')[-300:]}")
    return x.copy()


def has_audio(path: str) -> bool:
    """True if the file has at least one audio stream."""
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    return "audio" in p.stdout


def video_info(path: str) -> dict:
    """{fps, frames, w, h} via OpenCV (raises if unreadable)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    info = {"fps": cap.get(cv2.CAP_PROP_FPS) or 24.0, "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return info


class FFmpegWriter:
    """Pipe BGR frames into ffmpeg → H.264 (yuv420p, +faststart) with optional audio mux in the same pass."""

    def __init__(self, out: str, wh: tuple[int, int], fps: float, audio: str | None = None, crf: int = 18,
                 preset: str = "medium", audio_bitrate: str = "192k"):
        require_ffmpeg()
        w, h = int(wh[0]), int(wh[1])
        if w % 2 or h % 2:
            raise ValueError(f"output size must be even for yuv420p, got {w}x{h}")
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        self.out, self.wh, self.n = out, (w, h), 0
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
               "-r", f"{fps}", "-i", "-"]
        if audio:
            cmd += ["-i", audio, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", audio_bitrate, "-shortest"]
        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", out]
        self._p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape[1] != self.wh[0] or frame_bgr.shape[0] != self.wh[1]:
            raise ValueError(f"frame {frame_bgr.shape[1]}x{frame_bgr.shape[0]} != writer {self.wh}")
        try:
            self._p.stdin.write(np.ascontiguousarray(frame_bgr, dtype=np.uint8).tobytes())
        except BrokenPipeError as e:
            err = self._p.stderr.read().decode(errors="ignore")
            raise RuntimeError(f"ffmpeg died: {err[-500:]}") from e
        self.n += 1

    def close(self) -> str:
        if self._p.stdin and not self._p.stdin.closed:
            self._p.stdin.close()
        rc = self._p.wait()
        err = self._p.stderr.read().decode(errors="ignore") if self._p.stderr else ""
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed ({rc}): {err[-500:]}")
        return self.out

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            self.close()
        else:
            try:
                self._p.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            self._p.kill()
            self._p.wait()
