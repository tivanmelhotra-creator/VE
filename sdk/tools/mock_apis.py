"""Local mock of ElevenLabs / Gemini TTS / Gemini Omni REST APIs — test the integrations for free.
    python3 tools/mock_apis.py &            # serves http://127.0.0.1:8765
    then put  "base_url": "http://127.0.0.1:8765"  in spec.voice / spec.omni_cfg and any dummy keys."""
import base64, io, json, subprocess, sys, wave
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765


def wav_bytes(sec, rate=24000):
    t = np.arange(int(sec * rate)) / rate
    x = (0.3 * np.sin(2 * np.pi * 180 * t) * (np.sin(2 * np.pi * 3 * t) > 0)).astype(np.float32)
    x = np.concatenate([np.zeros(int(0.3 * rate), np.float32), x, np.zeros(int(0.2 * rate), np.float32)])
    b = io.BytesIO(); w = wave.open(b, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
    w.writeframes((x * 32767).astype("<i2").tobytes()); w.close(); return b.getvalue()


def mp4_bytes():
    p = "/tmp/_mock_omni.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=720x1280:r=24:d=5",
                    "-pix_fmt", "yuv420p", p], check=True)
    return open(p, "rb").read()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, ctype="application/json", raw=None):
        b = raw if raw is not None else json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if "alt=media" in self.path:
            return self._send(None, "video/mp4", mp4_bytes())
        self._send({"id": "x", "status": "completed"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = self.rfile.read(n)
        try:
            js = json.loads(body)
        except Exception:
            js = {}
        ok_key = "xi-api-key" in self.headers or "x-goog-api-key" in self.headers
        if not ok_key:
            self.send_response(401); self.end_headers(); return
        if "/with-timestamps" in self.path:
            text = js["text"]; d = 0.07 * len(text)
            mp3 = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "mp3", "pipe:1"],
                                 input=wav_bytes(d, 44100), capture_output=True).stdout
            chars = list(text); st = [0.3 + i * d / len(chars) for i in range(len(chars))]
            en = [s + d / len(chars) for s in st]
            return self._send({"audio_base64": base64.b64encode(mp3).decode(),
                               "alignment": {"characters": chars, "character_start_times_seconds": st,
                                             "character_end_times_seconds": en}})
        if self.path.startswith("/v1beta/interactions"):
            if js.get("response_format", {}).get("type") == "audio":
                txt = js["input"][0]["content"][0]["text"]
                return self._send({"id": "tts1", "status": "completed", "steps": [{"type": "model_output", "content": [
                    {"type": "audio", "mime_type": "audio/wav",
                     "data": base64.b64encode(wav_bytes(0.075 * len(txt))).decode()}]}]})
            return self._send({"id": "omni_123", "status": "completed", "steps": [{"type": "model_output", "content": [
                {"type": "video", "mime_type": "video/mp4",
                 "uri": f"http://127.0.0.1:{PORT}/v1beta/files/abc:download"}]}]})
        self._send({})


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", PORT), H).serve_forever()
