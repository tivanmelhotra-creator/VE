"""
mograph.keys — API keys for the optional cloud services (voice + Omni video).

Lookup order: env var → <project>/.mograph_keys.json → ~/.mograph/keys.json → a dropped keys FILE
(mograph_keys.txt / .env / keys.txt / json in the project, home, AI Drive or uploads) — see HOWTO_FA.
Keys are NEVER printed, logged, or written into spec/README/git (.mograph_keys.json is git-ignored).

    python -m mograph keys                      # status (set / missing) — no values shown
    python -m mograph keys set ELEVENLABS_API_KEY sk_...
    python -m mograph keys set GEMINI_API_KEY AIza...

If a key is missing, the engine raises MissingKey whose message is a ready-to-send request
to the user (in Persian) — the agent must ASK the user, never guess or skip silently.
"""
from __future__ import annotations

import json
import os

SERVICES = {
    "ELEVENLABS_API_KEY": {
        "what": "ElevenLabs (natural voice-over, Persian supported by eleven_v3)",
        "where": "https://elevenlabs.io/app/settings/api-keys",
        "ask_fa": "برای ساخت صدای راوی با ElevenLabs به API Key نیاز دارم. "
                  "از https://elevenlabs.io/app/settings/api-keys یک کلید بسازید و برایم بفرستید "
                  "(دسترسی Text to Speech و Voices کافی است). اگر Voice ID خاصی هم مدنظرتان است بفرستید.",
    },
    "GEMINI_API_KEY": {
        "what": "Google AI Studio (Gemini TTS voice + Gemini Omni Flash video)",
        "where": "https://aistudio.google.com/apikey",
        "ask_fa": "برای صدای Gemini TTS یا ساخت/ادیت ویدیوی واقع‌گرایانه با Gemini Omni به API Key گوگل "
                  "AI Studio نیاز دارم. از https://aistudio.google.com/apikey یک کلید بسازید و برایم بفرستید. "
                  "(هزینه‌ی تولید روی همان حساب گوگل محاسبه می‌شود.)",
    },
}


class MissingKey(RuntimeError):
    def __init__(self, name):
        info = SERVICES.get(name, {})
        super().__init__(
            f"API key '{name}' is not configured ({info.get('what', '')}).\n"
            f"AGENT: stop and ask the user — send them this (Persian):\n  {info.get('ask_fa', name)}\n  {HOWTO_FA}\n"
            f"Then run:  python -m mograph keys import <uploaded file>   (or: keys set {name} <key>)")
        self.name = name


def _paths(project_dir=None):
    ps = []
    if project_dir:
        ps.append(os.path.join(project_dir, ".mograph_keys.json"))
    ps.append(os.path.join(os.getcwd(), ".mograph_keys.json"))
    ps.append(os.path.expanduser("~/.mograph/keys.json"))
    return ps


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_key(name, project_dir=None, required=True):
    v = os.environ.get(name)
    if v:
        return v.strip()
    for p in _paths(project_dir):
        v = _read(p).get(name)
        if v:
            return str(v).strip()
    hit = scan_key_files(project_dir).get(name)     # a dropped keys file (txt/.env/json)
    if hit:
        set_key(name, hit[0])
        return hit[0]
    if required:
        raise MissingKey(name)
    return None


def set_key(name, value, scope="user"):
    path = os.path.expanduser("~/.mograph/keys.json") if scope == "user" else os.path.join(os.getcwd(), ".mograph_keys.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = _read(path)
    d[name] = value.strip()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    os.chmod(path, 0o600)
    return path


def status():
    lines = []
    for name, info in SERVICES.items():
        ok = get_key(name, required=False) is not None
        lines.append(f"{'✓' if ok else '✗'} {name:20s} {info['what']}" + ("" if ok else f"   → {info['where']}"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# v7: hand keys over as a FILE (no pasting into chat needed — works in any agent)
# ---------------------------------------------------------------------------
HOWTO_FA = ("روش امن تحویل کلید (بدون نوشتن در متن چت): یک فایل متنی به اسم mograph_keys.txt بسازید و داخلش "
            "بنویسید  ELEVENLABS_API_KEY=...  و/یا  GEMINI_API_KEY=...  بعد همان فایل را به‌صورت «فایل پیوست» "
            "بفرستید یا در AI Drive بگذارید. کلید فقط در حافظه‌ی امن (~/.mograph/keys.json) ذخیره می‌شود، "
            "هیچ‌جا چاپ نمی‌شود و وارد پروژه/بسته/گیت نمی‌شود.")
KEY_FILES = ["mograph_keys.txt", "mograph_keys.json", "mograph.env", ".env", "keys.txt", "api_keys.txt"]
KEY_DIRS = ["{project}", "{cwd}", "~", "/mnt/aidrive", "/mnt/aidrive/keys", "/home/user/uploaded_files",
            "/home/user", "/mnt/user-data", "/mnt/user-data/uploads"]
ALIASES = {"ELEVENLABS_API_KEY": ("ELEVEN", "XI_API"),
           "GEMINI_API_KEY": ("GEMINI", "GOOGLE", "AISTUDIO", "AI_STUDIO", "GOOGLEAI")}


def parse_key_text(txt):
    """Forgiving: NAME=value / NAME: value / JSON / export NAME="value" / bare keys recognised by shape."""
    raw = {}
    txt = (txt or "").strip()
    if txt.startswith("{"):
        try:
            raw = {str(k): str(v) for k, v in json.loads(txt).items()}
        except Exception:
            raw = {}
    else:
        for line in txt.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("export "):
                line = line[7:]
            for sep in ("=", ":"):
                if sep in line:
                    k, v = line.split(sep, 1)
                    raw[k.strip()] = v.strip().strip('"').strip("'")
                    break
    out = {}
    for k, v in raw.items():
        kk = k.upper().replace("-", "_").replace(" ", "_")
        if not v:
            continue
        if kk in SERVICES:
            out[kk] = v
            continue
        for canon, al in ALIASES.items():
            if any(a in kk for a in al):
                out.setdefault(canon, v)
                break
    for tok in txt.replace(",", " ").split():
        tok = tok.strip().strip('"').strip("'")
        if tok.startswith("sk_") and len(tok) > 30:
            out.setdefault("ELEVENLABS_API_KEY", tok)
        elif tok.startswith("AIza") and len(tok) > 30:
            out.setdefault("GEMINI_API_KEY", tok)
    return out


def scan_key_files(project_dir=None):
    found = {}
    for d in KEY_DIRS:
        d = os.path.expanduser(d.format(project=project_dir or os.getcwd(), cwd=os.getcwd()))
        for f in KEY_FILES:
            p = os.path.join(d, f)
            try:
                if os.path.isfile(p) and os.path.getsize(p) < 20000:
                    for k, v in parse_key_text(open(p, encoding="utf-8", errors="ignore").read()).items():
                        found.setdefault(k, (v, p))
            except OSError:
                continue
    return found


def import_file(path, delete=False):
    keys = parse_key_text(open(path, encoding="utf-8", errors="ignore").read())
    for k, v in keys.items():
        set_key(k, v)
    if delete and keys:
        try:
            os.remove(path)
        except OSError:
            pass
    return sorted(keys)
