"""Build dist/motion_creator_brain_sdk_<ver>.zip (v9: ships demo voice/media + the bundled .onnx model;
excludes renders, caches, matte caches and keys).   python3 tools/package.py v9 [--with-demo path.mp4]"""
import os, sys, zipfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ver = sys.argv[1] if len(sys.argv) > 1 else "v12"
demo = sys.argv[sys.argv.index("--with-demo") + 1] if "--with-demo" in sys.argv else None
out = os.path.join(ROOT, "dist", f"motion_creator_brain_sdk_{ver}.zip")
os.makedirs(os.path.dirname(out), exist_ok=True)
SKIP_DIRS = {".git", "__pycache__", "out", "dist", "deliverables", "node_modules", ".wrangler", "_work", "tests_out"}
n = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for d, dirs, files in os.walk(ROOT):
        import re as _re
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not _re.search(r"\.(matte|track|depth|rig|trace)(_[0-9a-f]+)?$", x)]
        for f in files:
            if ".enh_" in f or f.endswith((".shots.json", ".enh2.wav", ".enh.wav")) or f.startswith(".beat_"):
                continue
            if f.endswith((".pyc", ".prompts.json")) or _re.search(r"\.trace(_[0-9a-f]+)?\.json$", f) or f in ("render.log", ".mograph_keys.json"):
                continue
            p = os.path.join(d, f)
            z.write(p, os.path.relpath(p, ROOT)); n += 1
    if demo and os.path.exists(demo):
        z.write(demo, os.path.basename(demo)); n += 1
print(f"{out}  ({n} files, {os.path.getsize(out)//1024} KB)")
