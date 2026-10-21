"""
python -m mograph <command> spec.json [options]

  qa       spec.json                         lint + overlap/out-of-frame report (seconds, no video)
  frame    spec.json --t 12.5 [--out f.png]  render ONE frame (instant check of a moment)
  sheet    spec.json [--out sheet.jpg]       contact sheet of hero frames of every chapter
  preview  spec.json [--out p.mp4]           half-res 15fps video, all CPU cores (fast iteration)
  render   spec.json [--out final.mp4]       full quality + soundtrack + QA gate
  review   final.mp4 [--spec spec.json]       post-render check of the REAL video (black/frozen/audio/LUFS)
  align    vo.wav --text script.txt          word timestamps from your narration (needs faster-whisper)
  icons    [query]                           list/search icons (Persian or English)
  catalog  [--out catalog.png]               render a catalog of all icons
  analyze  ref.mp4 [-o blueprint.json] [--into spec.json]   v9: cuts, cadence, palette, LUFS, BPM + beats (b:N)
  srt2subs file.srt                          print subtitles JSON from an .srt
  cues     spec.json                         readable sound score: every SFX with time + reason (check sync)
  join     a.mp4 b.mp4 ... --out final.mp4   join acts (one spec per style) (re-encoded, -14 LUFS kept per act)
  keys     [set NAME VALUE | import FILE [--yes]]   key status / store / import a keys file (--yes deletes it)
  voice    spec.json [--provider elevenlabs|gemini|draft]   natural VO from the subtitles →
                                             voice/vo.wav + spec.voiced.json (word-exact, retimed)
  voices   elevenlabs|gemini                 list available voices
  omni     spec.json [--yes] [--job ID] [--provider draft]  Gemini Omni realistic shots (dry run
                                             without --yes: shows the paid plan, asks nothing)
  track    spec.json                         v11: precompute every footage tracker (+ debug sheet per clip)
  track    clip.mp4 --kind hand|pose|face|point|planar|color [--at u,v] [--point index_tip]   one-off track
  prompts  spec.json                         v11: image / video / music / voice jobs for missing media (gen blocks)
  cut      spec.json                         v12: edit decision list (jump cuts, shot cuts, captions in edited time)
  matte    spec.json                         v11: precompute person mattes of every footage element (cached)
  extract3d sheet.jpg --names a,b,c,d [--grid 2x2]  v13.5: slice & alpha-matte 3D sprite sheet into mograph/assets_3d/
  dna-plan <topic_or_spec>                   v13.5: generate Visual-DNA 5-part blueprint & 3D asset prompts
  before-after raw.mp4 after.mp4 --out out.mp4  v13.8: render @novadent-style BEFORE/AFTER split-card showcase
  img2mesh <image.png> --out model.ply       v14.4: 2D image → volumetric .ply point cloud (normals + colour)
  turntable3d <three_model|script.js|image.png> [--out sheet.png]  v20: 3D audit sheet (10 yaw/pitch/explode angles)
  three3d-list  ·  three3d-sheet <model|script.js> [--out s.png]  ·  three3d-selftest   v20: Procedural Three.js 3D engine
  three3d-anim-sheet <hourglass_loop|laptop_open|assemble> [--out s.png]           v20.1: LOOK at an object animation
  pbr3d-search <words> [hdri]  ·  pbr3d-fetch <id ...>  ·  pbr3d-sheet <id|model.glb>   v14.6: PHOTOREAL scanned PBR 3D (EffectCraft)
  sculpt-check <tree.json|preset>            v14.4: strict ObjectSculptSpec gate (ids, materials, ≥3 parts …)
  sculpt-compare <tree.json> <ref.png> [--t yaw] [--out cmp.png]  v14.4: auto camera-match + reference | render | IoU
  --from/--to  render a time range only (e.g. re-render chapter 4 only)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mograph", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd")
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--t", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--from", dest="t_from", type=float, default=0.0)
    ap.add_argument("--to", dest="t_to", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--force", action="store_true", help="render even if QA finds errors")
    ap.add_argument("--text", default=None, help="script text file for align")
    ap.add_argument("--no-overlap", action="store_true", help="skip the (slower) overlap scan")
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--yes", action="store_true", help="omni: actually call the paid API")
    ap.add_argument("--job", default=None)
    ap.add_argument("-o", dest="out_o", default=None)
    ap.add_argument("--into", default=None, help="analyze: write beats into this spec")
    ap.add_argument("--dedupe-fillers", action="store_true", help="voice: excise repeated / filler words")
    ap.add_argument("--kind", default=None, help="track: tracker kind")
    ap.add_argument("--at", default=None, help="track: u,v seed point")
    ap.add_argument("--point", default=None, help="track: landmark name")
    ap.add_argument("--names", default=None, help="extract3d: comma-separated sprite names")
    ap.add_argument("--grid", default="2x2", help="extract3d: grid layout e.g. 2x2, 1x1, 2x3")
    ap.add_argument("--bg", default="auto", help="extract3d: background color mode auto|black|green|white")
    a = ap.parse_args(argv)
    if a.out_o and not a.out:
        a.out = a.out_o
    if a.out and a.cmd not in ("review",) and os.path.dirname(a.out):
        os.makedirs(os.path.dirname(a.out), exist_ok=True)   # out/ may be gone after a sandbox reset

    if a.cmd == "keys":
        from .keys import set_key, status
        if a.spec == "set" and len(a.rest) == 2:
            print(f"saved {a.rest[0]} → {set_key(a.rest[0], a.rest[1])}")
            return 0
        if a.spec == "import" and a.rest:
            from .keys import import_file
            names = import_file(a.rest[0], delete=a.yes)
            print(f"imported {names or 'nothing (no recognisable keys)'} → ~/.mograph/keys.json"
                  + ("; source file deleted" if a.yes and names else ""))
            return 0 if names else 1
        print(status())
        return 0
    if a.cmd == "voices":
        from .voice import list_voices
        print("\n".join(list_voices(a.spec or "gemini")))
        return 0
    if a.cmd in ("voice", "omni"):
        from .keys import MissingKey
        try:
            if a.cmd == "voice":
                from .voice import voice_spec
                voice_spec(a.spec, provider=a.provider, out_path=a.out, dedupe_fillers=a.dedupe_fillers)
                return 0
            from .omni import run
            return run(a.spec, yes=a.yes, only=a.job, provider=a.provider or "gemini")
        except MissingKey as e:
            print(str(e))
            return 3
    if a.cmd == "join":
        import subprocess, tempfile, os as _os
        parts = [a.spec] + list(a.rest)
        out = a.out or "joined.mp4"
        inputs = sum([["-i", p] for p in parts], [])
        n = len(parts)
        fc = "".join(f"[{i}:v][{i}:a]" for i in range(n)) + f"concat=n={n}:v=1:a=1[v][a]"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
                        "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "192k", out], check=True)
        print(out)
        return 0
    if a.cmd == "cues":
        from .scene import load_spec as _ls
        from .sound import score_report
        print(score_report(_ls(a.spec)))
        return 0
    if a.cmd == "icons":
        from .icons import ICONS, find_icons
        names = find_icons(a.spec) if a.spec else sorted(ICONS)
        for n in names:
            print(f"{n:18s} {' '.join(ICONS[n].get('tags', []))}")
        return 0
    if a.cmd == "catalog":
        from .catalog import render_catalog
        print(render_catalog(a.out or "catalog/icons_catalog.png"))
        return 0
    if a.cmd == "review":
        from .qa import review_video
        exp_d = exp_f = None
        if a.out and a.out.endswith(".json"):
            from .scene import load_spec as _ls
            sc = _ls(a.out); exp_d, exp_f = sc.duration, sc.fps
        sheet = os.path.splitext(a.spec)[0] + "_review.jpg"
        txt, n = review_video(a.spec, exp_d, exp_f, sheet_path=sheet)
        print(txt)
        return 1 if n else 0
    if a.cmd == "analyze":
        from .analyze import analyze
        analyze(a.spec, out=a.out or (a.rest[0] if a.rest else None), spec_path=a.into)
        return 0
    if a.cmd == "cut":                       # v12: print the edit decision list
        import json as _json, copy as _copy
        from .cut import apply_edit, report as _rep
        with open(a.spec, encoding="utf-8") as f:
            sp = _json.load(f)
        apply_edit(sp, os.path.dirname(os.path.abspath(a.spec)))
        print(_rep(sp))
        return 0
    if a.cmd == "prompts":
        from .prompts import run as _prun
        return _prun(a.spec, a.out)
    if a.cmd == "track" and a.spec and not a.spec.endswith(".json"):
        from .track import get_track, debug_sheet
        spec = {"kind": a.kind or "hand"}
        if a.at:
            spec["at"] = [float(v) for v in a.at.split(",")]
        if a.point:
            spec["point"] = a.point
        tr = get_track(a.spec, spec, log=print)
        out = a.out or os.path.splitext(a.spec)[0] + f"_{spec['kind']}_track.jpg"
        print(debug_sheet(a.spec, {spec["kind"]: tr}, out))
        return 0
    if a.cmd in ("track", "matte"):
        from .scene import load_spec as _ls
        sc = _ls(a.spec)
        from .fx import footage_src, footage_tracker
        from .track import debug_sheet
        n = 0
        for ch in sc.chapters:
            for el in ch["elements"]:
                if el.get("type") != "footage":
                    continue
                if a.cmd == "track":
                    trs = {}
                    for name in (el.get("trackers") or {}):
                        trs[name] = footage_tracker(el, sc, name)
                    if el.get("reframe") == "face":
                        trs["face"] = footage_tracker(el, sc, "face")
                    if trs:
                        out = os.path.join(sc.base_dir, "out", f"{el['id']}_tracks.jpg")
                        os.makedirs(os.path.dirname(out), exist_ok=True)
                        print(debug_sheet(footage_src(el), trs, out)); n += len(trs)
                else:
                    from .media import reader
                    from .matte import cached_matte
                    import cv2 as _cv2
                    r = reader(footage_src(el))
                    mh = int(el.get("matte_h", 512))
                    for i in range(0, max(1, r.n)):
                        fr = r.frame(i / r.fps)
                        ms = min(1.0, mh / fr.shape[0])
                        small = _cv2.resize(fr, (int(fr.shape[1] * ms), int(fr.shape[0] * ms))) if ms < 1 else fr
                        cached_matte(r.path, i, small, el.get("matte") if isinstance(el.get("matte"), str) else "auto",
                                     el.get("key"))
                    n += 1
                    print(f"[matte] {el['id']}: {r.n} frames cached")
        print(f"[{a.cmd}] {n} done")
        return 0
    if a.cmd == "align":
        from .align import align_cli
        return align_cli(a.spec, a.text, a.out)
    if a.cmd == "extract3d":
        from .extract3d import extract_3d_sheet
        if not a.spec or not a.names:
            print("Usage: python3 -m mograph extract3d <sheet.jpg> --names a,b,c,d [--grid 2x2]")
            return 1
        names = [x.strip() for x in a.names.split(",") if x.strip()]
        saved = extract_3d_sheet(a.spec, names, grid=a.grid, bg_mode=a.bg, out_dir=a.out)
        print("\n".join(saved))
        return 0
    if a.cmd in ("before-after", "before_after"):
        from .creative_reels import render_before_after_video
        if not a.spec or not a.rest:
            print("Usage: python3 -m mograph before-after <raw.mp4> <after.mp4> --out showcase.mp4")
            return 1
        out_p = a.out or "before_after_showcase.mp4"
        render_before_after_video(a.spec, a.rest[0], out_p)
        print(out_p)
        return 0
    if a.cmd in ("img2mesh", "img2ply"):
        from .sam3d_bridge import export_image_to_ply
        from .obj3d import resolve_asset
        src = resolve_asset(a.spec) or a.spec
        out_p = a.out or (os.path.splitext(os.path.basename(a.spec))[0] + ".ply")
        export_image_to_ply(src, out_p)
        print(out_p)
        return 0
    if a.cmd in ("sculpt-check", "sculpt_check"):
        from .sculpt3d import get_tree, validate_tree
        errs, warns = validate_tree(get_tree(a.spec, a.spec))
        for w in warns:
            print("  warn ", w)
        for e in errs:
            print("  ERROR", e)
        print("sculpt-check:", "BLOCKED" if errs else "PASS")
        return 1 if errs else 0
    if a.cmd in ("sculpt-compare", "sculpt_compare"):
        from .sculpt3d import get_tree, comparison_sheet
        ref = a.rest[0] if a.rest else None
        if not ref:
            print("usage: sculpt-compare tree.json reference.png [--t yaw]")
            return 2
        out_p = a.out or "sculpt_compare.png"
        iou = comparison_sheet(get_tree(a.spec, a.spec), ref, out_p, yaw=a.t)   # no --t → auto camera match
        print(f"{out_p}  silhouette IoU {iou:.3f}")
        return 0
    if a.cmd in ("pbr3d-fetch", "pbr3d_fetch"):          # v14.6: CC0 scanned models / HDRIs from Poly Haven
        from . import pbr3d
        for n in [a.spec] + list(a.rest or []):
            print(pbr3d.fetch(n))
        return 0
    if a.cmd in ("pbr3d-search", "pbr3d_search"):
        from . import pbr3d
        kind = "hdris" if (a.rest and a.rest[0] == "hdri") else "models"
        for k, name, cats in pbr3d.search(a.spec or "", kind=kind):
            print(f"{k:34s} {name:34s} {', '.join(cats[:4])}")
        return 0
    if a.cmd in ("pbr3d-sheet", "pbr3d_sheet"):
        from . import pbr3d
        print(pbr3d.audit_sheet(a.spec, a.out))
        return 0
    if a.cmd in ("three3d-list", "three3d_list"):
        from . import three3d
        for k in three3d.list_models():
            sc = three3d.get_model(k)
            print(f"{k:24s} {len(sc.parts):2d} parts · {sc.tri_count:5d} tris")
        return 0
    if a.cmd in ("three3d-sheet", "three3d_sheet"):
        from . import three3d
        out_p = a.out or "three3d_sheet.png"
        print(three3d.audit_sheet(a.spec or "hourglass_3d", out_p))
        return 0
    if a.cmd in ("three3d-anim-sheet", "three3d_anim_sheet"):
        from . import three_anim
        print(three_anim.sheet(a.spec or "hourglass_loop", a.out))
        return 0
    if a.cmd in ("three3d-selftest", "three3d_selftest"):
        from . import three3d
        return three3d.selftest()            # selftest() already returns a process exit code (0 = all pass)
    if a.cmd in ("turntable3d", "audit3d"):
        from . import three3d
        out_p = a.out or "turntable3d.png"
        if a.spec and (three3d.resolve_model_name(a.spec) or str(a.spec).endswith(".js")):
            three3d.audit_sheet(a.spec, out_p)
        else:
            from .obj3d import audit_sheet
            audit_sheet(a.spec, out_p)
        print(out_p)
        return 0
    if a.cmd in ("dna-plan", "dnaplan"):
        from .visual_dna import generate_dna_plan_markdown
        md = generate_dna_plan_markdown(a.spec or "Custom Motion Reel")
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                f.write(md)
        print(md)
        return 0
    if a.cmd == "srt2subs":
        from .audio import subtitles_from_srt
        print(json.dumps(subtitles_from_srt(a.spec), ensure_ascii=False, indent=1))
        return 0

    from .scene import load_spec
    t0 = time.time()
    scene = load_spec(a.spec)
    stem = os.path.splitext(os.path.basename(a.spec))[0]
    outdir = os.path.join(scene.base_dir, "out")
    os.makedirs(outdir, exist_ok=True)

    if a.cmd == "qa":
        from .qa import report
        txt, errs = report(scene, overlaps=not a.no_overlap)
        print(txt)
        print(f"[qa] {time.time() - t0:.1f}s, duration={scene.duration:.2f}s, chapters={len(scene.chapters)}")
        return 1 if errs else 0
    if a.cmd == "frame":
        from .render import save_frame
        t = a.t if a.t is not None else 0.0
        out = a.out or os.path.join(outdir, f"{stem}_t{t:06.2f}.png")
        save_frame(scene, t, out, scale=a.scale)
        print(out)
        return 0
    if a.cmd == "sheet":
        from .qa import contact_sheet
        out = a.out or os.path.join(outdir, f"{stem}_sheet.jpg")
        contact_sheet(scene, out)
        print(out)
        return 0
    if a.cmd in ("preview", "render"):
        from .qa import report
        from .render import render_video
        txt, errs = report(scene, overlaps=(a.cmd == "render" and not a.no_overlap))
        print(txt)
        if errs and not a.force:
            print("[qa] errors found → fix the spec (or pass --force).")
            return 1
        audio = None
        if not a.no_audio:
            from .audio import build_soundtrack
            audio = build_soundtrack(scene, os.path.join(outdir, f"{stem}_audio.wav"))
        preview = a.cmd == "preview"
        out = a.out or os.path.join(outdir, f"{stem}_{'preview' if preview else 'final'}_{int(time.time()) % 100000}.mp4")
        render_video(scene, out, preview=preview, workers=a.workers, start=a.t_from, end=a.t_to,
                     audio_path=audio)
        print(out)
        if not preview and a.t_to is None and a.t_from == 0:
            from .qa import review_video
            txt, n = review_video(out, scene.duration, scene.fps, sheet_path=os.path.splitext(out)[0] + "_review.jpg")
            print(txt)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
