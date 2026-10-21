"""
tools/build_aefx_gallery.py — static, searchable HTML gallery of all 703 AE-FX presets (animated WebP previews).

    python3 tools/build_aefx_gallery.py [--out ../public/aefx] [--only motion|text]
    → public/aefx/index.html (+ media/*.webp, ~7 MB, git-ignored; regenerate any time, ~2 min)

Served by the Hono app at /aefx/ (wrangler pages dev) or open the file directly.  Each card shows the measured tags
from catalog.json and a copy-ready JSON snippet ({"motion": key} / "preset": key).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mograph.aefx import motion as AM, textanim as TA   # noqa: E402
from mograph.aefx.__main__ import _demo_plate          # noqa: E402

FPS = 12


def _save(frames, path):
    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(path, "WEBP", save_all=True, append_images=ims[1:], duration=int(1000 / FPS), loop=0, quality=55,
                method=4)


def motion_webp(key, path):
    plate = cv2.resize(_demo_plate(), (192, 108), interpolation=cv2.INTER_AREA)
    p = AM.get(key)
    n = int(min(p["dur"], 4.0) * FPS) + 6
    _save([AM.apply_preset(plate, key, i / FPS, FPS, motion_blur=0)[:, :, ::-1] for i in range(n)], path)


def aecomp_webp(name, path, lower_third=False, title=False):
    from mograph.aecomp import generators as GEN, lowerthirds as LT, titles as TT
    from mograph.aecomp.render import render_rgb
    if title:
        c = TT.make(name, "مقدمه را آخر بنویس", "نکته نگارش")
        frames = [render_rgb(c, i / FPS, 360, 640)[160:400] for i in range(int(c["dur"] * FPS) + 3)]
    elif lower_third:
        c = LT.make(name, "دکتر سارا احمدی", "متخصص ارتودنسی")
        frames = [render_rgb(c, i / FPS, 360, 640)[400:600] for i in range(int(c["dur"] * FPS) + 3)]
    else:
        c = GEN.make(name, seed=1)
        frames = [render_rgb(c, i / FPS, 200, 200) for i in range(int(c["dur"] * FPS) + 4)]
    _save(frames, path)


def text_webp(key, path, text="آموزش *ویدیو*"):
    import cairo
    from mograph.v14 import glyphs as G
    b = G.typeset(text, 64)
    an = TA.Animator(key)
    frames = []
    for i in range(int(an.dur * FPS) + 6):
        s = cairo.ImageSurface(cairo.FORMAT_ARGB32, 240, 110)
        c = cairo.Context(s)
        c.set_source_rgb(0.09, 0.09, 0.11)
        c.paint()
        st = an.frame(b, i / FPS)
        G.draw_block(c, b, 120, 66, motion=lambda g, _s=st: _s.apply(g, G.GS()), scale=0.5)
        s.flush()
        frames.append(np.ndarray((110, s.get_stride() // 4, 4), np.uint8, s.get_data())[:, :240, 2::-1].copy())
    _save(frames, path)


PAGE = """<!doctype html><html lang="fa" dir="ltr"><head><meta charset="utf-8"><title>AE-FX presets</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:14px system-ui;background:#0e0f13;color:#ddd;margin:0}header{position:sticky;top:0;background:#16171d;
padding:10px 16px;display:flex;gap:10px;flex-wrap:wrap;align-items:center;z-index:2}input,select{background:#22242c;
color:#eee;border:1px solid #333;border-radius:6px;padding:6px 8px}main{display:grid;grid-template-columns:
repeat(auto-fill,minmax(250px,1fr));gap:12px;padding:16px}.card{background:#17181e;border-radius:10px;padding:8px}
.card img{width:100%;border-radius:6px;background:#000}.k{font-family:monospace;color:#7fd8ff}.t{color:#9a9;font-size:12px}
code{display:block;background:#0b0c10;padding:4px 6px;border-radius:4px;font-size:11px;cursor:pointer;margin-top:4px}
</style></head><body><header><b>AE-FX</b><input id="q" placeholder="search: glitch, in, spin, decode, soft…" size="34">
<select id="kind"><option value="">all</option><option>motion</option><option>text</option><option>shape</option><option>lower_third</option><option>title</option></select>
<span id="n"></span><span class="t">click a snippet to copy · data: sdk/mograph/aefx/presets/catalog.json</span></header>
<main id="g"></main><script>const D=__DATA__;const g=document.getElementById('g');
function draw(){const q=document.getElementById('q').value.toLowerCase().split(/\\s+/).filter(Boolean),
k=document.getElementById('kind').value;let h='',n=0;for(const d of D){if(k&&d.kind!==k)continue;
const s=(d.key+' '+d.label+' '+d.tags).toLowerCase();if(!q.every(w=>s.includes(w)))continue;n++;
h+=`<div class="card"><img loading="lazy" src="media/${d.key}.webp"><div class="k">${d.key}</div>
<div>${d.label}</div><div class="t">${d.tags}</div><code onclick="navigator.clipboard.writeText(this.textContent)">${d.snip}</code></div>`}
g.innerHTML=h;document.getElementById('n').textContent=n+' presets'}
document.getElementById('q').oninput=draw;document.getElementById('kind').onchange=draw;draw()</script></body></html>"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "public", "aefx"))
    ap.add_argument("--only")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    media = os.path.join(a.out, "media")
    os.makedirs(media, exist_ok=True)
    cat = json.load(open(os.path.join(os.path.dirname(AM.__file__), "presets", "catalog.json"), encoding="utf-8"))
    data = []
    for k, v in cat["motion"].items():
        data.append({"kind": "motion", "key": k, "label": f"{v['cat']} · {v['dir']} · {v['dur']}s",
                     "tags": " ".join(v["tags"]), "snip": json.dumps({"motion": k})})
        if a.only in (None, "motion"):
            p = os.path.join(media, k + ".webp")
            if a.force or not os.path.exists(p):
                motion_webp(k, p)
    for k, v in cat["text"].items():
        data.append({"kind": "text", "key": k, "label": f"{v['cat']}/{v['sub']} · {v['name']}",
                     "tags": " ".join(v["tags"] + v["props"]), "snip": f'"preset": "{k}"'})
        if a.only in (None, "text"):
            p = os.path.join(media, k + ".webp")
            if a.force or not os.path.exists(p):
                text_webp(k, p)
    from mograph.aecomp import generators as GEN, lowerthirds as LT
    for k, (_, gcat, doc) in GEN.GENERATORS.items():                 # v17 AE-Comp shape elements
        data.append({"kind": "shape", "key": "gen_" + k, "label": f"{gcat} · {doc}", "tags": f"aecomp shape {gcat}",
                     "snip": json.dumps({"type": "aecomp", "gen": k, "seed": 1, "size": 420})})
        if a.only in (None, "aecomp"):
            p = os.path.join(media, "gen_" + k + ".webp")
            if a.force or not os.path.exists(p):
                aecomp_webp(k, p)
    for k in LT.styles():
        data.append({"kind": "lower_third", "key": "lt_" + k, "label": "lower third · box-free RTL",
                     "tags": "aecomp lower third name title", "snip": json.dumps(
                         {"type": "aecomp", "lower_third": k, "title": "…", "role": "…"}, ensure_ascii=False)})
        if a.only in (None, "aecomp"):
            p = os.path.join(media, "lt_" + k + ".webp")
            if a.force or not os.path.exists(p):
                aecomp_webp(k, p, lower_third=True)
    from mograph.aecomp import titles as TT
    for k in TT.styles():
        data.append({"kind": "title", "key": "tt_" + k, "label": "full-frame title · AE text animators",
                     "tags": "aecomp title hero kicker", "snip": json.dumps(
                         {"type": "aecomp", "title_style": k, "hero": "…", "kicker": "…"}, ensure_ascii=False)})
        if a.only in (None, "aecomp"):
            p = os.path.join(media, "tt_" + k + ".webp")
            if a.force or not os.path.exists(p):
                aecomp_webp(k, p, title=True)
    with open(os.path.join(a.out, "index.html"), "w", encoding="utf-8") as f:
        f.write(PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    tot = sum(os.path.getsize(os.path.join(media, x)) for x in os.listdir(media))
    print(f"gallery: {len(data)} presets → {os.path.abspath(a.out)}/index.html  (media {tot / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
