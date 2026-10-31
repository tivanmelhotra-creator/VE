"""
toonface.kit.redesign — v33 turn reference expressions into RIG-READY turnaround drawings, per view.

Why: an expression cut straight from a reference sheet is front-only, may hide a brow under the hair and carries
hair remnants.  The rig needs every expression drawn on a clean (bald) head in the views the head turns through
(front → 3/4 → 90° profile), with both brows complete.  An image model redraws each expression in the character's
own style; this module makes that step mechanical for an agent:

    1. prepare  : reference sheet(s) → one job per expression (reference crop + exact prompt)        [no model]
    2. (agent)  : call the image model with job.prompt and [job.ref_url, plate_url]  → turnaround png
    3. ingest   : turnaround png → 3 per-view decals (front / q34 / profile) in face units, with facing side
    4. decalcomp: places the decal of the matching view on the tracked blank face

CLI
    python3 -m mograph.motion.toonface.kit.redesign prepare --lib /tmp/lib --out /tmp/jobs
    python3 -m mograph.motion.toonface.kit.redesign ingest  --png turn.png --name smug --out /tmp/rig_lib
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from .sheetlib import Expression, ExpressionLibrary, detect_faces, extract_decal, split_parts

VIEWS = ("front", "q34", "profile")

# what must be preserved for each expression (the "feeling") — read from the reference sheets
DESCRIBE = {
    "roaring": "eyes squeezed shut as two '>' '<' chevrons, brows pulled down hard, huge open roaring mouth with "
               "upper teeth and tongue visible",
    "star_eyed": "both eyes are big white five-point stars, brows raised high, wide open happy smile showing tongue",
    "terrified": "huge round white eyes with tiny pinpoint pupils, brows raised and bent with worry, clenched "
                 "rectangular mouth showing gritted teeth",
    "pleading": "huge round glossy puppy eyes with big dark pupils and two white highlights each, worried brows "
                "raised in the middle, tiny gentle closed smile",
    "shy": "eyes looking down to the side with worried raised brows, small wavy embarrassed mouth, little blush "
           "hatching lines on both cheeks",
    "smug": "half-closed bored eyes looking sideways, one brow raised and the other lowered, lopsided smirk",
    "shocked": "huge round white eyes with tiny pupils, brows shot up, wide open mouth with jagged teeth",
    "dizzy": "both eyes are black spirals, wobbly zig-zag mouth",
    "mischievous": "narrowed scheming eyes with brows slanted down to the centre, very wide toothy grin",
    "yawning": "eyes closed as relaxed downward arcs, sleepy brows, tall open oval yawning mouth",
    "thinking": "half-lidded eyes looking up and to the side, one brow raised, small pursed mouth pushed to one "
                "side",
    "eating": "eyes closed as happy upward arcs, puffed full cheeks, small mouth closed with crumbs around it",
    "disgusted": "squinting eyes, brows scrunched down, wavy grimacing mouth pulled to one side, wrinkled nose",
    "eureka": "wide bright round eyes, brows raised, open delighted smile",
    "sulking": "angry brows slanted down to the nose, eyes looking sideways with pupils in the corner, small "
               "frowning mouth",
    "whistling": "eyes closed as happy arcs, small round puckered whistling mouth",
    "freezing": "wide round eyes with small pupils, worried shivering brows, chattering clenched teeth",
    "shocked_2": "huge round white eyes with tiny pupils, brows shot up, small open round 'o' mouth",
    "joy": "eyes closed as happy upward arcs, huge open laughing mouth with tongue",
    "excited": "eyes closed as happy upward arcs, huge open laughing mouth with tongue",
    "angry": "furious brows slanted sharply down to the nose, narrowed eyes with small pupils, wide clenched "
             "snarling mouth showing teeth",
    "crying": "eyes squeezed shut, brows bent up in sorrow, wobbly crying mouth, two streams of tears",
    "sneaky": "narrow sly eyes looking sideways under slanted brows, wide sneaky grin",
    "dizzy_2": "one eye an X and the other a spiral, tongue hanging out of a crooked open mouth",
    "excited_2": "eyes are black with gold stars inside, brows raised, huge grin showing all teeth",
    "scared": "round eyes with small pupils, brows raised in fear, open wobbly frightened mouth, sweat drop",
}

PROMPT = (
    "Rig-ready expression turnaround of the cartoon boy. Image 1 shows the EXPRESSION to copy exactly "
    "({NAME}: {DESC}, small nose). Image 2 shows the character's blank head shape and drawing style (front view and "
    "3/4 view). Redraw the same head completely BALD (no hair, no ears) as a smooth head with exactly the same face "
    "outline and chin as image 2, thick clean black ink outlines, flat peach skin, same line weight. BOTH eyebrows "
    "fully visible. Three views of the same head with the same expression side by side on plain white background, "
    "same size, evenly spaced: 1) front view, 2) three-quarter view turned 45 degrees toward the viewer's right, "
    "3) full side profile facing the viewer's right. No neck, no body, no shading, no blush, no text."
)


def prompt_for(name: str, desc: str | None = None) -> str:
    base = name.split("@")[0]
    d = desc or DESCRIBE.get(base) or DESCRIBE.get(base.rstrip("_0123456789")) or base.replace("_", " ")
    return PROMPT.format(NAME=base.replace("_", " ").upper(), DESC=d)


# ================================================================================================ prepare
def prepare(lib_dir: str, sheets_dir: str, out_dir: str, names=None):
    """one job per expression: reference crop png + prompt.  Returns the job list (also jobs.json)."""
    lib = ExpressionLibrary.load(lib_dir)
    os.makedirs(out_dir, exist_ok=True)
    jobs = []
    for nm in names or lib.names():
        e = lib.items[nm]
        im = cv2.imread(os.path.join(sheets_dir, e.source))
        x0, y0, x1, y1 = e.cell_bbox
        w, h = x1 - x0, y1 - y0
        crop = im[max(0, int(y0 - 0.35 * h)):min(im.shape[0], int(y1 + 0.08 * h)),
                  max(0, int(x0 - 0.2 * w)):min(im.shape[1], int(x1 + 0.2 * w))]
        ref = os.path.join(out_dir, f"{nm}_ref.png")
        cv2.imwrite(ref, crop)
        jobs.append(dict(name=nm, ref=ref, prompt=prompt_for(nm), out=os.path.join(out_dir, f"{nm}_turn.png")))
    with open(os.path.join(out_dir, "jobs.json"), "w") as fh:
        json.dump(jobs, fh, indent=1)
    return jobs


# ================================================================================================ ingest
def _facing(feat: np.ndarray, face_reg: np.ndarray) -> int:
    """+1 = drawing looks toward the viewer's right, −1 = left, 0 = frontal.
    Features centroid vs the head silhouette's bbox centre (robust to the widest-row cx of a turned head)."""
    ys, xs = np.nonzero(feat)
    hy, hx = np.nonzero(face_reg)
    if len(xs) == 0 or len(hx) == 0:
        return 0
    off = (xs.mean() - 0.5 * (hx.min() + hx.max())) / max(1.0, hx.max() - hx.min())
    return 0 if abs(off) < 0.05 else (1 if off > 0 else -1)


def _clean_turned(rgba, feat, face_reg, line_px, facing, view):
    """turned views: drop ears / back-of-head marks and thin jaw-construction strokes near the silhouette.
    q34 keeps both eyes; only the profile drops everything on the back half."""
    n, cl, st, ce = cv2.connectedComponentsWithStats(feat.astype(np.uint8), connectivity=8)
    if n <= 1:
        return rgba, feat
    hy, hx = np.nonzero(face_reg)
    cxh, wh = 0.5 * (hx.min() + hx.max()), max(1.0, hx.max() - hx.min())
    dist = cv2.distanceTransform(face_reg.astype(np.uint8), cv2.DIST_L2, 3)
    big = np.median(st[1:, cv2.CC_STAT_AREA])
    L = cv2.cvtColor(np.ascontiguousarray(rgba[..., :3]), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    drop = np.zeros(feat.shape, bool)
    for j in range(1, n):
        cj = cl == j
        area = st[j, cv2.CC_STAT_AREA]
        rel = facing * (ce[j][0] - cxh) / wh               # >0 = front side, <0 = back side
        near_sil = float(dist[cj].min()) < 2.5 * line_px + 2
        thin = area / max(1.0, np.hypot(st[j, 2], st[j, 3])) < 1.8 * line_px
        pocket = near_sil and float(L[cj].mean()) > 225 and float(dist[cj].min()) < 1.2 * line_px + 2
        if area < 40 or pocket or (near_sil and thin and area < 0.6 * big and rel < 0.25) \
                or (view == "profile" and rel < -0.10):
            drop |= cj
    if drop.any():                                       # include the soft anti-aliased rim of dropped marks
        rim = cv2.dilate(drop.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        keep = cv2.dilate((feat & ~drop).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        rgba[rim & ~keep, 3] = 0
        feat = feat & ~drop
    return rgba, feat


def mirror(rgba: np.ndarray, origin_px):
    """horizontal flip of a decal about its face-unit origin → (rgba, origin_px)."""
    return rgba[:, ::-1].copy(), (rgba.shape[1] - 1 - origin_px[0], origin_px[1])


def ingest(png: str, name: str, out_dir: str, unit_px: float = 360.0, views=VIEWS):
    """turnaround png (3 heads left→right = front, q34, profile) → Expression per view in out_dir/library.json."""
    from .facefind import face_frame_metrics
    os.makedirs(out_dir, exist_ok=True)
    lib = ExpressionLibrary.load(out_dir) if os.path.exists(os.path.join(out_dir, "library.json")) \
        else ExpressionLibrary(out_dir, {})
    im = cv2.imread(png)
    faces, skin, hair, lbl = detect_faces(im, max_frac=0.3)
    faces = sorted(faces, key=lambda f: -f["area"])[:len(views)]
    faces.sort(key=lambda f: f["bbox"][0])
    if len(faces) != len(views):
        raise RuntimeError(f"{png}: found {len(faces)} heads, expected {len(views)}")
    made = []
    for view, fc in zip(views, faces):
        rgba, freg, (ox, oy), line_px, feat = extract_decal(im, lbl, fc, skin, hair)
        facing = _facing(feat, freg) if view != "front" else 0
        if view != "front":
            # the convex hull swallows background pockets at the silhouette (nose bump, chin notch): keep only the
            # real skin region + one stroke, so no white sliver of paper comes along
            from scipy import ndimage as _ndi
            h_, w_ = rgba.shape[:2]
            real = _ndi.binary_fill_holes(lbl[oy:oy + h_, ox:ox + w_] == fc["i"])
            k_ = int(max(3, 1.2 * line_px)) | 1
            real = cv2.dilate(real.astype(np.uint8), np.ones((k_, k_), np.uint8)) > 0
            rgba[~real, 3] = 0
            feat = feat & real
            freg = freg & real
            rgba, feat = _clean_turned(rgba, feat, freg, line_px, facing, view)
        cx, chin_y, top_y, width = face_frame_metrics(freg)
        k = unit_px / max(1.0, width)
        rgba = cv2.resize(rgba, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
        feat = cv2.resize(feat.astype(np.uint8), (rgba.shape[1], rgba.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        org = (cx * k, chin_y * k)
        key = f"{name}@{view}"
        fn = f"{name}__{view}.png"
        cv2.imwrite(os.path.join(out_dir, fn), rgba)

        def to_unit(p, org=org):
            p = np.asarray(p, np.float64)
            return np.stack([(p[:, 0] - org[0]) / unit_px, (org[1] - p[:, 1]) / unit_px], 1)

        e = Expression(name=key, source=os.path.basename(png), cell_bbox=tuple(int(v) for v in fc["bbox"]),
                       px_per_unit=unit_px, origin_px=(float(org[0]), float(org[1])),
                       skin_bgr=tuple(float(v) for v in skin), line_px=float(line_px * k),
                       parts=split_parts(feat, to_unit), decal=fn, view=view, facing=int(facing))
        lib.items[key] = e
        made.append(key)
    lib.save()
    return made


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prepare")
    p.add_argument("--lib", required=True)
    p.add_argument("--sheets", default="tests/fixtures/toonface/sheets")
    p.add_argument("--out", required=True)
    p.add_argument("--names", default="")
    q = sp.add_parser("ingest")
    q.add_argument("--png", required=True)
    q.add_argument("--name", required=True)
    q.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "prepare":
        jobs = prepare(a.lib, a.sheets, a.out, [n for n in a.names.split(",") if n] or None)
        print(f"[redesign] {len(jobs)} jobs → {a.out}/jobs.json")
    else:
        print("[redesign]", ingest(a.png, a.name, a.out))


if __name__ == "__main__":
    main()
