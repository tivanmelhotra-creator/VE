"""
DEV TOOL: validate the whole Shape-Elements reference pack in one go (parallel, per-file timeout).

    python3 tools/batch_validate_aecomp.py /path/to/NM_Library_Shapes_AE -o /tmp/batch.jsonl [--j 4] [--only Arcs]
    python3 tools/batch_validate_aecomp.py --report /tmp/batch.jsonl        # summary per category + worst items

For every Shape_Elements/<cat>/<n>.aep with a Previews/Shape_Elements/<cat>/<n>.mp4 it imports the comp
(tools/import_aep.py), measures IoU + F3 (tools/validate_aecomp.py) and appends one JSON line.
Nothing from the pack is written into the repo — only numbers.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))


def one(aep, mp4, n):
    import signal
    import numpy as np

    def _alarm(*_):
        raise TimeoutError("timeout")
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(240)
    rec = {"id": os.path.relpath(aep).split("Shape_Elements/")[-1][:-4]}
    try:
        import py_aep
        import import_aep
        import validate_aecomp as V
        app = py_aep.parse(aep)
        comp = import_aep.convert(app.project.compositions[0])
        rec["layers"] = len(comp["layers"])
        rec["fx"] = sorted({e for L in comp["layers"] for e in L.get("unsupported", [])})
        ious, reg = V.measure(comp, mp4, n)
        rec.update(iou=round(float(np.mean(ious)) if ious else 0.0, 3), f3=reg["f3"], framing=reg)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    finally:
        signal.alarm(0)
    return rec


def report(path, worst=25):
    import numpy as np
    R = [json.loads(x) for x in open(path) if x.strip()]
    ok = [r for r in R if "iou" in r]
    cats = {}
    for r in ok:
        cats.setdefault(r["id"].split("/")[0], []).append(r)
    print(f"{len(ok)} measured, {len(R) - len(ok)} errors")
    print(f"{'category':24s} {'n':>4s} {'IoU':>6s} {'F3':>6s} {'score':>6s}")
    for c, rs in sorted(cats.items()):
        sc = [max(r["iou"], r["f3"]) for r in rs]
        print(f"{c:24s} {len(rs):4d} {np.median([r['iou'] for r in rs]):6.3f} "
              f"{np.median([r['f3'] for r in rs]):6.3f} {np.median(sc):6.3f}")
    allsc = [max(r["iou"], r["f3"]) for r in ok]
    print(f"{'ALL':24s} {len(ok):4d} {np.median([r['iou'] for r in ok]):6.3f} "
          f"{np.median([r['f3'] for r in ok]):6.3f} {np.median(allsc):6.3f}   "
          f">=0.6: {sum(s >= 0.6 for s in allsc)}  <0.4: {sum(s < 0.4 for s in allsc)}")
    print("\nworst:")
    for r in sorted(ok, key=lambda r: max(r["iou"], r["f3"]))[:worst]:
        print(f"  {r['id']:28s} iou {r['iou']:.3f} f3 {r['f3']:.3f} L{r.get('layers')} {','.join(r.get('fx', []))}")
    errs = [r for r in R if "error" in r]
    if errs:
        print("\nerrors:")
        for r in errs:
            print(f"  {r['id']:28s} {r['error']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?")
    ap.add_argument("-o", "--out", default="/tmp/batch.jsonl")
    ap.add_argument("--j", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--only")
    ap.add_argument("--ids", help="file with one id per line (e.g. Slides/34) — measure only these")
    ap.add_argument("--report")
    a = ap.parse_args(argv)
    if a.report:
        return report(a.report)
    jobs = []
    for aep in sorted(glob.glob(os.path.join(a.root, "Shape_Elements", "*", "*.aep"))):
        cat, n = aep.split(os.sep)[-2], os.path.basename(aep)[:-4]
        if a.only and cat != a.only:
            continue
        if a.ids and f"{cat}/{n}" not in set(open(a.ids).read().split()):
            continue
        mp4 = os.path.join(a.root, "Previews", "Shape_Elements", cat, n + ".mp4")
        if os.path.exists(mp4):
            jobs.append((aep, mp4))
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(x)["id"] for x in open(a.out) if x.strip()}
    jobs = [j for j in jobs if j[0].split("Shape_Elements/")[-1][:-4] not in done]
    print(f"{len(jobs)} to run, {len(done)} cached", flush=True)
    with ProcessPoolExecutor(a.j) as ex, open(a.out, "a") as f:
        futs = [ex.submit(one, aep, mp4, a.n) for aep, mp4 in jobs]
        for i, fu in enumerate(as_completed(futs)):
            r = fu.result()
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 20 == 0:
                print(i, r, flush=True)


if __name__ == "__main__":
    main()
