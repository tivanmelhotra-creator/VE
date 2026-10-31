import sys, cv2, numpy as np, time
sys.path.insert(0, '.')
from mograph.motion.toonface.kit.facefind import find_face, measure_style
from mograph.motion.toonface.kit.facetrack import estimate_bg_color, segment
def frame(n, i):
    cap = cv2.VideoCapture(f'/home/user/uploaded_files/cafe_blankface_gestures_v31_4__{n}_blank_face_gestures.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, i); return cap.read()[1]
def recolor(img, mask, bgr):
    # realistic re-skin: every skin-coloured pixel (face + hands + neck), soft weight on anti-aliased edges
    sk = np.median(img[mask], axis=0).astype(np.float32)
    d = np.linalg.norm(img.astype(np.float32) - sk, axis=2)
    m = np.clip((45.0 - d) / 20.0, 0, 1)[..., None]
    out = img.copy().astype(np.float32)
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None] / 230.0
    out = out * (1 - m) + np.array(bgr, np.float32) * lum * m
    return out.clip(0, 255).astype(np.uint8)
def clutter(img, seed=0):
    rng = np.random.default_rng(seed); H, W = img.shape[:2]
    bg = np.full_like(img, 0)
    for _ in range(140):   # busy cafe-like shapes: shelves, cups, windows, lines
        c = tuple(int(v) for v in rng.integers(30, 255, 3)); x, y = rng.integers(0, W), rng.integers(0, H)
        k = rng.integers(0, 3)
        if k == 0: cv2.rectangle(bg, (x, y), (x + rng.integers(20, 200), y + rng.integers(20, 200)), c, -1)
        elif k == 1: cv2.circle(bg, (x, y), int(rng.integers(10, 80)), c, -1)
        else: cv2.line(bg, (x, y), (int(rng.integers(0, W)), int(rng.integers(0, H))), (20, 20, 20), int(rng.integers(2, 6)))
    bgc = estimate_bg_color(img)
    isbg = np.linalg.norm(img.astype(np.float32) - bgc, axis=2) < 30
    n, l = cv2.connectedComponents(isbg.astype(np.uint8), connectivity=4)
    border = np.unique(np.r_[l[0], l[:, 0], l[:, -1]]); isbg = np.isin(l, border[border > 0])
    return np.where(isbg[..., None], bg, img)
rows = []
for n in ['boy', 'girl']:
    for fi in (0, 60, 100):
        base = frame(n, fi)
        ref = find_face(base)
        cases = {'orig': base,
                 'brown': recolor(base, ref.mask | False, (60, 95, 150)),
                 'green': recolor(base, ref.mask, (90, 170, 100)),
                 'clutter': clutter(base, fi),
                 'brown+clutter': clutter(recolor(base, ref.mask, (60, 95, 150)), fi + 1)}
        for k, img in cases.items():
            t = time.time()
            try:
                f = find_face(img); iou = (f.mask & ref.mask).sum() / max(1, (f.mask | ref.mask).sum())
                st = measure_style(img, f.mask)
                rows.append((n, fi, k, round(iou, 3), round(st.line_px, 1), st.skin_bgr.astype(int).tolist(), round(time.time() - t, 2)))
            except Exception as e:
                rows.append((n, fi, k, 'FAIL', str(e)[:60]))
            if fi == 60: cv2.imwrite(f'/tmp/ff/{n}_{k}.jpg', cv2.resize(img, (180, 320)))
for r in rows: print(*r)
