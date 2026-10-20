import sys, os, math, numpy as np, cv2, subprocess, time
sys.path.insert(0, '/home/user/webapp/sdk')
from mograph import real3d as R3, real3d_models as M
from multiprocessing import Pool
W = H = 1080; FPS = 30; DUR = 8.0; N = int(FPS * DUR)
OUT = '/tmp/real3d_frames'; os.makedirs(OUT, exist_ok=True)
BG = sys.argv[1] if len(sys.argv) > 1 else 'green'
def ease(x): return 0.5 - 0.5 * math.cos(math.pi * min(1, max(0, x)))
def poses(t):
    p = t / DUR
    turn = 360 * ease(p)                       # one full, eased 360° turn
    tilt = 22 * math.sin(2 * math.pi * p)      # look from above, then from below
    bob = math.sin(2 * math.pi * p * 2) * 10
    return [  # model, cx, cy, size, yaw, pitch, roll
        ('pizza_slice', 290, 300, 470, -20 + turn, 30 + tilt, 0),
        ('soda_can', 800, 300, 420, 15 - turn, 14 + tilt * 0.8, 14),
        ('growth_card', 700, 800, 430, -25 + turn, 8 + tilt * 0.7, -6),
    ], bob
def bg():
    if BG == 'green':
        im = np.zeros((H, W, 3), np.float32); im[:] = (0.10, 0.85, 0.12)
        cv2.line(im, (0, 520), (W, 520), (0, 0, 0), 4); cv2.line(im, (520, 0), (520, 520), (0, 0, 0), 4)
        return im
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.hypot(xx - W / 2, yy - H * 0.45) / W
    g = np.clip(1 - r * 1.1, 0, 1)[..., None]
    return (np.array([0.05, 0.055, 0.07]) + np.array([0.14, 0.15, 0.18]) * g).astype(np.float32)
BGI = bg()
def frame(i):
    t = i / FPS
    im = BGI.copy()
    objs, bob = poses(t)
    for name, cx, cy, sz, yaw, pitch, roll in objs:
        cy = cy + bob
        rgba, D = R3.render(M.get_model(name), sz, yaw, pitch, roll, ssaa=2)
        # soft floor shadow
        sh = np.zeros((H, W), np.float32)
        cv2.ellipse(sh, (int(cx), int(cy + sz * 0.50)), (int(sz * 0.33), int(sz * 0.06)), 0, 0, 360, 1, -1)
        sh = cv2.GaussianBlur(sh, (0, 0), sz * 0.05)
        im *= (1 - 0.45 * sh[..., None])
        x0, y0 = int(round(cx - D / 2)), int(round(cy - D / 2))
        xs0, ys0 = max(0, -x0), max(0, -y0)
        xs1, ys1 = min(D, W - x0), min(D, H - y0)
        a = rgba[ys0:ys1, xs0:xs1, 3:4]
        c = rgba[ys0:ys1, xs0:xs1, :3][..., ::-1]
        roi = im[y0 + ys0:y0 + ys1, x0 + xs0:x0 + xs1]
        roi[:] = c * a + roi * (1 - a)
    cv2.imwrite(f'{OUT}/{BG}_{i:04d}.png', np.clip(im * 255, 0, 255).astype(np.uint8))
    return i
if __name__ == '__main__':
    t0 = time.time()
    with Pool(2) as p:
        for k in p.imap_unordered(frame, range(N)):
            if k % 30 == 0: print(k, round(time.time() - t0, 1), flush=True)
    out = f'/home/user/webapp/work/renders/demo_real3d_{BG}.mp4'
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-framerate', str(FPS), '-i', f'{OUT}/{BG}_%04d.png',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'slow', out], check=True)
    print(out, round(time.time() - t0, 1))
