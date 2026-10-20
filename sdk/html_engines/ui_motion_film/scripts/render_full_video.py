#!/usr/bin/env python3
import base64
import concurrent.futures
import os
import shutil
import subprocess
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

OUT_DIR = "/working_dir/c_23bc7a8152bef27e/artifacts/file_generation/ttl=63d/output"
FRAMES_DIR = "/tmp/ui_motion_frames_v2"
HTML_FILE = f"{OUT_DIR}/ui_motion_film.html"
REF_VIDEO = "/working_dir/c_23bc7a8152bef27e/user_uploaded_files/ttl=63d/user_uploaded/video.1377563a.mp4"
MP4_PATH = f"{OUT_DIR}/ui_motion_film_1080p60.mp4"

os.makedirs(FRAMES_DIR, exist_ok=True)


def render_chunk(worker_id, start_frame, end_frame, html_b64):
    opts = Options()
    opts.binary_location = "/usr/bin/chromium"
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--force-device-scale-factor=1")

    driver = webdriver.Chrome(options=opts)
    driver.get("data:text/html;base64," + html_b64)
    driver.execute_script("document.body.classList.add('clean-mode');")

    # Capture preview stills on worker 0 before batch loop
    if worker_id == 0:
        stills = [
            (0.20, f"{OUT_DIR}/still_1_toggle.png"),
            (1.60, f"{OUT_DIR}/still_2_slider.png"),
            (2.52, f"{OUT_DIR}/still_3_notifications.png"),
            (5.02, f"{OUT_DIR}/still_4_keyboard.png"),
            (8.30, f"{OUT_DIR}/still_5_world_return.png"),
        ]
        for t_val, s_path in stills:
            b64_still = driver.execute_script(f"""
                window.seek({t_val});
                return document.getElementById('stage').toDataURL('image/png').split(',')[1];
            """)
            with open(s_path, "wb") as fp:
                fp.write(base64.b64decode(b64_still))

    batch_size = 15
    for f0 in range(start_frame, end_frame, batch_size):
        f1 = min(end_frame, f0 + batch_size)
        b64_list = driver.execute_script(f"""
            const out = [];
            const stage = document.getElementById('stage');
            for (let f = {f0}; f < {f1}; f++) {{
                window.seek(f / 60.0);
                out.push(stage.toDataURL('image/png').split(',')[1]);
            }}
            return out;
        """)
        for idx, b64_str in enumerate(b64_list):
            f_num = f0 + idx
            out_file = f"{FRAMES_DIR}/frame_{f_num:04d}.png"
            with open(out_file, "wb") as fp:
                fp.write(base64.b64decode(b64_str))

    driver.quit()
    return worker_id, end_frame - start_frame


def main():
    t0 = time.time()
    html_b64 = base64.b64encode(open(HTML_FILE, "rb").read()).decode()

    total_frames = 600
    n_workers = 4
    chunk = total_frames // n_workers
    ranges = [(i, i * chunk, (i + 1) * chunk if i < n_workers - 1 else total_frames) for i in range(n_workers)]

    print(f"Rendering {total_frames} frames at 1920x1080 @ 60fps across {n_workers} Chromium workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(render_chunk, wid, s, e, html_b64) for wid, s, e in ranges]
        for fut in concurrent.futures.as_completed(futs):
            wid, cnt = fut.result()
            print(f"  Worker {wid} finished {cnt} frames ({time.time() - t0:.1f}s elapsed)")

    # Mux the 600 rendered 1080p60 frames with the exact reference audio track!
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "60",
        "-i", f"{FRAMES_DIR}/frame_%04d.png",
        "-i", REF_VIDEO,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "15",
        "-pix_fmt", "yuv420p",
        "-r", "60",
        "-c:a", "aac",
        "-b:a", "192k",
        "-t", "10.00",
        "-movflags", "+faststart",
        MP4_PATH
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sz_mb = os.path.getsize(MP4_PATH) / (1024 * 1024)
    print(f"Encoded {MP4_PATH} ({sz_mb:.2f} MB) in {time.time() - t0:.1f}s total!")

    shutil.rmtree(FRAMES_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
