#!/usr/bin/env python3
import base64
import os
import numpy as np
from PIL import Image
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

OUT_DIR = "/working_dir/c_23bc7a8152bef27e/artifacts/file_generation/ttl=63d/output"
INT_DIR = "/working_dir/c_23bc7a8152bef27e/artifacts/file_generation/ttl=63d/intermediate"
HTML_FILE = f"{OUT_DIR}/ui_motion_film.html"

html_b64 = base64.b64encode(open(HTML_FILE, "rb").read()).decode()

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

def capture_at(t, out_path):
    b64 = driver.execute_script(f"""
        window.seek({t});
        return document.getElementById('stage').toDataURL('image/png').split(',')[1];
    """)
    raw = base64.b64decode(b64)
    with open(out_path, "wb") as f:
        f.write(raw)
    return Image.open(BytesIO(raw)).convert("RGB")

stills = [
    (0.20, f"{OUT_DIR}/still_1_toggle.png"),
    (1.60, f"{OUT_DIR}/still_2_slider.png"),
    (2.60, f"{OUT_DIR}/still_3_notifications.png"),
    (5.02, f"{OUT_DIR}/still_4_keyboard.png"),
    (8.30, f"{OUT_DIR}/still_5_world_return.png"),
]

for t_val, path in stills:
    img = capture_at(t_val, path)
    print(f"Rendered {os.path.basename(path)} at t={t_val:.2f}s -> size={img.size}")

img_start = capture_at(0.0, f"{INT_DIR}/loop_start_0s.png")
img_end = capture_at(10.0, f"{INT_DIR}/loop_end_10s.png")

arr_start = np.array(img_start, dtype=np.int16)
arr_end = np.array(img_end, dtype=np.int16)
max_diff = np.max(np.abs(arr_start - arr_end))
print(f"Loop boundary check (t=0.000s vs t=10.000s): max pixel diff = {max_diff}")

driver.quit()
