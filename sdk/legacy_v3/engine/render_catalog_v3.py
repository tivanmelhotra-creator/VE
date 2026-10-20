import sys
sys.path.insert(0, "/working_dir/c_4fc7259f35079930/artifacts/file_generation/ttl=63d/output")
import cairo
import cv2
import numpy as np
from PIL import Image
import vector_motion_library as vml

W, H = 1240, 1180
surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
ctx = cairo.Context(surf)
ctx.set_source_rgb(0, 0, 0)
ctx.paint()

vml.draw_pango_text(ctx, "نسخه V3: آیکون‌ها و ویجت‌های چندلایه با لبه‌های نرم (Rounded Bezier + Multi-Layer Detail)",
                    W * 0.5, 36, size=22, color=(1, 1, 1, 1), bold=True, glow=True)

# 1. Left 64% (`x = 25..785`): Grid of 35 Soft-Edged Multi-Layered Icons (7 cols x 5 rows)
icon_keys = list(vml.VECTOR_ICONS.keys())
for idx, k in enumerate(icon_keys[:35]):
    r = idx // 7
    c = idx % 7
    cx = 75 + c * 108
    cy = 132 + r * 200
    # Subtle soft-rounded card behind each icon so the grid looks like a premium design system
    vml.draw_rounded_rect(ctx, cx, cy, 88, 88, r=20.0, progress=1.0,
                          color=(0.18, 0.18, 0.20, 0.85), stroke_w=1.2, fill_black=True)
    vml.draw_icon(ctx, k, cx, cy, scale=0.76, progress=1.0, color=(1, 1, 1, 1), stroke_w=2.3, show_details=True)
    vml.draw_pango_text(ctx, k, cx, cy + 62, size=12, color=(0.72, 0.72, 0.75, 1.0), bold=True)

# Vertical divider line
ctx.set_source_rgba(0.28, 0.28, 0.30, 1.0)
ctx.set_line_width(1.5)
ctx.move_to(815, 75)
ctx.line_to(815, H - 30)
ctx.stroke()

# 2. Right 36% (`x = 830..1220`): Soft-Edged High-Level Widgets
vml.draw_pango_text(ctx, "ویجت‌های چندلایه با گوشه‌های گرد (V3):", 1025, 88, size=18, color=(1, 1, 1, 1), bold=True, glow=True)

# Widget A: Soft-Cornered Legal Tablet Laser Scanner (`rx=22px`)
vml.widget_laser_scanner(ctx, 1025, 250, 310, 235, "قانون اساسی", "اصل ۴۵: انفال و ثروت‌های عمومی", scan_prog=0.62, open_prog=1.0)

# Widget B: Formula Stadium Pill (`widget_formula_pill`)
vml.widget_formula_pill(ctx, 1025, 435, "همه می‌دونن", "+ هر ادعایی ...", open_prog=1.0, lock_morph=0.75)

# Widget C: Live Strikethrough & Soft Double-Bordered Stamp
vml.widget_live_strikethrough(ctx, 1025, 615, "مالِ دولت", "در اختیار مردم", strike_prog=1.0, true_prog=1.0, size=23)
vml.widget_stamp(ctx, 1025, 695, "مستند", t=2.0, t0=1.0, angle_deg=-6.0, w=165, h=52)

# Widget D: Soft-Edged Everyday Micro-Story Dashboard
vml.widget_everyday_story_hud(ctx, 1025, 920, prog_all=1.0, t=2.0)

# Widget E: Callout with Double-Ring Anchor & Auto-Filleted Elbow
vml.widget_callout(ctx, (920, 1115), (965, 1085), 140, "لیبل با گوشه نرم", progress=1.0, color=(1, 1, 1, 1), size=17, glow=True)

buf = np.frombuffer(surf.get_data(), dtype=np.uint8).reshape((H, W, 4))
rgb = cv2.cvtColor(buf, cv2.COLOR_BGRA2RGB)
bright = np.where(rgb > 205, rgb, 0)
blur = cv2.GaussianBlur(bright, (15, 15), 4.5)
out = np.clip(rgb.astype(np.uint16) + (blur.astype(np.uint16) * 45) // 100, 0, 255).astype(np.uint8)
out_path = "/working_dir/c_4fc7259f35079930/artifacts/file_generation/ttl=63d/output/workspace_v3_soft_complex_catalog.jpg"
Image.fromarray(out).save(out_path, quality=95)
print("Saved", out_path)
