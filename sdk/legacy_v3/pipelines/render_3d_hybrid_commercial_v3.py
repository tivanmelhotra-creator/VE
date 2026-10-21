import os
import math
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 900, 1536
FPS = 30.0

BASE_DIR = "/working_dir/c_4fc7259f35079930/extracted_motion_v2"
VIDEO_IN = "/working_dir/c_4fc7259f35079930/user_uploaded_files/ttl=63d/user_uploaded/ویدیو ای که باید ادیت بشه.mp4"
OUT_DIR = "/working_dir/c_4fc7259f35079930/artifacts/file_generation/ttl=63d/output"
os.makedirs(OUT_DIR, exist_ok=True)
# Unique filename so the frontend player never serves an old cached blob!
VIDEO_OUT = os.path.join(OUT_DIR, "ayyildiz_motion_final_v3.mp4")

FONT_EN_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_FA_BOLD = "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf"

# ============================================================================
# 1. PHYSICS & EASING ENGINE
# ============================================================================
def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

def ease_out_cubic(t):
    t = clamp(t)
    return 1.0 - (1.0 - t) ** 3

def ease_in_out_cubic(t):
    t = clamp(t)
    return 4.0 * t * t * t if t < 0.5 else 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0

def spring(t, start, dur=0.56, freq=6.2, decay=5.2):
    if t <= start:
        return 0.0
    p = (t - start) / max(dur, 1e-4)
    if p >= 2.5:
        return 1.0
    return 1.0 - math.exp(-decay * p) * math.cos(freq * p)

def fade_in(t, start, dur=0.20):
    return ease_out_cubic((t - start) / max(dur, 1e-4))


# ============================================================================
# 2. CLEAN GREEN-SCREEN EXTRACTION & SCENE 3 CHECKMARK FIX
# ============================================================================
def extract_bg_panel(layers_path):
    img = cv2.imread(layers_path)
    h, w, _ = img.shape
    gray = img.mean(axis=(0, 2))
    div1 = np.where((np.arange(w) > 850) & (np.arange(w) < 950) & (gray < 35))[0]
    div2 = np.where((np.arange(w) > 1800) & (np.arange(w) < 1900) & (gray < 35))[0]
    p2_start = div1[-1] + 2
    p2_end = div2[0] - 2
    p2 = cv2.resize(img[:, p2_start:p2_end], (W, H), interpolation=cv2.INTER_LANCZOS4)
    return cv2.cvtColor(p2, cv2.COLOR_BGR2RGB)

def key_green_crop(bgr_crop, mode="default"):
    b = bgr_crop[:, :, 0].astype(np.float32)
    g = bgr_crop[:, :, 1].astype(np.float32)
    r = bgr_crop[:, :, 2].astype(np.float32)
    
    g_cap = 0.35 * r + 0.65 * b
    if mode == "pizza":
        g_cap = np.where((r > 100) & (b > 40), 0.62 * r + 0.38 * b, g_cap)
    elif mode == "cards_s4":
        g_cap = np.where((r > 70) & (b > 65), g * 0.96, g_cap)
    elif mode == "trails_s4":
        trail_strength = np.clip((r + b * 0.5) / 170.0, 0.0, 1.0)
        a_u8 = (trail_strength * 255.0).astype(np.uint8)
        r_u8 = np.full_like(a_u8, 255)
        g_u8 = np.clip(145 + trail_strength * 95, 0, 255).astype(np.uint8)
        b_u8 = np.clip(65 + trail_strength * 110, 0, 255).astype(np.uint8)
        return np.dstack([r_u8, g_u8, b_u8, a_u8])

    green_excess = np.maximum(0.0, g - g_cap)
    bg_trans = np.clip((green_excess - 8.0) / 220.0, 0.0, 1.0)
    max_rb = np.maximum(r, b)
    bg_trans = np.where((bg_trans > 0.88) & (max_rb < 28), 1.0, bg_trans)
    
    alpha = 1.0 - bg_trans
    alpha = np.where(alpha < 0.02, 0.0, alpha)
    
    r_pre = r * np.where(alpha > 0, 1.0, 0.0)
    b_pre = b * np.where(alpha > 0, 1.0, 0.0)
    g_pre = np.minimum(g, g_cap) * np.where(alpha > 0, 1.0, 0.0)
    
    safe_a = np.maximum(alpha, 1e-4)
    r_str = np.clip(r_pre / safe_a, 0, 255).astype(np.uint8)
    g_str = np.clip(g_pre / safe_a, 0, 255).astype(np.uint8)
    b_str = np.clip(b_pre / safe_a, 0, 255).astype(np.uint8)
    a_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return np.dstack([r_str, g_str, b_str, a_u8])

def tight_rgba(bgr_img, x1, y1, x2, y2, mode="default", poly_pts=None):
    crop = bgr_img[y1:y2, x1:x2].copy()
    rgba = key_green_crop(crop, mode=mode)
    if poly_pts is not None:
        m = np.zeros(rgba.shape[:2], dtype=np.uint8)
        pts_local = np.array([(px - x1, py - y1) for px, py in poly_pts], dtype=np.int32)
        cv2.fillPoly(m, [pts_local], 255)
        rgba[:, :, 3] = np.minimum(rgba[:, :, 3], m)
    # Zero out outer 3px boundary to guarantee no white box border ever leaks in
    rgba[:3, :, 3] = 0
    rgba[-3:, :, 3] = 0
    rgba[:, :3, 3] = 0
    rgba[:, -3:, 3] = 0

    ys, xs = np.where(rgba[:, :, 3] > 15)
    if len(xs) == 0:
        return rgba
    tx0, ty0 = max(0, xs.min() - 4), max(0, ys.min() - 4)
    tx1, ty1 = min(rgba.shape[1], xs.max() + 5), min(rgba.shape[0], ys.max() + 5)
    return rgba[ty0:ty1, tx0:tx1].copy()

def fix_scene3_toggle_checkmark(im3_bgr):
    """
    Fixes the backwards/upside-down white checkmark (`_|`) inside the orange toggle knob
    on the Left Half of `layer (3).jpg` (`x=680..860, y=1160..1290`), replacing it with
    a crisp, properly oriented 3D upright white checkmark (`✓`).
    """
    im3_fixed = im3_bgr.copy()
    sub = im3_fixed[1160:1290, 680:860].copy()
    b, g, r = sub[:, :, 0].astype(int), sub[:, :, 1].astype(int), sub[:, :, 2].astype(int)
    oys, oxs = np.where((r - b) > 85)
    kcx, kcy = int(oxs.mean()), int(oys.mean())

    yy, xx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    in_knob_inner = ((xx - kcx) ** 2 + (yy - kcy) ** 2) <= (27 ** 2)
    white_check = (in_knob_inner & (b > 130) & (g > 130)).astype(np.uint8) * 255
    white_check = cv2.dilate(white_check, np.ones((7, 7), np.uint8))

    sub_clean = cv2.inpaint(sub, white_check, 6, cv2.INPAINT_TELEA)

    pil_sub = Image.fromarray(cv2.cvtColor(sub_clean, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_sub, "RGBA")
    # Subtle warm dark-orange contact shadow under the white checkmark
    pts_shadow = [(kcx - 13, kcy + 1), (kcx - 3, kcy + 11), (kcx + 15, kcy - 7)]
    draw.line(pts_shadow, fill=(150, 46, 20, 175), width=6, joint="curve")
    # Crisp 3D white upright checkmark `✓`
    pts_check = [(kcx - 13, kcy - 1), (kcx - 3, kcy + 9), (kcx + 15, kcy - 9)]
    draw.line(pts_check, fill=(255, 255, 255, 252), width=6, joint="curve")

    im3_fixed[1160:1290, 680:860] = cv2.cvtColor(np.array(pil_sub), cv2.COLOR_RGB2BGR)
    return im3_fixed


# ============================================================================
# 3. COMPOSITING & PADDED SHADOW ENGINE (ZERO SHADOW CLIPPING LINES!)
# ============================================================================
def place_rgba(
    canvas_f32, sprite_rgba, cx, cy,
    target_w=None, target_h=None,
    scale_x=1.0, scale_y=1.0,
    angle_deg=0.0, opacity=1.0,
    add_shadow=False, shadow_offset=(0, 16), shadow_alpha=0.22,
    sheen_pos=None, sheen_strength=0.46,
    clip_rx=None, clip_ry=None,
    glow_color=None, glow_strength=0.0
):
    sh, sw = sprite_rgba.shape[:2]
    if target_w is not None and target_h is None:
        s_base = target_w / float(sw)
        nw = int(round(sw * s_base * scale_x))
        nh = int(round(sh * s_base * scale_y))
    elif target_h is not None and target_w is None:
        s_base = target_h / float(sh)
        nw = int(round(sw * s_base * scale_x))
        nh = int(round(sh * s_base * scale_y))
    elif target_w is not None and target_h is not None:
        nw = int(round(target_w * scale_x))
        nh = int(round(target_h * scale_y))
    else:
        nw = int(round(sw * scale_x))
        nh = int(round(sh * scale_y))
    if nw <= 2 or nh <= 2 or opacity <= 0.005:
        return

    sf = sprite_rgba.astype(np.float32)
    sa = sf[:, :, 3:4] / 255.0
    rgb = sf[:, :, :3]

    # Specular light sweep across object
    if sheen_pos is not None and -0.35 <= sheen_pos <= 1.35:
        yy, xx = np.mgrid[0:sh, 0:sw]
        diag = (xx / max(1.0, float(sw))) * 0.78 + (yy / max(1.0, float(sh))) * 0.22
        band = np.exp(-((diag - sheen_pos) ** 2) / (2.0 * (0.085 ** 2)))[:, :, None]
        rgb = np.clip(rgb + band * 255.0 * sheen_strength, 0.0, 255.0)

    # Progressive clip reveal: clip_rx / clip_ry is the VISIBLE fraction (0.0 = hidden, 1.0 = fully visible)
    if clip_rx is not None and clip_rx < 0.999:
        hide_x = int(round(sw * (1.0 - clamp(clip_rx))))
        if hide_x > 0:
            fade_w = min(18, hide_x)
            sa[:, :hide_x - fade_w] = 0.0
            if fade_w > 0:
                sa[:, hide_x - fade_w:hide_x, 0] *= np.linspace(0.0, 1.0, fade_w)[None, :]
    if clip_ry is not None and clip_ry < 0.999:
        if clip_ry >= 0.0:
            hide_y = int(round(sh * (1.0 - clamp(clip_ry))))
            if hide_y > 0:
                fade_h = min(24, hide_y)
                sa[:hide_y - fade_h, :] = 0.0
                if fade_h > 0:
                    sa[hide_y - fade_h:hide_y, :, 0] *= np.linspace(0.0, 1.0, fade_h)[:, None]
        else:
            vis_p = clamp(-clip_ry)
            keep_y = int(round(sh * vis_p))
            if keep_y < sh:
                fade_h = min(24, sh - keep_y)
                sa[keep_y + fade_h:, :] = 0.0
                if fade_h > 0:
                    sa[keep_y:keep_y + fade_h, :, 0] *= np.linspace(1.0, 0.0, fade_h)[:, None]

    pre = np.dstack([rgb * sa, sa * 255.0])
    resized = cv2.resize(pre, (nw, nh), interpolation=cv2.INTER_AREA if nw < sw else cv2.INTER_LINEAR)

    # ALWAYS pad by 42px of transparent border so GaussianBlur for shadows/glows NEVER clips at the bounding box edge!
    pad_b = 42
    resized = cv2.copyMakeBorder(resized, pad_b, pad_b, pad_b, pad_b, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    nh, nw = resized.shape[:2]

    if abs(angle_deg) > 0.05:
        pad = int(math.hypot(nw, nh) * 0.45)
        padded = cv2.copyMakeBorder(resized, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
        ph, pw = padded.shape[:2]
        M = cv2.getRotationMatrix2D((pw * 0.5, ph * 0.5), angle_deg, 1.0)
        resized = cv2.warpAffine(padded, M, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        nh, nw = resized.shape[:2]

    x0 = int(round(cx - nw * 0.5))
    y0 = int(round(cy - nh * 0.5))

    # Optional colored outer glow
    if glow_color is not None and glow_strength > 0.01:
        gl_a = cv2.GaussianBlur(resized[:, :, 3], (35, 35), 12.0) * glow_strength * opacity
        gx0, gy0 = max(0, x0), max(0, y0)
        gx1, gy1 = min(W, x0 + nw), min(H, y0 + nh)
        if gx1 > gx0 and gy1 > gy0:
            sub_ga = (gl_a[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0, None] / 255.0)
            gcol = np.array(glow_color, dtype=np.float32)[None, None, :]
            canvas_f32[gy0:gy1, gx0:gx1] = np.clip(canvas_f32[gy0:gy1, gx0:gx1] * (1.0 - sub_ga * 0.5) + gcol * sub_ga, 0, 255)

    # Optional ambient ground contact shadow (only for grounded objects, with 42px padding so zero flat line!)
    if add_shadow:
        sh_a = cv2.GaussianBlur(resized[:, :, 3], (45, 45), 15.0) * shadow_alpha * opacity
        sx0, sy0 = x0 + int(shadow_offset[0]), y0 + int(shadow_offset[1])
        cx0, cy0 = max(0, sx0), max(0, sy0)
        cx1, cy1 = min(W, sx0 + nw), min(H, sy0 + nh)
        if cx1 > cx0 and cy1 > cy0:
            sub_sa = (sh_a[cy0 - sy0:cy1 - sy0, cx0 - sx0:cx1 - sx0, None] / 255.0)
            canvas_f32[cy0:cy1, cx0:cx1] = canvas_f32[cy0:cy1, cx0:cx1] * (1.0 - sub_sa) + 28.0 * sub_sa

    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x0 + nw), min(H, y0 + nh)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sub = resized[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    wa = (sub[:, :, 3:4] / 255.0) * float(opacity)
    w_rgb = sub[:, :, :3] * float(opacity)
    canvas_f32[cy0:cy1, cx0:cx1] = canvas_f32[cy0:cy1, cx0:cx1] * (1.0 - wa) + w_rgb


# ============================================================================
# 4. SPEAKER VIDEO LOADER
# ============================================================================
class SpeakerVideoReader:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            # Clean upper studio crop (x=25..535, y=25..815) and remove thin white @AYYILDIZ text at right edge
            crop = cv2.cvtColor(frame[25:815, 25:535], cv2.COLOR_BGR2RGB)
            sub_wm = crop[250:390, 420:510]
            wm_mask = ((sub_wm[:, :, 0] > 180) & (sub_wm[:, :, 1] > 180) & (sub_wm[:, :, 2] > 180)).astype(np.uint8) * 255
            if wm_mask.sum() > 0:
                wm_mask = cv2.dilate(wm_mask, np.ones((5, 5), np.uint8))
                crop[250:390, 420:510] = cv2.inpaint(sub_wm, wm_mask, 4, cv2.INPAINT_TELEA)
            cf = np.clip((crop.astype(np.float32) - 125.0) * 1.07 + 130.0, 0, 255).astype(np.uint8)
            self.frames.append(cf)
        self.cap.release()
        print(f"Preloaded {len(self.frames)} clean speaker frames.")

    def get_frame(self, idx):
        if not self.frames:
            return np.zeros((790, 510, 3), dtype=np.uint8)
        i = min(max(0, idx), len(self.frames) - 1)
        return self.frames[i]


# ============================================================================
# 5. LOAD ALL PRE-SEPARATED V2 SCENE ASSETS (WITH CLEAN CROPS & FIXED CHECKMARK)
# ============================================================================
def load_v2_assets():
    assets = {}

    # SCENE 1 (Note: receipt cropped at y2=1330, well above the y=1360 white box line!)
    im1 = cv2.imread(f"{BASE_DIR}/layer (1).jpg")
    assets[1] = {
        "bg": extract_bg_panel(f"{BASE_DIR}/1-layers.jpg"),
        "hl": tight_rgba(im1, 50, 240, 900, 405),
        "pill_bg": tight_rgba(im1, 110, 420, 890, 595),
        "pill_txt": tight_rgba(im1, 140, 635, 860, 780),
        "pizza": tight_rgba(im1, 1085, 62, 1595, 610, mode="pizza"),
        "soda": tight_rgba(im1, 1652, 62, 1980, 610),
        "receipt": tight_rgba(im1, 1085, 672, 1595, 1330),
        "arrow": tight_rgba(im1, 1085, 1425, 1595, 1980),
        "conn": tight_rgba(im1, 1652, 1425, 1980, 1980),
    }

    # SCENE 2
    im2 = cv2.imread(f"{BASE_DIR}/layer (2).jpg")
    assets[2] = {
        "bg": extract_bg_panel(f"{BASE_DIR}/2-layers.jpg"),
        "hl": tight_rgba(im2, 45, 240, 915, 395),
        "pill_bg": tight_rgba(im2, 250, 405, 760, 565),
        "pill_txt": tight_rgba(im2, 300, 585, 715, 715),
        "badge_bg": tight_rgba(im2, 370, 780, 650, 1055),
        "badge_txt": tight_rgba(im2, 400, 1085, 615, 1195),
        "hoodie": tight_rgba(im2, 1085, 62, 1980, 980),
        "swatches": tight_rgba(im2, 1085, 1040, 1980, 1980),
    }

    # SCENE 3 (Fix the backwards checkmark on `im3` before extracting `hero_all`!)
    im3 = fix_scene3_toggle_checkmark(cv2.imread(f"{BASE_DIR}/layer (3).jpg"))
    assets[3] = {
        "bg": extract_bg_panel(f"{BASE_DIR}/3-layers.jpg"),
        "hl": tight_rgba(im3, 45, 290, 915, 405),
        "pill_bg": tight_rgba(im3, 170, 445, 850, 580),
        "pill_txt": tight_rgba(im3, 210, 600, 810, 680),
        "hero_all": tight_rgba(im3, 5, 765, 1008, 1855),
    }

    # SCENE 4
    im4 = cv2.imread(f"{BASE_DIR}/layer (4).jpg")
    assets[4] = {
        "bg": extract_bg_panel(f"{BASE_DIR}/4-layers.jpg"),
        "hl": tight_rgba(im4, 40, 185, 935, 345),
        "pill_bg": tight_rgba(im4, 170, 350, 855, 455),
        "pill_txt": tight_rgba(im4, 195, 470, 820, 550),
        "word": tight_rgba(im4, 340, 1680, 670, 1875),
        "box": tight_rgba(im4, 1042, 1595, 2005, 2005),
        "cards": tight_rgba(
            im4, 1040, 30, 2035, 1235, mode="cards_s4",
            poly_pts=[(1040, 30), (2035, 30), (2035, 730), (1515, 730), (1515, 1235), (1040, 1235)]
        ),
        "plane": tight_rgba(im4, 1555, 770, 2030, 1290),
        "trails": tight_rgba(im4, 1555, 1335, 2030, 1535, mode="trails_s4"),
    }

    # SCENE 5 (Logo cropped at y2=750, Pedestal cropped strictly inside white borders)
    im5 = cv2.imread(f"{BASE_DIR}/layer (5).jpg")
    assets[5] = {
        "bg": extract_bg_panel(f"{BASE_DIR}/5-layers.jpg"),
        "hl": tight_rgba(im5, 35, 220, 930, 375),
        "pill_bg": tight_rgba(im5, 105, 1645, 900, 1845),
        "pill_txt": tight_rgba(im5, 135, 1870, 865, 1975),
        "logo": tight_rgba(im5, 1085, 60, 1990, 750),
        "vbadge": tight_rgba(im5, 1728, 825, 2022, 1215),
        "ped": tight_rgba(
            im5, 1085, 1095, 1990, 1990,
            poly_pts=[(1085, 1095), (1695, 1095), (1695, 1255), (1990, 1255), (1990, 1990), (1085, 1990)]
        ),
    }
    return assets


# ============================================================================
# 6. SPEAKER CUT OVERLAY (ZERO SPOILERS & ZERO DUPLICATE HEADLINES!)
# ============================================================================
def render_speaker_cut_overlay(canvas_f32, spk_frame, s_idx, st, cut_dur, intro_phrase):
    """
    During the opening Speaker Cut (`st < cut_dur`), shows the Woman Speaking
    with ONLY the exact spoken intro phrase (e.g. «دیگه الان...» or «ولی من...»)
    so we NEVER spoil the upcoming motion scene's headline/price and NEVER
    repeat the motion scene's text twice!
    """
    if cut_dur <= 0.01:
        return

    trans_window = 0.24
    if st < cut_dur - trans_window:
        tr = 0.0
    elif st < cut_dur:
        tr = ease_in_out_cubic((st - (cut_dur - trans_window)) / trans_window)
    else:
        tr = 1.0

    # 1. Full-Screen Studio Presenter Cut (`tr < 1.0`)
    if tr < 0.999:
        spk_opacity = 1.0 - tr
        zoom = 1.0 + 0.04 * (st / max(cut_dur, 0.4)) + 0.12 * tr
        target_h = 1285
        zw = int(round(W * zoom))
        zh = int(round(target_h * zoom))
        resized_spk = cv2.resize(spk_frame, (zw, zh), interpolation=cv2.INTER_LINEAR)
        ox = (zw - W) // 2
        oy = int((zh - target_h) * 0.35)
        cropped_spk = resized_spk[oy:oy + target_h, ox:ox + W].astype(np.float32)

        slide_y = int(round(-tr * 240.0))
        vis_h = max(0, min(H, target_h + slide_y))
        if vis_h > 0:
            src_slice = cropped_spk[target_h - vis_h:target_h]
            alpha_grad = np.ones((vis_h, W, 1), dtype=np.float32) * spk_opacity
            feather = min(210, vis_h)
            if feather > 10:
                ramp = np.linspace(1.0, 0.0, feather)[:, None, None] ** 1.5
                alpha_grad[vis_h - feather:vis_h] *= ramp
            canvas_f32[:vis_h] = src_slice * alpha_grad + canvas_f32[:vis_h] * (1.0 - alpha_grad)

        # Sleek compact studio caption card showing ONLY the spoken intro phrase (e.g. «دیگه الان...»)
        if intro_phrase:
            sp_c = spring(st, 0.02, dur=0.40)
            card_dy = int(round(tr * 260.0 + (1.0 - sp_c) * 90.0))
            pil_ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            pdraw = ImageDraw.Draw(pil_ov, "RGBA")
            c_alpha = int(235 * spk_opacity)
            cy1, cy2 = 1210 + card_dy, 1385 + card_dy
            if cy1 < H:
                # Soft drop shadow
                pdraw.rounded_rectangle([145, cy1 + 10, W - 145, cy2 + 10], radius=42, fill=(25, 22, 20, int(42 * spk_opacity)))
                # Frosted pill card
                pdraw.rounded_rectangle([140, cy1, W - 140, cy2], radius=42, fill=(246, 244, 240, c_alpha),
                                        outline=(255, 255, 255, int(245 * spk_opacity)), width=3)
                # Accent terracotta top indicator bar
                pdraw.rounded_rectangle([380, cy1, 520, cy1 + 8], radius=4, fill=(217, 83, 47, int(255 * spk_opacity)))
                # Render the spoken intro phrase in bold Persian typography
                f_fa = ImageFont.truetype(FONT_FA_BOLD, 66)
                bb = pdraw.textbbox((0, 0), intro_phrase, font=f_fa, direction="rtl", language="fa")
                tw, th = bb[2] - bb[0], bb[3] - bb[1]
                tx = (W - tw) * 0.5 - bb[0]
                ty = (cy1 + cy2 - th) * 0.5 - bb[1] - 4
                pdraw.text((tx, ty), intro_phrase, font=f_fa, fill=(32, 32, 35, int(255 * spk_opacity)),
                           direction="rtl", language="fa")

            ov_arr = np.array(pil_ov, dtype=np.float32)
            oa = ov_arr[:, :, 3:4] / 255.0
            canvas_f32[:] = canvas_f32 * (1.0 - oa) + ov_arr[:, :, :3] * oa

    # 2. Floating Circular Speaker PIP Badge in top-left during the Motion Scene (`tr > 0.15`)
    if s_idx in [1, 2, 3, 4] and tr > 0.15:
        pip_op = clamp((tr - 0.15) / 0.85)
        pip_r = 58
        pip_cx, pip_cy = 122, 118
        face_crop = cv2.resize(spk_frame[115:395, 130:410], (pip_r * 2, pip_r * 2), interpolation=cv2.INTER_AREA)

        pil_pip = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        pdraw = ImageDraw.Draw(pil_pip, "RGBA")
        pulse_r = pip_r + 5 + int(4 * abs(math.sin(st * 9.5)))
        pdraw.ellipse([pip_cx - pulse_r, pip_cy - pulse_r, pip_cx + pulse_r, pip_cy + pulse_r],
                      outline=(217, 83, 47, int(145 * pip_op)), width=3)
        pdraw.ellipse([pip_cx - pip_r, pip_cy - pip_r + 5, pip_cx + pip_r, pip_cy + pip_r + 5],
                      fill=(25, 22, 20, int(50 * pip_op)))
        pdraw.ellipse([pip_cx - pip_r - 3, pip_cy - pip_r - 3, pip_cx + pip_r + 3, pip_cy + pip_r + 3],
                      fill=(255, 255, 255, int(245 * pip_op)))

        pip_arr = np.array(pil_pip, dtype=np.float32)
        pa = pip_arr[:, :, 3:4] / 255.0
        canvas_f32[:] = canvas_f32 * (1.0 - pa) + pip_arr[:, :, :3] * pa

        yy, xx = np.mgrid[0:pip_r * 2, 0:pip_r * 2]
        dist = np.sqrt((xx - pip_r) ** 2 + (yy - pip_r) ** 2)
        c_mask = np.clip(pip_r - dist, 0.0, 1.0)[:, :, None] * pip_op
        x0, y0 = pip_cx - pip_r, pip_cy - pip_r
        canvas_f32[y0:y0 + pip_r * 2, x0:x0 + pip_r * 2] = (
            canvas_f32[y0:y0 + pip_r * 2, x0:x0 + pip_r * 2] * (1.0 - c_mask) +
            face_crop.astype(np.float32) * c_mask
        )


# ============================================================================
# 7. AMBIENT FLOOR RIPPLE, HUD BRACKETS & SPARKLE PARTICLES
# ============================================================================
def apply_floor_ripple(bg_rgb, st, trigger_t=0.22, cx=450, cy=1140):
    dt = st - trigger_t
    if dt <= 0.0 or dt >= 1.45:
        return bg_rgb.astype(np.float32)
    out = bg_rgb.astype(np.float32)
    radius = dt * 490.0
    amp = 7.0 * (1.0 - dt / 1.45) ** 1.6
    yy, xx = np.mgrid[700:H, 0:W]
    dx = xx - cx
    dy = (yy - cy) * 1.85
    dist = np.sqrt(dx * dx + dy * dy)
    band = np.exp(-((dist - radius) ** 2) / (2.0 * (40.0 ** 2)))
    wave = np.sin((dist - radius) * 0.14) * amp * band
    map_x = xx.astype(np.float32)
    map_y = (yy + wave).astype(np.float32)
    warped = cv2.remap(out, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    glow = (band * (1.0 - dt / 1.45))[:, :, None]
    accent = np.array([217.0, 83.0, 47.0], dtype=np.float32)[None, None, :]
    out[700:H] = np.clip(warped * (1.0 - glow * 0.15) + accent * (glow * 0.15), 0, 255)
    return out

def draw_hud_and_particles(canvas_f32, st, global_t, accent_rgb=(217, 83, 47)):
    pil_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(pil_img, "RGBA")

    sp = spring(st, 0.0, dur=0.42, freq=6.0, decay=5.5)
    inset = int(round(48 + (1.0 - sp) * 28))
    arm = 72
    col = (35, 35, 38, int(clamp(st / 0.18) * 225))
    for (x1, y1, x2, y2, x3, y3) in [
        (inset, inset + arm, inset, inset, inset + arm, inset),
        (W - inset - arm, inset, W - inset, inset, W - inset, inset + arm),
        (inset, H - inset - arm, inset, H - inset, inset + arm, H - inset),
        (W - inset - arm, H - inset, W - inset, H - inset, W - inset, H - inset - arm),
    ]:
        draw.line([(x1, y1), (x2, y2), (x3, y3)], fill=col, width=3)
        draw.ellipse([x2 - 4, y2 - 4, x2 + 4, y2 + 4], fill=(*accent_rgb, col[3]))

    for i in range(8):
        px = (140 + i * 95 + int(math.sin(global_t * 1.4 + i * 1.7) * 28)) % (W - 120) + 60
        py = int(1150 - ((global_t * (55 + i * 7) + i * 130) % 720))
        pr = 3 + (i % 3)
        pa = int(50 * math.sin(((1150 - py) / 720.0) * math.pi))
        draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=(*accent_rgb, pa))

    prog = clamp(global_t / 16.57)
    bar_y = H - 18
    draw.line([(inset, bar_y), (W - inset, bar_y)], fill=(200, 196, 190, 110), width=4)
    draw.line([(inset, bar_y), (inset + int((W - 2 * inset) * prog), bar_y)], fill=(*accent_rgb, 235), width=4)

    ov = np.array(pil_img, dtype=np.float32)
    oa = ov[:, :, 3:4] / 255.0
    canvas_f32[:] = canvas_f32 * (1.0 - oa) + ov[:, :, :3] * oa


# ============================================================================
# 8. SPEECH-LOCKED TIMELINE & SCENE RENDERER (ZERO SPOILERS, ZERO BUGS)
# ============================================================================
# Exact speech-locked timeline verified from `/tmp/speech_timeline_grid.jpg`:
# (scene_idx, t_start, t_end, speaker_cut_dur, speaker_intro_phrase)
SCENE_CONFIG = [
    # Scene 1: 0.00-0.90s woman says «دیگه الان...», 0.90-2.20s «یه پیتزا بخری و نوشابه», 2.20-3.55s «میشه ۱,۴۰۰!»
    (1, 0.00, 3.55, 0.90, "دیگه الان..."),
    # Scene 2: 3.55-4.35s woman says «ولی من...», 4.35-5.35s «۴۰ مدل هودی رو», 5.35-7.45s «فقط بهت میدم ۱.۴۰۰!»
    (2, 3.55, 7.45, 0.80, "ولی من..."),
    # Scene 3: 7.45-9.55s woman says «از سایز کوچیک تا بزرگ!» (starts immediately on «از سایز کوچیک»)
    (3, 7.45, 9.55, 0.00, ""),
    # Scene 4: 9.55-10.22s woman points down, 10.22-12.60s «کامنت بذار هودی تا مدلاش برات ارسال شه!»
    (4, 9.55, 12.60, 0.67, "همین الان..."),
    # Scene 5: 12.60-16.57s Brand Finale («پوشاک سایز بزرگ آیلیدیز | @AYYILDIZ_COMPANYY»)
    (5, 12.60, 16.57, 0.00, ""),
]

def render_scene_3d(assets, s_idx, st, global_t, cut_dur):
    sd = assets[s_idx]
    mt = max(0.0, st - max(0.0, cut_dur - 0.22))

    ripple_t = 0.20 if s_idx == 5 else 0.15
    canvas = apply_floor_ripple(sd["bg"], mt, trigger_t=ripple_t, cx=455, cy=1120 if s_idx != 3 else 1020)

    # ------------------------------------------------------------------------
    # SCENE 1:
    # - mt=0.00s (global_t=0.90s): «یه پیتزا و نوشابه» + floating Pizza & Soda (NO shadow!) + Glass Receipt (WITH ground shadow!)
    # - global_t=2.20s (when she says «میشه یک و چهارصد!»): «۱,۴۰۰,۰۰۰ تومان؟!» (NO shadow!) + Connector + Rising Arrow!
    # ------------------------------------------------------------------------
    if s_idx == 1:
        sp_hl = spring(mt, 0.02, dur=0.50)
        place_rgba(canvas, sd["hl"], 455, 250 - (1.0 - sp_hl) * 45.0, target_w=580,
                   scale_x=0.85 + 0.15 * sp_hl, scale_y=0.85 + 0.15 * sp_hl,
                   opacity=fade_in(mt, 0.0, 0.18), add_shadow=False)

        # Floating 3D Silver Soda Can & 3D Pizza Slice — NO SHADOW (`add_shadow=False`) because they are floating!
        sp_soda = spring(mt, 0.08, dur=0.58, freq=6.2, decay=5.0)
        sp_piz = spring(mt, 0.14, dur=0.60, freq=6.4, decay=4.8)
        soda_y = 665 - (1.0 - sp_soda) * 85.0 + math.cos(mt * 3.2) * 8.0
        piz_y = 675 - (1.0 - sp_piz) * 90.0 + math.sin(mt * 3.2) * 9.0

        place_rgba(canvas, sd["soda"], 595 + (1.0 - sp_soda) * 70.0, soda_y,
                   target_h=305, scale_x=0.78 + 0.22 * sp_soda, scale_y=0.78 + 0.22 * sp_soda,
                   angle_deg=(1.0 - sp_soda) * 14.0 + math.sin(mt * 2.5) * 2.5,
                   opacity=fade_in(mt, 0.08, 0.18), sheen_pos=(mt - 0.45) / 0.90, add_shadow=False)

        place_rgba(canvas, sd["pizza"], 368 - (1.0 - sp_piz) * 80.0, piz_y,
                   target_w=315, scale_x=0.75 + 0.25 * sp_piz, scale_y=0.75 + 0.25 * sp_piz,
                   angle_deg=(1.0 - sp_piz) * (-16.0) + math.cos(mt * 2.5) * 3.0,
                   opacity=fade_in(mt, 0.12, 0.18), add_shadow=False)

        # Grounded 3D Frosted Glass Chart/Receipt Pad — WITH soft padded ground shadow (`add_shadow=True`) and ZERO line under it!
        sp_rec = spring(mt, 0.22, dur=0.58)
        rec_sheen = (mt - 0.65) / 0.90 if 0.65 <= mt <= 1.65 else None
        place_rgba(canvas, sd["receipt"], 465, 1150 + (1.0 - sp_rec) * 105.0 + math.sin(mt * 2.6) * 3.0,
                   target_h=385, scale_x=0.82 + 0.18 * sp_rec, scale_y=0.82 + 0.18 * sp_rec,
                   angle_deg=(1.0 - sp_rec) * 6.0, opacity=fade_in(mt, 0.20, 0.20),
                   sheen_pos=rec_sheen, add_shadow=True, shadow_offset=(0, 22), shadow_alpha=0.25)

        # Price Punchline («۱,۴۰۰,۰۰۰ تومان؟!») + Connector + Rising Arrow ONLY trigger when she says «میشه ۱,۴۰۰» (global_t >= 2.15s)!
        price_t = global_t - 2.15
        if price_t > 0.0:
            sp_pl = spring(price_t, 0.0, dur=0.48, freq=6.8, decay=4.8)
            sp_txt = spring(price_t, 0.08, dur=0.45, freq=7.2, decay=4.8)
            sheen_pill = (price_t - 0.20) / 0.75 if 0.20 <= price_t <= 1.10 else None
            # Floating price capsule — NO shadow (`add_shadow=False`)!
            place_rgba(canvas, sd["pill_bg"], 455, 390 + (1.0 - sp_pl) * 25.0, target_w=580,
                       scale_x=0.55 + 0.45 * sp_pl, scale_y=0.55 + 0.45 * sp_pl,
                       opacity=fade_in(price_t, 0.0, 0.15), sheen_pos=sheen_pill, add_shadow=False)
            place_rgba(canvas, sd["pill_txt"], 455, 390, target_w=480,
                       scale_x=0.65 + 0.35 * sp_txt, scale_y=0.65 + 0.35 * sp_txt,
                       opacity=fade_in(price_t, 0.06, 0.15), add_shadow=False)

            conn_prog = ease_in_out_cubic(price_t / 0.42)
            place_rgba(canvas, sd["conn"], 435, 858, target_h=225,
                       clip_ry=-conn_prog, opacity=fade_in(price_t, 0.0, 0.12),
                       glow_color=(217, 83, 47), glow_strength=0.42, add_shadow=False)

            if price_t > 0.22:
                sp_arr = spring(price_t, 0.22, dur=0.48, freq=7.4, decay=4.6)
                arr_bump = math.sin((price_t - 0.22) * 5.5) * 5.0
                place_rgba(canvas, sd["arrow"],
                           452 - (1.0 - sp_arr) * 40.0,
                           1095 + (1.0 - sp_arr) * 50.0 - arr_bump,
                           target_w=185, scale_x=0.45 + 0.55 * sp_arr, scale_y=0.45 + 0.55 * sp_arr,
                           angle_deg=(1.0 - sp_arr) * (-16.0), opacity=fade_in(price_t, 0.22, 0.14),
                           sheen_pos=(price_t - 0.35) / 0.60, add_shadow=False,
                           glow_color=(217, 83, 47), glow_strength=0.32)

        # Fizz bubbles above soda
        pil_fx = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        fdraw = ImageDraw.Draw(pil_fx, "RGBA")
        if mt > 0.25:
            for b_i in range(6):
                bp = ((mt * 0.95 + b_i * 0.19) % 1.0)
                bx = 605 + int(math.sin(bp * 6.2 + b_i) * 24)
                by = int(510 - bp * 85)
                br = int(3 + (1.0 - bp) * 4)
                ba = int(math.sin(bp * math.pi) * 165)
                fdraw.ellipse([bx - br, by - br, bx + br, by + br],
                              fill=(255, 255, 255, ba), outline=(217, 83, 47, ba // 2))
        fx_arr = np.array(pil_fx, dtype=np.float32)
        fa = fx_arr[:, :, 3:4] / 255.0
        canvas[:] = canvas * (1.0 - fa) + fx_arr[:, :, :3] * fa

    # ------------------------------------------------------------------------
    # SCENE 2:
    # - mt=0.00s (global_t=4.35s, «۴۰ مدل هودی رو»): «۴۰ مدل هودی متنوع» + Hoodie + Swatches + `40x` Badge
    # - global_t=5.35s («فقط بهت میدم ۱.۴۰۰!»): Orange Capsule «فقط ۱.۴۰۰!» pops in!
    # ------------------------------------------------------------------------
    elif s_idx == 2:
        sp_hl = spring(mt, 0.02, dur=0.50)
        place_rgba(canvas, sd["hl"], 455, 235 - (1.0 - sp_hl) * 40.0, target_w=690,
                   scale_x=0.86 + 0.14 * sp_hl, scale_y=0.86 + 0.14 * sp_hl,
                   opacity=fade_in(mt, 0.0, 0.18), add_shadow=False)

        # 1. Unified Fanned Swatch Cards layer unfolds radially behind the hoodie
        if mt > 0.18:
            sp_sw = spring(mt, 0.18, dur=0.68, freq=6.0, decay=4.8)
            sw_sheen = ((mt - 0.75) / 1.15) if 0.75 <= mt <= 2.05 else None
            place_rgba(canvas, sd["swatches"],
                       465, 815 + (1.0 - sp_sw) * 80.0 + math.sin(mt * 2.4) * 4.0,
                       target_w=795,
                       scale_x=0.45 + 0.55 * sp_sw, scale_y=0.45 + 0.55 * sp_sw,
                       angle_deg=(1.0 - sp_sw) * (-22.0) + math.sin(mt * 2.0) * 1.2,
                       opacity=fade_in(mt, 0.18, 0.20), sheen_pos=sw_sheen, add_shadow=False)

        # 2. Separated 3D Folded Beige Hoodie lands on studio floor (grounded -> add_shadow=True)
        sp_hd = spring(mt, 0.08, dur=0.58, freq=6.2, decay=5.2)
        place_rgba(canvas, sd["hoodie"],
                   470, 1015 - (1.0 - sp_hd) * 90.0 + math.cos(mt * 2.6) * 4.0,
                   target_w=555,
                   scale_x=0.78 + 0.22 * sp_hd, scale_y=0.78 + 0.22 * sp_hd,
                   angle_deg=(1.0 - sp_hd) * 7.0,
                   opacity=fade_in(mt, 0.08, 0.18), add_shadow=True, shadow_offset=(0, 20), shadow_alpha=0.24)

        # 3. Separated 3D Orange Circle Badge + Checkmark, then `40x` Text!
        if mt > 0.36:
            sp_bg = spring(mt, 0.36, dur=0.52, freq=7.2, decay=4.6)
            sp_bt = spring(mt, 0.48, dur=0.46, freq=7.8, decay=4.8)
            by = 875 + math.sin(mt * 3.4) * 5.0

            pil_fx = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            fdraw = ImageDraw.Draw(pil_fx, "RGBA")
            rp = ((mt - 0.38) * 1.1) % 1.0
            rr = int(75 + rp * 46)
            fdraw.ellipse([658 - rr, int(by) - rr, 658 + rr, int(by) + rr],
                          outline=(217, 83, 47, int((1.0 - rp) * 145)), width=3)
            fx_arr = np.array(pil_fx, dtype=np.float32)
            fa = fx_arr[:, :, 3:4] / 255.0
            canvas[:] = canvas * (1.0 - fa) + fx_arr[:, :, :3] * fa

            place_rgba(canvas, sd["badge_bg"], 658, by, target_w=162,
                       scale_x=max(0.05, sp_bg), scale_y=max(0.05, sp_bg),
                       angle_deg=(1.0 - sp_bg) * (-25.0),
                       opacity=fade_in(mt, 0.36, 0.14), sheen_pos=(mt - 0.80) / 0.75, add_shadow=False)
            if mt > 0.46:
                place_rgba(canvas, sd["badge_txt"], 652, by + 10, target_w=112,
                           scale_x=max(0.05, sp_bt), scale_y=max(0.05, sp_bt),
                           opacity=fade_in(mt, 0.46, 0.12), add_shadow=False)

        # 4. Price Punchline («فقط ۱.۴۰۰!») ONLY pops in at global_t >= 5.30s when she says «فقط بهت میدم ۱.۴۰۰»!
        p2_t = global_t - 5.30
        if p2_t > 0.0:
            sp_pl = spring(p2_t, 0.0, dur=0.48, freq=6.8, decay=4.8)
            sp_txt = spring(p2_t, 0.08, dur=0.45, freq=7.2, decay=4.8)
            place_rgba(canvas, sd["pill_bg"], 455, 365 + (1.0 - sp_pl) * 26.0, target_w=390,
                       scale_x=0.55 + 0.45 * sp_pl, scale_y=0.55 + 0.45 * sp_pl,
                       opacity=fade_in(p2_t, 0.0, 0.15), sheen_pos=(p2_t - 0.18) / 0.75, add_shadow=False)
            place_rgba(canvas, sd["pill_txt"], 455, 365, target_w=310,
                       scale_x=0.65 + 0.35 * sp_txt, scale_y=0.65 + 0.35 * sp_txt,
                       opacity=fade_in(p2_t, 0.06, 0.15), add_shadow=False)

    # ------------------------------------------------------------------------
    # SCENE 3: SMALL SIZE TO PLUS SIZE (`7.45s – 9.55s`)
    # With the FIXED upright `✓` checkmark on the toggle switch!
    # ------------------------------------------------------------------------
    elif s_idx == 3:
        sp_hl = spring(mt, 0.02, dur=0.45)
        place_rgba(canvas, sd["hl"], 455, 270 - (1.0 - sp_hl) * 35.0, target_w=705,
                   scale_x=0.86 + 0.14 * sp_hl, scale_y=0.86 + 0.14 * sp_hl,
                   opacity=fade_in(mt, 0.0, 0.16), add_shadow=False)

        # Subtitle capsule pops in at mt > 0.55s (global_t >= 8.0s)
        if mt > 0.52:
            sp_pl = spring(mt, 0.52, dur=0.46)
            sp_txt = spring(mt, 0.60, dur=0.44)
            place_rgba(canvas, sd["pill_bg"], 455, 392 + (1.0 - sp_pl) * 24.0, target_w=590,
                       scale_x=0.60 + 0.40 * sp_pl, scale_y=0.60 + 0.40 * sp_pl,
                       opacity=fade_in(mt, 0.52, 0.15), sheen_pos=(mt - 0.70) / 0.75, add_shadow=False)
            place_rgba(canvas, sd["pill_txt"], 455, 392, target_w=510,
                       scale_x=0.68 + 0.32 * sp_txt, scale_y=0.68 + 0.32 * sp_txt,
                       opacity=fade_in(mt, 0.58, 0.15), add_shadow=False)

        # Unified 3D Draped Fabric + Size Blocks platform (with FIXED upright `✓` checkmark!)
        sp_hero = spring(mt, 0.08, dur=0.54, freq=5.8, decay=5.2)
        hero_y = 965 + (1.0 - sp_hero) * 75.0 + math.sin(mt * 2.8) * 3.5
        place_rgba(canvas, sd["hero_all"], 450, hero_y, target_w=900,
                   scale_x=0.88 + 0.12 * sp_hero, scale_y=0.88 + 0.12 * sp_hero,
                   opacity=fade_in(mt, 0.06, 0.18),
                   sheen_pos=(mt - 0.25) / 1.05, sheen_strength=0.40,
                   add_shadow=True, shadow_offset=(0, 18), shadow_alpha=0.22)

        # Progressive glowing size indicator pills above each block (`S -> M -> L -> XL -> 2XL -> 3XL+`)
        pil_fx = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        fdraw = ImageDraw.Draw(pil_fx, "RGBA")
        f_tag = ImageFont.truetype(FONT_EN_BOLD, 22)
        tags = [
            ("S", 162, 1020, 0.16, False),
            ("M", 272, 945, 0.26, False),
            ("L", 380, 868, 0.36, False),
            ("XL", 482, 755, 0.48, True),
            ("2XL", 575, 645, 0.60, True),
            ("3XL+", 685, 545, 0.72, True),
        ]
        pts_line = []
        for lbl, tx, ty, t_trig, is_plus in tags:
            if mt > t_trig:
                sp_t = spring(mt, t_trig, dur=0.42, freq=6.8, decay=5.0)
                cy_t = int(ty + (1.0 - sp_t) * 35.0)
                pts_line.append((tx, cy_t))
                pw_t = 34 if len(lbl) <= 2 else 46
                col_bg = (217, 83, 47, 238) if is_plus else (42, 40, 38, 215)
                fdraw.rounded_rectangle([tx - pw_t, cy_t - 18, tx + pw_t, cy_t + 18],
                                        radius=14, fill=col_bg, outline=(255, 255, 255, 225), width=2)
                bb = fdraw.textbbox((0, 0), lbl, font=f_tag)
                fdraw.text((tx - (bb[2] - bb[0]) * 0.5, cy_t - (bb[3] - bb[1]) * 0.5 - 2),
                           lbl, font=f_tag, fill=(255, 255, 255, 255))
        if len(pts_line) >= 2:
            fdraw.line(pts_line, fill=(217, 83, 47, 130), width=3)
        fx_arr = np.array(pil_fx, dtype=np.float32)
        fa = fx_arr[:, :, 3:4] / 255.0
        canvas[:] = canvas * (1.0 - fa) + fx_arr[:, :, :3] * fa

    # ------------------------------------------------------------------------
    # SCENE 4: COMMENT «هودی» -> DIRECT DM! (`9.55s – 12.60s`)
    # Both «هودی» on the box and the 7 cascading hoodies stay 100% visible!
    # ------------------------------------------------------------------------
    elif s_idx == 4:
        sp_hl = spring(mt, 0.02, dur=0.46)
        place_rgba(canvas, sd["hl"], 455, 205 - (1.0 - sp_hl) * 38.0, target_w=725,
                   scale_x=0.86 + 0.14 * sp_hl, scale_y=0.86 + 0.14 * sp_hl,
                   opacity=fade_in(mt, 0.0, 0.16), add_shadow=False)

        # 1. Unified 7-Hoodie Card Cascade + Glowing Trails + Paper Plane launch on button click (mt > 0.48s)
        if mt > 0.48:
            sp_pl = spring(mt, 0.48, dur=0.48)
            sp_txt = spring(mt, 0.56, dur=0.45)
            place_rgba(canvas, sd["pill_bg"], 455, 315 + (1.0 - sp_pl) * 26.0, target_w=585,
                       scale_x=0.58 + 0.42 * sp_pl, scale_y=0.58 + 0.42 * sp_pl,
                       opacity=fade_in(mt, 0.48, 0.15), sheen_pos=(mt - 0.65) / 0.80, add_shadow=False)
            place_rgba(canvas, sd["pill_txt"], 455, 315, target_w=535,
                       scale_x=0.66 + 0.34 * sp_txt, scale_y=0.66 + 0.34 * sp_txt,
                       opacity=fade_in(mt, 0.54, 0.15), add_shadow=False)

            sp_c = spring(mt, 0.48, dur=0.64, freq=6.0, decay=4.8)
            reveal_c = ease_out_cubic((mt - 0.48) / 0.48)
            place_rgba(canvas, sd["cards"],
                       595 - (1.0 - sp_c) * 50.0,
                       795 + (1.0 - sp_c) * 130.0 + math.sin(mt * 2.6) * 4.5,
                       target_w=605,
                       scale_x=0.62 + 0.38 * sp_c, scale_y=0.62 + 0.38 * sp_c,
                       angle_deg=(1.0 - sp_c) * 12.0,
                       clip_ry=reveal_c,
                       opacity=fade_in(mt, 0.48, 0.16),
                       sheen_pos=(mt - 0.85) / 1.05, add_shadow=False)

            tr_prog = ease_out_cubic((mt - 0.54) / 0.48)
            if tr_prog > 0.01:
                place_rgba(canvas, sd["trails"], 565, 960, target_w=340,
                           clip_ry=tr_prog, opacity=fade_in(mt, 0.54, 0.14) * 0.95,
                           glow_color=(255, 165, 70), glow_strength=0.65, add_shadow=False)

            sp_pl3d = spring(mt, 0.56, dur=0.62, freq=5.8, decay=4.8)
            px = 718 - (1.0 - sp_pl3d) * 190.0 + math.sin(mt * 3.5) * 6.0
            py = 765 + (1.0 - sp_pl3d) * 230.0 + math.cos(mt * 3.5) * 7.0
            place_rgba(canvas, sd["plane"], px, py, target_w=175,
                       scale_x=0.45 + 0.55 * sp_pl3d, scale_y=0.45 + 0.55 * sp_pl3d,
                       angle_deg=(1.0 - sp_pl3d) * 22.0 + math.sin(mt * 3.0) * 3.5,
                       opacity=fade_in(mt, 0.56, 0.14), add_shadow=False,
                       glow_color=(217, 83, 47), glow_strength=0.32)

        # 2. Separated 3D White Comment Box + RTL Typing of «هودی» (stays 100% visible!)
        sp_bx = spring(mt, 0.08, dur=0.50, freq=6.4, decay=5.2)
        click_squish = 0.0
        if 0.40 <= mt <= 0.58:
            click_squish = math.sin(((mt - 0.40) / 0.18) * math.pi) * 0.07
        box_y = 1205 + (1.0 - sp_bx) * 85.0
        place_rgba(canvas, sd["box"], 455, box_y, target_w=525,
                   scale_x=(0.75 + 0.25 * sp_bx) * (1.0 - click_squish),
                   scale_y=(0.75 + 0.25 * sp_bx) * (1.0 + click_squish * 0.6),
                   opacity=fade_in(mt, 0.06, 0.16), sheen_pos=(mt - 0.45) / 0.85,
                   add_shadow=True, shadow_offset=(0, 18), shadow_alpha=0.22)

        if mt > 0.16:
            type_prog = ease_out_cubic((mt - 0.16) / 0.26)
            place_rgba(canvas, sd["word"], 425, box_y, target_w=205,
                       clip_rx=type_prog, opacity=fade_in(mt, 0.16, 0.10), add_shadow=False)

        pil_fx = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        fdraw = ImageDraw.Draw(pil_fx, "RGBA")
        if 0.16 <= mt <= 0.46:
            cur_x = int(525 - ease_out_cubic((mt - 0.16) / 0.26) * 205)
            fdraw.rounded_rectangle([cur_x - 3, 1165, cur_x + 3, 1242], radius=3, fill=(217, 83, 47, 235))
        if 0.44 <= mt <= 1.05:
            cp = (mt - 0.44) / 0.61
            cr = int(38 + cp * 115)
            fdraw.ellipse([645 - cr, 1202 - cr, 645 + cr, 1202 + cr],
                          outline=(217, 83, 47, int((1.0 - cp) * 210)), width=4)
        fx_arr = np.array(pil_fx, dtype=np.float32)
        fa = fx_arr[:, :, 3:4] / 255.0
        canvas[:] = canvas * (1.0 - fa) + fx_arr[:, :, :3] * fa

    # ------------------------------------------------------------------------
    # SCENE 5: AYYILDIZ PLUS-SIZE APPAREL FINALE (`12.60s – 16.57s`)
    # Clean Frosted Glass Pedestal with ZERO lines or dark shadow smudges!
    # ------------------------------------------------------------------------
    elif s_idx == 5:
        sp_hl = spring(mt, 0.04, dur=0.52)
        place_rgba(canvas, sd["hl"], 455, 235 - (1.0 - sp_hl) * 45.0, target_w=755,
                   scale_x=0.86 + 0.14 * sp_hl, scale_y=0.86 + 0.14 * sp_hl,
                   opacity=fade_in(mt, 0.02, 0.20), sheen_pos=(mt - 0.35) / 0.90, add_shadow=False)

        # 1. Clean separated 3D Frosted Glass Pedestal rises from the floor (grounded -> add_shadow=True)
        sp_ped = spring(mt, 0.14, dur=0.64, freq=5.8, decay=5.2)
        ped_sheen = ((mt - 1.10) / 1.25) if 1.10 <= mt <= 2.50 else None
        place_rgba(canvas, sd["ped"], 455, 985 + (1.0 - sp_ped) * 115.0 + math.sin(mt * 2.4) * 3.0,
                   target_w=750,
                   scale_x=0.78 + 0.22 * sp_ped, scale_y=0.78 + 0.22 * sp_ped,
                   opacity=fade_in(mt, 0.12, 0.22), sheen_pos=ped_sheen,
                   add_shadow=True, shadow_offset=(0, 22), shadow_alpha=0.22)

        # 2. Separated unified 3D Crimson `Ayyildiz` + Crescent & Star (`add_shadow=False` so glass stays crystal-clear!)
        if mt > 0.32:
            sp_lg = spring(mt, 0.32, dur=0.66, freq=6.4, decay=4.8)
            lg_sheen = ((mt - 1.35) / 1.15) if 1.35 <= mt <= 2.65 else None
            place_rgba(canvas, sd["logo"],
                       455, 755 - (1.0 - sp_lg) * 135.0 + math.sin(mt * 2.4) * 3.5,
                       target_w=735,
                       scale_x=0.72 + 0.28 * sp_lg, scale_y=0.72 + 0.28 * sp_lg,
                       angle_deg=(1.0 - sp_lg) * (-6.0),
                       opacity=fade_in(mt, 0.32, 0.20), sheen_pos=lg_sheen,
                       add_shadow=False, glow_color=(200, 29, 37), glow_strength=0.22)

        # 3. Separated 3D Red Scalloped Verified Badge (`✓`)
        if mt > 0.58:
            sp_vb = spring(mt, 0.58, dur=0.60, freq=7.2, decay=4.6)
            vb_y = 632 + math.cos(mt * 3.2) * 5.0

            pil_fx = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            fdraw = ImageDraw.Draw(pil_fx, "RGBA")
            rp = ((mt - 0.62) * 0.95) % 1.0
            rr = int(46 + rp * 48)
            fdraw.ellipse([778 - rr, int(vb_y) - rr, 778 + rr, int(vb_y) + rr],
                          outline=(200, 29, 37, int((1.0 - rp) * 165)), width=3)
            fx_arr = np.array(pil_fx, dtype=np.float32)
            fa = fx_arr[:, :, 3:4] / 255.0
            canvas[:] = canvas * (1.0 - fa) + fx_arr[:, :, :3] * fa

            place_rgba(canvas, sd["vbadge"], 778, vb_y, target_w=96,
                       scale_x=max(0.05, sp_vb), scale_y=max(0.05, sp_vb),
                       angle_deg=(1.0 - sp_vb) * (-180.0),
                       opacity=fade_in(mt, 0.58, 0.16), add_shadow=False)

        # 4. Separated Glossy Black Capsule + `@AYYILDIZ_COMPANYY` Text
        if mt > 0.72:
            sp_bp = spring(mt, 0.72, dur=0.56, freq=6.4, decay=5.2)
            sp_bt = spring(mt, 0.86, dur=0.50, freq=7.0, decay=5.0)
            bp_sheen = ((mt - 1.50) / 1.10) if 1.50 <= mt <= 2.75 else None
            place_rgba(canvas, sd["pill_bg"], 455, 1365 + (1.0 - sp_bp) * 80.0, target_w=705,
                       scale_x=0.68 + 0.32 * sp_bp, scale_y=0.68 + 0.32 * sp_bp,
                       opacity=fade_in(mt, 0.72, 0.18), sheen_pos=bp_sheen,
                       add_shadow=True, shadow_offset=(0, 16), shadow_alpha=0.22)
            if mt > 0.84:
                place_rgba(canvas, sd["pill_txt"], 455, 1365 + (1.0 - sp_bt) * 25.0, target_w=575,
                           scale_x=0.75 + 0.25 * sp_bt, scale_y=0.75 + 0.25 * sp_bt,
                           opacity=fade_in(mt, 0.84, 0.16), add_shadow=False)

    accent = (200, 29, 37) if s_idx == 5 else (217, 83, 47)
    draw_hud_and_particles(canvas, mt, global_t, accent_rgb=accent)
    return canvas


def render_master_frame(assets, spk_reader, frame_idx, global_t):
    active = SCENE_CONFIG[-1]
    for cfg in SCENE_CONFIG:
        if cfg[1] <= global_t <= cfg[2]:
            active = cfg
            break
    s_idx, t_start, t_end, cut_dur, intro_phrase = active
    st = global_t - t_start
    dur = t_end - t_start

    canvas = render_scene_3d(assets, s_idx, st, global_t, cut_dur)

    if s_idx in [1, 2, 3, 4]:
        spk_frame = spk_reader.get_frame(frame_idx)
        if cut_dur > 0.01:
            render_speaker_cut_overlay(canvas, spk_frame, s_idx, st, cut_dur, intro_phrase)
        else:
            # In Scene 3 (no full-screen cut_dur), still show the sleek top-left circular PIP avatar!
            render_speaker_cut_overlay(canvas, spk_frame, s_idx, 1.0, 0.02, "")

    rem = dur - st
    if s_idx < 5 and rem < 0.20:
        p_tr = ease_in_out_cubic((0.20 - rem) / 0.20)
        next_cfg = SCENE_CONFIG[s_idx]
        next_canvas = render_scene_3d(assets, next_cfg[0], p_tr * 0.20, global_t, next_cfg[3])
        if next_cfg[0] in [1, 2, 3, 4]:
            if next_cfg[3] > 0.01:
                render_speaker_cut_overlay(next_canvas, spk_reader.get_frame(frame_idx), next_cfg[0], p_tr * 0.20, next_cfg[3], next_cfg[4])
            else:
                render_speaker_cut_overlay(next_canvas, spk_reader.get_frame(frame_idx), next_cfg[0], 1.0, 0.02, "")
        z_cur = 1.0 + 0.06 * p_tr
        cw, ch = int(W * z_cur), int(H * z_cur)
        cur_z = cv2.resize(canvas, (cw, ch), interpolation=cv2.INTER_LINEAR)
        cur_crop = cur_z[(ch - H) // 2:(ch - H) // 2 + H, (cw - W) // 2:(cw - W) // 2 + W]
        canvas = cur_crop * (1.0 - p_tr) + next_canvas * p_tr

    return np.clip(canvas, 0, 255).astype(np.uint8)


def main():
    print("Loading V3 bug-free assets & speaker video...")
    assets = load_v2_assets()
    spk_reader = SpeakerVideoReader(VIDEO_IN)

    total_frames = len(spk_reader.frames)
    tmp_raw = "/tmp/ayyildiz_v3_noaudio.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_raw, fourcc, FPS, (W, H))

    verify_strip = []
    # Sample keyframes across all Speaker Cuts, Pre-Punchline states, and Post-Punchline states!
    sample_times = [0.40, 1.50, 2.80, 3.90, 4.90, 6.40, 8.60, 9.85, 11.60, 14.80]
    sample_indices = {int(round(t * FPS)) for t in sample_times}

    for f_idx in range(total_frames):
        t = f_idx / FPS
        frame_rgb = render_master_frame(assets, spk_reader, f_idx, t)
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        if f_idx in sample_indices:
            verify_strip.append(cv2.resize(frame_rgb, (240, 410)))
        if f_idx % 100 == 0:
            print(f"  Rendered frame {f_idx}/{total_frames} (t={t:.2f}s)")

    writer.release()
    if verify_strip:
        Image.fromarray(np.hstack(verify_strip)).save("/tmp/v3_keyframes_strip.jpg", quality=94)
        print("Saved /tmp/v3_keyframes_strip.jpg")

    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_raw,
        "-i", VIDEO_IN,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        VIDEO_OUT
    ]
    subprocess.run(cmd, check=True)
    print(f"Successfully rendered V3 video to: {VIDEO_OUT}")

if __name__ == "__main__":
    main()
