# 🧬 USER TASTE & VISUAL-DNA CONSTITUTION (قانون اساسی سلیقهٔ کارفرما و دی‌ان‌ای بصری پروژه)
> **v21 HARMONISATION:** read together with `USER_TASTE_GUIDE.md` (6 Core Creative Laws) and
> `style_bank/STYLE_BANK_PLAYBOOK.md`. **Supreme law: copying and repetition are forbidden.** The techniques below
> are a vocabulary of principles, not a template. Use each one only when the meaning calls for it, and never
> repeat the same combination from one video to the next.
>
> **MANDATORY CROSS-SESSION MEMORY FOR ALL AI AGENTS:**
> Read this file FIRST whenever a new session begins. This document permanently records the User's exact aesthetic taste, creative direction rules, and strict prohibitions established across our production sessions.

---

## 🚫 LAW 1: THE STRICT ZERO-BOX MANDATE (قانون مطلقِ حذف باکس‌های دوبعدی و مستطیلی)
1. **NO Flat 2D Boxes, Rectangles, or Capsules Around Content:**
   - **NEVER** put headlines, titles, prices/counters (`+۸ میلیون`), brand logos (`Apple`, `Surface`, `HP`, `DELL`), stickers (`نکته خیلی مهم!`), or callouts inside flat 2D rectangular boxes, colored slabs, or plain pills!
   - Putting everything inside rectangular boxes feels amateur, stiff, and "low-level" (`بی‌لول و خشک`).
2. **The ONLY Two Permitted Exceptions to Rectangular/Cubic Shapes:**
   - **Exception A — True 3D Tactile Objects (`Tactile 3D Volume`):** A cube, block, or hardware monument is ONLY allowed if it is a **real 3D physical object** with perspective depth, beveled edges, studio reflections, and tactile materials (frosted translucent glass, brushed anodized metal, matte ceramic, polished acrylic—such as `grade_a_3d.png`, `grade_b_3d.png`, `grade_c_3d.png`, or `iso_cube`).
   - **Exception B — Intrinsic Game-Style Loading / Telemetry Bars (`@element("hud_loading_bar")`):** The ONLY 2D rectangular graphic allowed is when the concept itself is a **game-style loading bar or cybernetic progress meter** (`[████████░░░] ۸۵٪`) with viewfinder corner brackets `┌ ┐ └ ┘` and segmented energy cells.

---

## 🎬 LAW 2: HOW TO MAKE TEXT & GRAPHICS POP ON TALKING-HEAD REELS (بدون استفاده از باکس!)
When editing a raw talking-head video (`footage`), raw room backgrounds often have poor contrast for text. **Never solve poor background contrast by putting a box behind the text!** Instead, use these two signature techniques implemented in `mograph v13.5`:

### Technique A — Behind-Speaker Full-Environment Coordinate Grid Pass (`"behind": "clip"`)
> v21: an OPTIONAL technique for engineering/data topics, at most once per video. It is no longer drawn behind
> every element (stage grid default = 0). The default contrast tool is soft scrim + coloured light pool + glow.
- Segment the speaker using `portrait_seg.onnx` (`mograph/matte.py`).
- Transform the **entire background behind the speaker** (`_draw_dna_grid_environment` in `reelfx.py`) into a color-graded dark studio environment with:
  - A subtle `0.6px` engineering coordinate grid + `+` intersection tick marks,
  - A diffused radial key-fill spotlight harmonized with the speaker's skin/color grade,
  - Floating 3D extruded Persian typography (`KalamehFaNum-Black` + `DigiGhasedak-Bold` handwritten subtitle) and isolated 3D studio props (`laptop_3d.png`, `cash_3d.png`, `shield_3d.png`) hovering freely in the open gridded space behind the speaker's shoulders!

### Technique B — Full-Frame $f/2.8$ Depth-of-Field Speaker Defocus (`"dof_blur"` on `footage`)
- When a critical hero statement or warning (`نکته خیلی مهم!`) needs maximum visual impact:
  - Smoothly blur and dark-matte the **entire video frame including the speaker himself** (`"dof_blur": [{"t": 8.0, "dur": 1.35, "amount": 0.88}]` on `el_footage`).
  - Overlay a full-screen golden/emerald coordinate grid and stage the giant 3D extruded headline + handwritten subtitle + rotating 3D icon (`warn_3d.png`) right across the center of the full `1080x1920` screen with zero boxes!

---

## 💎 LAW 3: TACTILE 3D VOLUME & TRUE 3D PERSPECTIVE SPIN (`_blit_3d_perspective`)
1. **Zero Flat 2D Vector Icons for Hero Subjects:**
   - Never draw a flat red circle for a location pin, a stick-line triangle for an airplane, or plain text for brand logos!
   - Always use isolated, high-resolution **3D Studio RGBA Sprites** (`mograph/assets_3d/*.png`) or generate new ones on a `#000000` black background using the native workspace `image_gen:generate_image` tool and slice them with:
     ```bash
     python3 -m mograph extract3d sheet.jpg --names item1_3d,item2_3d,item3_3d,item4_3d --grid 2x2
     ```
2. **True 3D Perspective Homography Rotation (`cv2.warpPerspective`):**
   - When 3D brand logos (`logo_apple_3d.png`, `logo_surface_3d.png`, `logo_hp_3d.png`, `logo_dell_3d.png`), 3D grade blocks (`grade_a_3d.png`), or `tactile_3d` objects enter or hover on screen, rotate them in **true 3D perspective space (`yaw_deg`, `pitch_deg`, `roll_deg` + extruded 3D thickness layers)** using `_blit_3d_perspective`!

---

## ✒️ LAW 4: MULTI-FONT PERSIAN KINETIC TYPOGRAPHY & ONE THREAD OF COLOR

> ✅ **v21: multi-font pairing is ACTIVE again** (USER_TASTE_GUIDE Law 6). The v14 "Doran only" lock is retired
> (`font_lock` false by default). Pair one structural family with one expressive hero voice per video, and vary
> the pairing between videos. Colours are still designed from the footage.
1. **Multi-Font Pairing (Never Monotonous Single-Font Layouts):**
   - Pair heavy 3D-extruded display headlines (`KalamehFaNum-Black`, `Peyda-Black`) with expressive handwritten/calligraphic accents (`DigiGhasedak-Bold`, `Delbar-Bold`, `Shoko-Bold`, `Miremad` Nastaliq) and crisp numeric counters (`PelakFA-Black`).
2. **Glowing Caliper Underlines Instead of Boxes (`_draw_glow_underline`):**
   - Anchor floating typography and counters (`+۸ میلیون`) using a thin glowing `@accent` caliper line with a central diamond node (`◆`) underneath—never a bounding box that text could poke out of!

---

## خلاصهٔ فارسی قوانین دائمی سلیقهٔ شما (ثبت‌شده در حافظهٔ پروژه برای تمام سشن‌های بعدی)
0. **قانون برتر (v21): کپی‌کاری و تکرار اکیداً ممنوع.** رفرنس‌ها، تمپلیت‌ها و پروژه‌های قبلی فقط برای سبک‌شناسی و الگوبرداری هستند. هر ویدیو هویت بصری خودش را دارد و `pro originality` آن را بررسی می‌کند.
1. **ممنوعیت مطلق باکس‌های مستطیلی دوبعدی (`Zero-Box Law`):** هیچ نوشته، تیتر، قیمت، لوگو یا آیکونی نباید داخل باکس یا کپسول مستطیلی تخت قرار بگیرد. تنها استثناها: (الف) **احجام واقعاً سه‌بعدی** از جنس شیشهٔ مات و فلز تراش‌خورده (مثل بلوک‌های سه‌بعدی `A+ / B / C`)، و (ب) **نوار لودینگ به سبک بازی‌ها (`hud_loading_bar`)** وقتی که خودِ مفهوم، بارگذاری یا لودینگ باشد.
2. **گرید مهندسی (اختیاری، حداکثر یک بار در هر ویدیو؛ پیش‌فرض خاموش):** به جای گذاشتن باکس پشت متن، کل پس‌زمینهٔ پشت شخص دوربری‌شده به محیط استودیویی تیره با گرید مهندسی ظریف (`0.6px`) و نورپردازی هم‌رنگ با گوینده تبدیل می‌شود.
3. **تار کردن کل تصویر گوینده (`dof_blur`) برای نکات مهم:** در لحظات اوج (مثل «نکته خیلی مهم!»)، کل تصویر گوینده به نرمی تار و مات (`f/2.8 Blur`) شده و تیتر سه‌بعدی در مرکزِ کل صفحه بدون هیچ باکسی می‌درخشد.
4. **لوگوها و المان‌های سه‌بعدی چرخان (`Tactile 3D + Perspective Spin`):** تمام لوگوهای برندها و آیکون‌های اصلی به صورت آبجکت سه‌بعدی واقعی با چرخش پرسپکتیو سه‌بعدی (`yaw/pitch`) و ترکیب فونت‌های متنوع فارسی (`KalamehFaNum-Black` + `DigiGhasedak-Bold` دست‌نویس + `PelakFA-Black`) اجرا می‌شوند.
