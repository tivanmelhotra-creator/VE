# 🎬 CREATIVE REELS PLAYBOOK — `v13.9 @novadent.ir 100/100 Master Edition`
> **MANDATORY REFERENCE FOR ALL AI AGENTS:**
> This playbook documents the 8 forensic laws dissected from the User's favorite `@novadent.ir` Before/After showcase reel (`Screenshots 211857..212043`) and the 2 companion viral reels, all implemented in `mograph/creative_reels.py`, `mograph/captions.py`, `mograph/sound.py`, and `mograph/fx.py`.

---

## 🌟 THE 8 FORENSIC LAWS OF THE `@novadent.ir` 100/100 AESTHETIC

### Law 1. Extreme Inline Word Contrast (`تضاد شدید فونت، ضخامت، اندازه و رنگ بین کلمات یک جمله یا تایتل`)
* **Never render all words in a title, badge, or subtitle with the same font, weight, size, and color!**
* Always pair **Ultra-Thin / Light weights (`IRANYekanThin.ttf` / `IRANYekanLight.ttf`)** at `1.0x` scale in neutral white/silver (`#E2E8F0`) right next to **One Hero Word (`یک کلمهٔ کلیدی در میان متن`)** set in either:
  * **Lush Calligraphic / Display Script (`Delbar-Bold.ttf` / `Miremad.ttf` / `DigiGhasedak-Bold.ttf`)** at `1.38x–2.0x` larger size in glowing gold (`#FBBF24`) or crimson (`#FF2E42`), OR
  * **Ultra-Heavy Geometric Black (`IRANYekanExtraBlack2.ttf` / `KalamehFaNum-Black.ttf`)** at `1.38x–2.0x` larger size!
* Use `draw_mixed_font_line(ctx, "و **دهان** رو __خشک__ میکنن", ...)` in `mograph/creative_reels.py` (automatically active in `hud.subs.style = "novadent_jelly"`).

### Law 2. Dashed Golden Semi-Circle + Real Glassmorphic Backdrop-Blur Dome + Split Trapezoidal Panels (`@element("novadent_split_dome")` — Screenshot `211857`)
* Behind the segmented speaker (`behind: "clip"`):
  * The upper background (above a crisp horizontal white line at shoulder height `y_split`) is split by an angled diagonal into **two trapezoidal product panels** (`SPARTINA / اسپارتینا` in dark emerald on the left, `OZEMPIC / اوزمپیک` with a tilted 3D product/pen cutout on the right).
  * Over the top center sits a **semi-circular dome with real Glassmorphic Backdrop-Blur (`_paint_backdrop_blur`)**, bounded by a **glowing golden dashed/dotted semi-circular arc**, containing `آمپول‌های` (`IRANYekanThin.ttf`) + giant sweeping **`لاغری`** (`Delbar-Bold.ttf`).

### Law 3. Translucent Dark Vinyl/CD Ring + Polar-Warped Curved Persian Text + Dynamic Collar-to-Hand Red Light Spill (`@element("novadent_cd_dial")` — Screenshots `211908` & `211917`)
* A translucent dark-glass CD donut ring on the speaker's chest with a white center disc (`بیشتر از` in `IRANYekanExtraBlack2.ttf` + `این کـه` in `IRANYekanLight.ttf`).
* Uses **Polar-to-Cartesian Strip Warping (`_polar_arc_surface` via `cv2.remap`)** so every Persian word and cursive connection curves continuously along the circular track (`روی کاهش وزن اثر بذارن!` on top, `روی سلامت دهان و دندونتون` on bottom).
* **State 1 (`t < rotate_at`):** Only the **top half of the outer circumference** has a glowing red rim line (bottom half has zero red!), top curved text glows white, and an `OPERATOR_SCREEN` crimson light spill physically tints the speaker's **white shirt collar and neck** above the circle!
* **State 2 (`t >= rotate_at`):** Smooth `Ease-In-Out` (`_cubic_ease_in_out`) `180°` rotation! Top rim loses red, top curved text becomes **dark embossed charcoal (`embossed=True`)**, bottom outer rim glows red, bottom curved text glows golden-yellow, and the **red optical light spill shifts downward onto the speaker's hands and white shirt cuff**!

### Law 4. Tilted Rounded-Square Wireframe + Border-Breaking 3D Golden Calligraphy + 45° Rotated Diamond (`◇`) Badge (`@element("novadent_diamond_square")` — Screenshot `211936`)
* A `-16°` tilted rounded-square vector outline above the speaker's head where the **`م` of `مورد اول` intentionally extends outside the right wireframe border**, and the right/left tips of giant 3D-shaded golden **`خشکی`** (`Delbar-Bold.ttf`) **break across both borders**!
* Pinned to the top-left corner is a **45° rotated rounded-square Diamond (`◇`) frame** with a double white/glass border and clipped medical/3D illustration inside.

### Law 5. Elastic Jelly Glassmorphic Subtitle Bar + `"jelly"` Sound Synthesizer (`hud.subs.style = "novadent_jelly"` in `captions.py` & `sound.py` — Screenshot `211936`)
* Real Backdrop-Blur Glassmorphic dark-emerald capsule on the speaker's chest with **glowing cyan-teal (`#2DD4BF`) crescent rim highlights concentrated on the left and right curved end-caps**.
* As subtitle text length changes, the capsule **elastically squashes and stretches (`Jelly Squash & Stretch` with volume conservation `h ~ w^-0.42`)** accompanied by a synthesized **frequency-modulated watery/rubbery `"jelly"` SFX** in `sound.py`!

### Law 6. Behind-Chair Occlusion & Self-Illuminating Golden Light-Wrap (`@element("novadent_chair_glow_text")` & `@element("novadent_edge_light_spill")` — Screenshot `212019`)
* Stacks `ماهایی که` (thin white) + giant extra-bold golden **`زدیم`** (`IRANYekanExtraBlack2.ttf`) + `چیکار / کنیم` (thin white) behind the speaker & chair (`behind: "clip"`) so the right tip of **`ز`** in `زدیم` tucks **behind the black chair backrest** and casts a warm golden `OPERATOR_SCREEN` light-wrap onto the chair/shoulder edge!

### Law 7. Concentric Two-Tone Red CD Halo Behind Speaker & Chair (`@element("novadent_red_halo")` — Screenshot `212032`)
* Darkens the room background to moody charcoal (`#0B0C10`) and places a **two-layer concentric red CD halo** (outer dark-crimson ring `#3B070B` with bright neon-red outer stroke `#FF2E3B` + wide red wall glow; inner vivid-crimson disc `#DC1E28`) behind the speaker's head and chair, displaying `اصلاً` (thin) + stretched **`ســــــخت`** + `نیست` (thin).

### Law 8. True Backdrop-Blur Glassmorphic Hexagon + Top Vertex Specular Light + 3D Head-Orbiting Titles (`@element("novadent_glass_hex")` & `@element("orbiting_title_3d")` — Screenshot `212043`)
* `novadent_glass_hex`: Real `_paint_backdrop_blur` inside a rounded hexagon over the speaker's chest (visibly blurring the speaker's clothes underneath), with a `2.2px` golden-amber border, a **bright white-gold radial specular light bloom on the top vertex**, and silky `Ease-In-Out` open/close animation.
* `orbiting_title_3d`: Orbits a 3D title around the speaker's head (`half="back"` behind the head + `half="front"` in front of the head) or swoops like a bird from behind the speaker down into the subtitle slot (`mode="bird_swoop_to_sub"`).
