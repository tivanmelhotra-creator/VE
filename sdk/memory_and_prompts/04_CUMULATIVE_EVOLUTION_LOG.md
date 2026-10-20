# 📈 04_CUMULATIVE_EVOLUTION_LOG.md
## دفترچه تکامل و انباشت تجربیات موتور موشن‌گرافیک (Living Memory Log)

این فایل تاریخچه تکامل «مغز خالق موشن‌گرافیک» در جلسات مختلف است. هر ایجنت در پایان هر جلسه کاری جدید باید دستاوردها، آیکون‌های جدید، ویجت‌های اضافه‌شده و باگ‌های برطرف‌شده را به انتهای این سند اضافه کند و فایل زیپ جدید تحویل دهد.

---

### نسخه V1 — پایه‌گذاری استخراج لایه‌های پرده سبز و موشن سه‌بعدی
* پیاده‌سازی Chroma Keying با حذف هاله سبز (`Despill`) و حفظ سایه برای شیت‌های ۳ پنله (`N-layers.jpg`).
* کشف مشکل برش دستی روی آبجکت‌های هم‌پوشان (کالیته‌های پارچه و زنجیره هودی‌ها) و وضع **قانون طلایی ۱**: «فقط باکس‌های جداشده در شیت پرده سبز مستقل انیمیت شوند؛ آبجکت‌های هم‌پوشان یک لایه واحد هستند».

### نسخه V2 — تدوین ترکیبی گوینده + قفل کلمه‌به‌کلمه + کالبدشکافی سبک وکتور `@ENSAN.BESHAM`
* **رفع باگ اسپویل زیرنویس (تقلب دادن به مخاطب) و دوباره‌نویسی:** نمونه‌برداری ۰.۲ ثانیه‌ای از ویدیوی گوینده؛ نمایش صرفاً کلمات آغازین («دیگه الان...»، «ولی من...») در کات گوینده و مخفی نگه‌داشتن کپسول قیمت تا لحظه ادا شدن عدد.
* **تفکیک فیزیک سایه‌های معلق و زمینی:** حذف سایه از آبجکت‌های معلق (`add_shadow=False`) و افزودن `42px` حاشیه شفاف قبل از `GaussianBlur` برای حذف کامل خط افقی زیر صفحات شیشه‌ای.
* **مهندسی معکوس کامل سبک وکتور سیاه‌وسفید (`@ENSAN.BESHAM`):** پیاده‌سازی سناریوی ۱۰ فصلی «محاکمه انسان از زبان پول» (۱۴۸ ثانیه) به همراه سینت‌سایزر صدای محیطی دارک (`Dark Ambient Drone + Piano + Micro-SFX`).

### نسخه V3 (نسخه فعلی) — مهاجرت به موتور زیرپیکسلی `PyCairo` + لبه‌های نرم بزیه + طراحی چندلایه پرجزئیات
* **مهاجرت از `cv2.line` به `PyCairo`:** رندر اعشاری زیرپیکسلی (`Sub-Pixel AA`) بدون پرش در زوم‌های نرم و پشتیبانی بومی از منحنی‌های بزیه (`C / Q`) و `Trim Path` (`ctx.set_dash`).
* **لبه‌های نرم و گرد خودکار (`Auto-Fillet & Rounded Bezier Geometry`):** افزودن `rounded_rect_svg`، `draw_rounded_rect` و گرد کردن خودکار گوشه‌های تیز پلی‌گون‌ها (`_build_filleted_poly`) برای حذف حس خشک و تیز چهارگوش‌ها.
* **طراحی چندلایه پرجزئیات (`Multi-Layer Detailed Vector Art`):** ارتقای ۳۵ آیکون استاندارد در `engine/vector_motion_library.py` به ۲ لایه مجزا: لایه اصلی (`#FFFFFF` با ضخامت `2.3px`) + لایه جزئیات ظریف داخلی (`DETAIL:` با رنگ `#7A7A80` و ضخامت `1.4px`).
* **ایمپورتر مستقیم فایل‌های خارجی:** افزودن `load_external_svg_paths`، `load_lottie_json_paths` و `find_system_svg` (دسترسی به ۶۴۷ آیکون SVG لینوکس).
* **موتور متن دوگانه بدون باکس خالی (`▯`):** سوئیچ هوشمند بین `NotoSansArabic-Bold` و `DejaVuSans-Bold` برای پشتیبانی هم‌زمان از فارسی `RTL` و علائم `() % $ ✓ ✕` و متون انگلیسی `LTR`.

---

### قالب ثبت برای جلسات بعدی (نسخه V4 و بالاتر)
* **تاریخ / جلسه:** `[توسط ایجنت در جلسه بعد پر شود]`
* **آیکون‌های جدید اضافه‌شده به `VECTOR_ICONS`:** `[...]`
* **ویجت‌های جدید اضافه‌شده به `vector_motion_library.py`:** `[...]`
* **تجربیات و نکات کارگردانی جدید:** `[...]`

---

### Version V4 (2026-10-18): declarative engine `mograph` + three-pass thinking split
**Core problem:** the agent was writing the scenario, drawing, coding, and syncing all at once, and starting from scratch every time (v3's 1300-line pipeline was hard-coded).
* **Declarative JSON spec:** the scenario is written as data, and 19 ready-made elements (icon, text, box, callout, stamp, strike, counter, dot_grid, icon_grid, bars, ring, timeline, pill, divider, arrow, line, circle, path, image) draw it. New elements are added as plugins with `@element`.
* **Word anchors** (`"t": "w:انجام"`): every animation is locked to the narration word instead of a guessed second. Diacritics, ZWNJ, and punctuation are normalised.
* **Exact Trim Path:** the SVG parser covers M/L/H/V/C/S/Q/T/A in both absolute and relative form, true path length is measured and cached (v3 used `approx_len = 460*scale`, so drawing speed was wrong), and multi-part paths are drawn sequentially.
* **Fully vector text** (HarfBuzz + Vazirmatn plus DejaVu/Symbols fallback): sharp under zoom, built-in bidi (`1%`, `@HANDLE`, `۱۲٫۵٪`), and `missing_glyphs` for QA. v3 drew text as raster sprites, which went blurry when the camera zoomed.
* **6× faster bloom** (1/8 and 1/32 resolution pyramid): frame time dropped from 147 ms to 24 ms at 1080×1920.
* **Parallel renderer:** each worker pipes into its own ffmpeg, then `concat -c copy`. Worker count adapts to RAM (with 1 GB, two full-res x264 encoders cause a BrokenPipe, so it falls back to 1 worker automatically).
* **Automatic QA:** lint (glyphs, ids, icons, HUD zone, pacing > 3.5 s with no new event) plus overlaps/out-of-frame measured from real vector geometry, and a contact sheet of every chapter's hero frames.
* **Automatic sound:** a default SFX per element type, ambient bed with piano at chapter changes, VO ducking, and −14 LUFS.
* **Icons:** 50 → 80, with Persian/English tags and search (`mograph icons دارو`), saved in `icons.json`, plus `import_svg_icon`.

**Lessons:**
- Bloom (not vector drawing) was 90% of render time, so always profile before optimising.
- In the overlap check, text that sits *inside* a container (box/divider) by design should not count as a collision.
- Run QA on elements only in their stable state (raw_p ≥ 0.5, not exiting); otherwise the transitions produce false alarms.
- The sandbox environment can reset: always keep code in `/home/user/webapp` and commit regularly.

### Version V5 (2026-10-18): sequences instead of primitives — widgets, morph, thread, post-FX, real-video review
**Core problem:** v4 removed the drawing load, but the agent still assembled every beat from 19 primitives, stacked duplicate counters for changing numbers, left orphan leader lines when a target exited, and only checked the spec — never the real mp4.
* **6 compound widgets** (`quote_rig flow_loop laser_scan invoice strata scale_rig`) in `mograph/widgets.py`: one element = a full sequence with internal beats (highlights, drops, rows, scan, drill, flips) and their own SFX cues.
* **Vector morph** (`morph` element + `icon.morph_to`): resample to 64 pts, greedy sub-path pairing by centroid/size, ring alignment → clean in-betweens (naive index pairing produced crumpled shapes).
* **Value tracks** (`tracks.value`) with automatic scale punch → one counter changes in place.
* **Auto-cascade out + `group`/`group_out`**: dependents inherit their target's exit.
* **Font roles** (`font: title/document/handwritten/nastaliq/counter/subtitle/note/ui/body`) over 24 Persian fonts; Vazirmatn first in the fallback chain.
* **Thread transition**: S-curve comet line from the focal element of chapter N into N+1, tail retracts after landing, ring pulse at the target.
* **Post-FX**: vignette, film grain, exposure + bloom pulse on stamp slams.
* **QA**: unknown-key typo detection (difflib), orphan-callout lint, cross-fade overlap check, widget-aware pacing; `review` command checks the real video (black/frozen, LUFS, true peak, silence) and runs automatically after `render`.
* **VO alignment** (`mograph align`): faster-whisper word timestamps snapped to the script's spelling.
* **Open-source evaluation**: Remotion / OpenMontage / SeeCut / NullMotion reviewed. Not integrated (Node + headless Chrome is too heavy for a 1 GB sandbox; licences: Remotion company licence, SeeCut PolyForm-Noncommercial, NullMotion unlicensed). Their good ideas were adopted natively instead: mandatory post-render self-review (OpenMontage), word-level alignment (WhisperX-style), director "skills" as docs + rigs.

**Lessons:**
- Role names must be resolvable everywhere a font path is expected (`F()` passed `"body"` as a path → crash); resolve roles in `_font` itself.
- Pacing lint must count widget-internal beats, otherwise rich widgets look "dead" to QA.
- A connecting line that stays on screen after the transition reads as a bug — always retract it.
- Keep `out/` creation inside the CLI paths; after a sandbox reset, ignored folders are gone.

### Version V6 (2026-10-18): real narrator voice + realistic shots, still one engine
**Core problem:** the vector engine had SFX/ambient but no narrator, and some beats need photoreal material that code can't draw.
* **`mograph voice`** (ElevenLabs eleven_v3 with character timestamps / Gemini TTS via Interactions API with generateContent fallback / offline draft): subtitles = script, per-line cache by content hash, previous/next-text continuity, whisper or syllable alignment fallback, piecewise-linear retiming of chapters/cameras/numeric times, `audio.vo` ducking.
* **`mograph omni`** (Gemini Omni Flash, REST, no SDK): tasks edit / first_last / from_frame / text; the engine renders its own frames as input; dry run = approval gate; content-hash cache; plates rendered under or over the vector world with cross-fade; HUD/subs stay vector; missing clip → vector fallback.
* **`video` element** (OpenCV sequential reader, blend screen/multiply, noir grade, rounded/border).
* **`mograph keys`**: env → project → ~/.mograph; MissingKey prints a Persian message for the user (exit 3).
* **`tools/mock_apis.py`**: local fake of all 3 APIs → every integration is testable for free.
* Reviewed HyperFrames (Apache-2.0, Node + Chrome) and gbro-collage-broll (MIT, Omni skill): integrated their *ideas* (gates, first/last frames, English prompts, "keep everything else the same"), not their stacks.

**Lessons:**
- Never make paid generation implicit: dry run by default, `--yes` only after the user approves the plan.
- Text belongs to the vector layer; generative video garbles Persian letters.
- Retime with the real voice, not the guessed one — word anchors make this nearly free.

### V7 — styles, event-locked sound, fonts, keys file (2026-10)
* **Sound engine `sound.py`:** each element type has a score function that reuses the renderer's timing (ease
  derivative, spring contact frame, the exact frames where counter digits change) → SFX lands on the visual
  event (measured: stamp −1 ms, counter +6 ms). Stereo pan by x, small room reverb, soft-clip bus. `mograph cues`.
* **Styles:** `meta.style` lineart / swiss / kinetic + palettes + `@tokens`; Swiss grid (`col`/`row`, RTL), wipe/slide
  transitions with an accent band; Swiss elements block/headline/rule/big_number/index/disc; `kinetic` element.
* **Fonts:** every draw_text has an explicit role; fallback chain starts with Aria (Persian symbols); ZWNJ no
  longer switches fonts; QA warns on fallback characters.
* **Keys file** handover (`keys import`, auto-scan) + **`voice.provider: file`** (platform TTS + word timestamps).

**Lessons:**
- The user hears desync before seeing it: sound must be derived from the same maths as the motion, never from start times.
- A user who loves certain fonts notices one default-font glyph immediately → explicit roles everywhere + QA.
- Real voice is possible without the user's key if the platform has a TTS tool → `provider: file`.
- Swiss frames go "dead" in long silences — add a slow camera push or a beat during pauses (review catches it).

### V7.1 — glyph safety + scene-aware contrast (2026-10)
* **Bug (user review):** Swiss/kinetic reveals cut Persian dots, kasra and ی/ن tails (fixed `size*1.5` mask
  rectangles); karaoke spoken word turned white on the cream Swiss paper and vanished.
* **Fix:** `text.ink_box` / `mask_rect` (real outlines) in every mask; ink-aware leading; `contrast.py` (WCAG) +
  HUD samples the real rendered pixels per zone; `styles.fit_palette` adapts tokens to the actual bg.
* **Guard:** QA text audit (`glyph clipped`, `low contrast`) — proven to catch the old bugs on the old code.

**Lessons:**
- Never size anything around Persian text with fixed ratios of the font size — measure the ink.
- A colour is only correct relative to what is behind it; decide it from pixels, not from the spec.
- Every bug the user finds must become an automatic QA check, so no agent can reintroduce it.

### Template for next sessions (V8+)
* **Date / session:** `[...]`
* **New icons / elements:** `[...]`
* **Bugs fixed / new lessons:** `[...]`

## v8 Studio Horizon (2026-10-18)
- styles.world_for(): per-chapter style worlds (palette fitted to that chapter's bg, fonts, HUD, sound, post-FX);
  chapters without a transition use their own style default. text.use_roles(): switch roles per chapter without cache
  clears (layout cache keyed by the concrete font file).
- render: region transitions paint each world's bg inside its region; portal (iris / slab, focal x,y or `from` id);
  bg colour + FX blended across fades; bloom forced off on light grounds.
- horizon.py: waffle_grid, swiss_bars (inside/outside type by final length + contrast.ensure), versus_table, waveform,
  roll_number (mechanical carry, static layout bbox, column auto-fit). typo: kinetic masonry (row partition search,
  equal width, ink-exact row stacking) + highlight_sweep (slab + inverted overdraw).
- z parallax relative to the chapter's rest view; QA bboxes parallax-aware. scene.vo_env (VO RMS 60 Hz / subtitle cadence).
- QA: per-chapter HUD safe zones & font roles, new keys, nested row/item glyph checks, AUDIT_MUTE for transition frames,
  odometer wheels and marker overdraw (intentional clips). Old examples render pixel-identical to v7.2.
- Lesson: sandbox resets lose work — commit after each step and keep a tarball off-box.


## v9 Studio Apex (2026-10-18)
- ctypes libharfbuzz shaper (hbc.py): uharfbuzz-compatible subset; fontTools BasePen with glyphSet (composite glyphs!) —
  bug found: Vazirmatn medial forms are composites → empty outlines until pen.glyphSet is set.
- draw_text gained style= (outline / extrude_3d / knockout) + kashida= (tatweel bar scaled, glyphs to the left shifted).
  Pitfall: never reuse the name `st` inside draw_text (word_style dict shadows it).
- promo style + apex.py (backdrops, bokeh / warp_stars / grid, whip blur via cv2.filter2D on an offscreen layer, whip_pan).
- promo.py widgets; matte.py (MediaPipe selfie .onnx 462 KB; chroma auto-detect must require a uniform saturated border —
  a blue brick wall was a false positive); talking_head with behind-layer, rim, PIP, punch-in on `_cuts`.
- edit.py (max_pause / dedupe / cut ranges, zero-crossing + raised-cosine splices); spectral carve (complementary bands,
  perfect reconstruction); analyze.py + b:N / bar:N; meta.aspect.
- Sandbox lesson: the sandbox can be reset mid-turn → pycairo must be re-installed (`pip install pycairo`) and
  git-ignored assets (voice/, media/, *.voiced.json) are lost — v9 whitelists the shipped demo assets in .gitignore.

## v11 Studio Pantheon (2026-10-19)
- Goal: every reference reel (ref01–09: talking-head beauty edits, editor showcases, neon/noir line art, radial
  formula) re-creatable from a JSON spec + the user's checklist (tracking, planar, roto, edge glow, hue keys,
  tracked callout, depth masking, speed ramps).
- fx.py `footage`: time map built once per element (ramps / remap / beat_ramp / freeze integrate speed → src_t);
  normalize_footage converts ALL nested sub-times to clip-local at scene build (word anchors resolved there).
  Clip audio follows the same time map. Hue shift needs a saturation floor (smin 110) or skin shifts too.
- track.py: LK + forward-backward error, RANSAC homography for planar (chained to the reference frame = no drift),
  mediapipe 1.1.1 tasks (CPU delegate; needs apt libegl1), One-Euro smoothing; tracks are normalised (u,v) so placement,
  zoom and reframe are applied at sample time (fx.resolve_follow).
- render: containers (`device_frame children`) are flattened by showcase.expand_children; children carry `parent`
  and are drawn in the parent's virtual-screen transform; each frame's border is drawn right after its own children
  (otherwise borders paint over later frames). social_outro forced to layer 1000.
- QA: contrast audit is pixel-based → faint decorative type needs `decor: true`; text over moving footage should
  use a shadow or `hud.subs.box`. `gen` is a legal key on every element (prompts.py).
- genimage: auto_trace uses cv2.ximgproc.thinning (needs opencv-contrib); `color: "source"` collided with hex parsing
  → renamed `color_mode`. depth_plate falls back to saliency + grabCut when no person is present.
- Backups: keep only 720p proxies of references, MP3 voices, squash git history; zips of inputs are re-downloadable.
- Sandbox lesson (again): resets lose pip packages → `bash setup_env.sh` restores pycairo/mediapipe/onnxruntime + libegl1.

## v12 Studio Cut (2026-10-19)
- Trigger: user judged v11 demo not professional; supplied a raw 480p 20 s take (laptop reselling, 3 camera setups).
- cut.py: EDL from word timestamps (speech runs, shot-aware, beat snap) → subtitles re-timed into edited time so
  word anchors survive the edit. Last segment must keep playing into the tail (frozen frame otherwise).
- Audio splice needs ≥ 30 ms raised-cosine fades (12 ms clicked per external review).
- Reviewer feedback loop (analyze_media_content with gemini-3.1-pro) worked well: blurry shot transitions felt dated
  → whip + hard punch; skin went magenta under auto levels → skin_correct.
- Caption contrast audit must ignore stroke/shadow passes (false lows); outlined captions are muted from the audit.
- Springs start at 0 → cairo scale(0) = singular matrix: always clamp scale (reelfx._S).
- Paid music generation unavailable (credits) → beat.py procedural bed; first version judged harsh (hats, clipping):
  darker/quieter hats, swing, kick sidechain, vocal-pocket EQ, HP sweep + 1/16 gap before the drop, headroom.
- Sandbox reset mid-turn twice: restore from the user's backup tar, re-apply only uncommitted patch. Commit often.
