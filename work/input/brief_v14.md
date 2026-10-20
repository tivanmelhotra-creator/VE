# کالبدشکافی موشکافانهٔ ۷ کات طلایی `@novadent.ir` (نسخهٔ ۱۰۰ از ۱۰۰) + موتور تضاد فونت و وزن کلمات درون‌متنی (`v13.9 @novadent Master`) + پرامپت جامع `v14` برای مدل رنک ۱

این سند بر اساس نقد دقیق، ریزبینانه و کاملاً درست شما روی ۷ اسکرین‌شات ویدیوی Before / After پیج `@novadent.ir` و نکتهٔ کلیدی شما دربارهٔ **«تضاد فونت، ضخامت/نازکی، اندازه و رنگ یک کلمه در میان سایر کلمات متن»** به‌روزرسانی شده و شامل ۳ بخش است:
1. **کالبدشکافی مو‌به‌موی هر ۸ تکنیک ظریف در ۷ اسکرین‌شات `@novadent.ir`** (از نیم‌دایرهٔ نقطه‌چین شیشه‌ای مات و پنل‌های ذوزنقه‌ای، تا تابش نور قرمز نوار سی‌دی روی یقهٔ سفید و انتقال آن به دست، بیرون زدن حروف از مربع خطی + لوزی ۴۵ درجه، رفتن حرف «ز» پشت صندلی با تابش نور طلایی، زیرنویس ژله‌ای با افکت صدای ژله‌ای، هالهٔ دو لایهٔ قرمز پشت صندلی، شش‌ضلعی گلس‌مورفیسم و چرخش سه‌بعدی تایتل دور سر).
2. **گزارش دقیق پیاده‌سازی همین ۸ تکنیک در موتور خودمان (`motion_creator_brain_sdk_v13_9_novadent_100_master.zip`) و مقایسهٔ شات‌به‌شات با ۷ اسکرین‌شات اصلی شما.**
3. **پرامپت جامع و تکمیل‌شدهٔ انگلیسی (`Co-Engineering Brief: Auditing v13.9 & Building v14 @novadent 100/100 Master`)** آمادهٔ کپی و ارسال به مدل رنک ۱.

---

## بخش اول: کالبدشکافی موشکافانهٔ ۸ تکنیک طلایی در ۷ اسکرین‌شات `@novadent.ir` (چرا امتیاز این کار ۱۰۰ از ۱۰۰ است؟)

همان‌طور که اشاره کردید، در نگاه اول شاید کسی فکر کند این ویدیو فقط چند خط و زیرنویس ساده دارد، اما رازِ شاهکار بودن آن در **۸ جزئیات میکروسکوپیِ نوری، تایپوگرافی و فیزیک حرکت (`Ease-In-Out` و `Jelly Squash-Stretch`)** است که همگی را در نسخهٔ **`v13.9`** پیاده‌سازی کردیم:

### ۱. قانون اول (نکتهٔ طلایی شما): تضاد شدید یک کلمه در میان متن (`Inline Multi-Font, Weight, Size & Color Contrast`)
* در تمام بخش‌های این ویدیو (چه در تایتل‌های بالای سر، چه داخل دایره‌ها و شش‌ضلعی‌ها، و چه حتی داخل یک خط جمله یا زیرنویس)، **هیچ‌وقت تمام کلمات با یک فونت، یک ضخامت و یک اندازه نوشته نمی‌شوند!**
* **فرمول طلایی:** کلمات اطراف با **فونت بسیار نازک و ظریف (`Ultra-Thin / Light` مثل `IRANYekanThin` و `IRANYekanLight`)**، اندازهٔ کوچک‌تر و رنگ سفید/نقره‌ای ملایم نوشته می‌شوند؛ اما **یک کلمهٔ کلیدی در میان آن‌ها** ناگهان:
  1. **فونتش کاملاً عوض می‌شود** (یا به فونت خوشنویسی/فانتزی مثل `Delbar-Bold` / `Miremad` تغییر می‌کند، یا به فونت فوق‌سنگین مثل `IRANYekanExtraBlack2` / `KalamehFaNum-Black`)،
  2. **ضخامتش (Bolder) و اندازه‌اش (`1.4x` تا `2.2x` بزرگ‌تر)** می‌شود،
  3. **رنگش متمایز می‌شود** (طلایی کهربایی `#FBBF24`، قرمز یاقوتی `#FF2E42` یا سفید درخشان با هالهٔ نور).
* **مثال‌های عینی در ۷ اسکرین‌شات شما:**
  * اسکرین‌شات ۱: `آمپول‌های` (نازک، ظریف و کوچک‌تر) در برابر **`لاغری`** (فوق‌ضخیم، خوشنویسی کشیده و ۲ برابر بزرگ‌تر).
  * اسکرین‌شات ۲ و ۳ (هستهٔ سفید وسط سی‌دی): **`بیشتر از`** (فوق‌ضخیم و مشکی) در برابر `این که` (فوق‌نازک `Thin` و خاکستری روشن در زیر آن).
  * اسکرین‌شات ۴: `مورد اول` (نازک و سفید) در برابر **`خشکی`** (خوشنویسی طلایی سه‌بعدی بسیار درشت) در برابر `دهان` (سفید متوسط)؛ و حتی در زیرنویس ژله‌ای: `و` (نازک) + **`دهان`** (درشت و بولد) + `رو خشک میکنن` (نازک).
  * اسکرین‌شات ۵: `ماهایی که` (نازک و سفید) در برابر **`زدیم`** (فوق‌سنگین، ۲ برابر بزرگ‌تر و طلاییِ نورانی) در برابر `چیکار / کنیم` (نازک و کوچک).
  * اسکرین‌شات ۷ (داخل شش‌ضلعی شیشه‌ای): `از آدامس‌های` (نازک و ریز) در برابر **`بدون قند`** (فوق‌ضخیم و درشت در مرکز) در برابر `برای تحریک بزاق استفاده کنید` (بسیار نازک و ریز در پایین).

---

### ۲. اسکرین‌شات اول (`211857.png` — نیم‌دایرهٔ نقطه‌چین شیشه‌ای مات + برش ذوزنقه‌ای تصاویر پشت گوینده)
* **نیم‌دایرهٔ نقطه‌چین طلایی با پس‌زمینهٔ گلس‌مورفیسم (`Frosted Glass Semi-Circle Dome`):** بالای سر گوینده یک نیم‌دایره نیست که توخالی باشد؛ بلکه فضای داخل این نیم‌دایره **شیشه‌ای مات (`Backdrop Blur Glassmorphism`)** است و دور قوس آن یک **نقطه‌چین/خط‌چین طلایی ظریف و درخشان** کشیده شده که عبارت دو فونتی `آمپول‌های / لاغری` روی آن نشسته است.
* **برش ذوزنقه‌ای/مورب پس‌زمینه برای دو محصول (`Split Trapezoidal Product Panels`):** به محض نام بردن از دو دارو، فضای پشت شانهٔ گوینده (تا خط افقی سفید در تراز شانه) با یک برش زاویه‌دار ذوزنقه‌ای به ۲ قاب تقسیم می‌شود: سمت چپ تصویر و واترمارک `SPARTINA / اسپارتینا` و سمت راست تصویر قلم داروی `OZEMPIC / اوزمپیک` که سرِ قلم کمی از کادر بیرون زده است.
* **ویجت ساخته‌شده در `v13.9`:** `@element("novadent_split_dome")` در `mograph/creative_reels.py`.

---

### ۳. اسکرین‌شات دوم و سوم (`211908.png` و `211917.png` — نوار سی‌دی شیشه‌ای + تابش نور قرمز روی یقه و انتقال به دست!)
* **ساختار دونات شیشه‌ای تیره (شبیه نوار سی‌دی / وینیل):** یک دایرهٔ شیشه‌ای تیره که وسطش یک دایرهٔ سفید کوچک‌تر (`بیشتر از / این که`) قرار دارد.
* **خم شدن واقعی حروف فارسی روی قوس دایره (`Polar-Warped Curved Persian Text`):** جملهٔ `روی کاهش وزن اثر بذارن!` واقعاً حرف‌به‌حرف (با حفظ اتصال حروف فارسی!) روی قوس نیم‌دایرهٔ بالایی خم شده است.
* **فقط نیمهٔ بالاییِ لبهٔ بیرونی قرمز است + تابش نور قرمز روی یقهٔ سفید گوینده (`Optical Red Light Spill on Collar`):**
  * در اسکرین‌شات دوم (`211908.png`)، خط نوری قرمز **فقط روی نیم‌دایرهٔ بالاییِ لبهٔ بیرونی** وجود دارد (پایین دایره هیچ خط قرمزی ندارد!) و دقیقاً از همان لبهٔ قرمز بالا، یک **نور قرمز نرم روی بخش‌های سفید یقه و گردن خانم گوینده تابیده است (`Screen Light Bleed`)**!
* **چرخش ۱۸۰ درجه‌ای با `Ease-In-Out` و انتقال نور قرمز به روی دست گوینده (`211917.png`):**
  * در کات بعدی، همان دایره با حرکت نرم `Ease-In-Out` می‌چرخد: این بار **قرمزیِ بالا کاملاً حذف می‌شود** و نوشتهٔ بالا به رنگ خاکستری تیرهٔ برجسته (`Embossed Dark-on-Dark`) درمی‌آید؛ در عوض، **نیمهٔ پایینیِ دایره قرمز می‌شود**، جملهٔ `روی سلامت دهان و دندونتون` به رنگ طلایی روشن می‌شود، و **تابش نور قرمز این بار از بالای یقه رفته و دقیقاً روی دست و آستین سفید گوینده در پایین دایره می‌افتد!**
* **ویجت ساخته‌شده در `v13.9`:** `@element("novadent_cd_dial")` + موتور `_polar_arc_surface` (خم‌کنندهٔ قطبی متون فارسی با `cv2.remap`).

---

### ۴. اسکرین‌شات چهارم (`211936.png` — مربع خطی کج با بیرون‌زدگی حروف + لوزی ۴۵ درجه + زیرنویس ژله‌ای با صدای ژله‌ای)
* **مربع برداری کج (`-16°`) و بیرون زدن عمدی حروف از خط کادر:** بالای سر گوینده یک بردار مربعی با گوشه‌های گرد و زاویهٔ کج قرار دارد که در آن، حرف **«م»** از عبارت `مورد اول` از خط راست مربع بیرون زده، و دنبالهٔ حرف **«خ»** و کاسهٔ حرف **«ی»** از کلمهٔ طلایی و سه‌بعدی **`خشکی`** از هر دو طرف خط مربع بیرون زده‌اند تا حس عمق و رهایی از کادر بدهند!
* **لوزی ۴۵ درجه (`45° Rotated Diamond Badge ◇`) در گوشهٔ بالا-چپ:** قاب تصویرِ خشکی دهان یک دایره نیست، بلکه **یک مربع با گوشه‌های گرد است که ۴۵ درجه چرخیده و شکل لوزی (`◇`) با کادر دولایهٔ سفید** پیدا کرده و روی گوشهٔ چپِ مربع اصلی نشسته است.
* **زیرنویس گلس‌مورفیسم ژله‌ای (`Elastic Jelly Glassmorphic Subtitle`) + افکت صدای ژله‌ای (`sfx="jelly"`):**
  * کپسول زیرنویس روی سینه (`و دهان رو خشک میکنن`) یک شیشهٔ مات تیره با **دو هلال نوری فیروزه‌ای-سبز (`#2DD4BF`) در دو لبهٔ چپ و راست** است.
  * با کوتاه و بلند شدن متن هر جمله، این کپسول شیشه‌ای مثل **ژله (`Jelly Squash & Stretch`)** کش می‌آید یا جمع می‌شود و هم‌زمان یک **افکت صدای ژله‌ای/حبابی نرم (`syn("jelly")` در `mograph/sound.py`)** پخش می‌کند!
* **ویجت‌های ساخته‌شده در `v13.9`:** `@element("novadent_diamond_square")` + استایل زیرنویس `"style": "novadent_jelly"` در `captions.py` + صدای `"jelly"` در `sound.py`.

---

### ۵. اسکرین‌شات پنجم (`212019.png` — رفتن حرف «ز» از «زدیم» به پشت صندلی + تابش نور طلایی روی صندلی)
* در جملهٔ `ماهایی که / زدیم / چیکار / کنیم`، کلمهٔ **`زدیم`** بسیار درشت، بولد و طلایی است و طوری پشت لایهٔ دوربری‌شدهٔ گوینده و صندلی (`behind: "clip"`) قرار گرفته که **لبهٔ راست حرف «ز» به پشتِ پشتیِ مشکی صندلی رفته** و هم‌زمان **نور طلاییِ کلمهٔ `زدیم` روی لبهٔ چرم مشکی صندلی و شانهٔ گوینده می‌تابد (`novadent_edge_light_spill`)**!

---

### ۶. اسکرین‌شات ششم (`212032.png` — نوار سی‌دی دو لایهٔ قرمز تیره و قرمز روشن پشت سر و صندلی مثل هاله)
* پس‌زمینهٔ اتاق کاملاً تاریک و دودی می‌شود و درست **پشت سرِ گوینده و پشت صندلی (`behind: "clip"`)**، یک دایرهٔ دو لایه مثل هاله باز می‌شود:
  * **لایهٔ بیرونی:** حلقهٔ قرمز تیره (`#3B070B`) با خط دور نئونی قرمز روشن و هالهٔ نوری قرمز که به دیوار تاریک پشت پخش شده است.
  * **لایهٔ درونی:** دایرهٔ قرمز روشن (`#DC1E28`) که داخل نیمهٔ بالایی آن عبارت `اصلاً` (نازک) + **`ســــــخت`** (کشیده و درشت) + `نیست` (نازک) درست بالای موهای گوینده می‌درخشد.
* **ویجت ساخته‌شده در `v13.9`:** `@element("novadent_red_halo")`.

---

### ۷. اسکرین‌شات هفتم (`212043.png` — شش‌ضلعی گلس‌مورفیسم واقعی با نور راس بالا + چرخش سه‌بعدی تایتل دور سر)
* **شش‌ضلعی شیشه‌ای مات (`novadent_glass_hex`):** روی سینه و دست گوینده یک **شش‌ضلعی با گوشه‌های گرد** باز می‌شود که داخل آن با `_paint_backdrop_blur` **واقعاً لباس سیاه‌وسفید زیر خودش را تار و شیشه‌ای (`Glassmorphism`) کرده است**، دورش خط طلایی ظریف دارد، **بالای راس آن یک نقطهٔ نورانی سفید-طلایی (`Top Vertex Specular Bloom`)** می‌درخشد، و با حرکت ابریشمی `Ease-In-Out` باز و بسته می‌شود.
* **موشن چرخش سه‌بعدی تایتل دور سر گوینده و فرود پرنده‌وار به زیرنویس (`@element("orbiting_title_3d")`):**
  * تایتل روی یک مدار بیضوی سه‌بعدی دور سر گوینده می‌چرخد (در نیم‌دور پشتی با `half: "back"` از **پشت سر گوینده** رد می‌شود و در نیم‌دور جلویی با `half: "front"` از **جلوی صورت** عبور می‌کند!) یا با حالت `mode: "bird_swoop_to_sub"` مثل یک پرنده از پشت شانهٔ گوینده قوس می‌زند و نرم داخل کادر زیرنویس پایین فرود می‌آید!

---

## بخش دوم: پرامپت جامع و نهایی انگلیسی برای ارسال به مدل رنک ۱ (**V14 @novadent 100/100 Master & Visual-DNA Engine**)

فایل زیپ جدید و کم‌حجم **`motion_creator_brain_sdk_v13_9_novadent_100_master.zip` (`24.6 MB`)** را به همراه پرامپت انگلیسی زیر برای مدل رنک ۱ بفرستید تا هم کدهای `v13.9` ما را بررسی و پولیش کند و هم قابلیت‌های تکمیلی `v14` را روی آن بسازد:

```markdown
# CO-ENGINEERING BRIEF: Auditing `mograph v13.9 @novadent 100/100 Master` & Building **v14 Studio Perfection**
*(Forensic 8-Law `@novadent.ir` Aesthetic + Inline Multi-Font/Weight/Size/Color Word Contrast + Polar-Warped Curved Persian Arc Text + Optical Collar/Hand Light Spill + Real Backdrop-Blur Glassmorphic Hexagons & Jelly Subtitles + 3D Head-Orbiting Titles)*

Hello! Our personal workspace agent has upgraded the SDK from `v12` to **`v13.9 @novadent 100/100 Master`** (attached as `motion_creator_brain_sdk_v13_9_novadent_100_master.zip`) after performing a deep forensic analysis of 7 high-resolution screenshots from the User's favorite `@novadent.ir` Before/After showcase reel (`Screenshots 211857..212043`) alongside our **Zero-Box Visual-DNA Constitution**.

We want a true two-way co-engineering collaboration:
1. **First, read `USER_TASTE_AND_DNA_LAWS.md` and `CREATIVE_REELS_PLAYBOOK.md` at the root of the SDK**, and audit, debug, and optimize everything our personal agent built in `v13.5..v13.9` (`mograph/creative_reels.py`, `mograph/captions.py`, `mograph/sound.py`, `mograph/fx.py`, `mograph/visual_dna.py`, `mograph/reelfx.py`, `mograph/extract3d.py`, `mograph/qa.py`, and `templates/15..17`).
2. **Second, implement the 7 next-level `v14` upgrades detailed in Part C below** so our engine achieves 100/100 broadcast perfection on every single detail!

---

## PART A — THE 8 FORENSIC LAWS OF THE `@novadent.ir` 100/100 AESTHETIC (`CREATIVE_REELS_PLAYBOOK.md`)
*(MANDATORY AESTHETIC CONSTITUTION — NEVER VIOLATE OR REVERT)*

1. **Law A1 — Extreme Inline Word Contrast (`تضاد شدید فونت، ضخامت/نازکی، اندازه و رنگ یک کلمه در میان متن`):**
   * **Never render all words in a title, badge, or subtitle with the same font, weight, size, and color!**
   * Always pair **Ultra-Thin / Light weights (`IRANYekanThin.ttf` / `IRANYekanLight.ttf`)** at `1.0x` scale in neutral white/silver (`#E2E8F0`) right next to **One Hero Word (`یک کلمهٔ کلیدی در میان متن`)** that is **much bolder, `1.38x–2.2x` larger, in a contrasting font family** (Calligraphic `Delbar-Bold.ttf` / `Miremad.ttf` or Ultra-Heavy `IRANYekanExtraBlack2.ttf` / `KalamehFaNum-Black.ttf`), and **in a glowing accent color** (`#FBBF24` gold, `#FF2E42` crimson, or `#2DD4BF` cyan-teal)!
2. **Law A2 — Dashed Golden Semi-Circle + Real Glassmorphic Backdrop-Blur Dome + Split Trapezoidal Panels (`novadent_split_dome` — Screenshot `211857`):**
   * Behind the segmented speaker (`behind: "clip"`), split the upper wall above shoulder level (`y_split`) into two angled trapezoidal product panels (`SPARTINA / اسپارتینا` left, `OZEMPIC / اوزمپیک` right with a 2.5D cutout pen breaking out of the top), and overlay a **semi-circular dome with real Glassmorphic Backdrop-Blur (`_paint_backdrop_blur`)** bounded by a **glowing golden dashed/dotted arc** containing thin `آمپول‌های` + bold calligraphic **`لاغری`**.
3. **Law A3 — Dark Vinyl/CD Donut Ring + Polar-Warped Curved Persian Text + Dynamic Collar-to-Hand Red Light Spill (`novadent_cd_dial` — Screenshots `211908` & `211917`):**
   * Dark-glass CD donut ring on the speaker's chest with a white center disc (`بیشتر از` bold black + `این کـه` thin light gray).
   * **State 1 (`t < rotate_at`):** Only the **top half of the outer circumference** has a glowing red rim line (bottom half has ZERO red!), top polar-warped curved Persian text (`روی کاهش وزن اثر بذارن!`) glows white, and a directional `OPERATOR_SCREEN` red light spill physically tints the speaker's **white shirt collar and neck** above the circle!
   * **State 2 (`t >= rotate_at`):** Smooth `Ease-In-Out` (`cubic-bezier(0.65, 0, 0.35, 1)`) `180°` rotation! Top outer rim loses red, top curved text becomes **dark embossed charcoal (`embossed=True`)**, bottom outer rim glows red, bottom curved text (`روی سلامت دهان و دندونتون`) glows golden-yellow, and the **red optical light spill shifts downward onto the speaker's hands and white shirt cuff**!
4. **Law A4 — Tilted Rounded-Square Wireframe + Border-Breaking 3D Golden Calligraphy + 45° Rotated Diamond (`◇`) Badge (`novadent_diamond_square` — Screenshot `211936`):**
   * A `-16°` tilted rounded-square vector outline above the speaker's head where the **`م` of `مورد اول` intentionally extends outside the right wireframe border**, and the right/left tips of giant 3D-shaded golden **`خشکی`** (`Delbar-Bold.ttf`) **break across both borders**, paired with a **45° rotated rounded-square Diamond (`◇`) photo badge** on the top-left corner!
5. **Law A5 — Elastic Jelly Glassmorphic Subtitle Bar + `"jelly"` Sound Synthesizer (`hud.subs.style = "novadent_jelly"` in `captions.py` & `sound.py` — Screenshot `211936`):**
   * Real Backdrop-Blur Glassmorphic dark-emerald capsule on the speaker's chest with **glowing cyan-teal (`#2DD4BF`) crescent rim highlights on the left and right curved end-caps**.
   * As subtitle text length changes, the capsule **elastically squashes and stretches (`Jelly Squash & Stretch` with volume conservation `h ~ w^-0.42`)** while emitting a synthesized **frequency-modulated watery/rubbery `"jelly"` SFX** in `sound.py`!
6. **Law A6 — Behind-Chair Occlusion & Self-Illuminating Golden Light-Wrap (`novadent_chair_glow_text` & `novadent_edge_light_spill` — Screenshot `212019`):**
   * Stacks `ماهایی که` (thin white) + giant extra-bold golden **`زدیم`** (`IRANYekanExtraBlack2.ttf`) + `چیکار / کنیم` (thin white) behind the speaker & chair (`behind: "clip"`) so the right tip of **`ز`** in `زدیم` tucks **behind the black chair backrest** and casts a warm golden `OPERATOR_SCREEN` light-wrap onto the chair/shoulder edge!
7. **Law A7 — Concentric Two-Tone Red CD Halo Behind Speaker & Chair (`novadent_red_halo` — Screenshot `212032`):**
   * Darkens the room background to moody charcoal (`#0B0C10`) and places a **two-layer concentric red CD halo** (outer dark-crimson ring `#3B070B` with bright neon-red outer stroke `#FF2E3B` + wide red wall glow; inner vivid-crimson disc `#DC1E28`) behind the speaker's head and chair, displaying `اصلاً` (thin) + stretched **`ســــــخت`** + `نیست` (thin).
8. **Law A8 — True Backdrop-Blur Glassmorphic Hexagon + Top Vertex Specular Light + 3D Head-Orbiting Titles (`novadent_glass_hex` & `orbiting_title_3d` — Screenshot `212043`):**
   * `novadent_glass_hex`: Real `_paint_backdrop_blur` inside a rounded hexagon over the speaker's chest (visibly blurring the speaker's clothes underneath), with a `2.2px` golden-amber border, a **bright white-gold radial specular light bloom on the top vertex**, and silky `Ease-In-Out` open/close animation.
   * `orbiting_title_3d`: Orbits a 3D title around the speaker's head (`half="back"` behind the head + `half="front"` in front of the head) or swoops like a bird from behind the speaker down into the subtitle slot (`mode="bird_swoop_to_sub"`).

---

## PART B — WHAT OUR PERSONAL AGENT ALREADY IMPLEMENTED IN `v13.9` (PLEASE AUDIT & REFINE)

Please inspect and refine these modules in `motion_creator_brain_sdk_v13_9_novadent_100_master.zip`:
1. **`mograph/creative_reels.py`:**
   * `_paint_backdrop_blur`: Real Cairo `ctx.get_target()` surface sampling + `cv2.GaussianBlur` + glass tint clipped to any polygon/dome/hexagon path.
   * `_polar_arc_surface` & `draw_curved_arc_text`: Renders shaped Persian text into a flat strip (preserving 100% of cursive ligatures) and polar-warps it (`cv2.remap`) onto a circular arc (`top_arc` or `bottom_arc`, active glowing or `embossed=True` dark-on-dark engraved).
   * `draw_mixed_font_line`: Inline multi-font, multi-weight (`IRANYekanThin` / `IRANYekanLight` vs `Delbar-Bold` / `IRANYekanExtraBlack2`), multi-size (`1.38x–2.0x`), and multi-color Persian sentence renderer.
   * All 7 `@novadent.ir` master widgets (`novadent_split_dome`, `novadent_cd_dial`, `novadent_diamond_square`, `novadent_chair_glow_text`, `novadent_edge_light_spill`, `novadent_red_halo`, `novadent_glass_hex`, `orbiting_title_3d`) + `compose_before_after_frame` & `render_before_after_video`.
2. **`mograph/captions.py` & `mograph/sound.py`:**
   * `draw_novadent_jelly`: Real backdrop-blur glassmorphic capsule with cyan-teal crescent end-caps (`#2DD4BF`), volume-preserving Jelly Squash-and-Stretch physics, and inline `draw_mixed_font_line` word contrast.
   * `syn("jelly")` in `sound.py`: Frequency-modulated elastic watery/rubbery squash-stretch sound effect automatically scored on every subtitle transition when `hud.subs.style == "novadent_jelly"`.
3. **`mograph/fx.py` & `mograph/render.py`:**
   * Added `"headroom": px` (seamless top room-wall reflection/blur extension so speakers filmed too close to the top of the frame get natural studio headroom for behind-speaker titles) and `"bg_dim"` / `"bg_tint"` (moody studio background grading behind the segmented speaker).
   * Fixed `render.py` worker default (`workers=1` when `footage` is present) so OpenCV/ONNX never hits multiprocessing fork deadlocks.

---

## PART C — THE 7 UPGRADES TO BUILD IN `mograph v14 Studio Perfection`

Building directly on top of `motion_creator_brain_sdk_v13_9_novadent_100_master.zip`, please implement these **7 upgrades**:

### Upgrade 1. Stateful Cross-Phrase Jelly Width Interpolation & Per-Word Elastic Ripple in `draw_novadent_jelly` (`mograph/captions.py`)
* Right now `draw_novadent_jelly` computes `jelly_osc` around `target_w` of the current subtitle `sub`.
* Upgrade `draw_novadent_jelly(ctx, W, H, cfg, sub, t, alpha, scene=None)` so it looks up the **previous subtitle's exact pixel width `w_prev`** (from `scene.subs[idx - 1]`) and smoothly interpolates `w(t)` from `w_prev` to `target_w` at the transition boundary `sub["t"]` using an under-damped spring (`stiffness=380, damping=14`) with exact area conservation (`h(t) = target_h * (target_w / w(t))**0.38`)!
* When the capsule shrinks from a long sentence to a 2-word phrase, it visibly squashes horizontally and plumps vertically with a higher-pitched `"jelly"` sound; when it expands from a short phrase to a long sentence, it stretches wide and thin before settling with a deeper `"jelly"` bounce!

### Upgrade 2. Universal Inline Mixed-Font Markup Across ALL `mograph` Text Elements (`mograph/text.py`)
* Bring the inline multi-font/multi-weight/multi-scale contrast power of `draw_mixed_font_line` directly into `mograph/text.py` (`layout_text` & `draw_text`) so ANY element in `mograph` supports inline contrast tokens:
  * Plain words in a `"mixed_contrast": true` element use `thin_font` (`IRANYekanLight.ttf`, `1.0x` scale, `@ink`),
  * `**word**` automatically switches that word to `callig_font` (`Delbar-Bold.ttf`, `1.45x` scale, `@accent` with soft radial glow) and shifts its baseline neatly so the larger calligraphic word sits in harmony with the thin words!
  * `__word__` automatically switches that word to `bold_font` (`IRANYekanExtraBlack2.ttf` / `KalamehFaNum-Black.ttf`, `1.38x` scale, `@accent`).

### Upgrade 3. True 3D Cylindrical Ribbon Text Orbiting Around the Speaker's Head (`@element("orbiting_title_3d")` in `mograph/creative_reels.py`)
* Upgrade `orbiting_title_3d` so that in addition to orbiting a flat/tilted phrase, when `"mode": "cylinder_ring_3d"` is set:
  * Wraps the Persian sentence around a **true 3D horizontal cylinder encircling the speaker's head** `(x = cx + R*sin(θ + ω*t), y = cy + tilt_y*R*cos(θ + ω*t), z = R*cos(θ + ω*t))` using column-wise perspective projection!
  * Renders the back half (`z < 0`) during the `behind: "clip"` pass (behind the speaker's head/hair) and the front half (`z >= 0`) on top of the speaker, and when `"land_to_sub_at"` is reached, unwraps the 3D ring along a silky `Ease-In-Out` Bezier flight path straight into the lower-third subtitle pill!

### Upgrade 4. Automatic Foreground Edge Light-Wrap (`"auto_light_wrap": true` in `mograph/fx.py`)
* Whenever ANY bright or glowing element (`novadent_chair_glow_text`, `novadent_red_halo`, `novadent_split_dome`, `wireframe_badge`) is rendered in the `behind: "clip"` layer, automatically extract the bright emissive pixels of the behind-layer right along the boundary of the speaker/chair mask (`edge_band = cv2.GaussianBlur(person, (31, 31), 10) * person`), blur those behind-layer colors (`cv2.GaussianBlur(behind_rgb, (45, 45), 16)`), and screen-blend (`OPERATOR_SCREEN`) them onto the outer 18px rim of the foreground speaker and chair!
* That way, if `زدیم` glows gold behind the left edge of the black chair, only that exact spot of the chair edge automatically receives a golden light-wrap; if `novadent_red_halo` glows crimson behind her hair and shoulders, her hair and shoulders automatically receive a realistic crimson rim-wrap!

### Upgrade 5. First-Class `"showcase": {"mode": "before_after"}` in `scene.py` & `render.py`
* Support `spec["showcase"] = {"mode": "before_after", "before_label": "BEFORE", "after_label": "AFTER", "full_bleed_at": [[t0, t1], ...]}` directly inside `render_frame(scene, t)` so any reel can be rendered as a `@novadent.ir` split-card Before/After showcase in a single pass, with smooth `Ease-In-Out` zoom-ins to full screen during climax windows!

### Upgrade 6. Hand-Tracked `glass_hud_frame`, `branch_tree`, `crew_pop_stack` & Speaker `"look": "silhouette_noir"`
* Complete the companion viral reel features in `creative_reels.py` and `fx.py`:
  * Bind `glass_hud_frame` to left/right hand trackers (`track.py`),
  * Add `@element("branch_tree")` (`راهکار دو مرحله‌ای → ① / ②`) and `@element("crew_pop_stack")` (behind-speaker pop-up characters + foreground desk diamond indicators),
  * Add `"look": "silhouette_noir"` on `el_footage` (black speaker silhouette + glowing white sunglasses + boiling brain scribble on a halftone stage).

### Upgrade 7. Deliverable Verification for `v14 Studio Perfection`
1. Audit and polish all `v13.9` modules and implement Upgrades 1–6.
2. Render a flagship showcase video **`demo_v14_novadent_and_viral_master.mp4`** demonstrating all 8 `@novadent.ir` forensic laws (including stateful Jelly Squash-and-Stretch subtitles with `"jelly"` SFX, inline thin-vs-bold word contrast, 3D cylindrical head-orbiting titles, and automatic foreground chair/hair light-wrap) inside the `before_after` split-card showcase!
3. Verify that all examples and all 17 templates pass `python3 -m mograph qa` and `review`, and package **`motion_creator_brain_sdk_v14.zip`**.
```
