# کاتالوگ جامع ۱۰ تمپلیت آماده و ۵ سبک استودیویی (`mograph v11 Studio Pantheon`)

این پوشه شامل **۱۰ تمپلیت آماده، تست‌شده و استاندارد (`QA clean ✓`)** برای پوشش تمام سبک‌های موشن‌گرافیک استودیویی (مستند، تحلیلی، تبلیغاتی، پادکست، معرفی محصول/اپلیکیشن و تیزر سینمایی) است.

## ۵ سبک اصلی پشتیبانی‌شده در هسته موتور (`meta.style`)
1. **`lineart` (خطی نوار / `@ENSAN.BESHAM`):** پس‌زمینه مشکی، خطوط سفید درخشان، کالبدشکافی نقل‌قول و پرونده‌های فلسفی/اقتصادی.
2. **`swiss` (گرید سوئیسی و دیتاجورنالیسم / Vox & FT Style):** شبکه ستونی ۶×۱۲، پس‌زمینه کاغذی، اعداد چرخان غول‌پیکر، ماتریس وافل ۱۰×۱۰ و جداول مقایسه.
3. **`kinetic` (تایپوگرافی جنبشی و پوستر زنده):** چیدمان بنتو (`masonry`)، ماژیک معکوس‌کننده رنگ (`highlight_sweep`) و کوبش کلمات روی صدا.
4. **`promo` (تیزر پرومو، سه‌بعدی و کشیده زنده / `@MOHAMADESFEHANIYAN` Style):** اسپات‌لایت شعاعی، ذرات بوکه، متن سه‌بعدی (`extrude_3d`)، حروف کشیده زنده (`kashida`)، حباب سیال (`liquid_blob`)، مکعب سه‌بعدی (`iso_cube`) و چشمی دوربین.
5. **`ui_physics` (فیزیک رابط کاربری و تعامل زنده / Apple & Linear Rube-Goldberg UI Style — جدید در V10):** دنیای نقطه‌چین بی‌نهایت (`dot_matrix`)، کلید تغییر وضعیت (`ui_toggle`)، اسلایدر با حباب شمارنده (`ui_slider`)، کارت‌های نوتیفیکیشن شیشه‌ای با ضربه‌گیر فنری (`ui_notification_stack`)، کیبورد مکانیکی با کلید Return پله‌ای (`ui_keyboard`)، کپسول پاسخ زنده (`ui_reply_pill`) و توپ فیزیکی با فشردگی و کشش (`physics_ball` / `ui_film_world`).

---

## فهرست ۱۰ تمپلیت آماده در پوشه `templates/`

| کد | نام فایل تمپلیت | سبک (`style`) | نسبت تصویر | کاربرد و المان‌های کلیدی |
| :--- | :--- | :--- | :--- | :--- |
| **01** | `01_ui_physics_rube_goldberg_16x9.json` | `ui_physics` (`apple_blue`) | `16:9` | فیلم ۱۰ ثانیه‌ای تعامل زنجیره‌ای المان‌های رابط کاربری (کلیک سوئیچ $\to$ اسلایدر $\to$ سقوط روی ۳ کارت نوتیفیکیشن $\to$ تایپ روی کیبورد $\to$ زوم بک و لوپ بی‌نقص). |
| **02** | `02_ui_physics_persian_app_reel_9x16.json` | `ui_physics` (`emerald_os`) | `9:16` | ریلز عمودی معرفی اپلیکیشن، سایت یا محصول نرم‌افزاری فارسی با نوتیفیکیشن‌های راست‌به‌چپ، اسلایدر فارسی و کیبورد. |
| **03** | `03_ensan_forensic_quote_dissection.json` | `lineart` (`noir`) | `9:16` | کالبدشکافی و راستی‌آزمایی نقل‌قول‌های خبری (`quote_rig` با دایره‌های شماره‌دار $\to$ نخ اتصال `thread` $\to$ چرخه `flow_loop` $\to$ اسکن قانون `laser_scan` $\to$ صورت‌حساب `invoice`). |
| **04** | `04_ensan_philosophical_trial_scale.json` | `lineart` (`noir`) | `9:16` | مستند پرونده‌محور و محاکمه مفهومی (`morph` تبدیل علامت سؤال به ترازو $\to$ ترازوی فیزیکی دوکفه `scale_rig` با سقوط وزنه‌ها $\to$ مهر نهایی `stamp`). |
| **05** | `05_swiss_economic_data_journalism.json` | `swiss` (`red`) | `9:16` | گزارش آماری و اقتصادی به سبک سوئیسی (`headline` + عدد چرخان `big_number` + ماتریس ۱۰۰ خانه‌ای `waffle_grid` + جدول مقایسه `versus_table`). |
| **06** | `06_kinetic_bento_manifesto_reel.json` | `kinetic` + `swiss` | `9:16` | ریلز انگیزشی و مانیفست برند با چیدمان مستطیلی بنتو (`masonry`)، موج صدا (`waveform`) و هایلایتر معکوس‌کننده (`highlight_sweep`). |
| **07** | `07_promo_ai_product_launch_crimson.json` | `promo` (`crimson` / `silver`) | `9:16` | تیزر پرانرژی معرفی ابزار هوش مصنوعی یا دوره آموزشی (`timeline_ui` با ضربدر قرمز $\to$ `prompt_box` $\to$ `iso_cube` $\to$ `liquid_blob` $\to$ کشیدگی زنده `kashida` $\to$ `viewfinder_hud` و `filmstrip`). |
| **08** | `08_hybrid_talking_head_commercial.json` | `promo` + `talking_head` | `9:16` | ویدیوی ترکیبی گوینده واقعی با جداسازی شبکه‌عصبی پس‌زمینه (`portrait_seg.onnx`)، قرارگیری تیتر سه‌بعدی **پشت سر و شانه گوینده** و تبدیل نرم گوینده به قاب دایره‌ای (`PIP`). |
| **09** | `09_podcast_audiogram_and_quote.json` | `lineart` (`ink`) | `9:16` | برش پادکست و مصاحبه با آواتار دایره‌ای گوینده در بالا، اسیلوسکوپ زنده صدا (`waveform`) در وسط و باکس نقل‌قول هایلایت‌شونده (`quote_rig`) در پایین. |
| **10** | `10_master_5style_anthology_showcase.json` | هر ۵ سبک در ۱ فایل | `9:16` | شואِیس جامع تغییر سبک زنده بین هر ۵ دنیای بصری (`lineart` $\to$ `swiss` $\to$ `kinetic` $\to$ `promo` $\to$ `ui_physics`). |

---

## موتور مکمل HTML5 Canvas / Chromium 60fps (`html_engines/ui_motion_film/`)
علاوه بر ماژول بومی `mograph/uiphysics.py`، فایل کامل موتور تعاملی مرورگر (`ui_motion_film.html` به همراه اسکریپت رندر موازی ۶۰ فریم بر ثانیه با Chromium) نیز در پوشه `html_engines/ui_motion_film/` قرار داده شده است.

---

## تمپلیت‌های ۱۱ تا ۱۹ — بازسازی ویدیوهای مرجع (جدید در v11)
هر تمپلیت دستور ساخت یک ویدیوی مرجع است (جزئیات در `REFERENCE_ANALYSIS.md`). همه از `tools/build_ref_templates.py` تولید می‌شوند؛ هر تغییری را در آن اسکریپت بدهید و دوباره اجرایش کنید.

| کد | فایل | مرجع | تکنیک‌های کلیدی |
| :--- | :--- | :--- | :--- |
| **11** | `11_ref01_talking_head_depth_beauty.json` | ref01 | گوینده واقعی (`footage`)، متن پشت سر گوینده (`behind`)، رنگ انتخابی، لوک نگاتیو، هاف‌تون، کارت عکس هشت‌ضلعی |
| **12** | `12_ref02_editor_showcase_raw_vs_edited.json` | ref02 | دو قاب گوشی خام/ادیت‌شده (`device_frame`)، کاغذ فیروزه‌ای، دکمه سه‌بعدی، هندل و اوتروی اینستاگرام |
| **13** | `13_ref03_storyboard_vs_motion.json` | ref03 | استوری‌بورد دست‌کش که زنده کشیده می‌شود (`auto_trace`)، قلب‌های ریزان (`emitter`)، نوتیفیکیشن و سوئیچ |
| **14** | `14_ref04_editing_process_timeline.json` | ref04 | مراحل تدوین روی یک کلیپ: اصلاح صدا، اصلاح رنگ کلیدفریمی، تایپوگرافی، تایم‌لاین با پلی‌هد قرمز |
| **15** | `15_ref05_tiles_phone_clip_timeline_tracking.json` | ref05 | ترکینگ دست (`trackers` + `follow`)، اسپید رمپ، کلمات پشت گوینده، تایم‌لاین کلیپ‌های رنگی |
| **16** | `16_ref06_before_after_neon_saber.json` | ref06 | قبل/بعد روی دود بنفش، درخشش لبه دور گوینده (saber)، تصویر لایه‌عمقی، مغز نئونی با مدار |
| **17** | `17_ref07_noir_lineart_comparison_story.json` | ref07 | سبک خطی نوآر انسان: سکو، نمودار، کال‌اوت خط‌چین، چشم تپنده، پن دوربین |
| **18** | `18_ref08_radial_viral_formula.json` | ref08 | چرخ فرمول شعاعی (`radial_formula`)، کمان پیشرفت، عدد شبح، آشکارسازی سانسور، مدار حلقه باز، REC |
| **19** | `19_ref09_neon_lineart_story_genimages.json` | ref09 | تصاویر AI به خطوط نئونی زنده (`auto_trace color_mode:source`)، عروسک (`sprite_rig`)، لرزش استرس، `mograph prompts` |

