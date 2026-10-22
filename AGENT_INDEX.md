# AGENT_INDEX — one page: "user asks X → open file Y, use key Z"   (VE / mograph v22 · Concat-inspired kit on v21 Style Bank)

> ⛔ **SUPREME LAW (v21) — قانون منع مطلق کپی و تکرار:** never copy a reference image, a template, an archived
> spec or a previous project (texts, shapes, widget sequence, colours, ordering). References are STYLE STUDIES
> (`sdk/style_bank/STYLE_BANK_PLAYBOOK.md`). Every video gets a new visual identity.
> `python3 -m mograph.pro originality edit.json` must say `originality ✓` before you render.
>
> 🎬 **Artistic framework** (how a senior editor thinks: narrative per cut, attention, easing, retime, sound,
> transitions, QA, delivery — in Persian): `sdk/MASTER_PROJECT_GUIDE.md`. This index tells you HOW in the engine.

## 0. Every session (2 commands)
```bash
cd /home/user/webapp/sdk && python3 -m mograph.pro doctor --fix      # deps · binaries · fonts · footage · tracking cache · 5 selftests
python3 -m mograph.motion resolve "<the user's sentence in Persian/English>"   # words → engine keys
python3 -m mograph.pro style-bank --list                # v21: principles of every reference (study, never copy)
```
`doctor` must end with `ALL GOOD ✓` (v14 9/9 · aefx 9/9 · aecomp 13/13 · motion 10/10 · pro 12/12 · three3d 8/8 · three_anim 7/7 · style_bank 3/3 · v22 10/10 = 81 tests).
Agents with MCP: `python3 -m mograph.mcp` (stdio) gives the same loop as tools with a warm renderer — `sdk/V22_GUIDE.md` §1.
The tracking cache (normally 9 min of MODNet/MediaPipe) is restored in seconds from `work/input/raw.mp4.v14pack/`.

## 1. Which engine?
| material | engine | spec | read |
|---|---|---|---|
| real talking-head footage (the normal case) | `mograph.v14` | `projects/<p>/edit_*.json` | `sdk/V14_GUIDE.md` |
| pure vector / infographic, no footage | `mograph` (v5–v13) | `scene spec` | `sdk/00_START_HERE_AGENT_BOOTLOADER.md`, `sdk/SPEC_REFERENCE.md` |

## 2. Request → key (v14 edit.json)
| the user says | key | LOOK first | guide |
|---|---|---|---|
| **رفرنس / سبک / «مثل این ویدیو»** | study → `style-bank --list`; add → drop file in `sdk/style_bank/references/` + `style-bank --sync` | `style-bank --sheet` | STYLE_BANK_PLAYBOOK |
| **بررسی کپی/تکرار** | `python3 -m mograph.pro originality edit.json` | | STYLE_BANK_PLAYBOOK §0 |
| تایتل / متن نوری خالص (پیش‌فرض ۶۰–۷۰٪ بیت‌ها) | `{"type":"kinetic_title","text":"…|*hero*","t":"w:word","follow":0.12,"stage":{"grid":0}}` | `v14 sheet` | V14_GUIDE §1–3 |
| مقایسهٔ دو مفهوم | two `kinetic_title` on a shared baseline + `morph_shape` (design it fresh) | | USER_TASTE_GUIDE law 3 |
| ترکیب فونت | `"styles":{"hero":{"font":"Delbar-Bold.ttf"}}` (font_lock off by default) | | USER_TASTE_GUIDE law 6 |
| سبک زیرنویس | `"subtitles":{"style":"minimal_clean|soft_scrim|glass_matte|jelly_crescent"}` (auto-hide under lower thirds) | | V14_GUIDE §3 |
| انیمیشن متن افترافکتی (حرف‌به‌حرف) | `"preset":"atp_23"` on kinetic_title | `aefx sheet text …` | AEFX_GUIDE |
| حرکت آماده (shake, glitch, zoom) روی هر لایه | `"fx":[{"motion":"shake_03","at":"w:…"}]` | `aefx suggest punchy` | AEFX_GUIDE |
| درخشش / گلو | `"fx":[{"glow":{…}}]` | | AEFX_GUIDE |
| نئون / شمشیر نوری / آتش / برق | `{"type":"saber","preset":"neon","core":"text"}` | `motion saber sheet` | MOTION_GUIDE §3 |
| شمشیر نوری توی دست گوینده | saber + `"start_anchor":"shoulder_r"`/`"hand_r"`, `"end_offset"` | `v14 frame` | PRO_EDITOR_GUIDE |
| اسلوموشن / فریز / ریمپ / عقب‌گرد / تایم‌لپس | `"source":{"retime":[{"op":"speed_ramp",…}]}` | `motion retime-plan` | MOTION_GUIDE §2 |
| ایزینگ / نرمی حرکت | `"ease":"out_back"` / `cubic-bezier(…)` | `motion ease graph` | MOTION_GUIDE §1 |
| کاراکتر کارتونی + لیپ‌سینک | `{"type":"character","rig":"bean","lip_sync":{…}}` | `motion character sheet` | MOTION_GUIDE §4–5 |
| انفجار / کانفتی / فلش / شکل‌های افترافکتی | `{"type":"aecomp","gen":"burst"}` | `aecomp sheet` | AECOMP_GUIDE |
| زیرنویس معرفی (lower third) بدون باکس | `{"type":"aecomp","lower_third":"line_reveal","title","role"}` | `aecomp lt-sheet` | AECOMP_GUIDE |
| تایتل تمام‌صفحه | `{"type":"aecomp","title_style":"burst_hero","hero"}` | `aecomp title-sheet` | AECOMP_GUIDE |
| حلقه متن دور سر / هاله / دیسک / شیشه / کادر (*legacy, ≤1 per video, only if meaningful*) | `orbit_ring` · `halo` · `disc` · `glass_shape` · `frame_box` | `v14 sheet` | V14_GUIDE §3 |
| برچسب ترک‌شده روی دست/شانه (*legacy, intentional gesture only*) | `tracked_tag` `"anchor":"hand_r"` | | V14_GUIDE §3 |
| **انیمیشن داخل شی سه‌بعدی** (ساعت شنی که شن می‌ریزد و برمی‌گردد، لپ‌تاپ باز شود، قطعات سر هم شوند) | `object_3d` `"three_model":"hourglass_3d","anim":"hourglass_loop","flips_at":["w:…"]` · `"anim":"laptop_open"` · `"anim":"assemble"` · `"explode_at":"w:…"` | `python3 -m mograph three3d-anim-sheet hourglass_loop` | V14_3D_GUIDE §0.1 |
| **شی سه‌بعدی واقعی (اسکریپتی Three.js / PBR)** | `object_3d` `"three_model":"hourglass_3d"` or `"three_script":"my.js:buildFn"` | `python3 -m mograph three3d-list` · `three3d-sheet` | V14_3D_GUIDE |
| شی سه‌بعدی اسکن‌شده (Poly Haven GLB) | `object_3d` `"model"` | `python3 -m mograph pbr3d-search` | V14_3D_GUIDE |
| خرد شدن کلمه / پاپت / مورف شکل | `shatter_text` · `puppet_sticker` · `morph_shape` | | V14_GUIDE v15.1 |
| **رنگ‌بندی / لوک سینمایی** | `"grade":{"look":"teal_orange"}` (10 looks) | `pro looks --spec edit.json` | PRO_EDITOR_GUIDE |
| **LUT مشتری** | `"grade":{"lut":"x.cube","lut_strength":.8}` | `pro lut-info x.cube` | PRO_EDITOR_GUIDE |
| **گرین / هاله‌ی فیلم / وایدسکرین** | `"grade":{"grain":.2,"halation":.15,"letterbox":2.39}` | | PRO_EDITOR_GUIDE |
| **بی‌رول / کات‌اوی** | `{"type":"broll","src":"media/x.mp4","mode":"full","in":"whip_left"}` | `pro frame --t` | PRO_EDITOR_GUIDE |
| **تصویر در تصویر (PiP)** | broll `"mode":"pip"` | | PRO_EDITOR_GUIDE |
| **عوض کردن پس‌زمینه پشت گوینده** | broll `"mode":"behind"` | | PRO_EDITOR_GUIDE |
| **اسپلیت اسکرین** | broll `"mode":"split_top"` | | PRO_EDITOR_GUIDE |
| **پرده سبز / کروما** | broll `"key":{"color":"#00ff00"}` | | PRO_EDITOR_GUIDE |
| **ترنزیشن** | broll `"in"/"out_fx"`: 12 names | | PRO_EDITOR_GUIDE |
| تطبیق رنگ با ویدیوی رفرنس | `"grade":{"match":{"ref":"…","strength":.7}}` | | V14_GUIDE v15.1 |
| حذف مکث / جامپ‌کات نامرئی | `"source":{"segments":[…],"morph_cuts":{…}}` | `v14 words` | V14_GUIDE §1 |
| تدوین خودکار از صفر (semantic, per-video style seed) | `python3 -m mograph.v14 auto-edit words.json media/raw.mp4 -o edit_auto.json` (`--legacy-showcase` = old widget cycle) | | V14_GUIDE v14.8 |
| حرکت دوربین / پانچ / لرزش | `"camera":{"keys":[…],"shakes":[…]}` | | V14_GUIDE §1 |
| **لرزش‌گیر ویدیو (دوربین روی دست)** | `"source":{"stabilize":true}` | `pro stabilize edit.json` | V22_GUIDE §6 |
| **ماسک سریع‌تر و پایدارتر (RVM)** | `"source":{"matte_engine":"rvm"}` + `v14.track … --engine rvm` | | V22_GUIDE §2 |
| **افزایش کیفیت تصویر / لوگوی بی‌کیفیت** | broll `"enhance":{"factor":2,"mix":.6}` · `"source":{"enhance":{"punch_in":1.3}}` | `pro enhance x.png` | V22_GUIDE §5 |
| **حذف پس‌زمینه عکس محصول** | broll `"cutout":true` | `pro cutout x.jpg` | V22_GUIDE §7 |
| **ماسک شکل (دایره/قلب/ستاره/مورب)** | broll `"mask":{"shape":"heart"}` · `"mode":"split_diag"` | | V22_GUIDE §8 |
| **گویندگی فارسی آفلاین (بدون API)** | `"voice":{"provider":"local","voice":"amir"}` | `python -m mograph voice spec.json` | V22_GUIDE §3 |
| **اصلاح زمان کلمات (زیرنویس زودتر از گفتار)** | `python3 -m mograph.vad raw.mp4 --words words.json` | | V22_GUIDE §4 |
| صدا: نویزگیری / موزیک / SFX / لاودنس | `"audio":{"denoise","music","sfx_gain","lufs"}` | `v14 render` | V14_GUIDE §6 |

## 3. Times (every "t" / "out" / "at")
`"w:word"` start of word · `"we:word+0.2"` end of word + offset · `"w:word#2"` 2nd occurrence · `3.2` / `"e:3.2"` edited
seconds (moves correctly through retime) · `"s:3.2"` raw on-screen seconds · `"src:12.4"` seconds in the ORIGINAL file.

## 4. The loop (never skip)
```bash
cd projects/<p> && export PYTHONPATH=../../sdk
python3 -m mograph.pro lint edit.json            # types / typos / files / anchors / hero overlaps
python3 -m mograph.v14 qa edit.json              # on-screen: face cover, edges, subtitles, collisions, originality
python3 -m mograph.pro originality edit.json     # v21: copy / clone / rotation audit → must be ✓
python3 -m mograph.v14 sheet edit.json           # LOOK at it (12 frames)
python3 -m mograph.pro render edit.json --preview  # client preview, half res
python3 -m mograph.pro render edit.json -j 2       # master + .srt/.vtt + timeline (resumable)
python3 -m mograph.pro reformat out/edit.mp4 --to 1:1 --spec edit.json   # other platforms
```
Delivery extras: `pro captions` · `pro edl` · `pro chapters` · `pro thumbnail` · `pro platform`.

## 5. Laws (user taste — never break) → `sdk/USER_TASTE_GUIDE.md` (6 Core Creative Laws)
1 **No copying, no repetition** (references = style studies; originality guard) · 2 **pure glow typography first**
(60–70 % of beats, no container around every sentence) · 3 **form follows meaning** (comparison / object /
warning) · 4 **world-locked stability** (follow ≈ 0.12–0.18; nothing on the face, forehead or hair) ·
5 **subtitle variety + zero collision** · 6 **curated font pairing**. Plus: zero flat 2D boxes · colours from the
footage (`@accent @alert @ink`) · grid off by default.

## 6. Repo rules
Work only in `/home/user/webapp` · commit after each step · repo < 100 MB (big binaries are fetched by doctor; v22 ML
models ONLY via `mograph/models_hub.py` into ~/.cache/mograph_models — never commit an .onnx) ·
`projects/maghale/` keeps ONE example (`edit_v14.json`); the 9 old clones live in `sdk/tests/legacy_maghale/`
(regression fixtures — never templates) · `projects/*/media` and `*.v14/` caches are git-ignored (doctor restores them) · do not run renders in parallel with
`v14.track` prep (RAM).
