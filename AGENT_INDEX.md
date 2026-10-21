# AGENT_INDEX — one page: "user asks X → open file Y, use key Z"   (VE / mograph v19 Pro Editor)

## 0. Every session (2 commands)
```bash
cd /home/user/webapp/sdk && python3 -m mograph.pro doctor --fix      # deps · binaries · fonts · footage · tracking cache · 5 selftests
python3 -m mograph.motion resolve "<the user's sentence in Persian/English>"   # words → engine keys
```
`doctor` must end with `ALL GOOD ✓` (v14 9/9 · aefx 9/9 · aecomp 13/13 · motion 10/10 · pro 12/12 = 53 tests).
The tracking cache (normally 9 min of MODNet/MediaPipe) is restored in seconds from `work/input/raw.mp4.v14pack/`.

## 1. Which engine?
| material | engine | spec | read |
|---|---|---|---|
| real talking-head footage (the normal case) | `mograph.v14` | `projects/<p>/edit_*.json` | `sdk/V14_GUIDE.md` |
| pure vector / infographic, no footage | `mograph` (v5–v13) | `scene spec` | `sdk/00_START_HERE_AGENT_BOOTLOADER.md`, `sdk/SPEC_REFERENCE.md` |

## 2. Request → key (v14 edit.json)
| the user says | key | LOOK first | guide |
|---|---|---|---|
| تایتل / متن سه‌بعدی / کلمه کلیدی | `{"type":"kinetic_title","text":"…|*hero*","t":"w:word"}` | `v14 sheet` | V14_GUIDE §2–3 |
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
| حلقه متن دور سر / هاله / دیسک / شیشه / کادر | `orbit_ring` · `halo` · `disc` · `glass_shape` · `frame_box` | `v14 sheet` | V14_GUIDE §3 |
| برچسب ترک‌شده روی دست/شانه | `tracked_tag` `"anchor":"hand_r"` | | V14_GUIDE §3 |
| شی سه‌بعدی واقعی | `object_3d` `"model"` | `python3 -m mograph pbr3d-search` | V14_3D_GUIDE |
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
| تدوین خودکار از صفر | `python3 -m mograph.v14 auto-edit words.json media/raw.mp4 -o edit_auto.json` | | V14_GUIDE v14.8 |
| حرکت دوربین / پانچ / لرزش | `"camera":{"keys":[…],"shakes":[…]}` | | V14_GUIDE §1 |
| صدا: نویزگیری / موزیک / SFX / لاودنس | `"audio":{"denoise","music","sfx_gain","lufs"}` | `v14 render` | V14_GUIDE §6 |

## 3. Times (every "t" / "out" / "at")
`"w:word"` start of word · `"we:word+0.2"` end of word + offset · `"w:word#2"` 2nd occurrence · `3.2` / `"e:3.2"` edited
seconds (moves correctly through retime) · `"s:3.2"` raw on-screen seconds · `"src:12.4"` seconds in the ORIGINAL file.

## 4. The loop (never skip)
```bash
cd projects/<p> && export PYTHONPATH=../../sdk
python3 -m mograph.pro lint edit.json            # types / typos / files / anchors / hero overlaps
python3 -m mograph.v14 qa edit.json              # on-screen: face cover, edges, subtitle lengths
python3 -m mograph.v14 sheet edit.json           # LOOK at it (12 frames)
python3 -m mograph.pro render edit.json --preview  # client preview, half res
python3 -m mograph.pro render edit.json -j 2       # master + .srt/.vtt + timeline (resumable)
python3 -m mograph.pro reformat out/edit.mp4 --to 1:1 --spec edit.json   # other platforms
```
Delivery extras: `pro captions` · `pro edl` · `pro chapters` · `pro thumbnail` · `pro platform`.

## 5. Laws (user taste — never break)
Zero-box (no flat 2D boxes behind text) · one hero idea at a time · hero = the spoken key word · Doran theme ·
colour tokens `@accent @alert @ink` · subtitles hidden under lower-third heroes · `sdk/USER_TASTE_AND_DNA_LAWS.md`.

## 6. Repo rules
Work only in `/home/user/webapp` · commit after each step · repo < 100 MB (big binaries are fetched by doctor) ·
`projects/*/media` and `*.v14/` caches are git-ignored (doctor restores them) · do not run renders in parallel with
`v14.track` prep (RAM).
