# STYLE BANK PLAYBOOK — Creative Synthesis Architecture (v21)

> **قانون اول و مطلق: کپی‌کاری و تکرار ممنوع است.**
> رفرنس‌ها «سبک‌شناسی» و «الگوبرداری» هستند، نه قالب. هیچ ویدیویی نباید شبیه هیچ‌کدام از تصاویر رفرنس،
> تمپلیت‌ها یا پروژه‌های قبلی باشد. از رفرنس فقط **اصل** (چرا کار می‌کند) استخراج می‌شود و برای هر ویدیو
> یک **فرم تازه** ساخته می‌شود.
>
> **Absolute law: copying and repetition are FORBIDDEN.** References are style studies. Extract the principle
> (why it works) and synthesise a NEW form for every video. Never reproduce a reference's frame, text, widget,
> shape, colour scheme or ordering. Never clone another project's spec, and never reuse your own last video's
> plan.

## 0. Enforcement (mechanical, not optional)
```bash
cd /home/user/webapp/sdk
python3 -m mograph.pro style-bank --list                         # read the principles (never a "recipe")
python3 -m mograph.pro originality ../projects/<p>/edit.json     # MUST end "originality ✓" before render
```
`originality` (`mograph/style_bank.py`) reports:

| check | level | meaning |
|---|---|---|
| TEXT COPY | ✗ error | on-screen text reproduces a reference's text (`forbidden_texts` in catalog.json) |
| REFERENCE PIXELS | ✗ error | a reference image/clip is used as media inside the spec |
| SPEC CLONE | ✗ error | hero sequence ≥ 80 % identical to another project/archive spec **and** same hero texts |
| ROTATION | ⚠ | hero forms repeat in a fixed cycle (a template, not direction) |
| MONOTONY / LEGACY WIDGET | ⚠ | one non-typo form > 50 % of beats · `tracked_tag/disc/halo/frame_box` more than once |
| PURE TYPO RATIO | ⚠ | container-free typography < 55 % of beats (target 60–70 %) |
| GRID REPEAT | ⚠ | the coordinate grid switched on for more than one layer |

`python3 -m mograph.v14 qa` runs it too. Errors block delivery. Warnings must be fixed, or explained in the report
when they are a deliberate choice (e.g. the user explicitly asked for a halo).

## 1. How to study a reference (the only allowed use)
For every reference, write three things and nothing else:
1. **Principle**: why it works (hierarchy, light, rhythm, depth, semantic fit, emotion).
2. **When**: which spoken meaning calls for it.
3. **Synthesis**: how to express the same principle with a *different* form, palette, position and motion that
   fits THIS speaker, THIS footage and THIS sentence.

If your design could be described as "the panel from ref_0X, with other words", it is a copy. Redesign it.

## 2. The seed references

### ref_01_pure_glow_typo_and_dual_nodes.jpg
- **Principle**: no container. A thin context line frames one heavy hero word, and *light* is the container: a
  soft coloured glow spills onto the dark clothes behind it. Colour carries emotion (cool for teaching, red only
  when the meaning turns to a problem). A comparison becomes two equal, symmetrical nodes with the shared word
  between them.
- **When**: explaining beats, one key word per sentence, A-vs-B comparisons.
- **Synthesis, not copy**: use your own hero word and a palette taken from the footage (`palette.py`). Move the
  glow and change it per section. For comparisons, build a new dual composition: two light pools, two answering
  `morph_shape` outlines, a split-tone grade key, two `kinetic_title`s on a shared baseline. **Not** two diamond
  pills.

### ref_02_state_flip_and_split_stage.jpg
- **Principle**: state A → state B. A stable centre stays while its context flips, and the light travels with the
  meaning. A title above the head is separated from the speaker by a thin luminous line. The plate behind her
  becomes two product panels while the title stays world-locked.
- **When**: cause/effect, myth/fact, before/after, sentences that pivot on «اما / ولی / بیشتر از اینکه», naming
  two products.
- **Synthesis, not copy**: express the flip with a text morph, a colour-temperature shift, a `shatter_text`
  break, a grade key or a `morph_shape`. A split stage uses `broll` `mode: behind | split_*` with your own media,
  panel geometry and transition. **Not** the circle badge with arc text.

### ref_03_multi_creator_diversity_matrix.jpg
- **Principle**: four creators, four systems, so **every video has its own identity**. Objects carry meaning
  (habit cards, loupe, chest B-roll window, a prop that interacts with the word). A warning changes the
  *environment*, not a sticker. Sometimes the best graphic is almost nothing (a box-free floating subtitle).
- **When**: choosing the identity of a new video, objects named in speech, a hook or warning, calm personal
  beats.
- **Synthesis, not copy**: choose a different treatment family than the last project. Use one clean prop per
  idea, and keep two ideas as two separate, spaced objects. Make warnings with fresh grade/dof/overlay work.
  **Not** the reference's shattered glass.

## 3. Director workflow (uses the bank, never copies it)
1. `style-bank --list`: re-read the principles. Look at `style_bank_sheet.jpg` for the quality bar only.
2. Read the transcript. Map tone and structure: hook → explanation → pivot → take-away.
3. Choose this video's **identity** (palette from footage, font pairing, motion temperament, subtitle style). It
   must differ from the closest existing project (`originality` prints the closest peer).
4. Make 60–70 % of beats **pure glow typography** (`kinetic_title`, grid off, world-stabilised,
   `follow ≈ 0.12–0.18`).
5. Use other forms only where the meaning asks for them:
   - comparison → a new dual composition
   - a tangible object → one 3D prop or cutout
   - a warning or climax → light, grade, depth or environment
   - showing something → `broll`
6. `lint` → `qa` (includes originality) → `sheet` → fix → render.
   `auto-edit` does steps 2–5 automatically from the meaning plus a per-video style seed. Treat its output as a
   starting point, not a template.

## 4. Adding new references (expandable)
1. Drop any screenshot or clip into `sdk/style_bank/references/` (jpg/png/webp/gif/mp4/mov/webm). Use descriptive
   snake_case names, e.g. `ref_04_kinetic_numbers_counter.jpg`.
2. `cd sdk && python3 -m mograph.pro style-bank --sync`. Stills are compressed (≤ 1400 px) so the bank stays
   under 2 MB, and each file gets a `catalog.json` entry with a `TODO` analysis stub.
3. Fill in `principles`, `when_to_use`, `synthesis`, `forbidden_texts` (texts visible in the reference, so they
   can never be reproduced) and `tags`. Describe **why**, never **how to copy**.
4. `python3 -m mograph.pro style-bank --sheet` regenerates the overview sheet. Commit.

## 5. Things that are always wrong
- The same widget sequence or the same opening as the previous video.
- A box, pill or hexagon around every sentence.
- Titles motion-tracked to every head bob.
- The cyan-bracket capsule as the default subtitle.
- The grid behind every element.
- Two 3D objects fused into one another.
- Reference text, colours or layout reproduced "because the user liked it". The user liked the *principle*.
