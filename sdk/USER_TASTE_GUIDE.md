# USER TASTE GUIDE — how to work with this user (v21, Creative Synthesis)

> Read this BEFORE anything visual. It is harmonised with `USER_TASTE_AND_DNA_LAWS.md` and
> `style_bank/STYLE_BANK_PLAYBOOK.md`. Where an older document conflicts, **this file wins**.
> v21 removed the old engine locks to the @novadent widgets: halo as the default, the stage grid behind every
> element, Doran-only fonts, and the fixed widget rotation.

## 1. Who the user is & how to communicate
- Persian speaker, often voice-to-text → typos/half-words; read for intent, not letters.
- **Reply in Persian, short, concrete.** End with: what changed + why + links.
- Standing order: **«سوال نپرس، به استدلالت اعتماد کن، برو.»** Never block on questions. Pick the professional
  default and state it in the report.
- They send reference images/videos. Treat each one as a **style study**: extract the principle, never the
  pixels (law 1).
- They praise rarely and briefly («بقیه عالی بود») — that means: don't touch the rest.
- The sandbox resets often; they send backup tarball links → restore, delete the tarball, **commit immediately**,
  continue.

## 2. The 6 Core Creative Laws of the user's taste

### Law 1 — THE ANTI-CLICHÉ & ANTI-REPETITION LAW (قانون منع مطلق تکرار و کپی)
- **Copying is strictly forbidden.** Never reproduce a reference image, a template, or a previous project. That
  covers its texts, exact shapes, widget types, colours and ordering.
- References (`sdk/style_bank/references/`), `sdk/templates/`, `projects/` and `sdk/tests/legacy_maghale/` are
  **study material only** (الگوبرداری و سبک‌شناسی، نه کپی).
- Every video gets its **own visual identity** from its topic and emotional tone: palette from the footage, a
  font pairing, a motion temperament, a subtitle style, and its own sequence of forms.
- Enforced by `python3 -m mograph.pro originality edit.json` (also inside `v14 qa`). Copy or clone = error.
  Fixed rotation, monotony or legacy-widget reuse = warning.

### Law 2 — PURE TYPOGRAPHY FIRST (تایپوگرافی نوری خالص بدون شکل‌های اضافه)
- Do NOT wrap every sentence in a circle, tilted rectangle, hexagon or dotted pill.
- **60–70 % of beats** are container-free layered typography: thin/clean context words frame ONE large
  bold/calligraphic hero (`*word*`). Use realistic optical glow and soft colour light-spill onto the clothes or
  background (`kinetic_title`, `stage: {"grid": 0}`, `post.spill`).
- Hero core stays near-white and readable. The colour lives in the glow and spill.

### Law 3 — CONTENT-DRIVEN FORM SYNTHESIS (تناسب معنایی فرم با محتوا)
Introduce a form only when the spoken meaning demands it, and design it fresh each time:
- **Comparing two concepts** → a custom symmetrical dual composition (two glow titles on a shared baseline,
  answering morph outlines, a split stage, a split-tone grade).
- **A tangible object or metaphor** → ONE clean 3D prop or cutout that interacts with the typography. Never two
  separate 3D objects fused on top of each other (`three3d.check_mesh_overlap`).
- **Warning / climax / emotional shift** → atmosphere: light shift, grade key, depth-of-field defocus, organic
  environmental overlay.

### Law 4 — WORLD-LOCKED STABILITY OVER HEAD-BOBBING TRACKING (منع موشن‌ترکینگِ افراطی به سر)
- Titles and graphics sit calmly in studio space (`"anchor": "screen"` or low parallax `"follow": 0.12`) behind
  the shoulders or over the chest.
- The engine default is now `follow 0.18` for `kinetic_title / frame_box / halo`. `"track": true` restores full
  tracking.
- Body point-tracking (`hand_r`, `hand_l`, `shoulder_*`) only for rare intentional moments, e.g. the speaker
  holds or points at something.
- **Nothing covers the face, forehead or hair crown.** The guard protects up to `head_top − 0.18·hw`. A title
  that can't fit above a high head slides to the side or shrinks.

### Law 5 — SUBTITLE VARIETY & ZERO COLLISION (تنوع زیرنویس و منع تداخل)
- No default cyan-bracket capsule. Choose a style for the mood: `minimal_clean` (default, box-free),
  `soft_scrim`, `glass_matte`. `jelly_crescent` is the legacy look, explicit only.
- Subtitles **auto-hide** whenever a lower third (`aecomp lower_third`) or a chest/belly hero element is on
  screen. QA flags any remaining collision.

### Law 6 — CURATED FONT PAIRING (آزادی ترکیب فونت‌های حرفه‌ای)
- The Doran-only lock is retired (`font_lock` is false by default; `"meta": {"font_lock": true}` brings it back).
- Pair a structural family (Doran, Peyda, KalamehFaNum, Morabba, Pelak, YekanBakh) with an expressive hero voice
  (Delbar, Miremad, IranNastaliq `FONT_2417`, DigiGhasedak, a clean Latin script accent):
  `"styles": {"hero": {"font": "Delbar-Bold.ttf"}}`.
- At most 2–3 families per video. Hierarchy by weight × size × colour inside each.

## 3. Permanent craft rules (unchanged)
- Zero flat 2D boxes behind text (see `USER_TASTE_AND_DNA_LAWS.md` Law 1).
- Motion is monotonic and intentional. Any back-and-forth jerk is a bug.
- Colours are designed from the footage (`palette.py auto_palette`), never copied from a reference.
  `@alert` red only for alert/emphasis.
- Clean audio (DeepFilterNet3), −14 LUFS.
- The coordinate grid is **off by default**. Use it at most once per video, when it means something
  (engineering / data).

## 4. Delivery checklist
1. `pro lint` + `v14 qa` clean (includes originality), contact sheet reviewed by you.
2. `pro originality edit.json` → `originality ✓`. The report names the closest previous project and shows how
   this one differs.
3. Render, upload via UploadFileWrapper; Persian report: changes / why / links.
4. Log lessons in `memory_and_prompts/05_EXPERIENCE_LOG.md` (SKILL_DEV_LOOP.md).
