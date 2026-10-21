# DIRECTOR_WORKFLOW — splitting the thinking: Director → Designer → Engineer

The main problem with earlier versions: the agent had to write the scenario, draw vectors, write code, and sync timing **all at once**.
Since v4 the work is split into **three separate passes**, and each one has a single focus:

## Pass 0 (v7): choose the style
| The video is mostly… | style |
|---|---|
| a story with objects, metaphors, mechanisms (money flows, layers, laws) | `lineart` |
| numbers, facts, lists, a strong statement — "poster" / editorial feel | `swiss` |
| a monologue / quote where the WORDS are the image | `kinetic` |
**v8: mix styles inside ONE spec** — set `style` / `palette` / `bg` on a chapter (it gets that style's full world:
colours, fonts, HUD, sound, post-FX). Rules for a multi-style video:
- change style only at an **act boundary** (a new argument / tone), not every chapter; 2–4 worlds per video max;
- bridge worlds with `portal` (from the focal element of the old world: `"from": "<id>"`) or `wipe` — the next world's
  background is revealed, never a black frame; use `cut` only inside kinetic;
- keep one recurring motif across worlds (e.g. the accent colour, a number, the same word) so it feels like ONE film.
(`mograph join` still works for separately rendered acts.)

## Pass 1: Director (thinking only, no code)
For each chapter, fill in a **Beat Sheet** table. One row per narration sentence:

| # | Narration phrase | Key word (anchor) | What we see (one main idea) | Element | Leftover from the previous beat |
|---|---|---|---|---|---|
| 1 | «شما می‌گین از منِ پول…» | `منِ` | coin at the top + row of people | icon, icon_grid | — |
| 2 | «ریشه‌ی همه‌ی فسادها منم» | `ریشه‌ی` | camera zooms on 3 people | camera, highlight_at | coin stays |
| 3 | «مستندتره» | `مستندتره؛` | case file + «مستند» stamp | box(tab), stamp | people fade out |

**v5 — pick a rig before inventing:** for each chapter ask "which widget *is* this idea?"
| Idea in the narration | Rig |
|---|---|
| a quote / official statement to dissect | `quote_rig` (highlights on the damning words) |
| money/power circulating between 3 actors | `flow_loop` |
| a law/contract clause that matters | `laser_scan` |
| hidden costs, a price breakdown | `invoice` (+ stamp punchline) |
| what lies beneath / extraction / hidden flows | `strata` |
| comparing worth, guilt, trade-offs | `scale_rig` |
| one thing turning into another | `morph` |
| a number that changes | one `counter` with `tracks.value` |
| something that must look REAL (object, material, crowd, 3D in space) | `omni` job on that beat (edit task) — max 1–2 per video |
v8 rigs: `waffle_grid` (a share of a population — "1 % of you"), `swiss_bars` (category comparison, editorial),
`versus_table` (claim vs reality), kinetic `masonry` (a punchline sentence as a poster), `highlight_sweep` (one damning
word in a sentence), `"roll": true` on a number that changes, `waveform` / `idle:"vo_pulse"` when the VOICE itself is the
subject, `"z"` on background (−) / foreground (+) layers of any chapter with a camera move.
Use `thread` transitions when the next chapter is a *consequence* of the previous one (cause → effect); `pan` for a new topic.

**Director's rules:**
- One focal point per beat. If there are two ideas, split them into two beats.
- Every visual change is tied to a **narration word** (`w:`), not to a guessed second.
- "Visual surprise" at the punchline: stamp, strike, counter, or a sudden camera zoom-out.
- Don't clear the screen: the previous beat goes dim (`dim_at`) or moves aside (`tracks`).
- Each chapter has a single hero moment (it will show up in the contact sheet).

## Pass 2: Designer (layout only)
- Content area: `x 80…1000`, `y 180…1500`. Header at the top, dots, and subtitles below are off-limits.
- 3×4 layout grid: columns `x = 270 / 540 / 810`, rows `y = 420 / 700 / 980 / 1260`.
- Main icon size 220–320, secondary 110–160. Main text 44–56, labels 30–36.
- RTL: the start of a story goes on the right, the result on the left. Timelines run right to left.
- Before inventing anything new: `python3 -m mograph icons <keyword>`. If it doesn't exist, use `add_icon` (100 box).

## Pass 2.5: Voice (v6)
Write subtitles as the narration script first (they are the script). Iterate with `"voice": {"provider": "draft"}`;
when the user approves the story, switch to elevenlabs/gemini, run `mograph voice`, and render the `.voiced.json`.
Use `"say"` for pronunciation fixes and ElevenLabs tags (`[sarcastic]`, `[whispers]`) on punchlines only.

## Pass 3: Engineer (spec + QA)
1. Write the spec, then `qa`. Fix every `error` and `overlap`.
2. Run `sheet`, view the image, and confirm every chapter's hero frame is correct.
3. Run `preview` to check motion and timing.
4. Run `render` — `review` runs automatically on the real mp4 (black/frozen frames, LUFS, peaks). Look at the review frame sheet.

## Self-review checklist (before handing over)
- [ ] `qa` = clean
- [ ] Every punchline appears exactly on its word (not earlier = no spoilers)
- [ ] No text sits on top of other text (overlap report)
- [ ] No black frame between chapters (pan/fade)
- [ ] Every chapter has at least one element entering in every 2-second window (no dead frames)
- [ ] Final audio: −14 LUFS, SFX not too loud, narration clear
- [ ] `review` = clean (no black/frozen stretches, −14 LUFS ±1, peak < −1 dBTP)
- [ ] Font roles match tone (titles = title, legal = document, twist = nastaliq)
- [ ] Output filename is new


---

## 🧬 v13.5 Visual-DNA & Zero-Box Creative Workflow (`mograph/visual_dna.py` & `mograph/reelfx.py`)
Always read `USER_TASTE_AND_DNA_LAWS.md` and `VISUAL_DNA_ARCHITECTURE.md`.
- **Standalone / Hybrid Visual-DNA Elements (`mograph/visual_dna.py`):**
  - `{"type": "dna_stage", "mode": "dark | light", "accent": "#10B981", "grid_step": 64, "telemetry": true}`
  - `{"type": "dna_connector", "from": [180, 960], "to": [900, 960], "device": "caliper_rail | circuit_bus | oscillating_wave"}`
  - `{"type": "tactile_3d", "sprite": "laptop_3d", "x": 540, "y": 780, "size": 280, "sub": "کیفیت استودیویی", "title": "نسل جدید مک‌بوک"}`
  - `{"type": "hud_loading_bar", "x": 540, "y": 1320, "w": 620, "h": 44, "from": 0, "to": 92, "label": "LOADING // پردازش موتور سه‌بعدی"}`
- **1-Command 3D Sprite Sheet Extraction (`mograph/extract3d.py`):**
  - `python3 -m mograph extract3d sheet.jpg --names logo1_3d,logo2_3d,logo3_3d,logo4_3d --grid 2x2`
