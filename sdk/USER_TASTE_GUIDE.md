# USER TASTE GUIDE — how to work with this user (written from experience, v14.3)

> Read this BEFORE anything visual. It overrides older laws where they conflict
> (e.g. LAW 4 "multi-font" in USER_TASTE_AND_DNA_LAWS.md is SUPERSEDED by §3 here).

## 1. Who the user is & how to communicate
- Persian speaker, often voice-to-text → typos/half-words; read for intent, not letters.
- **Reply in Persian, short, concrete.** End with: what changed + why + links.
- Standing order: **"سوال نپرس، به استدلالت اعتماد کن، برو."** Never block on questions;
  pick the professional default, state it in the report.
- They send **reference images/videos** and describe details ("دقت کنی..."). Each such
  detail is a spec item — implement ALL of them, then turn it into an engine default.
- They praise rarely and briefly ("بقیه عالی بود") — that means: don't touch the rest.
- Sandbox resets often; they send backup tarball links → restore, delete tarball, continue.

## 2. What "professional" means to them
- Looks like a top Persian Instagram editor (Novadent-style talking head): depth sandwich,
  text behind the head, physical motion (springs/jelly), no cheap templates.
- **Nothing covers the face/mouth.** Ever. (guard + QA enforce it.)
- Motion must be **monotonic & intentional** — any back-and-forth jerk is a bug.
- Text never collides with itself (rings: coverage ≤40%, facing fade).
- Behind every element: dark scrim + subtle world-locked grid + coloured light falloff
  that matches the hero word colour (stage.py). This is THEIR taste — keep it.
- Clean audio: no river/room noise (DeepFilterNet3 default). −14 LUFS.

## 3. Typography
- **Doran family ONLY**, horizontally stretched **+12%** (`STRETCH={"doran":1.12}`).
- Hierarchy by weight/size/colour inside Doran (Thin → ExtraBold), lockups with a hero word.
- Text that belongs to a shape goes INSIDE the shape (disc band, halo inner disc).
  Off-centre text inside a circle bends along the circle edge (arc field), centred text
  may be a stacked lockup (small / HERO / small).

## 4. Colour
- **Designed from the footage** (`palette.py auto_palette`), never copied from the reference.
- Accent avoids the clothing hues; WCAG contrast; warm ink for text.
- **Red (@alert) = alert/emphasis** moments (halo, "نه", "آخر صف").

## 5. Using references
- Extract the LOGIC (anatomy, layering, timing), not the pixels/colours.
- Halo anatomy learned from their ref: small bright red disc + bigger translucent disc,
  both centred slightly ABOVE the head; text inside the bright disc; head occludes the
  lower part; soft VFX (glow pulse / rim bloom / light sweep).

## 6. Delivery checklist
1. QA clean (`python3 -m mograph.v14 qa spec`), frame sheets reviewed by you.
2. Full render (render_chunks.sh), CRF 20 compress, showcase sheet, SDK zip.
3. Upload via UploadFileWrapper; Persian report: changes / why / links.
4. Log lessons in `memory_and_prompts/05_EXPERIENCE_LOG.md` (SKILL_DEV_LOOP.md).
