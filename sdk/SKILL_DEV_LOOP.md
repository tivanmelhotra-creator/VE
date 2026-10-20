# SKILL_DEV_LOOP — the project is ALWAYS under development (mandatory for every agent, every session)

The user's #1 meta-rule: **making a video and improving the engine are the same job.** Every session must leave
the SDK smarter than it found it. A weaker/cheaper model will run this project later — whatever you learn and do
not write down or automate is lost, and the user will have to repeat themselves (they hate that).

## The loop (do it continuously, not at the end)
1. **Notice** — while editing, any time you: hit a bug · tune a number by hand · think "the agent has to remember
   to …" · get user feedback · see a reference detail the engine can't do.
2. **Classify**
   - *Rule a model could forget* → turn it into CODE (solver, guard, default, QA check). Docs are the fallback.
   - *Missing capability* → build it if it fits the turn; otherwise add it to `SKILL_TOOLBOX.md` §5 backlog.
   - *Taste insight about the user* → `USER_TASTE_GUIDE.md`.
   - *Bug you can't fix now* → backlog with repro (file, time, symptom).
3. **Implement** with an automatic default (weak agents never opt in to optional features).
4. **Verify** with your own QC (frames sheet, motion strip, audio QC + numbers). Never claim a fix you didn't look at.
5. **Record** in the same commit:
   - `memory_and_prompts/05_EXPERIENCE_LOG.md` — dated entry: trigger → root cause → fix → how verified → lesson.
   - `SKILL_TOOLBOX.md` row if a tool was added/changed; run `python3 sdk/tools/build_tools_index.py`.
   - `V14_GUIDE.md` if the spec format changed.
6. **Commit** after every step (`git add -A && git commit -m "…"`). The sandbox resets without warning.

## Root-cause habits (lessons that cost the user time before)
- A visual bug is usually a MATH bug: e.g. "ring jerks back" = a spring overshoot fed into an angle; "text runs into
  itself" = no length-vs-circumference limit. Fix the formula, then add a guard so it can't come back.
- Don't tune for one video. Ask "what rule would make ANY footage right?" (e.g. palette from the plate, not copied
  reference colours).
- Measure before trusting a model's opinion (audio QC listeners are inconsistent; band energies & noise floor aren't).
- Prefer a strong proven open-source component over a home-made weak one (RNNoise couldn't kill a river;
  DeepFilterNet3 could). Keep it light: static binaries / ONNX, no torch, total < ~50 MB, put it inside the repo.
- Look at motion as a strip of frames (0.07 s apart), not one still — stutters are invisible in a single frame.

## Entry template (05_EXPERIENCE_LOG.md)
```
## YYYY-MM-DD — <short title>
- Trigger: <user feedback / bug / observation>
- Root cause: <the real reason>
- Fix: <files, functions, default behaviour>
- Verified: <how: sheet, strip, numbers, listener>
- Lesson / rule for next agents: <one line>
```
