# INTEGRATIONS — voice (ElevenLabs / Gemini TTS) and realistic shots (Gemini Omni)

Both are **optional layers**. The engine works 100% offline without them; they never change how you write the spec.
Read this file only when the user wants a real narrator voice or a realistic shot.

## 0. Keys — always ASK the user, never guess
```bash
python3 -m mograph keys                                  # ✓ / ✗ status (values are never printed)
python3 -m mograph keys set ELEVENLABS_API_KEY <key>     # stored in ~/.mograph/keys.json (chmod 600)
python3 -m mograph keys set GEMINI_API_KEY <key>         # one Google AI Studio key = Gemini TTS + Omni
```
**The secure way (preferred): a keys file — the user never pastes a key in the chat.**
Ask the user to create a plain text file `mograph_keys.txt`:
```
ELEVENLABS_API_KEY=sk_...
GEMINI_API_KEY=AIza...
```
(any of `.txt` / `.env` / `.json`; names like "eleven labs:" or `export GOOGLE_API_KEY="..."` and even bare keys are recognised)
and upload it as an attachment or drop it in AI Drive. Then:
```bash
python3 -m mograph keys import /path/to/mograph_keys.txt --yes   # stores in ~/.mograph/keys.json (600), deletes the file
```
If you forget, `get_key` scans the usual upload folders (project, cwd, ~, /mnt/aidrive, uploaded_files…) automatically.
Works the same for any agent that receives this package.

A missing key makes the command stop with **exit code 3** and print a ready Persian message for the user
(where to create the key). Send that message to the user and wait. Never put keys in spec, git, or README.

## 1. Narrator voice — `mograph voice`
Add a `voice` block; the subtitles you already wrote ARE the script.
```jsonc
"voice": {"provider": "elevenlabs", "voice": "<voice_id>", "model": "eleven_v3", "lang": "fa",
          "stability": 0.5, "speed": 1.0, "pause": 0.25}
// or: {"provider": "gemini", "voice": "Charon", "model": "gemini-3.1-flash-tts-preview",
//      "style": "calm, ironic Persian documentary narrator, low voice"}
// or: {"provider": "draft"}   ← free offline placeholder with realistic pacing (use while iterating)
```
Per subtitle line (optional): `"say"` = what is spoken if it differs from the display text
(audio tags `[whispers] [sarcastic] [pause]` for ElevenLabs v3, phonetic spelling for numbers/names), `"style"` = per-line delivery (Gemini).
```bash
python3 -m mograph voice spec.json                 # → voice/vo.wav  +  spec.voiced.json
python3 -m mograph render spec.voiced.json         # render the VOICED spec (never edit it by hand)
python3 -m mograph voices elevenlabs               # list voice ids on the user's account
```
What it does for you:
- one API call per line (with previous/next text for natural continuity), **cached by content**:
  edit one line → only that line is re-synthesised (costs nothing for the rest)
- word-level timing: ElevenLabs character timestamps → exact words; Gemini → faster-whisper if installed, otherwise syllable weighting
- **retiming**: the real voice is slower/faster than your guessed times → chapters, cameras and every numeric
  time are moved through the same map; `w:` word anchors re-lock to the real spoken words automatically
- `audio.vo` is set → music ducks under the voice, mix normalised to −14 LUFS

Which provider: **ElevenLabs** = most natural Persian, exact word timing, voice cloning on the user's account.
**Gemini** = cheaper, 30 voices, style controlled by a sentence. Draft first, real voice once the timing is approved.

## 1b. Voice made outside the SDK — `provider: "file"`
If the agent platform has its own TTS tool (or the user sends a recording), use it without any key:
```jsonc
"voice": {"provider": "file", "pause": 0.45, "clips": [
  {"audio": "voice_src/act1.mp3", "words": "voice_src/act1.words.json", "lines": [0, 3]}]}
```
`words` = word timestamps from any transcriber (`{"words":[{"text","start","end"}]}` or `[{"w","t","end"}]`);
`lines` = which subtitle lines (inclusive, by order) that clip speaks. The clip is cut at word boundaries per line,
snapped to your script spelling, and then everything works as with ElevenLabs (retiming, word anchors, ducking).
Tip: write pauses as `...` between sentences in the TTS text — they become the gaps between lines.

## 2. Realistic shots — `mograph omni` (Gemini Omni Flash, `gemini-omni-1.1-flash`)
Use only where vector line-art is the wrong tool (a real coin, oil gushing, a crowd, a 3D object in space).
**The engine stays the director:** it renders its OWN frames (your exact layout and camera) and sends them to Omni;
the returned clip goes back on the timeline as a **plate**. HUD, subtitles and post-FX stay vector-crisp on top.
```jsonc
"omni": [{"id": "coin_real", "t": "w:سکه", "end": "w:سکه+5",     // 3..10 s
          "task": "edit",        // edit       : restyle the engine's own clip (keeps motion/layout) ← best
                                 // first_last : engine frame at t + at end → Omni animates between
                                 // from_frame : continue from the engine frame at t
                                 // text       : pure b-roll from the prompt
          "prompt": "Turn the white line-art coin into a real heavy gold coin rotating on black velvet.",
          "mode": "replace",     // replace (over the world) | under (behind the vector layer)
          "grade": "noir", "fade": 0.25}]
```
```bash
python3 -m mograph omni spec.json          # DRY RUN — prints the plan; show it to the user (it costs money)
python3 -m mograph omni spec.json --yes    # generate → media/omni/<id>.mp4 (cached; re-runs only on changes)
python3 -m mograph render spec.json        # plates are picked up automatically
```
Rules (Google guidance + gbro-collage-broll experience):
- prompt in **English**, short, ONE change; the engine appends "keep framing/timing … no text/logos/dialogue"
- **never ask Omni for Persian text** — text stays in the vector layer (QA warns)
- one idea per clip, 3–10 s; `edit` keeps your composition best; locked camera unless motion is the point
- edit/extend of uploaded video is unavailable in EEA/CH/UK → use `first_last`/`from_frame` there
- clip not generated yet → the vector version renders in its place (QA warns), so nothing ever breaks

Footage you already have (stock, Omni, the user's own): element `video`
`{"type":"video","src":"media/x.mp4","x":540,"y":900,"w":720,"blend":"screen","grade":"noir","radius":24,"border":true}`
(`"fill":true` = full screen; `blend:"screen"` on black = only the bright parts glow into the line art.)

## 3. Testing without paying
`python3 tools/mock_apis.py &` → local fake ElevenLabs/Gemini/Omni at http://127.0.0.1:8765. Put
`"base_url": "http://127.0.0.1:8765"` in `voice` / `omni_cfg` and any dummy keys. `--provider draft` also works offline.

## Why not HyperFrames / gbro as dependencies
HyperFrames (HeyGen, Apache-2.0) is excellent but is a second engine (Node 22 + headless Chrome + GSAP): two
engines = a confused agent and a heavy sandbox. We took its ideas instead (lint → snapshot → render loop,
deterministic frames, registry of blocks = our widgets). gbro-collage-broll (MIT) taught the Omni workflow:
approval gates before paid generation, first/last-frame interpolation, English prompts, and the "keep everything else the same" guard.
Those are built into `mograph omni` (dry run = gate).
