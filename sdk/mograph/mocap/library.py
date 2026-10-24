"""
mograph.mocap.library — where the motion comes from. Nothing is committed to git; everything lands in
~/.cache/mograph_models/mocap (override MOGRAPH_MODELS).

Built-in packs (free, fetched on first use):
    quaternius   CC0-1.0   46 clips, Rigify rig: Walk_Loop, Walk_Formal_Loop, Jog_Fwd_Loop, Sprint_Loop,
                           Idle_Loop, Idle_Talking_Loop, Sitting_Enter/Idle/Talking/Exit, Crouch_*, Jump_*,
                           Fixing_Kneeling, PickUp_Table, Interact, Push_Loop, Punch_*, Roll, Dance_Loop, …
    threejs      MIT       Mixamo-captured samples from three.js: Xbot (walk, run, idle, agree, headShake,
                           sad_pose, sneak_pose), Michelle (SambaDance), Soldier (Idle, Walk, Run)
    bvh_sample   MIT       three.js pirouette.bvh (ballet, BVH reader test)

Your own clips: any Mixamo download — "FBX Binary, Without Skin" works best — or .glb / .gltf / .bvh:
    python3 -m mograph.mocap import ~/Downloads/Talking.fbx --name talking
FBX is converted once with FBX2glTF (godotengine build, fetched on demand, BSD-3) → <cache>/user/<name>.glb.

A motion reference in a spec is "pack:clip" ("quaternius:Walk_Loop", "threejs:Xbot/agree", "user:talking"),
a file path ("media/dance.glb" + "clip"), or a short alias from ALIASES ("walk", "talk", "sit_talk", …).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile

CACHE = os.path.join(os.environ.get("MOGRAPH_MODELS") or os.path.expanduser("~/.cache/mograph_models"), "mocap")

_Q = "https://raw.githubusercontent.com/J-Ponzo/gltf-universal-animation-library/HEAD/glTF/"
_T = "https://cdn.jsdelivr.net/gh/mrdoob/three.js@r160/examples/models/"
PACKS = {
    "quaternius": {"licence": "CC0-1.0", "main": "AnimationLibrary_Godot_Standard.gltf",
                   "files": {"AnimationLibrary_Godot_Standard.gltf": _Q + "AnimationLibrary_Godot_Standard.gltf",
                             "AnimationLibrary_Godot_Standard.bin": _Q + "AnimationLibrary_Godot_Standard.bin"}},
    "threejs": {"licence": "MIT (three.js examples, Mixamo captures)",
                "files": {"Xbot.glb": _T + "gltf/Xbot.glb", "Michelle.glb": _T + "gltf/Michelle.glb",
                          "Soldier.glb": _T + "gltf/Soldier.glb"}},
    "bvh_sample": {"licence": "MIT (three.js examples)", "main": "pirouette.bvh",
                   "files": {"pirouette.bvh": _T + "bvh/pirouette.bvh"}},
}

# short names an editor actually types → (pack ref). Pick the most human take available in the packs.
ALIASES = {
    "walk": "threejs:Xbot/walk", "walk_relaxed": "quaternius:Walk_Loop", "walk_formal": "quaternius:Walk_Formal_Loop",
    "jog": "quaternius:Jog_Fwd_Loop", "run": "threejs:Xbot/run", "sprint": "quaternius:Sprint_Loop",
    "idle": "threejs:Xbot/idle", "idle_breath": "quaternius:Idle_Loop", "talk": "quaternius:Idle_Talking_Loop",
    "agree": "threejs:Xbot/agree", "nod": "threejs:Xbot/agree", "head_shake": "threejs:Xbot/headShake",
    "sad": "threejs:Xbot/sad_pose", "sneak": "threejs:Xbot/sneak_pose",
    "sit_down": "quaternius:Sitting_Enter", "sit": "quaternius:Sitting_Idle_Loop",
    "sit_talk": "quaternius:Sitting_Talking_Loop", "stand_up": "quaternius:Sitting_Exit",
    "kneel_work": "quaternius:Fixing_Kneeling", "pick_up": "quaternius:PickUp_Table", "interact": "quaternius:Interact",
    "push": "quaternius:Push_Loop", "crouch": "quaternius:Crouch_Idle_Loop", "crouch_walk": "quaternius:Crouch_Fwd_Loop",
    "jump_start": "quaternius:Jump_Start", "jump_air": "quaternius:Jump_Loop", "jump_land": "quaternius:Jump_Land",
    "dance": "threejs:Michelle/SambaDance", "dance_loop": "quaternius:Dance_Loop", "punch": "quaternius:Punch_Cross",
    "jab": "quaternius:Punch_Jab", "roll": "quaternius:Roll", "hit": "quaternius:Hit_Chest",
    "drive": "quaternius:Driving_Loop", "pirouette": "bvh_sample:take",
}


def _download(url, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(tmp, dst)


def fetch(pack: str) -> str:
    P = PACKS[pack]
    d = os.path.join(CACHE, pack)
    for fn, url in P["files"].items():
        p = os.path.join(d, fn)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            print(f"[mocap] fetching {pack}/{fn} …", flush=True)
            _download(url, p)
    return d


PACK_DIR = os.path.join(os.path.dirname(__file__), "packs")       # IN THE REPO (animation-only GLBs, ~20 MB)
LOCAL = {   # pack name → file inside PACK_DIR  (see packs/README.md for licences and clip lists)
    "ual1": "quaternius_ual1.glb", "ual2": "quaternius_ual2.glb", "xbot": "mixamo_xbot.glb",
    "michelle": "mixamo_michelle.glb", "soldier": "mixamo_soldier.glb", "pirouette": "pirouette.bvh",
}
ALIASES.update({
    "walk_relaxed": "ual1:Walk_Loop", "walk_formal": "ual1:Walk_Formal_Loop", "jog": "ual1:Jog_Fwd_Loop",
    "sprint": "ual1:Sprint_Loop", "idle_breath": "ual1:Idle_Loop", "talk": "ual1:Idle_Talking_Loop",
    "sit_down": "ual1:Sitting_Enter", "sit": "ual1:Sitting_Idle_Loop", "sit_talk": "ual1:Sitting_Talking_Loop",
    "stand_up": "ual1:Sitting_Exit", "kneel_work": "ual1:Fixing_Kneeling", "pick_up": "ual1:PickUp_Table",
    "interact": "ual1:Interact", "push": "ual1:Push_Loop", "crouch": "ual1:Crouch_Idle_Loop",
    "crouch_walk": "ual1:Crouch_Fwd_Loop", "jump_start": "ual1:Jump_Start", "jump_air": "ual1:Jump_Loop",
    "jump_land": "ual1:Jump_Land", "dance_loop": "ual1:Dance_Loop", "punch": "ual1:Punch_Cross",
    "fold_arms": "ual2:Idle_FoldArms_Loop", "yes": "ual2:Yes", "no": "ual2:Idle_No_Loop",
    "phone": "ual2:Idle_TalkingPhone_Loop", "carry_walk": "ual2:Walk_Carry_Loop", "climb": "ual2:ClimbUp_1m",
    "lay_to_idle": "ual2:LayToIdle", "open_chest": "ual2:Chest_Open", "consume": "ual2:Consume",
    "throw": "ual2:OverhandThrow", "plant_seed": "ual2:Farm_PlantSeed", "water_plants": "ual2:Farm_Watering",
    "walk": "xbot:walk", "run": "xbot:run", "idle": "xbot:idle", "agree": "xbot:agree", "nod": "xbot:agree",
    "head_shake": "xbot:headShake", "sad": "xbot:sad_pose", "sneak": "xbot:sneak_pose",
    "dance": "michelle:SambaDance", "pirouette": "pirouette:take",
    # CMU (real captures, 120 fps → 30 fps): social / acting
    "comfort_hand_on_shoulder": "cmu:22_04", "comforted": "cmu:23_04", "comfort_one_hand": "cmu:22_07",
    "shelter_child": "cmu:22_10", "sheltered_child": "cmu:23_10", "conversation": "cmu:18_08",
    "conversation_walk": "cmu:18_09", "walk_shake_hands": "cmu:18_01", "slow_walk_stop": "cmu:16_33",
    "slow_walk_stop2": "cmu:16_34", "slow_walk": "cmu:07_04", "slow_walk2": "cmu:08_04", "sad_walk": "cmu:82_10",
    "happy_walk": "cmu:82_12", "happy_go_lucky_walk": "cmu:104_11", "sit_stool_stand": "cmu:13_01",
    "sit_chin_in_hand": "cmu:13_04", "sit_fidget": "cmu:13_05", "sit_hand_on_chin": "cmu:14_30",
    "sit_down_get_up": "cmu:143_18", "sit_down_stretch": "cmu:114_16", "jump_reach": "cmu:13_10",
    "lean_reach": "cmu:15_07", "walk_turn_back": "cmu:36_02", "very_happy": "cmu:79_69", "happy": "cmu:142_08",
    "sad_body": "cmu:142_15", "sad_walk2": "cmu:91_13", "wave": "cmu:13_26", "bow": "cmu:111_02",
    "scoop_up": "cmu:02_06", "face_in_hands": "cmu:22_03", "kneel_comfort": "cmu:23_03",
})


def resolve(ref: str, clip: str | None = None, base_dir: str | None = None) -> tuple[str, str | None]:
    """motion reference → (file path, clip name)."""
    ref = ALIASES.get(ref, ref)
    if ":" in ref and not os.path.exists(ref) and not ref[1:3] == ":\\":
        pack, rest = ref.split(":", 1)
        if pack == "cmu":
            return os.path.join(PACK_DIR, "cmu", f"cmu_{rest}.glb"), clip
        if pack in LOCAL:
            return os.path.join(PACK_DIR, LOCAL[pack]), clip or (None if rest == "take" else rest)
        if pack == "user":
            p = os.path.join(CACHE, "user", rest if rest.endswith((".glb", ".gltf", ".bvh")) else rest + ".glb")
            return p, clip
        d = fetch(pack)
        if "/" in rest:
            fn, c = rest.split("/", 1)
            for ext in ("", ".glb", ".gltf", ".bvh"):
                if os.path.exists(os.path.join(d, fn + ext)):
                    return os.path.join(d, fn + ext), clip or c
        main = PACKS[pack].get("main")
        return os.path.join(d, main), clip or (None if rest == "take" and main.endswith(".bvh") else rest)
    p = ref
    if base_dir and not os.path.isabs(p):
        p = os.path.join(base_dir, p)
    p = os.path.expanduser(p)
    if p.lower().endswith(".fbx"):
        p = convert_fbx(p)
    return p, clip


# --------------------------------------------------------------------------------------------- FBX
FBX2GLTF_URL = "https://github.com/godotengine/FBX2glTF/releases/download/v0.13.1/FBX2glTF-linux-x86_64.zip"


def fbx2gltf_bin() -> str:
    exe = shutil.which("FBX2glTF")
    if exe:
        return exe
    b = os.path.join(CACHE, "bin", "FBX2glTF")
    if not os.path.isfile(b):
        z = os.path.join(CACHE, "bin", "fbx2gltf.zip")
        print("[mocap] fetching FBX2glTF (godotengine, BSD-3) …", flush=True)
        _download(FBX2GLTF_URL, z)
        with zipfile.ZipFile(z) as zz:
            for n in zz.namelist():
                if n.endswith("FBX2glTF-linux-x86_64") or n.endswith("/FBX2glTF"):
                    with zz.open(n) as s, open(b, "wb") as f:
                        shutil.copyfileobj(s, f)
        os.chmod(b, 0o755)
        os.remove(z)
    return b


def convert_fbx(src: str, dst: str | None = None) -> str:
    dst = dst or os.path.join(CACHE, "user", os.path.splitext(os.path.basename(src))[0] + ".glb")
    if os.path.isfile(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "o")
        subprocess.run([fbx2gltf_bin(), "--binary", "--input", src, "--output", out], check=True,
                       stdout=subprocess.DEVNULL)
        got = out + ".glb" if os.path.exists(out + ".glb") else out
        shutil.move(got, dst)
    return dst


def import_clip(src: str, name: str | None = None) -> str:
    name = name or os.path.splitext(os.path.basename(src))[0]
    ext = os.path.splitext(src)[1].lower()
    dst = os.path.join(CACHE, "user", name + (".glb" if ext in (".fbx", ".glb") else ext))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if ext == ".fbx":
        convert_fbx(src, dst)
    else:
        shutil.copy(src, dst)
        if ext == ".gltf":                            # copy sidecar buffers
            import json
            for b in json.load(open(src)).get("buffers", []):
                u = b.get("uri", "")
                if u and not u.startswith("data:"):
                    shutil.copy(os.path.join(os.path.dirname(src), u), os.path.join(os.path.dirname(dst), u))
    return dst


def list_clips(pack=None) -> dict:
    from .retarget import load_motion
    out = {}
    for p in ([pack] if pack else list(PACKS) + ["user"]):
        if p == "user":
            d = os.path.join(CACHE, "user")
            files = [os.path.join(d, f) for f in sorted(os.listdir(d))] if os.path.isdir(d) else []
            files = [f for f in files if f.endswith((".glb", ".gltf", ".bvh"))]
        else:
            d = fetch(p)
            files = [os.path.join(d, f) for f in PACKS[p]["files"] if f.endswith((".glb", ".gltf", ".bvh"))]
        for f in files:
            sk = load_motion(f)
            key = p if (p != "threejs" and p != "user") else f"{p}:{os.path.splitext(os.path.basename(f))[0]}"
            out.setdefault(key, {}).update({k: round(v.duration, 2) for k, v in sk.clips.items()})
    return out
