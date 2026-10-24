# mocap packs (animation-only GLB, in the repo, ~21 MB)

| file | source | licence | clips |
|---|---|---|---|
| quaternius_ual1.glb | Quaternius Universal Animation Library 1 | CC0-1.0 | 43: walk, jog, sprint, idle, talking, sitting enter/idle/talk/exit, crouch, jump, kneel (Fixing_Kneeling), pick up, push, punch, roll, dance… |
| quaternius_ual2.glb | Quaternius Universal Animation Library 2 | CC0-1.0 | 43: fold arms, yes/no, phone talk, carry walk, climb, lay-to-idle, consume, throw, farm, sword… |
| mixamo_xbot / michelle / soldier.glb | three.js examples (Mixamo captures) | MIT (three.js) | walk, run, idle, agree, headShake, sad, sneak, samba, … |
| cmu/*.glb | CMU Graphics Lab Motion Capture DB (BVH by una-dinosauria), resampled 30 fps | free for any use (CMU terms) | 49 acting clips incl. paired 2-person takes (comfort / hand on shoulder, sit together, conversation, handshake, quarrel) |
| pirouette.bvh | three.js examples | MIT | ballet |

Use by alias (`mograph/mocap/library.py` ALIASES, 80+) or `pack:clip` (`ual1:Walk_Loop`, `cmu:22_04`).
Add your own: `python3 -m mograph.mocap.strip in.glb out.glb` (Mixamo FBX → `library.convert_fbx`).
