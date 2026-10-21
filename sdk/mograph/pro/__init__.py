"""
mograph.pro — v19 PRO EDITOR (agent-first): the everyday tools of a professional video editor on top of v14.
    color.py    looks (10 built-in) · .cube LUTs (read/write) · white balance · lift/gamma/gain · skin protect ·
                grain · halation · sharpen · letterbox           →  spec "grade": {...}
    broll.py    B-roll cutaways · picture-in-picture · background replace behind the speaker · split screen ·
                12 transitions · Ken Burns · chroma key           →  layer {"type": "broll", ...}
    export.py   SRT / VTT / ASS captions · EDL timeline · chapters · thumbnails · reformat · platform checks
    lint.py     v14 spec linter (unknown types / typo'd keys / bad anchors / missing files) with suggestions
    render.py   parallel, resumable, frame-exact master render (+ captions + timeline report)
    doctor.py   one-shot environment check + restore after a sandbox reset (tracking cache pack/unpack)
Guide: sdk/PRO_EDITOR_GUIDE.md      CLI: python3 -m mograph.pro --help      Tests: python3 -m mograph.pro selftest
"""
