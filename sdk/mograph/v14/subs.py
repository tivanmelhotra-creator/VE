"""
mograph.v14.subs — STATEFUL jelly glass subtitles (the @novadent capsule, done properly).

v13.9 audit: the capsule wobbled around the CURRENT phrase width only, so it popped between phrases.
v14: the capsule keeps physical state across phrases:
  * width springs from the previous phrase's width to the new one (under-damped, k=380 c=14)
  * height conserves area (h ∝ w^-0.38) → shrinking = plumps up, growing = stretches thin
  * real frosted glass (blurs the speaker's clothes), teal crescent rim highlights on both end caps
  * karaoke: words pop in on their exact timestamp (jelly spring); the active word gets the accent colour;
    emphasised words ("emph" list or *markup*) use the hero font → inline weight contrast inside every phrase
  * a soft "jelly" SFX is scheduled on each phrase change (pitch from the width delta) — see audio.py

"subtitles": {"anchor": "chest", "offset": [0, 0.35], "size": 46, "max_words": 4, "emph": ["مقدمه", ...],
              "phrases": [["w:درسته", "we:مقاله"], ...]  (optional manual phrasing; default = auto by gaps/punct),
              "hide": [[t0, t1], ...]  (no subs while a big title owns the screen), "rim": "#2DD4BF"}

v21 — SUBTITLE VARIETY & ZERO COLLISION
  "style": "minimal_clean"   (NEW DEFAULT) box-free crisp text, soft legibility shadow, active word in @accent
           "soft_scrim"      feathered dark gradient behind the text (blur 28), no border, no brackets
           "glass_matte"     neutral frosted capsule, hairline rgba(255,255,255,.12) border, NO coloured crescents
           "jelly_crescent"  the legacy @novadent teal-bracketed jelly capsule (only when asked for explicitly)
  AUTO-HIDE: subtitles are hidden (±0.12 s) while an aecomp lower_third, or a chest/belly disc · glass_shape ·
  kinetic_title is on screen — no lower-zone collision, ever.  "auto_hide": false disables it.
"""
from __future__ import annotations

import math

import cairo

from . import glyphs as G
from . import physics as ph
from .compositor import glass, norm_word

SUB_STYLES = ("minimal_clean", "soft_scrim", "glass_matte", "jelly_crescent")
HIDE_PAD = 0.12


def lower_zone_windows(R, pad=HIDE_PAD):
    """[[t0, t1], …] (edited seconds) of every layer that owns the lower zone: aecomp lower thirds and chest/belly
    hero elements.  Used by prepare_subs (auto-hide) and by qa (collision check)."""
    out = []
    for L in R.spec.get("layers", []):
        t = L.get("type")
        lower = (t == "aecomp" and L.get("lower_third")) or \
            (L.get("anchor") in ("chest", "belly") and t in ("disc", "glass_shape", "kinetic_title"))
        if not lower:
            continue
        try:
            a = R.tl.resolve(L.get("t", 0.0))
            b = R.tl.resolve(L.get("out"), R.duration + 1)
        except Exception:  # noqa: BLE001
            continue
        out.append([a - pad, b + pad])
    return out


def prepare_subs(cfg, R):
    S = dict(cfg)
    S["type"] = "subtitles"
    S["pass"] = "front"
    S["z"] = 1000
    S["_t0"], S["_t1"] = 0.0, R.duration + 1
    pal = G.PALETTE
    if cfg.get("palette"):
        pal = dict(G.PALETTE); pal.update(cfg["palette"])
    S["_pal"] = pal
    words = R.tl.words
    emph = {norm_word(w) for w in cfg.get("emph", [])}
    maxw = int(cfg.get("max_words", 4))
    phrases = []
    if cfg.get("phrases"):
        for a, b in cfg["phrases"]:
            ta, tb = R.tl.resolve(a), R.tl.resolve(b)
            ws = [w for w in words if ta - 0.01 <= w["t"] <= tb + 0.01]
            if ws:
                phrases.append(ws)
    else:
        cur = []
        for i, w in enumerate(words):
            if cur and (len(cur) >= maxw or w["t"] - cur[-1]["end"] > 0.35 or w["t"] in R.tl.cuts or w["end"] - cur[0]["t"] > 2.85):
                phrases.append(cur)
                cur = []
            cur.append(w)
            if w["w"].rstrip()[-1:] in "،,.!?؟":
                phrases.append(cur)
                cur = []
        if cur:
            phrases.append(cur)
        # SMART MERGE: a phrase shorter than 0.45 s (e.g. a lone «است،») is unreadable → glue it to its neighbour
        merged = []
        for ph_ in phrases:
            dur = ph_[-1]["end"] - ph_[0]["t"]
            if merged and (dur < 0.45 or len(ph_) == 1) and len(merged[-1]) + len(ph_) <= maxw + 1 \
                    and ph_[0]["t"] not in R.tl.cuts and (ph_[-1]["end"] - merged[-1][0]["t"]) <= 3.1:
                merged[-1] = merged[-1] + ph_
            else:
                merged.append(ph_)
        phrases = merged
    out = []
    for i, ws in enumerate(phrases):
        t0 = ws[0]["t"] - 0.06
        t1 = (phrases[i + 1][0]["t"] - 0.06) if i + 1 < len(phrases) else ws[-1]["end"] + 0.6
        t1 = min(t1, ws[-1]["end"] + 0.9, t0 + 3.40)
        txt = " ".join(("*" + w["w"].strip("،,.!?؟") + "*") if w["n"] in emph else w["w"].strip("،,.") for w in ws)
        b = G.typeset(txt, float(cfg.get("size", 46)), cfg.get("styles", {
            "thin": {"font": "Doran-Medium.ttf", "color": "@ink"},
            "hero": {"scale": 1.32, "glow": 0.35}}), word_gap=0.30, lockup=False)
        out.append({"t0": t0, "t1": t1, "words": ws, "blk": b, "w": b.w, "h": b.h})
    S["_phr"] = out
    S["_hide"] = [[R.tl.resolve(a), R.tl.resolve(b)] for a, b in cfg.get("hide", [])]
    if cfg.get("auto_hide", True):
        S["_hide"] += lower_zone_windows(R)
    st = cfg.get("style", "minimal_clean")
    S["_style"] = st if st in SUB_STYLES else "minimal_clean"
    return S


def _width_at(S, t, i):
    """Spring-interpolated capsule width at time t inside phrase i (stateful across phrases)."""
    P = S["_phr"]
    tgt = P[i]["w"]
    if i == 0:
        p = ph.spring(t - P[0]["t0"], 380, 14)
        return tgt * max(0.0, p), p
    prev = P[i - 1]["w"]
    gap = P[i]["t0"] - P[i - 1]["t1"]
    if gap > 0.4:                                      # long pause → grow from zero
        prev = 0.0
    p = ph.spring(t - P[i]["t0"], 380, 14)
    return prev + (tgt - prev) * p, p


def draw_subs(ctx, S, F):
    t = F.t
    for a, b in S["_hide"]:
        if a <= t <= b:
            return
    P = S["_phr"]
    i = next((k for k, p in enumerate(P) if p["t0"] <= t < p["t1"]), None)
    if i is None:
        return
    ph_ = P[i]
    hw = F.head_px()
    k = hw / float(S.get("head_ref", 300.0)) if S.get("head_scale", False) else 1.0
    ax, ay = F.anchor(S.get("anchor", "chest"))
    off = S.get("offset", [0, 0.35])
    x = ax + off[0] * hw * float(S.get("follow_x", 0.35))
    x = F.W / 2 + (x - F.W / 2) * float(S.get("follow_x", 0.35))
    y = ay + off[1] * hw
    y = min(max(y, F.H * 0.55), F.H * 0.86)
    w, p = _width_at(S, t, i)
    pad_x, pad_y = 0.62 * ph_["h"] + 26, 0.38 * ph_["h"] + 14
    W_ = (w + 2 * pad_x) * k
    base_h = (ph_["h"] + 2 * pad_y) * k
    ratio = (ph_["w"] + 2 * pad_x) / max(1.0, w + 2 * pad_x)
    H_ = base_h * min(1.5, max(0.7, ratio ** 0.38))
    # out (last phrase / before hide)
    a = 1.0
    nxt_gap = (P[i + 1]["t0"] - ph_["t1"]) if i + 1 < len(P) else 9
    if nxt_gap > 0.3 and t > ph_["t1"] - 0.18:
        q = ph.ease_in(min(1, (t - (ph_["t1"] - 0.18)) / 0.18))
        a = 1 - q
        W_ *= 1 - 0.3 * q
        H_ *= 1 - 0.5 * q
    if W_ < 8 or a <= 0.01:
        return
    r = H_ / 2
    x0, x1 = x - W_ / 2, x + W_ / 2

    def path(c):
        c.new_sub_path()
        c.arc(x1 - r, y, r, -math.pi / 2, math.pi / 2)
        c.arc(x0 + r, y, r, math.pi / 2, 3 * math.pi / 2)
        c.close_path()

    style = S.get("_style", "minimal_clean")
    if style == "jelly_crescent":
        glass(ctx, F, path, (x0, y - H_ / 2, x1, y + H_ / 2), blur=16,
              tint=tuple(S.get("tint", (0.02, 0.07, 0.06, 0.55))), lift=0.02, sat=0.7, alpha=a)
        # thin border + teal crescents on the two end caps (legacy look — explicit opt-in only)
        rim = G.rgba(S.get("rim", "@accent2"), S["_pal"])
        ctx.save()
        path(ctx)
        ctx.set_source_rgba(1, 1, 1, 0.14 * a)
        ctx.set_line_width(1.4)
        ctx.stroke()
        for (cx_, a0, a1) in ((x1 - r, -math.pi / 2 + 0.35, math.pi / 2 - 0.35), (x0 + r, math.pi / 2 + 0.35, 3 * math.pi / 2 - 0.35)):
            for lw, aa in ((7, .25), (2.6, 1.0)):
                ctx.arc(cx_, y, r - 2, a0, a1)
                ctx.set_source_rgba(rim[0], rim[1], rim[2], aa * a)
                ctx.set_line_width(lw)
                ctx.set_line_cap(cairo.LINE_CAP_ROUND)
                ctx.stroke()
        ctx.restore()
    elif style == "glass_matte":
        glass(ctx, F, path, (x0, y - H_ / 2, x1, y + H_ / 2), blur=18,
              tint=tuple(S.get("tint", (0.05, 0.05, 0.06, 0.42))), lift=0.03, sat=0.55, alpha=a, chroma=0.0)
        ctx.save()
        path(ctx)
        ctx.set_source_rgba(1, 1, 1, 0.12 * a)
        ctx.set_line_width(1.0)
        ctx.stroke()
        ctx.restore()
    elif style == "soft_scrim":
        # feathered dark pool: stacked translucent ellipses ≈ gaussian (blur ~28 px), zero edge, zero brackets
        ctx.save()
        ex, ey = W_ * 0.5 + 28, H_ * 0.5 + 28
        rg = cairo.RadialGradient(0, 0, 0, 0, 0, 1)
        dark = float(S.get("scrim", 0.55))
        rg.add_color_stop_rgba(0.0, 0, 0, 0, dark * a)
        rg.add_color_stop_rgba(0.55, 0, 0, 0, dark * 0.62 * a)
        rg.add_color_stop_rgba(1.0, 0, 0, 0, 0)
        ctx.translate(x, y)
        ctx.scale(ex, ey)
        ctx.arc(0, 0, 1, 0, 2 * math.pi)
        ctx.set_source(rg)
        ctx.fill()
        ctx.restore()
    # minimal_clean: no background at all — legibility comes from the text shadow below

    # text: per-word pop on its timestamp, active word accent
    ws = ph_["words"]
    act = max((j for j, w in enumerate(ws) if w["t"] <= t + 0.02), default=-1)
    accent = S.get("active", "@accent")

    def m(g):
        s = G.GS()
        wt = ws[g.word]["t"] if g.word < len(ws) else ph_["t0"]
        gl = t - wt + 0.08 - g.wpos * 0.008
        if gl <= 0:
            s.visible = True
            s.alpha = 0.72
            return s
        q = ph.sp(gl, "jelly")
        s.sx = s.sy = 0.55 + 0.45 * q
        s.ty = (1 - min(1, q)) * g.size * 0.25
        if g.word == act and g.sname == "thin":
            s.color = accent
        return s
    sx = min(1.0, w / max(1.0, ph_["w"]))
    shadow = {"alpha": .55, "dy": 2}
    if style == "minimal_clean":
        # legibility WITHOUT a box: three widening dark outline passes (≈ a soft 10–14 px halo) under the glyphs
        def m_dark(g):
            q = m(g)
            q.color = "#000000"
            q.fill_alpha = 0.0
            return q
        for wdt, al in ((14.0, 0.10), (9.0, 0.16), (5.0, 0.26)):
            G.draw_block(ctx, ph_["blk"], x, y + 0.03 * ph_["h"], motion=m_dark, pal=S["_pal"], ectx=None,
                         scale=k * (0.85 + 0.15 * sx), alpha=a * al,
                         stroke={"color": "#000000", "width": wdt})
        shadow = {"alpha": .6, "dy": 2}
    G.draw_block(ctx, ph_["blk"], x, y + 0.03 * ph_["h"], motion=m, pal=S["_pal"], ectx=None,
                 scale=k * (0.85 + 0.15 * sx), alpha=a, shadow=shadow)
