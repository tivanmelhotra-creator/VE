"""Example plugin: a speech bubble element. Enable with  "plugins": ["../plugins/example_plugin.py"]"""
from mograph.elements import element, xy
from mograph.geometry import draw_path
from mograph.text import draw_text, text_bbox


@element("speech_bubble", dur=0.6, enter="pop", sfx="pop")
def speech_bubble(ctx, el, s):
    x, y = xy(el, s.t)
    size = el.get("size", 40)
    tb = text_bbox(el["text"], 0, 0, size, max_width=el.get("max_width", 600))
    w, h = tb[2] - tb[0] + 70, tb[3] - tb[1] + 50
    tail = -1 if el.get("tail", "left") == "left" else 1
    d = (f"M {-w/2+24} {-h/2} H {w/2-24} Q {w/2} {-h/2} {w/2} {-h/2+24} V {h/2-24} Q {w/2} {h/2} {w/2-24} {h/2} "
         f"H {tail*w/4+20} L {tail*w/3} {h/2+36} L {tail*w/4-20} {h/2} H {-w/2+24} Q {-w/2} {h/2} {-w/2} {h/2-24} "
         f"V {-h/2+24} Q {-w/2} {-h/2} {-w/2+24} {-h/2} Z")
    ctx.save(); ctx.translate(x, y + s.dy); ctx.scale(s.scale, s.scale)
    draw_path(ctx, d, 0, 0, 1, anchor="none", stroke=s.color, width=2.6, fill="#000", alpha=s.alpha)
    draw_text(ctx, el["text"], 0, 0, size=size, color=s.color, alpha=s.alpha, max_width=el.get("max_width", 600))
    ctx.restore()
    return (x - w/2, y - h/2, x + w/2, y + h/2 + 36)
