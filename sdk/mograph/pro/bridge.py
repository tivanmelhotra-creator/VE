"""mograph.pro.bridge — registers the v19 PRO EDITOR layers in the v14 renderer (DRAW + PREPARE registries)."""
from __future__ import annotations


def _prep_broll(L, R):
    from . import broll
    return broll.prepare(L, R)


def _draw_broll(ctx, L, F):
    from . import broll
    return broll.draw_v14(ctx, L, F)


def register_v14(DRAW, PREPARE):
    DRAW["broll"] = _draw_broll
    PREPARE["broll"] = _prep_broll
