"""
mograph.mocap.film — render several mocap Actors together as ONE line drawing (shared depth buffer → characters
occlude each other correctly), with a keyframed camera, props and a finish.
"""
from __future__ import annotations

import math

import numpy as np

from ..human.lineart import Camera, Stroke3D, Style, render_lines


def actor_parts(actor, scene, t, pid_base):
    G, X = actor.posed(t, scene)
    ch = actor.ch
    parts = ch.parts(G, X)
    for p in parts:
        p.pid = pid_base + p.pid
    extra = []
    for s in ch.face_strokes(G, X) + (ch.fold_strokes(G, parts) if ch.folds else []):
        if s.pid:
            s.pid = s.pid if s.pid >= pid_base else pid_base + s.pid
        extra.append(s)
    return parts, extra, G, X


def render_scene(scene, t, cam: Camera, style: Style, ss=1.3, fig_px=None, seed=7, extra=(), hidden=()):
    parts, ex, posed = [], list(extra), {}
    for k, a in enumerate(scene.actors.values()):
        if a.name in hidden:
            continue
        p, e, G, X = actor_parts(a, scene, t, 20 * k)
        parts += p
        ex += e
        posed[a.name] = (G, X)
    lf = render_lines(parts, cam, style, extra=ex, ss=ss, fig_px=fig_px or cam.H * 0.8, seed=seed)
    return lf, posed
