"""
mograph.mocap.bvh — BVH motion-capture reader → the same Skeleton / Clip API as mograph.mocap.gltf.

BVH (CMU, Truebones, Rokoko, Mixamo-exported…): HIERARCHY of OFFSET / CHANNELS, then MOTION frames of Euler
angles (any channel order, intrinsic, degrees) and root positions. Units are whatever the file uses (CMU = inches
× something, Mixamo = cm); the retargeter only uses ratios, so units do not matter. The rest pose = all zero
rotations (the BVH convention); offsets give the bone directions.
"""
from __future__ import annotations

import math

import numpy as np

from .gltf import Channel, Clip, Skeleton


def _rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _mat_to_quat(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()          # xyzw


def load(path, name=None) -> Skeleton:
    with open(path) as f:
        tok = f.read().split()
    i = 0
    names, parent, offs, chans = [], [], [], []
    stack = []
    while tok[i] != "MOTION":
        t = tok[i]
        if t in ("ROOT", "JOINT"):
            names.append(tok[i + 1])
            parent.append(stack[-1] if stack else -1)
            offs.append(np.zeros(3))
            chans.append([])
            i += 2
        elif t == "End":                                # End Site → a leaf node (gives the last bone a direction)
            names.append(names[stack[-1]] + "_End")
            parent.append(stack[-1])
            offs.append(np.zeros(3))
            chans.append([])
            i += 2
        elif t == "{":
            stack.append(len(names) - 1)
            i += 1
        elif t == "}":
            stack.pop()
            i += 1
        elif t == "OFFSET":
            offs[-1] = np.array([float(tok[i + 1]), float(tok[i + 2]), float(tok[i + 3])])
            i += 4
        elif t == "CHANNELS":
            n = int(tok[i + 1])
            # the CHANNELS line belongs to the joint currently open (top of stack)
            chans[stack[-1]] = tok[i + 2:i + 2 + n]
            i += 2 + n
        else:
            i += 1
    i += 1
    nf = int(tok[i + 1])
    ft = float(tok[i + 4])
    i += 5
    ncol = sum(len(c) for c in chans)
    data = np.array(tok[i:i + nf * ncol], float).reshape(nf, ncol)
    N = len(names)
    times = np.arange(nf) * ft
    T = np.array(offs)
    R = np.tile([0, 0, 0, 1.0], (N, 1))
    S = np.ones((N, 3))
    channels = []
    col = 0
    for j in range(N):
        cs = chans[j]
        if not cs:
            continue
        block = data[:, col:col + len(cs)]
        col += len(cs)
        pos = np.tile(offs[j], (nf, 1))
        has_pos = False
        rot_order = [(k, c[0]) for k, c in enumerate(cs) if c.endswith("rotation")]
        for k, c in enumerate(cs):
            if c.endswith("position"):
                has_pos = True
                pos[:, "XYZ".index(c[0])] = block[:, k]
        if rot_order:                                  # BVH = intrinsic rotations in channel order (vectorised)
            from scipy.spatial.transform import Rotation
            seq = "".join(ax for _, ax in rot_order)
            rots = Rotation.from_euler(seq, block[:, [k for k, _ in rot_order]], degrees=True).as_quat()
        if has_pos:
            channels.append(Channel(j, "translation", times, pos))
        if rot_order:
            channels.append(Channel(j, "rotation", times, rots))
    sk = Skeleton(names, np.array(parent), T, R, S, list(range(N)), source=path)
    sk.clips[name or "take"] = Clip(name or "take", channels, float(times[-1]) if nf else 0.0)
    return sk
