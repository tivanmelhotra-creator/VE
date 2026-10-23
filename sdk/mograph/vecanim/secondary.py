"""Secondary motion (follow-through / overlapping action) on vector points.

Hair, loose strands, collars and hems are *driven* by the primary deformation: each region
carries its points through the ARAP solve like everything else, then a damped spring adds the
lagging part of the motion — the classic "drag" of 2D animation (Williams, Animator's Survival
Kit: overlapping action).  Deterministic: the spring is integrated from t = 0 on a fixed
sub-step each time a frame is asked for in order, cached for random access.

    SecondaryRegion(name, center, radius, root=(x,y), stiffness, damping, gain, wind)
    - weight per point = smooth falloff inside the radius x distance from `root` (tips swing
      more than roots, like a strand anchored at the scalp)
    - the driver is the mean displacement of the region's points produced by the primary
      stack; the spring state x follows it with lag; offset = (x - driver) * weight * gain
    - optional `wind`: AE-style wiggle added to the target (idle life)
"""
import numpy as np
from . import anim as A


class SecondaryRegion:
    def __init__(self, name, center, radius, root=None, stiffness=38.0, damping=7.0, gain=1.0,
                 wind=0.0, wind_freq=0.45, seed=0, max_off=14.0):
        self.name = name
        self.center = np.asarray(center, float)
        self.radius = float(radius)
        self.root = np.asarray(root if root is not None else center, float)
        self.k, self.c, self.gain = stiffness, damping, gain
        self.wind = A.wiggle(wind_freq, wind, seed=seed) if wind else None
        self.wind2 = A.wiggle(wind_freq * 0.7, wind * 0.6, seed=seed + 11) if wind else None
        self.max_off = max_off

    def weights(self, P0):
        d = np.linalg.norm(P0 - self.center, axis=1)
        u = np.clip(1 - d / self.radius, 0, 1)
        fall = u * u * (3 - 2 * u)
        reach = np.linalg.norm(P0 - self.root, axis=1)
        reach = np.clip(reach / max(reach[fall > 0].max() if (fall > 0).any() else 1, 1e-6), 0, 1)
        return fall * (0.15 + 0.85 * reach)


class Secondary:
    """Attached to a Character: `ch.secondary = Secondary(ch, regions)`."""
    def __init__(self, ch, regions, fps=30, sub=4):
        self.ch, self.regions, self.dt = ch, regions, 1.0 / (fps * sub)
        self.sub, self.fps = sub, fps
        self.W = [r.weights(ch.P0) for r in regions]
        self.idx = [np.nonzero(w > 1e-3)[0] for w in self.W]
        self.cache = {}
        self.state = None        # (t, x, v per region)

    def _driver(self, Q):
        return [Q[i].mean(0) - self.ch.P0[i].mean(0) if len(i) else np.zeros(2) for i in self.idx]

    def offsets(self, t, Q):
        """Q: primary-deformed points at time t. Returns additive offsets (N,2)."""
        key = round(t * self.fps * self.sub)
        drv = self._driver(Q)
        if self.state is not None and key - self.state[0] > self.sub * 2:
            self.state = None          # random access (preview): restart at rest on the driver
        if self.state is None or key < self.state[0]:
            self.state = (key, [d.copy() for d in drv], [np.zeros(2) for _ in drv], drv)
        k0, xs, vs, prev = self.state
        steps = key - k0
        for s in range(steps):
            a = (s + 1) / steps
            for j, r in enumerate(self.regions):
                tgt = prev[j] + (drv[j] - prev[j]) * a
                tt = (k0 + s + 1) * self.dt
                if r.wind is not None:
                    tgt = tgt + np.array([r.wind(tt), r.wind2(tt) * 0.5])
                acc = r.k * (tgt - xs[j]) - r.c * vs[j]
                vs[j] = vs[j] + acc * self.dt
                xs[j] = xs[j] + vs[j] * self.dt
        self.state = (key, xs, vs, drv)
        out = np.zeros_like(Q)
        for j, r in enumerate(self.regions):
            lag = xs[j] - drv[j]
            n = np.linalg.norm(lag)
            if n > r.max_off:
                lag = lag / n * r.max_off
            out += self.W[j][:, None] * lag[None] * r.gain
        return out
