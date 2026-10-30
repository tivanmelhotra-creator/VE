"""
toonface.kit — v31 HAND-CRAFTED 2.5D puppet kit (Moho / Toon Boom style), replaces the formula face for production.

    plate25.py     flat raster plate → clean plate (far eye fully removed + cheek contour re-stroked), alpha,
                   foreground hair RGBA (forelocks composited AFTER brows/eyes)
    shapes.py      authored key-shapes per character and per yaw key (same point count → morphable, CANDIDE-like
                   fixed topology)
    draw25.py      cairo renderer: wedge lids, lash wings, clipped pupils + glint, hair-coloured brows with ink rim,
                   hooked 3/4 nose, asymmetric crimson mouths with teeth band + tongue
    visemes.py     14 mouth key-shapes (same topology) + Persian G2P → viseme track
    acting.py      blinks / saccades / mood keys / head micro-bob from audio energy / stepped breathing (on 2s)
    gate.py        5 s quality-gate render + side-by-side comparison sheet
"""
