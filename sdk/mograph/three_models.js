// ============================================================================
// mograph/three_models.js — VE v20 Standalone Three.js Procedural 3D Generators
// Blender-style procedural 3D mesh + PBR material generators for WebGL / Canvas
// ============================================================================

function createStudioMaterials(THREE) {
  return {
    gold: new THREE.MeshStandardMaterial({
      color: 0xf7be48,
      metalness: 0.88,
      roughness: 0.16,
    }),
    brass: new THREE.MeshStandardMaterial({
      color: 0xd99b32,
      metalness: 0.82,
      roughness: 0.24,
    }),
    matteBlack: new THREE.MeshStandardMaterial({
      color: 0x1c1f26,
      metalness: 0.25,
      roughness: 0.35,
    }),
    chrome: new THREE.MeshStandardMaterial({
      color: 0xe2e8f0,
      metalness: 0.94,
      roughness: 0.10,
    }),
    glass: new THREE.MeshPhysicalMaterial({
      color: 0xe6f4ff,
      transparent: true,
      opacity: 0.36,
      roughness: 0.06,
      metalness: 0.05,
      reflectivity: 0.95,
      clearcoat: 1.0,
      clearcoatRoughness: 0.05,
    }),
    crystal: new THREE.MeshPhysicalMaterial({
      color: 0xd8e2f2,
      transparent: true,
      opacity: 0.68,
      roughness: 0.10,
      metalness: 0.35,
      clearcoat: 1.0,
      flatShading: true,
    }),
    emerald: new THREE.MeshPhysicalMaterial({
      color: 0x10b981,
      emissive: 0x043824,
      emissiveIntensity: 0.45,
      roughness: 0.08,
      metalness: 0.20,
      clearcoat: 1.0,
      flatShading: true,
    }),
    sand: new THREE.MeshStandardMaterial({
      color: 0xf59e0b,
      emissive: 0x78350f,
      emissiveIntensity: 0.35,
      roughness: 0.38,
      metalness: 0.15,
    }),
  };
}

// 1. PROCEDURAL 3D HOURGLASS (hourglass_3d)
function buildHourglass3D(THREE, mats) {
  const group = new THREE.Group();

  [-1, 1].forEach((sgn) => {
    const outerCap = new THREE.Mesh(new THREE.CylinderGeometry(0.68, 0.68, 0.08, 48), mats.gold);
    outerCap.position.y = sgn * 0.94;
    outerCap.userData.explodeVec = new THREE.Vector3(0, sgn * 0.45, 0);
    group.add(outerCap);

    const innerCap = new THREE.Mesh(new THREE.CylinderGeometry(0.58, 0.58, 0.07, 48), mats.brass);
    innerCap.position.y = sgn * 0.86;
    innerCap.userData.explodeVec = new THREE.Vector3(0, sgn * 0.32, 0);
    group.add(innerCap);

    const bezelRing = new THREE.Mesh(new THREE.TorusGeometry(0.58, 0.024, 16, 48), mats.gold);
    bezelRing.rotation.x = Math.PI / 2;
    bezelRing.position.y = sgn * 0.825;
    bezelRing.userData.explodeVec = new THREE.Vector3(0, sgn * 0.25, 0);
    group.add(bezelRing);
  });

  for (let k = 0; k < 4; k++) {
    const ang = (k * Math.PI) / 2 + Math.PI / 4;
    const px = Math.cos(ang) * 0.53;
    const pz = Math.sin(ang) * 0.53;
    const pillarGroup = new THREE.Group();
    pillarGroup.position.set(px, 0, pz);
    pillarGroup.userData.explodeVec = new THREE.Vector3(px * 0.7, 0, pz * 0.7);

    const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 1.72, 24), mats.gold);
    pillarGroup.add(rod);

    [-0.78, 0, 0.78].forEach((ry) => {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.048, 0.015, 12, 24), mats.brass);
      ring.rotation.x = Math.PI / 2;
      ring.position.y = ry;
      pillarGroup.add(ring);
    });
    group.add(pillarGroup);
  }

  const topSandPts = [
    new THREE.Vector2(0.001, 0.03),
    new THREE.Vector2(0.045, 0.08),
    new THREE.Vector2(0.28, 0.32),
    new THREE.Vector2(0.38, 0.48),
    new THREE.Vector2(0.001, 0.46),
  ];
  group.add(new THREE.Mesh(new THREE.LatheGeometry(topSandPts, 40), mats.sand));

  const botSandPts = [
    new THREE.Vector2(0.001, -0.81),
    new THREE.Vector2(0.42, -0.81),
    new THREE.Vector2(0.40, -0.66),
    new THREE.Vector2(0.20, -0.42),
    new THREE.Vector2(0.001, -0.32),
  ];
  group.add(new THREE.Mesh(new THREE.LatheGeometry(botSandPts, 40), mats.sand));

  const stream = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.022, 0.44, 16), mats.sand);
  stream.position.y = -0.16;
  group.add(stream);

  const glassPts = [];
  for (let i = 0; i <= 48; i++) {
    const y = -0.82 + (1.64 * i) / 48;
    const ay = Math.abs(y);
    const r = 0.062 + 0.395 * Math.pow(Math.sin(Math.min(1, ay / 0.82) * Math.PI * 0.82), 1.35);
    glassPts.push(new THREE.Vector2(r, y));
  }
  group.add(new THREE.Mesh(new THREE.LatheGeometry(glassPts, 56), mats.glass));

  return group;
}

// 2. PROCEDURAL 3D MICROBLADING PEN (microblading_pen_3d)
function buildMicrobladingPen3D(THREE, mats) {
  const pen = new THREE.Group();

  // profile runs bottom → top so LatheGeometry normals face OUTWARD (top → bottom renders inside-out)
  const capPts = [
    new THREE.Vector2(0.001, 0.74),
    new THREE.Vector2(0.18, 0.74),
    new THREE.Vector2(0.165, 0.93),
    new THREE.Vector2(0.09, 1.02),
    new THREE.Vector2(0.001, 1.04),
  ];
  const cap = new THREE.Mesh(new THREE.LatheGeometry(capPts, 40), mats.matteBlack);
  cap.userData.explodeVec = new THREE.Vector3(0, 0.45, 0);
  pen.add(cap);

  const goldBarrel = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.165, 0.56, 48), mats.gold);
  goldBarrel.position.y = 0.46;
  goldBarrel.userData.explodeVec = new THREE.Vector3(0, 0.25, 0);
  pen.add(goldBarrel);

  const midRing = new THREE.Mesh(new THREE.TorusGeometry(0.17, 0.015, 14, 40), mats.brass);
  midRing.rotation.x = Math.PI / 2;
  midRing.position.y = 0.18;
  pen.add(midRing);

  const midPts = [];
  for (let i = 0; i <= 24; i++) {
    const t = i / 24;
    const y = -0.34 + 0.52 * t;
    const r = 0.154 - 0.025 * Math.sin(t * Math.PI) + 0.012 * t;
    midPts.push(new THREE.Vector2(r, y));
  }
  pen.add(new THREE.Mesh(new THREE.LatheGeometry(midPts, 44), mats.matteBlack));

  const collar = new THREE.Mesh(new THREE.CylinderGeometry(0.158, 0.164, 0.12, 44), mats.gold);
  collar.position.y = -0.40;
  collar.userData.explodeVec = new THREE.Vector3(0, -0.15, 0);
  pen.add(collar);

  const gripGroup = new THREE.Group();
  gripGroup.userData.explodeVec = new THREE.Vector3(0, -0.30, 0);
  const gripCore = new THREE.Mesh(new THREE.CylinderGeometry(0.148, 0.128, 0.34, 40), mats.matteBlack);
  gripCore.position.y = -0.62;
  gripGroup.add(gripCore);
  for (let k = 0; k < 5; k++) {
    const gr = new THREE.Mesh(new THREE.TorusGeometry(0.144 - k * 0.003, 0.014, 12, 36), mats.matteBlack);
    gr.rotation.x = Math.PI / 2;
    gr.position.y = -0.49 - k * 0.06;
    gripGroup.add(gr);
  }
  pen.add(gripGroup);

  const nozzleGroup = new THREE.Group();
  nozzleGroup.userData.explodeVec = new THREE.Vector3(0, -0.55, 0);
  const chuck = new THREE.Mesh(new THREE.CylinderGeometry(0.116, 0.052, 0.24, 36), mats.chrome);
  chuck.position.y = -0.90;
  nozzleGroup.add(chuck);

  const bladeHolder = new THREE.Mesh(new THREE.BoxGeometry(0.022, 0.24, 0.068), mats.gold);
  bladeHolder.position.set(0.01, -1.08, 0);
  bladeHolder.rotation.z = 0.22;
  nozzleGroup.add(bladeHolder);
  pen.add(nozzleGroup);

  const wrapper = new THREE.Group();
  pen.rotation.z = -0.64;
  wrapper.add(pen);
  return wrapper;
}

// 3. PROCEDURAL 3D EMERALD & GOLD TREE (emerald_tree_3d)
function buildEmeraldTree3D(THREE, mats) {
  const group = new THREE.Group();

  const glassBase = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.46, 0.20, 48), mats.glass);
  glassBase.position.y = -0.88;
  glassBase.userData.explodeVec = new THREE.Vector3(0, -0.35, 0);
  group.add(glassBase);

  const goldPlinth = new THREE.Mesh(new THREE.CylinderGeometry(0.33, 0.36, 0.07, 48), mats.gold);
  goldPlinth.position.y = -0.74;
  goldPlinth.userData.explodeVec = new THREE.Vector3(0, -0.20, 0);
  group.add(goldPlinth);

  const branchPaths = [
    { pts: [[0, -0.74, 0], [0.04, -0.45, 0.03], [-0.04, -0.12, -0.02], [0.02, 0.25, 0.03], [0, 0.85, 0]], r: 0.085 },
    { pts: [[-0.02, -0.34, 0], [-0.26, -0.22, 0.09], [-0.54, -0.04, 0.16], [-0.74, 0.18, 0.12]], r: 0.048 },
    { pts: [[0.02, -0.30, 0], [0.28, -0.18, -0.09], [0.56, 0.0, -0.16], [0.76, 0.22, -0.12]], r: 0.048 },
    { pts: [[-0.03, -0.08, 0.02], [-0.28, 0.10, -0.12], [-0.54, 0.34, -0.20], [-0.68, 0.58, -0.14]], r: 0.040 },
    { pts: [[0.03, -0.04, -0.02], [0.28, 0.14, 0.12], [0.54, 0.38, 0.20], [0.68, 0.62, 0.14]], r: 0.040 },
    { pts: [[-0.01, 0.18, 0.02], [-0.20, 0.40, 0.15], [-0.38, 0.66, 0.18], [-0.46, 0.82, 0.14]], r: 0.032 },
    { pts: [[0.01, 0.20, -0.02], [0.20, 0.42, -0.15], [0.38, 0.68, -0.18], [0.46, 0.84, -0.14]], r: 0.032 },
  ];

  branchPaths.forEach((b) => {
    const vecs = b.pts.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
    const curve = new THREE.CatmullRomCurve3(vecs);
    const tubeGeo = new THREE.TubeGeometry(curve, 28, b.r, 16, false);
    group.add(new THREE.Mesh(tubeGeo, mats.gold));
  });

  const gemGeo = new THREE.OctahedronGeometry(0.12, 1);
  gemGeo.scale(0.82, 1.35, 0.72);
  const gemCoords = [
    [0, 0.94, 0, 0],
    [-0.75, 0.22, 0.12, 0.5], [0.77, 0.26, -0.12, -0.5],
    [-0.69, 0.62, -0.14, 0.4], [0.69, 0.66, 0.14, -0.4],
    [-0.46, 0.86, 0.14, 0.25], [0.46, 0.88, -0.14, -0.25],
    [-0.22, 0.58, -0.22, 0.2], [0.22, 0.60, 0.22, -0.2],
    [-0.52, -0.06, 0.18, 0.7], [0.54, -0.02, -0.18, -0.7],
  ];
  gemCoords.forEach((g) => {
    const gemGroup = new THREE.Group();
    gemGroup.position.set(g[0], g[1], g[2]);
    gemGroup.rotation.z = g[3];
    gemGroup.userData.explodeVec = new THREE.Vector3(g[0] * 0.45, Math.max(0.1, g[1] * 0.35), g[2] * 0.45);
    gemGroup.add(new THREE.Mesh(gemGeo, mats.emerald));
    const cup = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.02, 0.05, 16), mats.brass);
    cup.position.y = -0.11;
    gemGroup.add(cup);
    group.add(gemGroup);
  });

  const leafGeo = new THREE.ConeGeometry(0.055, 0.16, 12);
  leafGeo.scale(1.0, 1.0, 0.35);
  const leafCoords = [
    [-0.25, -0.12, 0.10, 0.7], [0.26, -0.08, -0.10, -0.7],
    [-0.44, 0.06, 0.16, -0.6], [0.46, 0.10, -0.16, 0.6],
    [-0.28, 0.20, -0.12, 0.5], [0.28, 0.24, 0.12, -0.5],
    [-0.50, 0.40, -0.18, -0.5], [0.50, 0.44, 0.18, 0.5],
    [-0.18, 0.42, 0.15, 0.4], [0.18, 0.44, -0.15, -0.4],
    [-0.34, 0.68, 0.18, -0.35], [0.34, 0.70, -0.18, 0.35],
    [-0.12, 0.66, 0.06, 0.3], [0.12, 0.68, -0.06, -0.3],
  ];
  leafCoords.forEach((l) => {
    const leaf = new THREE.Mesh(leafGeo, mats.gold);
    leaf.position.set(l[0], l[1], l[2]);
    leaf.rotation.set(0.35, l[3], l[3]);
    leaf.userData.explodeVec = new THREE.Vector3(l[0] * 0.3, l[1] * 0.2, l[2] * 0.3);
    group.add(leaf);
  });

  return group;
}

// 4. PROCEDURAL 3D FACETED STAR TROPHY (star_trophy_3d)
function createStarPolyhedronGeo(THREE, nPoints, rOuter, rInner, zApex) {
  const positions = [];
  const ring = [];
  for (let i = 0; i < nPoints * 2; i++) {
    const ang = (i * Math.PI) / nPoints;
    const r = i % 2 === 0 ? rOuter : rInner;
    const zr = i % 2 === 0 ? 0.04 : -0.04;
    ring.push([r * Math.sin(ang), r * Math.cos(ang), zr]);
  }
  const total = nPoints * 2;
  for (let i = 0; i < total; i++) {
    const p0 = ring[i];
    const p1 = ring[(i + 1) % total];
    positions.push(0, 0, zApex, p1[0], p1[1], p1[2], p0[0], p0[1], p0[2]);
    positions.push(0, 0, -zApex, p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.computeVertexNormals();
  return geo;
}

function buildStarTrophy3D(THREE, mats) {
  const group = new THREE.Group();

  const baseBlack = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.44, 0.18, 48), mats.matteBlack);
  baseBlack.position.y = -0.89;
  baseBlack.userData.explodeVec = new THREE.Vector3(0, -0.35, 0);
  group.add(baseBlack);

  const baseRing = new THREE.Mesh(new THREE.TorusGeometry(0.43, 0.022, 16, 48), mats.gold);
  baseRing.rotation.x = Math.PI / 2;
  baseRing.position.y = -0.80;
  baseRing.userData.explodeVec = new THREE.Vector3(0, -0.25, 0);
  group.add(baseRing);

  const baseGold = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.34, 0.08, 48), mats.gold);
  baseGold.position.y = -0.75;
  baseGold.userData.explodeVec = new THREE.Vector3(0, -0.18, 0);
  group.add(baseGold);

  const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.08, 0.24, 24), mats.gold);
  stem.position.y = -0.63;
  group.add(stem);

  const star1 = new THREE.Mesh(createStarPolyhedronGeo(THREE, 5, 0.94, 0.36, 0.26), mats.gold);
  star1.position.y = 0.10;
  star1.userData.explodeVec = new THREE.Vector3(0, 0.12, 0.25);
  group.add(star1);

  const star2 = new THREE.Mesh(createStarPolyhedronGeo(THREE, 5, 0.82, 0.38, 0.20), mats.crystal);
  star2.position.y = 0.10;
  star2.rotation.z = Math.PI / 5;
  star2.userData.explodeVec = new THREE.Vector3(0, 0.12, -0.30);
  group.add(star2);

  const star3 = new THREE.Mesh(createStarPolyhedronGeo(THREE, 5, 0.52, 0.21, 0.38), mats.brass);
  star3.position.y = 0.10;
  star3.userData.explodeVec = new THREE.Vector3(0, 0.18, 0.45);
  group.add(star3);

  return group;
}
