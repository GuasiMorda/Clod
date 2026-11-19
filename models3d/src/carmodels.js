// Procedural car models: BMW M5 (F90-style), Cybertruck-style pickup, Dodge Viper-style coupe.
// Units: metres. +x = front, +y = up, z = width. Origin on the ground under the car centre.
import * as THREE from 'three';
import { Brush, Evaluator, SUBTRACTION } from 'three-bvh-csg';

const lerp = (a, b, x) => a + (b - a) * x;
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));

// ---------- 1D key interpolation (Catmull-Rom, or linear) ----------
function interp(keys, u, linear = false) {
  if (u <= keys[0][0]) return keys[0][1];
  if (u >= keys[keys.length - 1][0]) return keys[keys.length - 1][1];
  let i = 0; while (u > keys[i + 1][0]) i++;
  const [u0, v0] = keys[i], [u1, v1] = keys[i + 1];
  const t = (u - u0) / (u1 - u0);
  if (linear) return lerp(v0, v1, t);
  const vm = i > 0 ? keys[i - 1][1] : v0, vp = i + 2 < keys.length ? keys[i + 2][1] : v1;
  const t2 = t * t, t3 = t2 * t;
  return 0.5 * ((2 * v0) + (-vm + v1) * t + (2 * vm - 5 * v0 + 4 * v1 - vp) * t2 + (-vm + 3 * v0 - 3 * v1 + vp) * t3);
}

// ---------- loft: superellipse cross-sections along x ----------
// prof: {x0, x1, yb(u), yt(u), hwB(u), hwT(u), n(u)}  with u in [0,1] from rear (x0) to front (x1)
function loft(prof, S = 90, M = 64, flatBottom = true) {
  const pos = [], uv = [], idx = [];
  for (let i = 0; i < S; i++) {
    const u = i / (S - 1);
    const x = lerp(prof.x0, prof.x1, u);
    const yb = prof.yb(u), yt = prof.yt(u), hwB = prof.hwB(u), hwT = prof.hwT(u), n = prof.n(u);
    const yc = (yb + yt) / 2, hh = Math.max(0.002, (yt - yb) / 2);
    for (let j = 0; j < M; j++) {
      const th = (j / M) * Math.PI * 2;
      const c = Math.cos(th), s = Math.sin(th);
      const zz = Math.sign(c) * Math.pow(Math.abs(c), 2 / n);
      let yy = Math.sign(s) * Math.pow(Math.abs(s), 2 / n);
      if (flatBottom && yy < -0.85) yy = -0.85 - (yy + 0.85) * 0.15;
      const hw = lerp(hwB, hwT, (yy + 1) / 2);
      pos.push(x, yc + yy * hh, zz * hw);
      uv.push(u, j / M);
    }
  }
  for (let i = 0; i < S - 1; i++) for (let j = 0; j < M; j++) {
    const a = i * M + j, b = i * M + (j + 1) % M, c = (i + 1) * M + j, d = (i + 1) * M + (j + 1) % M;
    idx.push(a, c, b, b, c, d);
  }
  // end caps
  for (const [ring, flip] of [[0, true], [S - 1, false]]) {
    const ci = pos.length / 3;
    let cx = 0, cy = 0, cz = 0;
    for (let j = 0; j < M; j++) { cx += pos[(ring * M + j) * 3]; cy += pos[(ring * M + j) * 3 + 1]; cz += pos[(ring * M + j) * 3 + 2]; }
    pos.push(cx / M, cy / M, cz / M); uv.push(ring ? 1 : 0, 0.5);
    for (let j = 0; j < M; j++) {
      const a = ring * M + j, b = ring * M + (j + 1) % M;
      if (flip) idx.push(ci, b, a); else idx.push(ci, a, b);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}
function secPt(prof, u, th, off = 0) {
  const x = lerp(prof.x0, prof.x1, u);
  const yb = prof.yb(u), yt = prof.yt(u), hwB = prof.hwB(u), hwT = prof.hwT(u), n = prof.n(u);
  const yc = (yb + yt) / 2, hh = (yt - yb) / 2 + off;
  const c = Math.cos(th), s = Math.sin(th);
  const zz = Math.sign(c) * Math.pow(Math.abs(c), 2 / n), yy = Math.sign(s) * Math.pow(Math.abs(s), 2 / n);
  return [x, yc + yy * hh, zz * (lerp(hwB, hwT, (yy + 1) / 2) + off)];
}
function patch(prof, ua, ub, ta, tb, off = 0.006, nu = 24, nt = 16) {
  const pos = [], idx = [];
  for (let i = 0; i <= nu; i++) for (let j = 0; j <= nt; j++) pos.push(...secPt(prof, lerp(ua, ub, i / nu), lerp(ta, tb, j / nt), off));
  for (let i = 0; i < nu; i++) for (let j = 0; j < nt; j++) {
    const a = i * (nt + 1) + j, b = a + 1, c = a + nt + 1, d = c + 1; idx.push(a, c, b, b, c, d);
  }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setIndex(idx); g.computeVertexNormals();
  return g;
}
const K = (keys, linear) => (u) => interp(keys, u, linear);
const C = (v) => () => v;

// ---------- CSG: cut wheel arches ----------
const evaluator = new Evaluator();
evaluator.attributes = ['position', 'normal', 'uv'];
function cutArches(geo, mat, wheels, r, depth, sides = 40, hexRot = 0) {
  const hasCol = !!geo.attributes.color;
  evaluator.attributes = hasCol ? ['position', 'normal', 'uv', 'color'] : ['position', 'normal', 'uv'];
  let brush = new Brush(geo, mat); brush.updateMatrixWorld();
  for (const [wx] of wheels) {
    const cyl = new THREE.CylinderGeometry(r, r, depth, sides, 1);
    cyl.rotateX(Math.PI / 2);
    if (hasCol) { const n = cyl.attributes.position.count; cyl.setAttribute('color', new THREE.Float32BufferAttribute(new Array(n * 3).fill(0.05), 3)); }
    if (hexRot) cyl.rotateZ(hexRot);
    const b = new Brush(cyl, mat); b.position.set(wx, wheels.y, 0); b.updateMatrixWorld();
    brush = evaluator.evaluate(brush, b, SUBTRACTION);
  }
  return brush;
}

// ---------- materials ----------
export function paint(color, opts = {}) {
  return new THREE.MeshPhysicalMaterial({ color, metalness: opts.metal ?? 0.55, roughness: opts.rough ?? 0.25, clearcoat: 1, clearcoatRoughness: 0.04,
    envMapIntensity: 1.4, flatShading: !!opts.flat, vertexColors: !!opts.vc, side: THREE.DoubleSide });
}
const glass = () => new THREE.MeshPhysicalMaterial({ color: 0x05070b, metalness: 0.1, roughness: 0.02, clearcoat: 1, envMapIntensity: 2.2 });
const black = (r = 0.5) => new THREE.MeshStandardMaterial({ color: 0x07070a, roughness: r, metalness: 0.2 });
const chrome = () => new THREE.MeshStandardMaterial({ color: 0xdadde3, metalness: 1, roughness: 0.12 });
const emissive = (c, i = 4) => new THREE.MeshStandardMaterial({ color: c, emissive: c, emissiveIntensity: i });

// ---------- wheels ----------
export function wheel(r, w, style = {}) {
  const g = new THREE.Group();
  const rimR = r * (style.rim ?? 0.72);
  // tyre: lathe of a rounded profile around z
  const prof = [];
  const sw = w / 2;
  for (let k = 0; k <= 16; k++) {
    const a = -Math.PI / 2 + (k / 16) * Math.PI;
    prof.push(new THREE.Vector2(r - 0.06 + Math.cos(a) * 0.06, Math.sin(a) * sw * 0.98));
  }
  prof.unshift(new THREE.Vector2(rimR, -sw * 0.92)); prof.push(new THREE.Vector2(rimR, sw * 0.92));
  const tyre = new THREE.Mesh(new THREE.LatheGeometry(prof, 64), new THREE.MeshStandardMaterial({ color: 0x111113, roughness: 0.85 }));
  tyre.rotation.x = Math.PI / 2; g.add(tyre);
  if (style.tread) {
    for (let k = 0; k < 48; k++) {
      const blk = new THREE.Mesh(new THREE.BoxGeometry(0.035, 0.018, w * 0.86), tyre.material);
      const a = k / 48 * Math.PI * 2; blk.position.set(Math.cos(a) * (r - 0.005), Math.sin(a) * (r - 0.005), 0); blk.rotation.z = a; g.add(blk);
    }
  }
  // rim barrel + face
  const rimMat = style.rimMat || new THREE.MeshStandardMaterial({ color: style.rimColor ?? 0x2a2c31, metalness: 0.9, roughness: 0.3 });
  const barrel = new THREE.Mesh(new THREE.CylinderGeometry(rimR, rimR, w * 0.86, 48, 1, true), rimMat);
  barrel.rotation.x = Math.PI / 2; g.add(barrel);
  const face = new THREE.Group(); face.position.z = sw * 0.42; g.add(face);
  if (style.cover) {           // aero cover (cybertruck style)
    const disc = new THREE.Mesh(new THREE.CylinderGeometry(rimR, rimR, 0.03, 6), new THREE.MeshStandardMaterial({ color: 0x1a1b1e, metalness: 0.5, roughness: 0.4, flatShading: true }));
    disc.rotation.x = Math.PI / 2; face.add(disc);
    for (let k = 0; k < 6; k++) {
      const tri = new THREE.Mesh(new THREE.CylinderGeometry(rimR * 0.55, rimR * 0.55, 0.035, 3), new THREE.MeshStandardMaterial({ color: 0x2c2e33, metalness: 0.7, roughness: 0.3, flatShading: true }));
      tri.rotation.x = Math.PI / 2; tri.rotation.y = k * Math.PI / 3; tri.position.z = 0.005; face.add(tri);
    }
  } else {
    const n = style.spokes ?? 10;
    for (let k = 0; k < n; k++) {
      const sp = new THREE.Mesh(new THREE.BoxGeometry(rimR * 0.92, rimR * (style.spokeW ?? 0.11), 0.05), rimMat);
      const a = k / n * Math.PI * 2;
      sp.position.set(Math.cos(a) * rimR * 0.5, Math.sin(a) * rimR * 0.5, 0); sp.rotation.z = a; face.add(sp);
      if (style.split) { const sp2 = sp.clone(); sp2.rotation.z = a + 0.18; sp2.position.set(Math.cos(a + 0.18) * rimR * 0.5, Math.sin(a + 0.18) * rimR * 0.5, -0.01); face.add(sp2); }
    }
    const hub = new THREE.Mesh(new THREE.CylinderGeometry(rimR * 0.2, rimR * 0.22, 0.07, 24), rimMat); hub.rotation.x = Math.PI / 2; face.add(hub);
    const lip = new THREE.Mesh(new THREE.TorusGeometry(rimR * 0.985, 0.012, 8, 64), chrome()); face.add(lip);
    // brake disc + caliper behind the spokes
    const disc = new THREE.Mesh(new THREE.CylinderGeometry(rimR * 0.82, rimR * 0.82, 0.03, 48), new THREE.MeshStandardMaterial({ color: 0x6b6d72, metalness: 0.9, roughness: 0.45 }));
    disc.rotation.x = Math.PI / 2; disc.position.z = sw * 0.15; g.add(disc);
    if (style.caliper) {
      const cal = new THREE.Mesh(new THREE.BoxGeometry(rimR * 0.5, rimR * 0.22, 0.08), new THREE.MeshStandardMaterial({ color: style.caliper, roughness: 0.35, metalness: 0.3 }));
      cal.position.set(rimR * 0.45, rimR * 0.35, sw * 0.22); cal.rotation.z = 0.7; g.add(cal);
    }
  }
  g.traverse(o => { if (o.isMesh) o.castShadow = true; });
  return g;
}

function addWheels(car, wheels, r, w, track, style) {
  car.userData.wheels = [];
  for (const [wx] of wheels) for (const s of [-1, 1]) {
    const wh = wheel(r, w, style);
    if (s < 0) wh.rotation.y = Math.PI;
    wh.position.set(wx, r, s * track);
    car.add(wh); car.userData.wheels.push(wh);
  }
  // dark wheel wells
  for (const [wx] of wheels) {
    const well = new THREE.Mesh(new THREE.CylinderGeometry(r + 0.03, r + 0.03, track * 2 - 0.15, 32, 1, true), new THREE.MeshStandardMaterial({ color: 0x050506, roughness: 1, side: THREE.BackSide }));
    well.rotation.x = Math.PI / 2; well.position.set(wx, r + 0.02, 0); car.add(well);
  }
}
function mesh(geo, mat, parent, cast = true) { const m = new THREE.Mesh(geo, mat); m.castShadow = cast; m.receiveShadow = true; parent.add(m); return m; }
function box(w, h, d, mat, x, y, z, parent, rx = 0, ry = 0, rz = 0) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat); m.position.set(x, y, z); m.rotation.set(rx, ry, rz); parent.add(m); return m;
}

// ======================================================================= BMW M5 (F90 style)
export function buildM5(color = 0x1b2a4a) {
  const car = new THREE.Group(); car.name = 'M5_F90';
  const L = 4.97, X0 = -L / 2, X1 = L / 2, WB = 2.98, R = 0.35;
  const wx = [-WB / 2 + 0.05, WB / 2 + 0.05]; const wheels = wx.map(x => [x]); wheels.y = R;
  const body = paint(color);
  const lower = loft({
    x0: X0, x1: X1,
    yb: K([[0, 0.32], [0.04, 0.2], [0.1, 0.17], [0.9, 0.16], [0.97, 0.2], [1, 0.3]]),
    yt: K([[0, 0.8], [0.03, 0.97], [0.12, 1.02], [0.22, 1.0], [0.5, 0.97], [0.66, 0.96], [0.8, 0.9], [0.93, 0.8], [0.985, 0.7], [1, 0.58]]),
    hwB: K([[0, 0.8], [0.05, 0.92], [0.15, 0.95], [0.5, 0.94], [0.85, 0.94], [0.95, 0.9], [1, 0.74]]),
    hwT: K([[0, 0.7], [0.05, 0.82], [0.2, 0.84], [0.6, 0.84], [0.9, 0.82], [1, 0.62]]),
    n: K([[0, 3.2], [0.1, 4.2], [0.9, 4.2], [1, 3.0]]),
  }, 120, 72);
  car.add(cutArches(lower, body, wheels, R + 0.05, 2.4));
  // greenhouse (glass) + body-coloured roof + pillars
  const cab = { x0: -1.45, x1: 1.12,
    yb: C(0.93),
    yt: K([[0, 0.99], [0.08, 1.12], [0.22, 1.38], [0.36, 1.46], [0.6, 1.46], [0.72, 1.38], [0.86, 1.16], [1, 0.95]]),
    hwB: K([[0, 0.8], [0.5, 0.84], [1, 0.82]]), hwT: K([[0, 0.48], [0.5, 0.66], [1, 0.5]]), n: C(3.0) };
  mesh(loft(cab, 90, 64), glass(), car);
  const PI = Math.PI;
  mesh(patch(cab, 0.25, 0.73, 0.33 * PI, 0.67 * PI, 0.008, 30, 20), body, car);                       // roof skin
  for (const [ta, tb] of [[0.04 * PI, 0.36 * PI], [0.64 * PI, 0.96 * PI]]) {
    mesh(patch(cab, 0.02, 0.24, ta, tb, 0.007), body, car);                                            // C-pillars
    mesh(patch(cab, 0.5, 0.53, ta, tb, 0.007, 4, 12), black(0.25), car);                               // B-pillars
    mesh(patch(cab, 0.72, 0.79, ta, tb, 0.007, 8, 12), body, car);                                     // A-pillars
    mesh(patch(cab, 0.02, 0.95, ta < PI / 2 ? 0.02 * PI : 0.96 * PI, ta < PI / 2 ? 0.05 * PI : 0.99 * PI, 0.01, 40, 2), chrome(), car); // window trim
  }
  // front: kidney grille (two rounded frames), intakes, headlights with twin LED rings
  const front = new THREE.Group(); front.position.set(X1 - 0.04, 0, 0); car.add(front);
  for (const s of [-1, 1]) {
    const sh = new THREE.Shape(); const gw = 0.24, gh = 0.2, rr = 0.06;
    sh.moveTo(-gw / 2 + rr, 0); sh.lineTo(gw / 2 - rr, 0); sh.quadraticCurveTo(gw / 2, 0, gw / 2, rr); sh.lineTo(gw / 2, gh - rr);
    sh.quadraticCurveTo(gw / 2, gh, gw / 2 - rr, gh); sh.lineTo(-gw / 2 + rr, gh); sh.quadraticCurveTo(-gw / 2, gh, -gw / 2, gh - rr);
    sh.lineTo(-gw / 2, rr); sh.quadraticCurveTo(-gw / 2, 0, -gw / 2 + rr, 0);
    const fr = new THREE.Mesh(new THREE.ExtrudeGeometry(sh, { depth: 0.05, bevelEnabled: true, bevelSize: 0.012, bevelThickness: 0.01 }), black(0.2));
    fr.rotation.y = Math.PI / 2; fr.position.set(0.02, 0.43, s * 0.15); front.add(fr);
    for (let k = 0; k < 6; k++) box(0.01, 0.17, 0.01, black(0.6), 0.075, 0.53, s * 0.15 - 0.09 + k * 0.036, front);
    // headlight housing + twin LED rings + bar
    const hl = new THREE.Group(); hl.position.set(-0.12, 0.66, s * 0.62); hl.rotation.y = -s * 0.35; front.add(hl);
    box(0.06, 0.11, 0.42, black(0.1), 0, 0, 0, hl);
    for (const dz of [-0.09, 0.07]) { const ring = new THREE.Mesh(new THREE.TorusGeometry(0.04, 0.006, 6, 6), emissive(0xeaf2ff, 6)); ring.rotation.y = Math.PI / 2; ring.position.set(0.035, 0, dz); hl.add(ring); }
    box(0.01, 0.012, 0.3, emissive(0xeaf2ff, 5), 0.036, -0.045, 0, hl);
    // side intakes
    box(0.05, 0.14, 0.32, black(0.5), 0.02, 0.3, s * 0.58, front);
  }
  box(0.06, 0.09, 0.9, black(0.5), 0.02, 0.24, 0, front);   // lower centre intake
  // rear: L-shaped tail lights, diffuser, quad exhaust
  for (const s of [-1, 1]) {
    const tl = new THREE.Group(); tl.position.set(X0 + 0.06, 0.9, s * 0.72); tl.rotation.y = s * 0.28; car.add(tl);
    box(0.04, 0.12, 0.36, new THREE.MeshPhysicalMaterial({ color: 0x300404, roughness: 0.1, clearcoat: 1 }), 0, 0, 0, tl);
    box(0.012, 0.016, 0.3, emissive(0xff1a12, 6), -0.022, 0.03, 0, tl);
    box(0.012, 0.08, 0.016, emissive(0xff1a12, 6), -0.022, -0.01, s * -0.14, tl);
    for (const dz of [0.32, 0.45]) {
      const ex = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.12, 24, 1, true), chrome()); ex.rotation.z = Math.PI / 2;
      ex.position.set(X0 + 0.05, 0.27, s * dz); car.add(ex);
      const inner = new THREE.Mesh(new THREE.CircleGeometry(0.04, 24), black(1)); inner.rotation.y = -Math.PI / 2; inner.position.set(X0 + 0.0, 0.27, s * dz); car.add(inner);
    }
  }
  box(0.08, 0.1, 1.3, black(0.6), X0 + 0.08, 0.24, 0, car);   // diffuser
  // mirrors, side skirt, door handles
  for (const s of [-1, 1]) {
    const mir = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 14), body); mir.scale.set(0.1, 0.055, 0.1); mir.position.set(0.86, 1.02, s * 0.93); car.add(mir);
    box(0.06, 0.02, 0.1, black(0.4), 0.88, 0.98, s * 0.86, car);
    box(3.0, 0.05, 0.03, black(0.4), 0.05, 0.2, s * 0.95, car);
    box(0.12, 0.02, 0.015, chrome(), 0.15, 0.93, s * 0.94, car); box(0.12, 0.02, 0.015, chrome(), -0.75, 0.93, s * 0.94, car);
  }
  addWheels(car, wheels, R, 0.27, 0.8, { spokes: 7, split: true, spokeW: 0.1, rimColor: 0x1d1e22, caliper: 0x1e4fd6 });
  return finish(car);
}

// ======================================================================= Cybertruck style
export function buildCybertruck() {
  const car = new THREE.Group(); car.name = 'Cybertruck';
  const L = 5.68, X0 = -L / 2, X1 = L / 2, WB = 3.63, R = 0.44;
  const wheels = [[-WB / 2 + 0.05], [WB / 2 + 0.05]]; wheels.y = R;
  const steel = new THREE.MeshPhysicalMaterial({ color: 0xa3a6ac, metalness: 1.0, roughness: 0.4, clearcoat: 0.0, flatShading: true, envMapIntensity: 1.1, side: THREE.DoubleSide });
  const apex = 0.56;
  const shell = loft({
    x0: X0, x1: X1,
    yb: K([[0, 0.62], [0.06, 0.5], [0.94, 0.5], [1, 0.62]], true),
    yt: K([[0, 1.28], [apex, 1.79], [0.985, 1.08], [1, 1.0]], true),
    hwB: K([[0, 0.98], [0.5, 1.0], [1, 0.97]], true), hwT: K([[0, 0.86], [0.5, 0.86], [1, 0.84]], true),
    n: C(14),
  }, 140, 40);
  car.add(cutArches(shell, steel, wheels, R + 0.12, 2.6, 6, Math.PI / 6));
  // glass: windshield on the front slope and side windows
  const topAt = (x) => { const u = (x - X0) / L; return u < apex ? lerp(1.28, 1.79, u / apex) : lerp(1.79, 1.08, (u - apex) / (0.985 - apex)); };
  const xa = X0 + apex * L;
  const x2 = xa + 1.3, ang = Math.atan2(1.79 - topAt(x2), x2 - xa);
  const ws = box(Math.hypot(x2 - xa, 1.79 - topAt(x2)), 0.012, 1.55, glass(), (xa + x2) / 2, (1.79 + topAt(x2)) / 2 + 0.01, 0, car, 0, 0, -ang);
  for (const s of [-1, 1]) {
    const sh = new THREE.Shape();
    const p = [[xa - 1.35, 1.24], [xa - 0.05, 1.24], [xa + 1.05, 1.24], [xa + 0.05, topAt(xa + 0.05) - 0.12], [xa - 1.35, topAt(xa - 1.35) - 0.1]];
    sh.moveTo(p[0][0], p[0][1]); for (const q of p.slice(1)) sh.lineTo(q[0], q[1]);
    const gm = glass(); gm.side = THREE.DoubleSide;
    const m = new THREE.Mesh(new THREE.ShapeGeometry(sh), gm); m.position.z = s * 0.905; car.add(m);
    box(0.02, 0.36, 0.01, steel, xa - 0.15, 1.42, s * 0.9, car);          // B-pillar
  }
  // light bars
  box(0.02, 0.025, 1.9, emissive(0xf2f6ff, 8), X1 - 0.005, 1.0, 0, car);
  box(0.02, 0.03, 1.9, emissive(0xff2216, 6), X0 - 0.005, 1.25, 0, car);
  box(0.04, 0.1, 1.95, black(0.6), X0 + 0.01, 0.62, 0, car);              // rear bumper strip
  box(0.04, 0.12, 1.9, black(0.6), X1 - 0.01, 0.58, 0, car);              // front lower strip
  // dark cladding around arches (polygonal)
  for (const [wx0] of wheels) for (const s of [-1, 1]) {
    const cl = new THREE.Mesh(new THREE.TorusGeometry(R + 0.14, 0.05, 4, 6, Math.PI), black(0.7)); cl.rotation.z = Math.PI / 6 - Math.PI / 6;
    cl.position.set(wx0, R, s * 1.0); car.add(cl);
  }
  addWheels(car, wheels, R, 0.33, 0.88, { cover: true, rim: 0.62, tread: true });
  return finish(car);
}

// ======================================================================= Dodge Viper style
export function buildViper(color = 0x1546c8) {
  const car = new THREE.Group(); car.name = 'Viper';
  const L = 4.46, X0 = -L / 2, X1 = L / 2, WB = 2.51, R = 0.34;
  const wheels = [[X0 + 1.0], [X0 + 1.0 + WB]]; wheels.y = R;
  const uR = 1.0 / L, uF = (1.0 + WB) / L;
  const body = paint(0xffffff, { vc: true, metal: 0.5 });
  const geo = loft({
    x0: X0, x1: X1,
    yb: K([[0, 0.3], [0.05, 0.16], [0.95, 0.14], [1, 0.2]]),
    yt: K([[0, 0.72], [0.05, 0.9], [0.12, 0.93], [0.25, 0.92], [0.45, 0.88], [0.6, 0.86], [0.75, 0.8], [0.9, 0.7], [0.97, 0.6], [1, 0.48]]),
    hwB: K([[0, 0.78], [0.08, 0.93], [uR, 0.98], [0.36, 0.95], [0.46, 0.86], [0.6, 0.88], [uF, 0.97], [0.9, 0.92], [1, 0.72]]),
    hwT: K([[0, 0.66], [uR, 0.82], [0.46, 0.7], [uF, 0.8], [1, 0.55]]),
    n: K([[0, 2.6], [0.5, 2.8], [1, 2.4]]),
  }, 140, 80);
  // GTS racing stripes via vertex colours (paint blue, stripes white, on the upper surface)
  const p = geo.attributes.position, cols = [];
  const base = new THREE.Color(color), white = new THREE.Color(0xf4f4f4);
  for (let i = 0; i < p.count; i++) {
    const z = Math.abs(p.getZ(i)), y = p.getY(i);
    const stripe = z > 0.06 && z < 0.2 && y > 0.62;
    const c = stripe ? white : base; cols.push(c.r, c.g, c.b);
  }
  geo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
  car.add(cutArches(geo, body, wheels, R + 0.05, 2.4));
  // low cabin, double-bubble roof
  const cab = { x0: X0 + 1.05, x1: X0 + 2.5, yb: C(0.86),
    yt: K([[0, 0.92], [0.2, 1.1], [0.42, 1.24], [0.6, 1.24], [0.78, 1.12], [1, 0.88]]),
    hwB: C(0.7), hwT: C(0.48), n: C(2.6) };
  mesh(loft(cab, 70, 56), glass(), car);
  for (const s of [-1, 1]) {
    const bub = { x0: X0 + 1.55, x1: X0 + 2.15, yb: (u) => cab.yt(lerp(0.34, 0.76, u)) - 0.03, yt: (u) => cab.yt(lerp(0.34, 0.76, u)) + 0.025,
      hwB: C(0.17), hwT: C(0.15), n: C(2.5) };
    const g2 = loft(bub, 24, 32, false); g2.translate(0, 0, s * 0.19);
    const bc = []; for (let i = 0; i < g2.attributes.position.count; i++) bc.push(base.r, base.g, base.b);
    g2.setAttribute('color', new THREE.Float32BufferAttribute(bc, 3));
    mesh(g2, body, car);
  }
  // long hood vents, side gills behind front wheels, wide snake grille
  for (const s of [-1, 1]) {
    for (let k = 0; k < 4; k++) box(0.24, 0.02, 0.03, black(0.4), X1 - 1.15, 0.775 - k * 0.012, s * (0.3 + k * 0.045), car, 0, 0, -0.08);
    for (let k = 0; k < 3; k++) box(0.03, 0.16, 0.03, black(0.4), X1 - 1.22 - k * 0.07, 0.52, s * 0.905, car, 0, 0, 0.35);
    // headlights: swept LED strip + projector
    const hl = new THREE.Group(); hl.position.set(X1 - 0.3, 0.6, s * 0.6); hl.rotation.set(0, -s * 0.5, -0.12); car.add(hl);
    box(0.36, 0.06, 0.2, black(0.1), 0, 0, 0, hl);
    box(0.3, 0.008, 0.012, emissive(0xeef4ff, 6), 0.02, 0.025, -s * 0.06, hl);
    const pj = new THREE.Mesh(new THREE.SphereGeometry(0.032, 16, 12), emissive(0xffffff, 3)); pj.position.set(0.06, 0, s * 0.03); hl.add(pj);
    // tail lights
    const tl = box(0.04, 0.035, 0.42, emissive(0xff1a12, 6), X0 + 0.07, 0.82, s * 0.5, car); tl.rotation.y = s * 0.25;
    // side exhaust under the doors
    const sx = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.5, 20), chrome()); sx.rotation.z = Math.PI / 2; sx.position.set(X0 + 2.0, 0.2, s * 0.97); car.add(sx);
    const mir = box(0.15, 0.06, 0.1, body, X0 + 2.42, 0.89, s * 0.84, car);
    mir.geometry.setAttribute('color', new THREE.Float32BufferAttribute(new Array(mir.geometry.attributes.position.count * 3).fill(0).map((_, i) => [base.r, base.g, base.b][i % 3]), 3));
  }
  const gr = new THREE.Shape(); gr.moveTo(-0.45, 0); gr.quadraticCurveTo(0, -0.06, 0.45, 0); gr.quadraticCurveTo(0.3, 0.13, 0, 0.14); gr.quadraticCurveTo(-0.3, 0.13, -0.45, 0);
  const grille = new THREE.Mesh(new THREE.ExtrudeGeometry(gr, { depth: 0.05, bevelEnabled: false }), black(0.6));
  grille.rotation.y = Math.PI / 2; grille.position.set(X1 - 0.03, 0.27, 0); car.add(grille);
  box(0.06, 0.06, 0.9, black(0.6), X0 + 0.04, 0.25, 0, car);   // rear diffuser
  addWheels(car, wheels, R, 0.3, 0.82, { spokes: 5, spokeW: 0.16, rimColor: 0x111214, caliper: 0xd21a1a });
  return finish(car);
}

function finish(car) {
  car.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  return car;
}
