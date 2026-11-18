import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const params = new URLSearchParams(location.search);
const W = +(params.get('w') || 1080), H = +(params.get('h') || 1920);
const S = [0, 3, 7, 11, 16, 20];
const SLOW = [13.5, 15.5];

const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const lerp = (a, b, x) => a + (b - a) * x;
const easeOut = x => 1 - (1 - clamp(x)) ** 3;
const easeInOut = x => { x = clamp(x); return 3 * x * x - 2 * x * x * x; };
function rnd(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setSize(W, H);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const pmrem = new THREE.PMREMGenerator(renderer);
const envTex = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

const composer = new EffectComposer(renderer);
const renderPass = new RenderPass(new THREE.Scene(), new THREE.PerspectiveCamera());
composer.addPass(renderPass);
const bloom = new UnrealBloomPass(new THREE.Vector2(W / 2, H / 2), 0.9, 0.5, 0.82);
composer.addPass(bloom);
composer.addPass(new OutputPass());

// ---------- helpers ----------
function canvasTex(w, h, draw) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4;
  return t;
}
const softDot = canvasTex(64, 64, (g, w, h) => {
  const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.5, 'rgba(255,255,255,0.6)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = gr; g.fillRect(0, 0, w, h);
});

function labelPlane(text, w, h, opts = {}) {
  const pw = 512, ph = Math.round(512 * h / w);
  const tex = canvasTex(pw, ph, (g) => {
    if (opts.bg) { g.fillStyle = opts.bg; g.fillRect(0, 0, pw, ph); }
    if (opts.border) { g.strokeStyle = opts.border; g.lineWidth = 14; g.strokeRect(7, 7, pw - 14, ph - 14); }
    g.fillStyle = opts.fg || '#fff';
    g.font = `bold ${opts.size || Math.round(ph * 0.6)}px DejaVu Sans, sans-serif`;
    g.textAlign = 'center'; g.textBaseline = 'middle';
    if (opts.stroke) { g.lineWidth = 10; g.strokeStyle = opts.stroke; g.strokeText(text, pw / 2, ph / 2); }
    g.fillText(text, pw / 2, ph / 2);
  });
  const m = new THREE.MeshBasicMaterial({ map: tex, transparent: !opts.bg, toneMapped: false });
  return new THREE.Mesh(new THREE.PlaneGeometry(w, h), m);
}

// falling snow around a box region (follows a camera)
function makeSnow(count, size, box, seed = 1) {
  const r = rnd(seed);
  const base = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) { base[i * 4] = r(); base[i * 4 + 1] = r(); base[i * 4 + 2] = r(); base[i * 4 + 3] = 0.5 + r(); }
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(count * 3);
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const mat = new THREE.PointsMaterial({ size, map: softDot, transparent: true, depthWrite: false, color: 0xeaf2ff, opacity: 0.9 });
  const pts = new THREE.Points(geo, mat); pts.frustumCulled = false;
  pts.userData.update = (t, center, wind = 0.6, speed = 1) => {
    for (let i = 0; i < count; i++) {
      const sp = base[i * 4 + 3] * speed;
      let y = (base[i * 4 + 1] - t * 0.12 * sp) % 1; if (y < 0) y += 1;
      let x = (base[i * 4] + t * 0.03 * wind * sp + 0.01 * Math.sin(t * 2 + i)) % 1; if (x < 0) x += 1;
      pos[i * 3] = center.x + (x - 0.5) * box.x;
      pos[i * 3 + 1] = center.y + (y - 0.5) * box.y;
      pos[i * 3 + 2] = center.z + (base[i * 4 + 2] - 0.5) * box.z;
    }
    geo.attributes.position.needsUpdate = true;
  };
  return pts;
}

// snow spray / smoke puffs emitted behind a moving point
function makeSpray(count, seed = 3, color = 0xf0f4ff) {
  const r = rnd(seed);
  const seeds = Array.from({ length: count }, () => [r(), r(), r(), r()]);
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(count * 3);
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const mat = new THREE.PointsMaterial({ size: 1.6, map: softDot, transparent: true, depthWrite: false, color, opacity: 0.55 });
  const pts = new THREE.Points(geo, mat); pts.frustumCulled = false;
  // emitter(time)->{p:Vector3, back:Vector3(dir), side}
  pts.userData.update = (t, emitter, life = 0.9, strength = 1) => {
    for (let i = 0; i < count; i++) {
      const [a, b, c, d] = seeds[i];
      const age = ((t / life + a) % 1) * life;
      const birth = t - age;
      const e = emitter(birth);
      const k = age / life;
      const sp = (4 + 8 * b) * strength;
      pos[i * 3] = e.p.x + e.back.x * sp * age + (c - 0.5) * 3 * age;
      pos[i * 3 + 1] = e.p.y + (2 + 3 * d) * age - 3 * age * age;
      pos[i * 3 + 2] = e.p.z + e.back.z * sp * age + (d - 0.5) * 3 * age;
      if (k > 0.98) pos[i * 3 + 1] = -100;
    }
    geo.attributes.position.needsUpdate = true;
  };
  return pts;
}

// ---------- car factory: extruded side profiles ----------
const PROFILES = {
  vaz: { body: [[0.02, 0.62], [0.03, 0.40], [0.20, 0.38], [0.30, 0.08], [0.70, 0.08], [0.80, 0.36], [0.98, 0.40], [0.99, 0.66], [0.02, 0.70]],
         win: [[0.32, 0.14], [0.68, 0.14], [0.76, 0.36], [0.24, 0.37]], wheels: [0.20, 0.80], L: 4.1, Hh: 1.45, Wd: 1.62 },
  amg: { body: [[0.0, 0.60], [0.02, 0.42], [0.18, 0.36], [0.40, 0.10], [0.62, 0.10], [0.80, 0.34], [0.99, 0.44], [1.0, 0.62], [0.0, 0.70]],
         win: [[0.42, 0.15], [0.61, 0.15], [0.74, 0.34], [0.26, 0.36]], wheels: [0.18, 0.82], L: 4.6, Hh: 1.35, Wd: 1.9 },
  charger: { body: [[0.0, 0.58], [0.01, 0.36], [0.22, 0.32], [0.36, 0.06], [0.60, 0.06], [0.70, 0.30], [0.99, 0.34], [1.0, 0.64], [0.0, 0.70]],
         win: [[0.38, 0.11], [0.58, 0.11], [0.66, 0.30], [0.28, 0.32]], wheels: [0.17, 0.83], L: 5.0, Hh: 1.4, Wd: 1.95 },
};

const glassMat = new THREE.MeshPhysicalMaterial({ color: 0x0a0f1a, metalness: 0.2, roughness: 0.05, clearcoat: 1, envMapIntensity: 1.5 });
const tireMat = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.9 });
const rimMat = new THREE.MeshStandardMaterial({ color: 0xc8ccd4, metalness: 1, roughness: 0.25 });
const headMat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xfff4d6, emissiveIntensity: 2.2 });
const tailMat = new THREE.MeshStandardMaterial({ color: 0xff2020, emissive: 0xff1010, emissiveIntensity: 2.5 });

function makeCar(kind, color, opts = {}) {
  const P = PROFILES[kind];
  const { L, Hh, Wd } = P;
  const clear = 0.18;
  const toV = ([x, y]) => new THREE.Vector2(x * L - L / 2, (1 - y) * Hh * 0.7 / 0.7 * 1.0 * (1 / 1) * 1 + clear - Hh * 0.30);
  const shape = new THREE.Shape(P.body.map(toV));
  const bev = 0.12;
  const bodyGeo = new THREE.ExtrudeGeometry(shape, { depth: Wd - 2 * bev, bevelEnabled: true, bevelThickness: bev, bevelSize: 0.08, bevelSegments: 4, curveSegments: 4, steps: 6 });
  bodyGeo.translate(0, 0, -(Wd - 2 * bev) / 2);
  // subdivide-ish: more vertices along length for deformation -> add tessellation by re-extruding not possible; fine.
  const bodyMat = new THREE.MeshPhysicalMaterial({ color, metalness: 0.55, roughness: 0.28, clearcoat: 1, clearcoatRoughness: 0.06, envMapIntensity: 1.3 });
  const body = new THREE.Mesh(bodyGeo, bodyMat);
  body.castShadow = true;
  const g = new THREE.Group();
  g.add(body);
  const winShape = new THREE.Shape(P.win.map(toV));
  const winGeo = new THREE.ExtrudeGeometry(winShape, { depth: Wd + 0.04, bevelEnabled: false });
  winGeo.translate(0, 0, -(Wd + 0.04) / 2);
  const win = new THREE.Mesh(winGeo, glassMat); g.add(win);
  // windshield & rear glass panels
  const wsF = [toV(P.win[1]), toV(P.win[2])], wsR = [toV(P.win[0]), toV(P.win[3])];
  for (const [a, b] of [wsF, wsR]) {
    const len = a.distanceTo(b);
    const pane = new THREE.Mesh(new THREE.BoxGeometry(len, 0.03, Wd - 0.15), glassMat);
    pane.position.set((a.x + b.x) / 2 + (a === wsF[0] ? 0.06 : -0.06), (a.y + b.y) / 2 + 0.04, 0);
    pane.rotation.z = Math.atan2(b.y - a.y, b.x - a.x);
    g.add(pane);
  }
  // wheels
  const wr = 0.36 * (Hh / 1.4) + 0.02;
  const wheels = [];
  for (const wx of P.wheels) for (const s of [-1, 1]) {
    const wg = new THREE.Group();
    const tire = new THREE.Mesh(new THREE.CylinderGeometry(wr, wr, 0.28, 24), tireMat);
    tire.rotation.x = Math.PI / 2; wg.add(tire);
    const rim = new THREE.Mesh(new THREE.CylinderGeometry(wr * 0.62, wr * 0.62, 0.3, 10), rimMat);
    rim.rotation.x = Math.PI / 2; wg.add(rim);
    for (let k = 0; k < 5; k++) {
      const sp = new THREE.Mesh(new THREE.BoxGeometry(wr * 1.1, 0.07, 0.32), new THREE.MeshStandardMaterial({ color: 0x333840, metalness: 0.8, roughness: 0.4 }));
      sp.rotation.z = k * Math.PI / 5; wg.add(sp);
    }
    wg.position.set(wx * L - L / 2, wr, s * (Wd / 2 - 0.08));
    wg.castShadow = true;
    g.add(wg); wheels.push(wg);
  }
  // lights
  const lights = [];
  for (const s of [-1, 1]) {
    const hl = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.14, 0.38), headMat);
    const frontY = toV(P.body[6]).y - 0.12;
    hl.position.set(L / 2 + 0.02, frontY, s * (Wd / 2 - 0.35)); g.add(hl); lights.push(hl);
    const tl = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.14, 0.45), tailMat);
    tl.position.set(-L / 2 - 0.02, toV(P.body[1]).y - 0.12, s * (Wd / 2 - 0.35)); g.add(tl);
  }
  if (opts.spot) {
    for (const s of [-1, 1]) {
      const sl = new THREE.SpotLight(0xfff2d8, opts.spot, 60, 0.45, 0.5, 1.2);
      sl.position.set(L / 2, 0.7, s * 0.6);
      sl.target.position.set(L / 2 + 10, -0.5, s * 1.2);
      g.add(sl); g.add(sl.target);
    }
  }
  // deformation support
  const orig = bodyGeo.attributes.position.array.slice();
  const lightsOrig = lights.map(l => l.position.clone());
  g.userData = {
    L, wheels, wr,
    spin(dist) { for (const w of wheels) w.rotation.z = -dist / wr; },
    crush(c) {
      const p = bodyGeo.attributes.position.array;
      for (let i = 0; i < p.length; i += 3) {
        const x = orig[i], y = orig[i + 1], z = orig[i + 2];
        const k = clamp((x - L * 0.15) / (L * 0.35));
        if (k <= 0 || c <= 0) { p[i] = x; p[i + 1] = y; p[i + 2] = z; continue; }
        const kk = k * k;
        p[i] = x - c * 0.9 * kk - c * 0.25 * kk * Math.sin(z * 5 + y * 7);
        p[i + 1] = y + c * 0.22 * kk * Math.sin(x * 9 + z * 3) - c * 0.12 * kk;
        p[i + 2] = z * (1 + c * 0.10 * kk) + c * 0.15 * kk * Math.sin(y * 11);
      }
      bodyGeo.attributes.position.needsUpdate = true;
      bodyGeo.computeVertexNormals();
      lights.forEach((l, j) => { l.position.x = lightsOrig[j].x - c * 0.9; l.rotation.y = c * 0.6 * (j ? 1 : -1); });
    },
  };
  return g;
}

// ---------- shared materials ----------
const snowMat = new THREE.MeshStandardMaterial({ color: 0xdfe7f5, roughness: 0.85 });
const roadMat = new THREE.MeshStandardMaterial({ color: 0x23262e, roughness: 0.55, metalness: 0.1 });
const dashMat = new THREE.MeshStandardMaterial({ color: 0xe8e8e0, emissive: 0x555550, roughness: 0.5 });

function makeTrees(scene, n, seed, xMin, xMax, zMin, zMax) {
  const r = rnd(seed);
  const trunkM = new THREE.MeshStandardMaterial({ color: 0x2a1d14, roughness: 1 });
  const leafM = new THREE.MeshStandardMaterial({ color: 0x14261f, roughness: 1 });
  const snowM = new THREE.MeshStandardMaterial({ color: 0xe6eef8, roughness: 0.8 });
  for (let i = 0; i < n; i++) {
    const side = r() < 0.5 ? -1 : 1;
    const x = side * lerp(xMin, xMax, r()), z = lerp(zMin, zMax, r());
    const h = 4 + r() * 6;
    const t = new THREE.Group();
    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.2, 1.2, 6), trunkM); trunk.position.y = 0.6; t.add(trunk);
    for (let k = 0; k < 3; k++) {
      const rr = (1.6 - k * 0.4) * h / 7;
      const cone = new THREE.Mesh(new THREE.ConeGeometry(rr, h * 0.45, 8), leafM);
      cone.position.y = 1 + k * h * 0.22 + h * 0.2; cone.castShadow = true; t.add(cone);
      const cap = new THREE.Mesh(new THREE.ConeGeometry(rr * 0.7, h * 0.18, 8), snowM);
      cap.position.y = cone.position.y + h * 0.16; t.add(cap);
    }
    t.position.set(x, 0, z); scene.add(t);
  }
}

// ======================= SCENE 1: HOOK =======================
const s1 = new THREE.Scene();
s1.background = new THREE.Color(0x070b1a);
s1.fog = new THREE.FogExp2(0x0a1022, 0.028);
s1.environment = envTex; s1.environmentIntensity = 0.15;
const cam1 = new THREE.PerspectiveCamera(55, W / H, 0.1, 400);
{
  s1.add(new THREE.HemisphereLight(0x5070b0, 0x101018, 0.5));
  const moon = new THREE.DirectionalLight(0x8fa8ff, 0.9); moon.position.set(-20, 30, -10);
  moon.castShadow = true; moon.shadow.mapSize.set(1024, 1024);
  Object.assign(moon.shadow.camera, { left: -20, right: 20, top: 20, bottom: -20 });
  s1.add(moon);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), snowMat); ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true; s1.add(ground);
  const road = new THREE.Mesh(new THREE.PlaneGeometry(9, 400), roadMat); road.rotation.x = -Math.PI / 2; road.position.y = 0.01; road.receiveShadow = true; s1.add(road);
  for (let z = -200; z < 20; z += 6) { const d = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.02, 2.5), dashMat); d.position.set(0, 0.02, z); s1.add(d); }
  makeTrees(s1, 90, 5, 7, 45, -180, 10);
}
const car1 = makeCar('charger', 0x8a1018, { spot: 25 }); s1.add(car1);
const snow1 = makeSnow(2500, 0.12, new THREE.Vector3(30, 20, 40), 11); s1.add(snow1);
const spray1 = makeSpray(260, 7); s1.add(spray1);
function car1Pose(lt) {
  const k = clamp(lt / 3);
  const z = lerp(-60, -9, Math.pow(k, 1.1));
  const x = 2.6 * Math.sin(lt * 2.0 - 0.4) * (1 - 0.4 * k);
  const yaw = -Math.PI / 2 + 0.55 * Math.sin(lt * 2.0 + 0.9);
  return { x, z, yaw };
}
function renderHook(t) {
  const lt = t;
  const p = car1Pose(lt);
  car1.position.set(p.x, 0, p.z); car1.rotation.y = p.yaw;
  car1.userData.spin(lt * 30);
  car1.rotation.z = 0.03 * Math.sin(lt * 6);
  cam1.position.set(-2.2 + 0.4 * Math.sin(lt), 1.0 + lt * 0.12, 0);
  cam1.lookAt(p.x * 0.7, 0.7, p.z);
  cam1.fov = lerp(28, 50, easeInOut(lt / 3)); cam1.updateProjectionMatrix();
  snow1.userData.update(t, new THREE.Vector3(0, 8, -10), 1.2, 1.6);
  spray1.userData.update(lt, (bt) => {
    const q = car1Pose(Math.max(0, bt));
    const back = new THREE.Vector3(-Math.cos(q.yaw), 0, Math.sin(q.yaw));
    const side = (Math.floor(bt * 37) % 2) ? 1 : -1;
    const sideV = new THREE.Vector3(Math.sin(q.yaw), 0, Math.cos(q.yaw)).multiplyScalar(side * 0.9);
    return { p: new THREE.Vector3(q.x, 0.3, q.z).add(back.clone().multiplyScalar(1.8)).add(sideV), back };
  }, 0.9, 0.8);
  bloom.strength = 0.55; bloom.threshold = 0.9;
  renderPass.scene = s1; renderPass.camera = cam1;
}

// ======================= SCENE 2: ADVENT =======================
const s2 = new THREE.Scene();
s2.background = new THREE.Color(0x0b0d18);
s2.environment = envTex; s2.environmentIntensity = 0.6;
const cam2 = new THREE.PerspectiveCamera(48, W / H, 0.1, 200);
const boxes = [];
{
  s2.add(new THREE.AmbientLight(0x404060, 0.6));
  const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(4, 6, 10); s2.add(key);
  const rim = new THREE.PointLight(0x6080ff, 40, 30); rim.position.set(-6, -3, 4); s2.add(rim);
  const warm = new THREE.PointLight(0xffa040, 40, 30); warm.position.set(6, 4, 4); s2.add(warm);
  // bokeh background lights
  const r = rnd(21);
  const bg = new THREE.BufferGeometry(); const n = 160; const pa = new Float32Array(n * 3); const ca = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    pa[i * 3] = (r() - 0.5) * 40; pa[i * 3 + 1] = (r() - 0.5) * 50; pa[i * 3 + 2] = -10 - r() * 15;
    const c = new THREE.Color().setHSL(r() < 0.5 ? 0.08 : 0.6, 0.8, 0.6); ca.set([c.r, c.g, c.b], i * 3);
  }
  bg.setAttribute('position', new THREE.BufferAttribute(pa, 3)); bg.setAttribute('color', new THREE.BufferAttribute(ca, 3));
  s2.add(new THREE.Points(bg, new THREE.PointsMaterial({ size: 1.4, map: softDot, vertexColors: true, transparent: true, depthWrite: false, opacity: 0.6 })));
  const boxCols = [0xb3122a, 0x1d4fb8, 0x0f7a45];
  const ribbonM = new THREE.MeshStandardMaterial({ color: 0xffd060, metalness: 1, roughness: 0.3 });
  const goldM = new THREE.MeshStandardMaterial({ color: 0xffc83a, metalness: 1, roughness: 0.2, emissive: 0x553300, emissiveIntensity: 1.5 });
  const starShape = new THREE.Shape();
  for (let i = 0; i < 10; i++) { const a = i / 10 * Math.PI * 2 + Math.PI / 2, rr = i % 2 ? 0.16 : 0.38; const v = [Math.cos(a) * rr, Math.sin(a) * rr]; i ? starShape.lineTo(...v) : starShape.moveTo(...v); }
  const starGeo = new THREE.ExtrudeGeometry(starShape, { depth: 0.12, bevelEnabled: true, bevelSize: 0.03, bevelThickness: 0.03 });
  const starM = new THREE.MeshStandardMaterial({ color: 0xc070ff, emissive: 0x9030ff, emissiveIntensity: 2.5 });
  const kinds = ['coin', 'car', 'star', 'coin', 'car', 'coin'];
  for (let i = 0; i < 24; i++) {
    const c = i % 4, rr = Math.floor(i / 4);
    const g = new THREE.Group();
    const m = new THREE.MeshPhysicalMaterial({ color: boxCols[(c + rr) % 3], roughness: 0.35, clearcoat: 0.6 });
    const base = new THREE.Mesh(new THREE.BoxGeometry(1, 0.85, 1), m); base.position.y = -0.075; g.add(base);
    const rb1 = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.86, 1.02), ribbonM); rb1.position.y = -0.075; g.add(rb1);
    const lid = new THREE.Group();
    const lidM = new THREE.Mesh(new THREE.BoxGeometry(1.08, 0.2, 1.08), m); lid.add(lidM);
    const rb2 = new THREE.Mesh(new THREE.BoxGeometry(0.17, 0.21, 1.1), ribbonM); lid.add(rb2);
    const bow = new THREE.Mesh(new THREE.TorusGeometry(0.14, 0.05, 8, 16), ribbonM); bow.position.y = 0.16; lid.add(bow);
    lid.position.y = 0.45; g.add(lid);
    const num = labelPlane(String(i + 1), 0.6, 0.45, { fg: '#ffffff', stroke: '#101830', size: 200 });
    num.position.set(0, -0.08, 0.515); g.add(num);
    let reward;
    const kind = kinds[i % kinds.length];
    if (kind === 'coin') { reward = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.32, 0.08, 32), goldM); reward.rotation.x = Math.PI / 2; const w = new THREE.Group(); w.add(reward); reward = w; }
    else if (kind === 'star') { reward = new THREE.Mesh(starGeo, starM); }
    else { reward = makeCar(['vaz', 'amg', 'charger'][i % 3], [0xd02030, 0xf0f0f0, 0x2050ff][i % 3]); reward.scale.setScalar(0.15); }
    reward.visible = false; g.add(reward);
    g.position.set((c - 1.5) * 1.3, (2.5 - rr) * 1.3 - 0.9, 0);
    s2.add(g);
    boxes.push({ g, lid, reward, kind });
  }
}
function renderAdvent(t) {
  const lt = t - S[1];
  const opened = 24 * Math.pow(clamp(lt / 3.6), 1.7);
  boxes.forEach((b, i) => {
    const o = clamp(opened - i);
    b.lid.position.y = 0.45 + easeOut(o) * 1.4;
    b.lid.rotation.x = -easeOut(o) * 1.1;
    b.lid.position.z = easeOut(o) * 0.6;
    b.lid.visible = o < 0.95;
    b.reward.visible = o > 0;
    const pop = o > 0 ? easeOut(o * 1.3) : 0;
    b.reward.position.set(0, 0.2 + pop * 0.55, 0.3);
    b.reward.rotation.y = lt * 4 + i;
    if (b.kind === 'car') b.reward.scale.setScalar(0.15 * (0.3 + 0.7 * pop));
    else b.reward.scale.setScalar(0.3 + 0.9 * pop);
    b.g.position.z = 0.15 * Math.sin(lt * 3 + i) + (o > 0 && o < 1 ? 0.6 * Math.sin(o * Math.PI) : 0);
  });
  cam2.position.set(1.2 * Math.sin(lt * 0.6) + 0.6, 0.6 + 0.3 * Math.sin(lt * 0.8), lerp(13.5, 11.5, easeInOut(lt / 4)));
  cam2.lookAt(0, -0.8, 0);
  cam2.rotation.z += 0.03 * Math.sin(lt * 1.3);
  bloom.strength = 0.5; bloom.threshold = 0.9;
  renderPass.scene = s2; renderPass.camera = cam2;
}

// ======================= SCENE 3: BATTLE PASS =======================
const s3 = new THREE.Scene();
s3.background = new THREE.Color(0x150826);
s3.fog = new THREE.Fog(0x150826, 12, 40);
s3.environment = envTex; s3.environmentIntensity = 0.8;
const cam3 = new THREE.PerspectiveCamera(50, W / H, 0.1, 200);
const cards = [];
const CARD_DX = 3.9;
{
  s3.add(new THREE.AmbientLight(0x8060c0, 0.5));
  const key = new THREE.DirectionalLight(0xffffff, 2); key.position.set(3, 5, 8); s3.add(key);
  const items = ['car', 'paint', 'plate', 'money', 'car', 'paint', 'plate', 'car', 'money', 'paint', 'car', 'plate', 'car', 'car', 'paint', 'car'];
  const cols = [0xd02030, 0x2080ff, 0xf5c518, 0x30c060, 0xf0f0f5];
  const kinds = ['amg', 'charger', 'vaz'];
  items.forEach((it, i) => {
    const prem = i % 3 === 2;
    const g = new THREE.Group();
    const cardM = new THREE.MeshPhysicalMaterial({ color: prem ? 0x6a4410 : 0x3a1e6e, roughness: 0.3, metalness: 0.3, clearcoat: 1, emissive: prem ? 0x150a00 : 0x0a0418 });
    const card = new THREE.Mesh(new THREE.BoxGeometry(2.8, 4.0, 0.12), cardM); g.add(card);
    const edgeM = new THREE.MeshStandardMaterial({ color: prem ? 0xffd050 : 0xb090ff, emissive: prem ? 0xffb020 : 0x8050ff, emissiveIntensity: 0.7 });
    for (const [w, h, x, y] of [[2.9, 0.08, 0, 2.02], [2.9, 0.08, 0, -2.02], [0.08, 4.1, 1.42, 0], [0.08, 4.1, -1.42, 0]]) {
      const e = new THREE.Mesh(new THREE.BoxGeometry(w, h, 0.16), edgeM); e.position.set(x, y, 0); g.add(e);
    }
    const lbl = labelPlane(`УР. ${(i + 1) * 5}`, 2.2, 0.55, { fg: '#ffffff', stroke: '#000000', size: 150 }); lbl.position.set(0, 1.55, 0.08); g.add(lbl);
    if (prem) { const pl = labelPlane('ПРЕМИУМ', 2.0, 0.4, { fg: '#ffd050', size: 110 }); pl.position.set(0, 1.05, 0.08); g.add(pl); }
    let obj;
    if (it === 'car') { obj = makeCar(kinds[i % 3], cols[i % 5]); obj.scale.setScalar(0.45); obj.position.y = -0.5; }
    else if (it === 'paint') {
      obj = new THREE.Group();
      [0xff3040, 0x30a0ff, 0xffd020, 0x40e080].forEach((c, k) => {
        const sph = new THREE.Mesh(new THREE.SphereGeometry(0.38, 32, 16), new THREE.MeshPhysicalMaterial({ color: c, metalness: 0.6, roughness: 0.2, clearcoat: 1 }));
        sph.position.set(Math.cos(k * Math.PI / 2) * 0.55, Math.sin(k * Math.PI / 2) * 0.55, 0); obj.add(sph);
      });
      const lb = labelPlane('ОКРАСКА', 2.0, 0.4, { fg: '#fff', size: 110 }); lb.position.set(0, -1.2, 0); obj.add(lb);
    } else if (it === 'plate') {
      obj = new THREE.Group();
      const pt = labelPlane('А777АА', 2.2, 0.5, { bg: '#f4f4f4', fg: '#0a0a0a', border: '#111', size: 170 });
      const back = new THREE.Mesh(new THREE.BoxGeometry(2.25, 0.55, 0.06), new THREE.MeshStandardMaterial({ color: 0xdddddd, metalness: 0.8, roughness: 0.3 }));
      back.position.z = -0.04; obj.add(back); obj.add(pt);
      const lb = labelPlane('НОМЕРА', 2.0, 0.4, { fg: '#fff', size: 110 }); lb.position.set(0, -1.2, 0); obj.add(lb);
    } else {
      obj = new THREE.Group();
      const gm = new THREE.MeshStandardMaterial({ color: 0x40d070, metalness: 0.3, roughness: 0.5, emissive: 0x0a3010 });
      for (let k = 0; k < 4; k++) { const st = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.18, 0.7), gm); st.position.set((k % 2) * 0.15 - 0.07, -0.6 + k * 0.2, 0); st.rotation.y = k * 0.2; obj.add(st); }
      const lb = labelPlane('$ 100 000', 2.2, 0.5, { fg: '#7dff9a', stroke: '#003010', size: 140 }); lb.position.set(0, 0.5, 0.3); obj.add(lb);
    }
    obj.position.z += 0.7; g.add(obj);
    g.position.set(i * CARD_DX, 0, 0);
    s3.add(g); cards.push({ g, obj, it });
  });
  // sparkles
  const r = rnd(33); const n = 300; const pa = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { pa[i * 3] = r() * 60 - 5; pa[i * 3 + 1] = (r() - 0.5) * 20; pa[i * 3 + 2] = -3 - r() * 12; }
  const sg = new THREE.BufferGeometry(); sg.setAttribute('position', new THREE.BufferAttribute(pa, 3));
  s3.add(new THREE.Points(sg, new THREE.PointsMaterial({ size: 0.25, map: softDot, color: 0xd0b0ff, transparent: true, depthWrite: false })));
}
function renderPass3(t) {
  const lt = t - S[2];
  const scroll = easeInOut(lt / 3.8) * (cards.length - 3) * CARD_DX;
  cards.forEach((c, i) => {
    const dx = c.g.position.x - scroll;
    c.g.rotation.y = clamp(-dx * 0.08, -0.4, 0.4);
    c.g.position.z = -Math.abs(dx) * 0.12;
    c.g.position.y = 0.4 + 0.1 * Math.sin(lt * 3 + i);
    c.obj.rotation.y = (c.it === 'car' ? 1 : 0.4) * Math.sin(lt * 2 + i) + (c.it === 'car' ? lt * 1.5 : 0);
    if (c.it === 'car') c.obj.userData.spin(lt * 10);
  });
  cam3.position.set(scroll, 0.6, 8.2);
  cam3.lookAt(scroll, 0.2, 0);
  cam3.rotation.z = 0.06 * Math.sin(lt * 1.7);
  bloom.strength = 0.45; bloom.threshold = 0.9;
  renderPass.scene = s3; renderPass.camera = cam3;
}

// ======================= SCENE 4a: SHOWROOM =======================
const s4 = new THREE.Scene();
s4.background = new THREE.Color(0x07070c);
s4.environment = envTex; s4.environmentIntensity = 0.45;
const cam4 = new THREE.PerspectiveCamera(42, W / H, 0.1, 200);
const showCars = [];
let turntable;
{
  s4.add(new THREE.AmbientLight(0x303040, 0.5));
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), new THREE.MeshStandardMaterial({ color: 0x15151c, roughness: 0.18, metalness: 0.6 }));
  floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; s4.add(floor);
  turntable = new THREE.Mesh(new THREE.CylinderGeometry(4.2, 4.3, 0.15, 64), new THREE.MeshStandardMaterial({ color: 0x2a2a34, metalness: 0.9, roughness: 0.25 }));
  turntable.position.y = 0.075; turntable.receiveShadow = true; s4.add(turntable);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(4.25, 0.04, 8, 96), new THREE.MeshStandardMaterial({ color: 0x40a0ff, emissive: 0x2080ff, emissiveIntensity: 1.5 }));
  ring.rotation.x = Math.PI / 2; ring.position.y = 0.16; s4.add(ring);
  for (const [x, z, c] of [[-5, 4, 0xffffff], [5, 3, 0xffffff], [0, -6, 0x80b0ff]]) {
    const sp = new THREE.SpotLight(c, 70, 40, 0.5, 0.6, 1.5); sp.position.set(x, 9, z); sp.target.position.set(0, 0, 0);
    sp.castShadow = true; sp.shadow.mapSize.set(1024, 1024); s4.add(sp); s4.add(sp.target);
  }
  // light strips on the walls
  for (let i = 0; i < 9; i++) {
    const st = new THREE.Mesh(new THREE.BoxGeometry(0.15, 6, 0.15), new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 1.2 }));
    const a = -Math.PI * 0.9 + i * Math.PI * 0.225; st.position.set(Math.cos(a) * 14, 3, Math.sin(a) * 14 - 2); s4.add(st);
  }
  [['vaz', 0xb0101a], ['amg', 0xa8acb4], ['charger', 0x16171c]].forEach(([k, c]) => {
    const car = makeCar(k, c); car.position.y = 0.15; car.visible = false; s4.add(car); showCars.push(car);
  });
}
function renderShowroom(t) {
  const lt = t - S[3];
  const idx = lt < 1 ? 0 : lt < 2 ? 1 : 2;
  const st = lt - idx;
  showCars.forEach((c, i) => c.visible = i === idx);
  const car = showCars[idx];
  car.rotation.y = 0.6 + st * 0.9 + idx * 2.1;
  turntable.rotation.y = car.rotation.y;
  const a = 0.8 + lt * 0.5 + idx * 1.3;
  const r = lerp(9.5, 8.2, easeOut(st * 2));
  cam4.position.set(Math.cos(a) * r, 1.6 + 0.3 * idx, Math.sin(a) * r);
  cam4.lookAt(0, 0.9, 0);
  bloom.strength = 0.4; bloom.threshold = 0.92;
  renderPass.scene = s4; renderPass.camera = cam4;
}

// ======================= SCENE 4b: DRIFT + CRASH =======================
const s5 = new THREE.Scene();
s5.background = new THREE.Color(0x060a18);
s5.fog = new THREE.FogExp2(0x0b1226, 0.022);
s5.environment = envTex; s5.environmentIntensity = 0.25;
const cam5 = new THREE.PerspectiveCamera(50, W / H, 0.1, 400);
const driftCar = makeCar('charger', 0xe0701a, { spot: 20 }); s5.add(driftCar);
const BARRIER_X = 62;
let debris;
{
  s5.add(new THREE.HemisphereLight(0x6080c0, 0x101018, 0.6));
  const moon = new THREE.DirectionalLight(0x9fb4ff, 1.0); moon.position.set(10, 30, 20); moon.castShadow = true; moon.shadow.mapSize.set(1024, 1024);
  Object.assign(moon.shadow.camera, { left: -30, right: 30, top: 30, bottom: -30 }); moon.target.position.set(50, 0, 0);
  s5.add(moon); s5.add(moon.target);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(500, 200), snowMat); ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true; s5.add(ground);
  const road = new THREE.Mesh(new THREE.PlaneGeometry(500, 10), roadMat); road.rotation.x = -Math.PI / 2; road.position.y = 0.01; road.receiveShadow = true; s5.add(road);
  // packed-snow patches on the track
  for (let x = -100; x < 120; x += 7) { const p = new THREE.Mesh(new THREE.PlaneGeometry(4, 6), new THREE.MeshStandardMaterial({ color: 0xb8c2d6, roughness: 0.9, transparent: true, opacity: 0.5 })); p.rotation.x = -Math.PI / 2; p.position.set(x + (x % 3), 0.02, (x % 5) - 2); s5.add(p); }
  for (let x = -100; x < 150; x += 14) {
    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.1, 7, 8), new THREE.MeshStandardMaterial({ color: 0x30323a, metalness: 0.7, roughness: 0.4 }));
    post.position.set(x, 3.5, -7); s5.add(post);
    const arm = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.1, 2), post.material); arm.position.set(x, 7, -6); s5.add(arm);
    const bulb = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.12, 0.8), new THREE.MeshStandardMaterial({ color: 0xffd090, emissive: 0xffc070, emissiveIntensity: 6 }));
    bulb.position.set(x, 6.9, -5.2); s5.add(bulb);
  }
  const pl = new THREE.PointLight(0xffc070, 120, 30); pl.position.set(BARRIER_X - 4, 6, -4); s5.add(pl);
  const tg = new THREE.Group(); makeTrees(tg, 70, 9, 10, 45, -100, 150); tg.rotation.y = Math.PI / 2; s5.add(tg);
  // swap tree layout: trees were placed along z; rotate group-less by moving them – acceptable as a forest backdrop
  // barrier: concrete blocks with red/white stripes
  for (let k = -2; k <= 2; k++) {
    const blk = new THREE.Mesh(new THREE.BoxGeometry(1.2, 1.1, 1.6), new THREE.MeshStandardMaterial({ color: k % 2 ? 0xd02020 : 0xeeeeee, roughness: 0.6 }));
    blk.position.set(BARRIER_X, 0.55, k * 1.6); blk.castShadow = true; blk.receiveShadow = true; s5.add(blk);
  }
  const r = rnd(77); const n = 120; const geo = new THREE.BoxGeometry(0.12, 0.05, 0.1);
  debris = [];
  for (let i = 0; i < n; i++) {
    const m = new THREE.Mesh(geo, i % 3 ? new THREE.MeshStandardMaterial({ color: 0xcfe0ff, roughness: 0.2, metalness: 0.5 }) : driftCar.children[0].material);
    m.userData.v = new THREE.Vector3(-r() * 6 - 1, r() * 6 + 2, (r() - 0.5) * 10); m.userData.w = r() * 20;
    m.visible = false; s5.add(m); debris.push(m);
  }
}
const snow5 = makeSnow(2500, 0.12, new THREE.Vector3(30, 20, 30), 5); s5.add(snow5);
const spray5 = makeSpray(320, 9); s5.add(spray5);
const CRASH_T = SLOW[0] + 0.6; // 14.1s real time
// game-time: slow-mo region runs 4x slower
function gameTime(t) {
  if (t < SLOW[0]) return t;
  if (t < SLOW[1]) return SLOW[0] + (t - SLOW[0]) * 0.25;
  return SLOW[0] + (SLOW[1] - SLOW[0]) * 0.25 + (t - SLOW[1]);
}
const G_CRASH = gameTime(CRASH_T);
const G0 = S[3] + 2.5;
function driftPose(g) {
  // car slides along +x, yaw angled into the drift; stops at the barrier
  const u = g - G0;
  const uc = Math.min(u, G_CRASH - G0);
  const v = 24;
  const x = BARRIER_X - 2.6 - v * (G_CRASH - G0) + v * uc - 0.0;
  const z = 1.2 * Math.sin(uc * 1.6);
  const yaw = 0.55 + 0.15 * Math.sin(uc * 2.2);
  return { x, z, yaw, u };
}
function renderDrift(t) {
  const g = gameTime(t);
  const p = driftPose(g);
  const crashed = g > G_CRASH;
  const ca = crashed ? easeOut((g - G_CRASH) / 0.35) : 0;
  driftCar.position.set(p.x - ca * 0.3, 0, p.z);
  driftCar.rotation.y = p.yaw + ca * 0.25;
  driftCar.rotation.z = ca * 0.05 * Math.sin((g - G_CRASH) * 30) * (1 - ca);
  driftCar.userData.spin(Math.min(g, G_CRASH) * 70);
  driftCar.userData.crush(ca);
  debris.forEach((d) => {
    const age = g - G_CRASH;
    d.visible = age > 0;
    if (age > 0) {
      const v = d.userData.v;
      d.position.set(BARRIER_X - 1 + v.x * age, Math.max(0.03, 1 + v.y * age - 9.8 * age * age), v.z * age * 0.5);
      d.rotation.set(age * d.userData.w, age * d.userData.w * 0.7, 0);
    }
  });
  // camera tracks alongside; in slow-mo it swings around the front
  const slowK = clamp((t - SLOW[0]) / (SLOW[1] - SLOW[0]));
  const ang = lerp(0.35, 1.25, easeInOut(slowK));
  const dist = lerp(10, 7, easeInOut(slowK));
  cam5.position.set(Math.min(p.x + Math.sin(ang) * dist * 0.7 - 3, BARRIER_X - 4), 1.6 + slowK * 0.8, 6 + Math.cos(ang) * dist);
  cam5.lookAt(p.x + 1.5, 0.7, p.z);
  snow5.userData.update(t < SLOW[0] ? t : g, new THREE.Vector3(p.x, 8, 0), 1.5, 1);
  spray5.userData.update(Math.min(g, G_CRASH + 0.2), (bt) => {
    const q = driftPose(Math.min(bt, G_CRASH));
    const back = new THREE.Vector3(-Math.cos(q.yaw), 0, Math.sin(q.yaw));
    const side = (Math.floor(bt * 41) % 2) ? 1 : -1;
    const sideV = new THREE.Vector3(Math.sin(q.yaw), 0, Math.cos(q.yaw)).multiplyScalar(side * 0.9);
    return { p: new THREE.Vector3(q.x, 0.3, q.z).add(back.clone().multiplyScalar(1.9)).add(sideV), back: back.add(new THREE.Vector3(0, 0, 0.6)) };
  }, 0.8, 1.1);
  bloom.strength = 0.55; bloom.threshold = 0.9;
  renderPass.scene = s5; renderPass.camera = cam5;
}

// ======================= SCENE 5: CITY CTA =======================
const s6 = new THREE.Scene();
s6.background = new THREE.Color(0x070a1c);
s6.fog = new THREE.FogExp2(0x0c1028, 0.012);
s6.environment = envTex; s6.environmentIntensity = 0.2;
const cam6 = new THREE.PerspectiveCamera(50, W / H, 0.1, 600);
const convoy = [];
{
  s6.add(new THREE.HemisphereLight(0x5060a0, 0x101018, 0.6));
  const moon = new THREE.DirectionalLight(0x9fb4ff, 0.6); moon.position.set(-20, 40, 10); s6.add(moon);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(200, 600), snowMat); ground.rotation.x = -Math.PI / 2; ground.position.z = -200; s6.add(ground);
  const road = new THREE.Mesh(new THREE.PlaneGeometry(14, 600), roadMat); road.rotation.x = -Math.PI / 2; road.position.set(0, 0.01, -200); s6.add(road);
  for (let z = 20; z > -400; z -= 6) { const d = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.02, 2.5), dashMat); d.position.set(0, 0.02, z); s6.add(d); }
  const r = rnd(99);
  const winTex = (seed) => canvasTex(128, 256, (g) => {
    const rr = rnd(seed); g.fillStyle = '#0d0f1a'; g.fillRect(0, 0, 128, 256);
    for (let y = 8; y < 256; y += 16) for (let x = 6; x < 128; x += 14) {
      if (rr() < 0.45) { g.fillStyle = rr() < 0.8 ? '#ffcf7a' : '#9fd0ff'; g.fillRect(x, y, 8, 9); }
    }
  });
  const texes = [1, 2, 3, 4, 5].map(winTex);
  const roofM = new THREE.MeshStandardMaterial({ color: 0xe8eef8, roughness: 0.8 });
  for (const side of [-1, 1]) {
    for (let z = 10; z > -380; z -= 14 + r() * 6) {
      const w = 10 + r() * 6, d = 10 + r() * 4, h = 14 + r() * 50;
      const tex = texes[Math.floor(r() * 5)].clone(); tex.wrapS = tex.wrapT = THREE.RepeatWrapping; tex.repeat.set(w / 10, h / 20); tex.needsUpdate = true;
      const m = new THREE.MeshStandardMaterial({ color: 0x15182a, emissive: 0xffffff, emissiveMap: tex, emissiveIntensity: 1.6, roughness: 0.8 });
      const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m);
      b.position.set(side * (14 + w / 2 + r() * 4), h / 2, z); s6.add(b);
      const roof = new THREE.Mesh(new THREE.BoxGeometry(w + 0.3, 0.5, d + 0.3), roofM); roof.position.set(b.position.x, h + 0.25, z); s6.add(roof);
    }
    for (let z = 10; z > -380; z -= 18) {
      const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.35, 12, 8), new THREE.MeshStandardMaterial({ color: 0xffd090, emissive: 0xffc070, emissiveIntensity: 8 }));
      bulb.position.set(side * 8, 6.5, z); s6.add(bulb);
      const post = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.12, 6.5, 6), new THREE.MeshStandardMaterial({ color: 0x222, metalness: 0.6 }));
      post.position.set(side * 8, 3.25, z); s6.add(post);
    }
  }
  const pal = [0xc0141c, 0xf0f0f0, 0x101114, 0x2050ff, 0xe0701a, 0x30c060, 0xf5c518];
  const kinds = ['charger', 'amg', 'vaz'];
  for (let i = 0; i < 7; i++) {
    const c = makeCar(kinds[i % 3], pal[i]); c.rotation.y = Math.PI / 2; // front -> -z
    s6.add(c); convoy.push(c);
  }
}
const snow6 = makeSnow(3000, 0.18, new THREE.Vector3(60, 40, 60), 13); s6.add(snow6);
function renderCity(t) {
  const lt = t - S[4];
  const speed = 14;
  const head = -lt * speed;
  convoy.forEach((c, i) => {
    const lane = (i % 2 ? 1.9 : -1.9);
    c.position.set(lane, 0, head - i * 7 - 6);
    c.userData.spin(lt * speed);
  });
  const rise = easeInOut(lt / 3.6);
  cam6.position.set(lerp(1.5, 0, rise), lerp(2.2, 38, rise), head + lerp(6, 30, rise));
  cam6.lookAt(0, lerp(1.5, 0, rise), head - lerp(20, 45, rise));
  snow6.userData.update(t, new THREE.Vector3(0, cam6.position.y - 5, head - 10), 0.6, 1);
  bloom.strength = 0.7; bloom.threshold = 0.85;
  renderPass.scene = s6; renderPass.camera = cam6;
}

window.renderAt = (t) => {
  if (t < S[1]) renderHook(t);
  else if (t < S[2]) renderAdvent(t);
  else if (t < S[3]) renderPass3(t);
  else if (t < S[3] + 2.5) renderShowroom(t);
  else if (t < S[4]) renderDrift(t);
  else renderCity(t);
  composer.render();
  return true;
};
window.ready = true;
