import * as THREE from 'three';
import { RectAreaLightUniformsLib } from 'three/addons/lights/RectAreaLightUniformsLib.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const W = 1080, H = 1920;
RectAreaLightUniformsLib.init();
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(1); renderer.setSize(W, H);
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 0.85;
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x120818);
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.04;

const cam = new THREE.PerspectiveCamera(52, W / H, 0.01, 50);
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, cam));
composer.addPass(new UnrealBloomPass(new THREE.Vector2(W / 2, H / 2), 0.5, 0.6, 0.9));
composer.addPass(new OutputPass());

function canvasTex(w, h, draw) {
  const c = document.createElement('canvas'); c.width = w; c.height = h; draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; return t;
}

// ---------------- room ----------------
const DESK_Y = 0.75;
const desk = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.04, 2.6),
  new THREE.MeshStandardMaterial({ color: 0x8a7f95, roughness: 0.32, metalness: 0.0 }));
desk.position.set(0, DESK_Y - 0.02, 0.15); desk.receiveShadow = true; scene.add(desk);

const wallMat = new THREE.MeshStandardMaterial({ color: 0xc9bcd6, roughness: 0.9 });
const wall = new THREE.Mesh(new THREE.PlaneGeometry(8, 5), wallMat); wall.position.set(0, 1.8, -2.4); scene.add(wall);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(8, 8), new THREE.MeshStandardMaterial({ color: 0x2a2030, roughness: 0.8 }));
floor.rotation.x = -Math.PI / 2; scene.add(floor);
// low white console along the wall
const consoleM = new THREE.MeshStandardMaterial({ color: 0xe6e1ea, roughness: 0.4 });
const cons = new THREE.Mesh(new THREE.BoxGeometry(2.6, 0.42, 0.45), consoleM); cons.position.set(-0.2, 0.98, -2.15); scene.add(cons);
// TV on the wall (generic streaming UI)
const tvTex = canvasTex(1280, 720, (g, w, h) => {
  g.fillStyle = '#05060a'; g.fillRect(0, 0, w, h);
  const cols = ['#c0392b', '#2e86de', '#f5b041', '#8e44ad', '#16a085', '#e67e22', '#d35400', '#5d6d7e'];
  for (let r = 0; r < 3; r++) for (let c = 0; c < 6; c++) {
    const gr = g.createLinearGradient(0, 0, 0, 200);
    gr.addColorStop(0, cols[(r * 6 + c) % cols.length]); gr.addColorStop(1, '#111');
    g.fillStyle = gr; g.fillRect(40 + c * 205, 140 + r * 190, 185, 160);
  }
  g.fillStyle = '#ddd'; g.font = 'bold 40px DejaVu Sans'; g.fillText('Главная', 40, 90);
});
const tv = new THREE.Group();
tv.add(new THREE.Mesh(new THREE.BoxGeometry(1.25, 0.72, 0.04), new THREE.MeshStandardMaterial({ color: 0x0a0a0c, roughness: 0.3 })));
const tvs = new THREE.Mesh(new THREE.PlaneGeometry(1.21, 0.68), new THREE.MeshBasicMaterial({ map: tvTex, toneMapped: true, color: 0x9a9a9a }));
tvs.position.z = 0.021; tv.add(tvs);
tv.position.set(-0.85, 2.0, -2.37); scene.add(tv);
const shelf = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.04, 0.2), new THREE.MeshStandardMaterial({ color: 0x1a1a1e }));
shelf.position.set(-0.9, 1.45, -2.3); scene.add(shelf);
// speaker on the console
const spk = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.08, 0.32, 24), new THREE.MeshStandardMaterial({ color: 0x18161c, roughness: 0.5 }));
spk.position.set(0.75, 1.35, -2.1); scene.add(spk);

// LED strips on the wall + purple wash
const ledM = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffd6ff, emissiveIntensity: 3 });
for (const x of [0.12, 0.62]) {
  const s = new THREE.Mesh(new THREE.BoxGeometry(0.02, 2.2, 0.01), ledM); s.position.set(x, 2.3, -2.38); scene.add(s);
  const ra = new THREE.RectAreaLight(0xb21cff, 6, 0.25, 2.4); ra.position.set(x, 2.2, -2.35); ra.lookAt(x, 2.2, 0); scene.add(ra);
}
const wash = new THREE.PointLight(0xa020ff, 2.5, 4, 1.5); wash.position.set(0.4, 1.5, -1.9); scene.add(wash);
const wash2 = new THREE.PointLight(0xe060ff, 0.8, 3, 1.5); wash2.position.set(-0.7, 1.3, -1.2); scene.add(wash2);
scene.add(new THREE.HemisphereLight(0x9050c0, 0x201020, 0.12));
const fill = new THREE.DirectionalLight(0xd0a0ff, 0.12); fill.position.set(1, 2, 2); scene.add(fill);

// ---------------- laptop ----------------
const alu = new THREE.MeshPhysicalMaterial({ color: 0x3a3a40, metalness: 0.9, roughness: 0.35, clearcoat: 0.3 });
const LW = 0.31, LD = 0.215, LT = 0.0155;
const laptop = new THREE.Group(); laptop.position.set(0, DESK_Y, -0.12); scene.add(laptop);
const base = new THREE.Mesh(new THREE.BoxGeometry(LW, LT, LD), alu); base.position.y = LT / 2; base.castShadow = true; laptop.add(base);
// keyboard well + keys
const well = new THREE.Mesh(new THREE.BoxGeometry(LW * 0.9, 0.001, LD * 0.43), new THREE.MeshStandardMaterial({ color: 0x0c0c0e, roughness: 0.8 }));
well.position.set(0, LT + 0.0002, -LD * 0.17); laptop.add(well);
const keyM = new THREE.MeshStandardMaterial({ color: 0x0e0e10, roughness: 0.55 });
const keyGeo = new THREE.BoxGeometry(1, 0.0012, 1);
const keys = []; const ks = 0.0168, kg = 0.0019;
const rows = [[14, 0.62], [14, 1], [14, 1], [13, 1], [12, 1], [9, 1]];
let ky = -LD * 0.17 - 2.5 * (ks + kg);
rows.forEach(([n, hScale], r) => {
  const total = LW * 0.86; const w = (total - (n - 1) * kg) / n;
  for (let i = 0; i < n; i++) {
    let kw = w; let kx = -total / 2 + i * (w + kg) + w / 2;
    if (r === 5 && i === 4) { kw = w * 3.5; }
    if (r === 5 && i > 4) kx += w * 2.5 + kg * 2;
    if (r === 5 && i > 4 && kx > total / 2) continue;
    keys.push([kx, ky + (r ? 0 : -ks * (1 - hScale) / 2), kw, ks * hScale]);
  }
  ky += ks + kg;
});
const inst = new THREE.InstancedMesh(keyGeo, keyM, keys.length);
const m4 = new THREE.Matrix4();
keys.forEach(([x, z, w, d], i) => { m4.compose(new THREE.Vector3(x, LT + 0.0009, z), new THREE.Quaternion(), new THREE.Vector3(w, 1, d)); inst.setMatrixAt(i, m4); });
laptop.add(inst);
// key legends
const legendTex = canvasTex(2048, 640, (g, w, h) => {
  g.clearRect(0, 0, w, h); g.fillStyle = 'rgba(210,210,220,0.75)'; g.font = '30px DejaVu Sans'; g.textAlign = 'center';
  const L = ['ESC F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 ⏻', '` 1 2 3 4 5 6 7 8 9 0 - = ⌫', '⇥ Й Ц У К Е Н Г Ш Щ З Х Ъ \\', '⇪ Ф Ы В А П Р О Л Д Ж Э ↵', '⇧ Я Ч С М И Т Ь Б Ю . ⇧', 'fn ctrl alt cmd    cmd alt ◀ ▶'];
  L.forEach((row, r) => { const it = row.split(' '); it.forEach((s, i) => g.fillText(s, (i + 0.5) * w / it.length, (r + 0.62) * h / 6)); });
});
const legend = new THREE.Mesh(new THREE.PlaneGeometry(LW * 0.86, 6 * (ks + kg)), new THREE.MeshBasicMaterial({ map: legendTex, transparent: true, opacity: 0.55, depthWrite: false }));
legend.rotation.x = -Math.PI / 2; legend.position.set(0, LT + 0.0017, -LD * 0.17); laptop.add(legend);
// trackpad
const pad = new THREE.Mesh(new THREE.BoxGeometry(LW * 0.42, 0.0004, LD * 0.33), new THREE.MeshPhysicalMaterial({ color: 0x45454c, metalness: 0.6, roughness: 0.25 }));
pad.position.set(0, LT + 0.0002, LD * 0.29); laptop.add(pad);
// lid
const lid = new THREE.Group(); lid.position.set(0, LT, -LD / 2); laptop.add(lid);
const LIDT = 0.006, LIDH = 0.205;
const lidShell = new THREE.Mesh(new THREE.BoxGeometry(LW, LIDH, LIDT), alu); lidShell.position.set(0, LIDH / 2, -LIDT / 2); lidShell.castShadow = true; lid.add(lidShell);
const bezel = new THREE.Mesh(new THREE.PlaneGeometry(LW - 0.004, LIDH - 0.004), new THREE.MeshStandardMaterial({ color: 0x050506, roughness: 0.15 }));
bezel.position.set(0, LIDH / 2, 0.0005); lid.add(bezel);
const SCW = 0.288, SCH = 0.18;
const screenM = new THREE.MeshBasicMaterial({ color: 0x000000 });
const screen = new THREE.Mesh(new THREE.PlaneGeometry(SCW, SCH), screenM);
screen.position.set(0, LIDH / 2 - 0.004, 0.001); lid.add(screen);
lid.rotation.x = -0.30;   // opened ~107°
// screen light onto keyboard / desk
const scrLight = new THREE.RectAreaLight(0x8060ff, 6, SCW, SCH);
scrLight.position.set(0, LIDH / 2, 0.01); lid.add(scrLight);
scrLight.rotation.y = 0; // faces +z of lid
// RectAreaLight faces -z by default; flip to face the user
scrLight.lookAt(new THREE.Vector3(0, LIDH / 2, 1).applyMatrix4(new THREE.Matrix4()));
const scrPoint = new THREE.PointLight(0x8060ff, 0.6, 0.6, 2); scrPoint.position.set(0, 0.1, 0.12); laptop.add(scrPoint);

// ---------------- handheld camera ----------------
function noise(t, f, ph) { return Math.sin(t * f + ph) * 0.6 + Math.sin(t * f * 2.13 + ph * 1.7) * 0.3 + Math.sin(t * f * 4.7 + ph * 0.3) * 0.1; }
function camAt(t) {
  const push = Math.min(1, t / 20);
  const p = new THREE.Vector3(0.0 + 0.004 * noise(t, 0.9, 1), 0.985 + 0.003 * noise(t, 1.1, 2), lerp(0.43, 0.395, push));
  cam.position.copy(p);
  const look = new THREE.Vector3(0.0 + 0.006 * noise(t, 0.7, 3), 0.86 + 0.005 * noise(t, 0.8, 5), -0.2);
  cam.lookAt(look);
  cam.rotateZ(0.006 * noise(t, 0.6, 7) - 0.012);
}
function lerp(a, b, x) { return a + (b - a) * x; }

const corners = [[-SCW / 2, SCH / 2], [SCW / 2, SCH / 2], [SCW / 2, -SCH / 2], [-SCW / 2, -SCH / 2]];
window.renderAt = (t, r, g, b) => {
  camAt(t);
  const col = new THREE.Color(r, g, b);
  const lum = 0.3 + (r + g + b);
  scrLight.color.copy(col); scrLight.intensity = 0.6 + 2.5 * lum;
  scrPoint.color.copy(col); scrPoint.intensity = 0.0;
  composer.render();
  scene.updateMatrixWorld(true);
  return corners.map(([x, y]) => {
    const v = new THREE.Vector3(x, y, 0).applyMatrix4(screen.matrixWorld).project(cam);
    return [(v.x + 1) / 2 * W, (1 - v.y) / 2 * H];
  });
};
window.ready = true;
