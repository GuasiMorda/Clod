import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { buildM5, buildCybertruck, buildViper } from './carmodels.js';

const P = new URLSearchParams(location.search);
const W = +(P.get('w') || 1080), H = +(P.get('h') || 1920);
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(1); renderer.setSize(W, H);
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07060b);
scene.fog = new THREE.Fog(0x07060b, 12, 30);
const pm = new THREE.PMREMGenerator(renderer);
scene.environment = pm.fromScene(new RoomEnvironment(), 0.03).texture;
scene.environmentIntensity = 0.35;

// studio: glossy dark floor, overhead soft box, purple rim lights (STATE RP accent)
const floor = new THREE.Mesh(new THREE.CircleGeometry(30, 64), new THREE.MeshPhysicalMaterial({ color: 0x050508, roughness: 0.5, metalness: 0.1, clearcoat: 0.25, clearcoatRoughness: 0.6 }));
floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);
const key = new THREE.SpotLight(0xffffff, 110, 30, 0.6, 0.8, 1.5); key.position.set(2, 9, 4); key.castShadow = true; key.shadow.mapSize.set(2048, 2048); scene.add(key); scene.add(key.target);
const rimA = new THREE.SpotLight(0x9a40ff, 22, 25, 0.5, 0.8, 1.5); rimA.position.set(-7, 4.5, -5); rimA.target.position.set(0, 0.9, 0); scene.add(rimA); scene.add(rimA.target);
const rimB = new THREE.SpotLight(0x5c7cff, 45, 25, 0.6, 0.8, 1.5); rimB.position.set(7, 2.5, -4); scene.add(rimB); scene.add(rimB.target);
const fill = new THREE.HemisphereLight(0x6050a0, 0x050408, 0.25); scene.add(fill);
// overhead light strips (reflect nicely on the paint)
for (let k = -2; k <= 2; k++) {
  const st = new THREE.Mesh(new THREE.BoxGeometry(6, 0.04, 0.12), new THREE.MeshBasicMaterial({ color: 0x9a9a9a }));
  st.position.set(0, 5.5, k * 0.9); scene.add(st);
}
const ring = new THREE.Mesh(new THREE.TorusGeometry(4.2, 0.03, 8, 128), new THREE.MeshStandardMaterial({ color: 0x8a3cff, emissive: 0x8a3cff, emissiveIntensity: 1.6 }));
ring.rotation.x = Math.PI / 2; ring.position.y = 0.01; scene.add(ring);

const cam = new THREE.PerspectiveCamera(30, W / H, 0.1, 200);
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, cam));
composer.addPass(new UnrealBloomPass(new THREE.Vector2(W / 2, H / 2), 0.35, 0.4, 0.95));
composer.addPass(new OutputPass());

const builders = { m5: () => buildM5(0x14213f), cyber: () => buildCybertruck(), viper: () => buildViper(0x1546c8) };
let car = null, which = null;
window.setCar = (name) => {
  if (car) scene.remove(car);
  car = builders[name](); which = name; scene.add(car);
  return car.name;
};
// shots: t in [0,1] turntable, or named beauty angles
window.renderShot = (mode, t) => {
  const len = which === 'cyber' ? 5.7 : which === 'viper' ? 4.5 : 5.0;
  const hf = Math.atan(Math.tan(THREE.MathUtils.degToRad(cam.fov / 2)) * cam.aspect);   // horizontal half-FOV
  const fit = (w) => (w / 2) / Math.tan(hf);
  let az, el, dist, ty = 0.65;
  if (mode === 'turn') { az = t * Math.PI * 2 + 0.6; el = 0.12 + 0.05 * Math.sin(t * Math.PI * 2); dist = fit(len * 1.02); }
  else if (mode === 'front') { az = 0.6; el = 0.12; dist = fit(len * 0.95); }
  else if (mode === 'rear') { az = Math.PI + 0.6; el = 0.14; dist = fit(len * 0.95); }
  else if (mode === 'side') { az = Math.PI / 2; el = 0.04; dist = fit(len * 1.12); }
  else if (mode === 'low') { az = 0.35; el = 0.03; dist = fit(len * 0.62); ty = 0.55; }
  dist = Math.max(dist, len * 1.1);
  cam.position.set(Math.cos(az) * dist, 0.35 + Math.sin(el) * dist, Math.sin(az) * dist);
  cam.lookAt(0, ty, 0);
  composer.render();
  return true;
};
window.exportGLB = async () => {
  const ex = new GLTFExporter();
  const buf = await ex.parseAsync(car, { binary: true });
  const bytes = new Uint8Array(buf); let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s);
};
window.ready = true;
