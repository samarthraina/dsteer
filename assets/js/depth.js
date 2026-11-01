import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { loadPair, WORLD, PAIR_LABEL, reducedMotion } from "./data.js";
import { glowMaterial, cloudGeometry, paint, COLORS, textSprite, whenVisible } from "./glow.js";

const GAP = 0.62;
const S = WORLD * 0.62;
const zOf = (l) => (l - 15.5) * GAP;

export async function initDepth() {
  const canvas = document.getElementById("depth-gl");
  const chip = document.getElementById("depth-chip");
  let P = await loadPair("mistral");
  const N = P.N, L = P.L;
  let active = 31;
  let pairName = "mistral";

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 200);
  camera.position.set(-4, 16, 24);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.enableZoom = true;
  controls.minDistance = 10;
  controls.maxDistance = 70;
  controls.target.set(0, 0, zOf(active));

  const geo = cloudGeometry(N * 2 * L);
  const pos = geo.attributes.position.array;
  const alpha = geo.attributes.aAlpha.array;
  const layerOf = new Uint8Array(N * 2 * L);
  for (let l = 0; l < L; l++) {
    paint(geo, (l * 2) * N, N, COLORS.it, 0.3, 3 + l);
    paint(geo, (l * 2 + 1) * N, N, COLORS.dpo, 0.3, 5 + l);
    layerOf.fill(l, l * 2 * N, (l + 1) * 2 * N);
  }
  const mat = glowMaterial({ size: 4.4, opacity: 1 });
  scene.add(new THREE.Points(geo, mat));

  // frames for each slice
  const frames = [];
  for (let l = 0; l < L; l++) {
    const steered = l >= 27;
    const g = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-4.6, -3.4, 0), new THREE.Vector3(4.6, -3.4, 0),
      new THREE.Vector3(4.6, 3.4, 0), new THREE.Vector3(-4.6, 3.4, 0),
    ]);
    const m = new THREE.LineLoop(g, new THREE.LineBasicMaterial({
      color: steered ? 0xa48bff : 0x5a6788, transparent: true, opacity: steered ? 0.5 : 0.14, depthWrite: false,
    }));
    m.position.z = zOf(l);
    m.userData.base = steered ? 0.5 : 0.14;
    frames.push(m);
    scene.add(m);
  }
  for (const l of [0, 8, 16, 24, 31]) {
    const s = textSprite(`L${l}`, { color: l >= 27 ? "#c9b8ff" : "#8b95ad", size: 0.5, font: "500 46px 'JetBrains Mono', monospace" });
    s.position.set(-5.3, 3.4, zOf(l));
    scene.add(s);
  }
  const steerLbl = textSprite("steered 27–31", { color: "#c9b8ff", size: 0.5, font: "500 44px 'JetBrains Mono', monospace", glow: "#a48bff" });
  steerLbl.position.set(2.6, 4.0, zOf(29));
  scene.add(steerLbl);

  // centroid rails: where the IT and DPO clouds sit along v̂ at every depth
  const railMat = (c) => new THREE.LineBasicMaterial({ color: c, transparent: true, opacity: 0.9, toneMapped: false });
  const railIT = new THREE.Line(new THREE.BufferGeometry(), railMat(COLORS.it));
  const railDPO = new THREE.Line(new THREE.BufferGeometry(), railMat(COLORS.dpo));
  scene.add(railIT, railDPO);

  function build() {
    for (let l = 0; l < L; l++) {
      for (let m = 0; m < 2; m++) {
        const o = (l * 2 + m) * N * 3;
        for (let i = 0; i < N; i++) {
          const k = o + i * 3;
          pos[k] = P.data[k] * S;
          pos[k + 1] = P.data[k + 1] * S;
          pos[k + 2] = zOf(l) + P.data[k + 2] * S * 0.12;
        }
      }
    }
    geo.attributes.position.needsUpdate = true;
    const rail = (key) => P.meta.layers.map((x, l) => new THREE.Vector3(x[key][0] * S, -3.4, zOf(l)));
    railIT.geometry.setFromPoints(rail("centroid_it"));
    railDPO.geometry.setFromPoints(rail("centroid_dpo"));
    chip.textContent = `${PAIR_LABEL[pairName]} · layer ${active}`;
  }
  build();

  let shown = active;
  function shade() {
    for (let i = 0; i < alpha.length; i++) {
      const d = Math.abs(layerOf[i] - shown);
      alpha[i] = d < 0.5 ? 1 : Math.max(0.12, 0.5 - d * 0.035);
    }
    geo.attributes.aAlpha.needsUpdate = true;
    frames.forEach((f, l) => {
      f.material.opacity = Math.abs(l - shown) < 0.5 ? 0.95 : f.userData.base;
    });
  }
  shade();

  const api = {
    setLayer(l) {
      active = l;
      chip.textContent = `${PAIR_LABEL[pairName]} · layer ${l}`;
    },
    async setPair(p) {
      pairName = p;
      P = await loadPair(p);
      build();
    },
  };

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    // keep the whole stack in frame: narrow screens need more distance
    const dir = camera.position.clone().sub(controls.target).normalize();
    camera.position.copy(controls.target).addScaledVector(dir, w / h < 1 ? 46 : 25);
  }
  resize();
  new ResizeObserver(resize).observe(canvas);

  let running = false;
  whenVisible(canvas, (v) => (running = v));
  const clock = new THREE.Clock();
  let lastShade = -1;
  function frame() {
    requestAnimationFrame(frame);
    if (!running || !canvas.clientWidth || !canvas.clientHeight) return;
    const dt = Math.min(clock.getDelta(), 0.05);
    shown += (active - shown) * Math.min(dt * (reducedMotion ? 60 : 6), 1);
    if (Math.abs(shown - lastShade) > 0.02) {
      shade();
      lastShade = shown;
    }
    controls.target.z += (zOf(shown) * 0.35 - controls.target.z) * Math.min(dt * 3, 1);
    controls.update();
    renderer.render(scene, camera);
  }
  frame();
  return api;
}
