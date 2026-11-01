import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { loadPair, fillLayer, WORLD, reducedMotion } from "./data.js";
import { glowMaterial, cloudGeometry, paint, COLORS, arrow, textSprite, whenVisible } from "./glow.js";

const LAYER = 31;
const COMETS = 64;
const TRAIL = 26;

export async function initHero(canvas) {
  const P = await loadPair("mistral");
  const N = P.N;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: true, powerPreference: "high-performance" });
  const coarse = matchMedia("(pointer: coarse)").matches;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, coarse ? 1.25 : 1.5));
  renderer.setClearColor(0x000000, 0);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 200);
  camera.position.set(0, 1.4, 15);

  const world = new THREE.Group();
  const spin = new THREE.Group();
  world.add(spin);
  spin.scale.setScalar(0.8);
  scene.add(world);

  // ── measured clouds ──
  const target = new Float32Array(N * 2 * 3);
  fillLayer(P, LAYER, 0, target.subarray(0, N * 3));
  fillLayer(P, LAYER, 1, target.subarray(N * 3));

  const geo = cloudGeometry(N * 2);
  paint(geo, 0, N, COLORS.it, 0.35, 7);
  paint(geo, N, N, COLORS.dpo, 0.35, 11);
  const sizes = geo.attributes.aSize.array;
  for (let i = 0; i < N * 2; i++) sizes[i] = 0.7 + Math.random() * 0.7;
  const pos = geo.attributes.position.array;

  const start = new Float32Array(N * 2 * 3);
  const delay = new Float32Array(N * 2);
  for (let i = 0; i < N * 2; i++) {
    const r = 9 + Math.random() * 10;
    const th = Math.random() * Math.PI * 2;
    const ph = Math.acos(2 * Math.random() - 1);
    start[i * 3] = r * Math.sin(ph) * Math.cos(th);
    start[i * 3 + 1] = r * Math.cos(ph);
    start[i * 3 + 2] = r * Math.sin(ph) * Math.sin(th);
    delay[i] = Math.random() * 0.9;
  }
  pos.set(reducedMotion ? target : start);

  const cloudMat = glowMaterial({ size: 5.2, opacity: 0.95 });
  spin.add(new THREE.Points(geo, cloudMat));

  // ── comets: real states translated by the mean IT→DPO displacement (install) and back (remove) ──
  const gap = P.meta.layers[LAYER].r * WORLD;
  const cometGeo = cloudGeometry(COMETS * 2 * TRAIL);
  const cometPos = cometGeo.attributes.position.array;
  const cometSize = cometGeo.attributes.aSize.array;
  const cometAlpha = cometGeo.attributes.aAlpha.array;
  paint(cometGeo, 0, COMETS * TRAIL, COLORS.dpo, 0, 3);
  paint(cometGeo, COMETS * TRAIL, COMETS * TRAIL, COLORS.rem, 0, 5);
  const comets = [];
  for (let k = 0; k < COMETS * 2; k++) {
    const install = k < COMETS;
    const i = Math.floor(Math.random() * N) + (install ? 0 : N);
    comets.push({
      x: target[i * 3], y: target[i * 3 + 1], z: target[i * 3 + 2],
      dx: install ? gap : -gap,
      phase: Math.random() * 3,
      speed: 0.32 + Math.random() * 0.12,
    });
  }
  const cometMat = glowMaterial({ size: 4.2, opacity: 1 });
  const cometPoints = new THREE.Points(cometGeo, cometMat);
  spin.add(cometPoints);

  // ── direction arrows above and below the clouds ──
  const cIT = new THREE.Vector3(P.meta.layers[LAYER].centroid_it[0] * WORLD, 0, 0);
  const cDPO = new THREE.Vector3(P.meta.layers[LAYER].centroid_dpo[0] * WORLD, 0, 0);
  const up = 3.35;
  const inst = arrow(cIT.clone().setY(up), cDPO.clone().setY(up), COLORS.dpo, { radius: 0.035, head: 0.34 });
  const rem = arrow(cDPO.clone().setY(-up), cIT.clone().setY(-up), COLORS.rem, { radius: 0.035, head: 0.34 });
  spin.add(inst, rem);
  const lInst = textSprite("+λv  install", { color: "#bff7e6", size: 0.6, glow: "#2fe3b4" });
  lInst.position.set(0, up + 0.5, 0);
  const lRem = textSprite("−λv  remove", { color: "#ffd0bf", size: 0.6, glow: "#ff6b3d" });
  lRem.position.set(0, -up - 0.5, 0);
  spin.add(lInst, lRem);

  // ── faint dust for depth ──
  const DUST = 900;
  const dustGeo = cloudGeometry(DUST);
  const dp = dustGeo.attributes.position.array;
  for (let i = 0; i < DUST; i++) {
    dp[i * 3] = (Math.random() - 0.5) * 70;
    dp[i * 3 + 1] = (Math.random() - 0.5) * 40;
    dp[i * 3 + 2] = -10 - Math.random() * 40;
  }
  paint(dustGeo, 0, DUST, new THREE.Color("#8ea2c8"), 0.6, 13);
  dustGeo.attributes.aAlpha.array.fill(0.35);
  const dust = new THREE.Points(dustGeo, glowMaterial({ size: 2.2, opacity: 0.5 }));
  scene.add(dust);

  // ── post ──
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.85, 0.55, 0.05);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  let wide = true;
  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    composer.setSize(w, h);
    camera.aspect = w / h;
    wide = w / h > 1.15;
    camera.fov = wide ? 42 : 58;
    camera.updateProjectionMatrix();
    // on narrow screens the cloud sits up behind the title and steps back from the body text
    world.position.set(wide ? 4.4 : 0.6, wide ? -0.2 : 4.4, 0);
    cloudMat.uniforms.uOpacity.value = wide ? 0.95 : 0.55;
    cometMat.uniforms.uOpacity.value = wide ? 1 : 0.6;
  }
  resize();
  new ResizeObserver(resize).observe(canvas);

  const mouse = { x: 0, y: 0, tx: 0, ty: 0 };
  window.addEventListener("pointermove", (e) => {
    mouse.tx = e.clientX / window.innerWidth - 0.5;
    mouse.ty = e.clientY / window.innerHeight - 0.5;
  });

  let running = true;
  whenVisible(canvas, (v) => (running = v));
  const clock = new THREE.Clock();
  let introT = 0;
  let scrollFade = 0;
  // drop bloom and resolution if this device can't hold ~30 fps
  let lowQuality = false, frames = 0, slowTime = 0;
  window.addEventListener("scroll", () => {
    scrollFade = Math.min(window.scrollY / window.innerHeight, 1);
  }, { passive: true });

  function frame() {
    requestAnimationFrame(frame);
    if (!running || !canvas.clientWidth || !canvas.clientHeight) return;
    const dt = Math.min(clock.getDelta(), 0.05);
    const t = clock.elapsedTime;

    if (!reducedMotion && introT < 2.8) {
      introT += dt;
      for (let i = 0; i < N * 2; i++) {
        let p = Math.min(Math.max((introT - delay[i]) / 1.7, 0), 1);
        p = 1 - Math.pow(1 - p, 4);
        for (let a = 0; a < 3; a++) pos[i * 3 + a] = start[i * 3 + a] + (target[i * 3 + a] - start[i * 3 + a]) * p;
      }
      geo.attributes.position.needsUpdate = true;
    }

    const show = reducedMotion ? 0 : Math.min(Math.max((introT - 1.6) / 1, 0), 1);
    const labels = reducedMotion ? 1 : show;
    for (let k = 0; k < comets.length; k++) {
      const c = comets[k];
      const cycle = (t * c.speed + c.phase) % 1.6;
      for (let j = 0; j < TRAIL; j++) {
        const idx = k * TRAIL + j;
        const p = cycle - j * 0.0075;
        const vis = p >= 0 && p <= 1 ? 1 : 0;
        const e = p <= 0 ? 0 : p >= 1 ? 1 : p * p * (3 - 2 * p);
        cometPos[idx * 3] = c.x + c.dx * e;
        cometPos[idx * 3 + 1] = c.y;
        cometPos[idx * 3 + 2] = c.z;
        const fade = Math.sin(Math.min(Math.max(p, 0), 1) * Math.PI);
        cometAlpha[idx] = vis * show * fade * Math.pow(1 - j / TRAIL, 1.6) * 0.8;
        cometSize[idx] = (1 - j / TRAIL) * 1.1 + 0.35;
      }
    }
    cometGeo.attributes.position.needsUpdate = true;
    cometGeo.attributes.aAlpha.needsUpdate = true;
    cometGeo.attributes.aSize.needsUpdate = true;

    const ar = show * (0.6 + 0.4 * Math.sin(t * 1.6));
    inst.userData.mat.opacity = rem.userData.mat.opacity = 0.25 + 0.7 * labels;
    lInst.material.opacity = lRem.material.opacity = labels;
    bloom.strength = 0.75 + 0.15 * ar;

    if (!reducedMotion) spin.rotation.y = Math.sin(t * 0.11) * 0.55 - 0.25;
    if (!reducedMotion) {
      mouse.x += (mouse.tx - mouse.x) * 0.04;
      mouse.y += (mouse.ty - mouse.y) * 0.04;
    }
    world.rotation.x = 0.12 + mouse.y * 0.25;
    world.rotation.y = mouse.x * 0.4;
    dust.rotation.y = t * 0.01;
    camera.position.z = 15 + scrollFade * 4;
    camera.position.y = 1.4 - scrollFade * 1.5;
    camera.lookAt(0, wide ? 0 : 0.6, 0);
    lowQuality ? renderer.render(scene, camera) : composer.render();

    if (frames === 0) document.documentElement.classList.add("gl-ready");
    if (!lowQuality && document.visibilityState === "visible" && ++frames > 30 && frames <= 150) {
      slowTime += dt;
      if (frames === 150 && slowTime / 120 > 1 / 30) {
        lowQuality = true;
        renderer.setPixelRatio(1);
        resize();
      }
    }
  }
  frame();
}
