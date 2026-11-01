import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { loadPair, fillLayer, WORLD, LAMBDA_STAR, reducedMotion } from "./data.js";
import { glowMaterial, cloudGeometry, paint, COLORS, arrow, setArrow, textSprite, whenVisible } from "./glow.js";

const LMAX = 0.4;
const LINKS = 150;
const FLOOR = -3.4;
const GRID_X = [-7, 7];
const BINS = 160;

const $ = (id) => document.getElementById(id);

export async function initLab() {
  const canvas = $("lab-gl");
  const state = { pair: "mistral", dir: "install", lambda: 0, shown: 0, layer: 31, links: false, touched: false };

  let P = await loadPair(state.pair);
  const N = P.N;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
  camera.position.set(3.2, 3.6, 12.5);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.minDistance = 6;
  controls.maxDistance = 22;
  controls.enablePan = false;
  // sway around the starting view rather than spin, so v̂ never turns end-on
  let idle = !reducedMotion;
  controls.addEventListener("start", () => (idle = false));
  const home = new THREE.Spherical().setFromVector3(camera.position);

  // ── floor grid + v̂ axis ──
  const grid = new THREE.GridHelper(12, 12, 0x2a3350, 0x161c2e);
  grid.position.y = FLOOR;
  grid.material.transparent = true;
  grid.material.opacity = 0.55;
  scene.add(grid);
  const axis = arrow(new THREE.Vector3(-6, FLOOR, 0), new THREE.Vector3(6.2, FLOOR, 0), new THREE.Color("#c9d3ea"), { radius: 0.016, head: 0.28 });
  axis.userData.mat.opacity = 0.6;
  scene.add(axis);
  const axisLabel = textSprite("v̂   IT → DPO", { color: "#c9d3ea", size: 0.36, font: "500 46px 'JetBrains Mono', monospace" });
  axisLabel.position.set(4.6, FLOOR + 0.45, 0.6);
  scene.add(axisLabel);

  // ── points: [IT | DPO | steered] ──
  const geo = cloudGeometry(N * 3);
  paint(geo, 0, N, COLORS.it, 0.3, 3);
  paint(geo, N, N, COLORS.dpo, 0.3, 5);
  const pos = geo.attributes.position.array;
  const alpha = geo.attributes.aAlpha.array;
  const sizes = geo.attributes.aSize.array;
  for (let i = 0; i < N * 3; i++) sizes[i] = i < 2 * N ? 0.9 : 1.05;
  const mat = glowMaterial({ size: 6.5, opacity: 1 });
  scene.add(new THREE.Points(geo, mat));

  // ── prompt-paired links ──
  const linkGeo = new THREE.BufferGeometry();
  const linkPos = new Float32Array(LINKS * 2 * 3);
  const linkCol = new Float32Array(LINKS * 2 * 3);
  for (let k = 0; k < LINKS; k++) {
    COLORS.it.toArray(linkCol, k * 6);
    COLORS.dpo.toArray(linkCol, k * 6 + 3);
  }
  linkGeo.setAttribute("position", new THREE.BufferAttribute(linkPos, 3));
  linkGeo.setAttribute("color", new THREE.BufferAttribute(linkCol, 3));
  const linkMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.22, blending: THREE.AdditiveBlending, depthWrite: false });
  const links = new THREE.LineSegments(linkGeo, linkMat);
  scene.add(links);
  const linkIdx = Array.from({ length: LINKS }, (_, k) => Math.floor((k * N) / LINKS));

  // ── centroid markers and the steering arrow ──
  const markerGeo = new THREE.SphereGeometry(0.11, 24, 16);
  const mk = (c) => new THREE.Mesh(markerGeo, new THREE.MeshBasicMaterial({ color: c, toneMapped: false }));
  const mIT = mk(COLORS.it), mDPO = mk(COLORS.dpo), mSteer = mk(COLORS.white);
  scene.add(mIT, mDPO, mSteer);
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.2, 0.26, 48), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8, side: THREE.DoubleSide, toneMapped: false }));
  scene.add(ring);
  const steerArrow = arrow(new THREE.Vector3(), new THREE.Vector3(1, 0, 0), COLORS.dpo, { radius: 0.045, head: 0.36 });
  scene.add(steerArrow);
  // floor shadows of the three centroids so the shift along v̂ reads at any camera angle
  const dropGeo = new THREE.CircleGeometry(0.16, 32);
  const drop = (c) => {
    const m = new THREE.Mesh(dropGeo, new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.85, toneMapped: false }));
    m.rotation.x = -Math.PI / 2;
    m.position.y = FLOOR + 0.01;
    scene.add(m);
    return m;
  };
  const dIT = drop(COLORS.it), dDPO = drop(COLORS.dpo), dSteer = drop(COLORS.white);

  // ── layer morph buffers ──
  const base = { it: new Float32Array(N * 3), dpo: new Float32Array(N * 3) };
  const from = { it: new Float32Array(N * 3), dpo: new Float32Array(N * 3) };
  const to = { it: new Float32Array(N * 3), dpo: new Float32Array(N * 3) };
  let morph = 1;
  const centroid = { it: new THREE.Vector3(), dpo: new THREE.Vector3() };

  function setTargets(initial) {
    from.it.set(base.it);
    from.dpo.set(base.dpo);
    fillLayer(P, state.layer, 0, to.it);
    fillLayer(P, state.layer, 1, to.dpo);
    if (initial) {
      base.it.set(to.it);
      base.dpo.set(to.dpo);
      morph = 1;
    } else morph = reducedMotion ? 1 : 0;
    computeDensity();
  }

  function mean(arr) {
    const v = new THREE.Vector3();
    for (let i = 0; i < N; i++) v.x += arr[i * 3], v.y += arr[i * 3 + 1], v.z += arr[i * 3 + 2];
    return v.multiplyScalar(1 / N);
  }

  // ── density along v̂ (side panel) ──
  const ridge = $("lab-ridge");
  const rctx = ridge.getContext("2d");
  const dens = { it: new Float32Array(BINS), dpo: new Float32Array(BINS) };
  function kde(arr, out) {
    const xs = new Float32Array(N);
    let m = 0;
    for (let i = 0; i < N; i++) m += xs[i] = arr[i * 3];
    m /= N;
    let v = 0;
    for (let i = 0; i < N; i++) v += (xs[i] - m) ** 2;
    const h = 1.06 * Math.sqrt(v / (N - 1)) * Math.pow(N, -0.2);
    const step = (GRID_X[1] - GRID_X[0]) / (BINS - 1);
    for (let b = 0; b < BINS; b++) {
      const x = GRID_X[0] + b * step;
      let s = 0;
      for (let i = 0; i < N; i++) s += Math.exp(-0.5 * ((x - xs[i]) / h) ** 2);
      out[b] = s / (N * h * Math.sqrt(2 * Math.PI));
    }
  }
  function computeDensity() {
    kde(to.it, dens.it);
    kde(to.dpo, dens.dpo);
  }
  function drawRidge() {
    const w = ridge.clientWidth, h = ridge.clientHeight, pr = Math.min(window.devicePixelRatio, 2);
    if (ridge.width !== w * pr) { ridge.width = w * pr; ridge.height = h * pr; }
    rctx.setTransform(pr, 0, 0, pr, 0, 0);
    rctx.clearRect(0, 0, w, h);
    const peak = Math.max(...dens.it, ...dens.dpo) * 1.1;
    const X = (x) => ((x - GRID_X[0]) / (GRID_X[1] - GRID_X[0])) * w;
    const Y = (d) => h - 6 - (d / peak) * (h - 12);
    const step = (GRID_X[1] - GRID_X[0]) / (BINS - 1);
    const shift = (state.dir === "install" ? 1 : -1) * state.shown * WORLD;
    const curve = (d, dx, stroke, fill, dash) => {
      rctx.beginPath();
      for (let b = 0; b < BINS; b++) {
        const x = X(GRID_X[0] + b * step + dx), y = Y(d[b]);
        b ? rctx.lineTo(x, y) : rctx.moveTo(x, y);
      }
      if (fill) {
        rctx.lineTo(X(GRID_X[1] + dx), h - 6);
        rctx.lineTo(X(GRID_X[0] + dx), h - 6);
        rctx.closePath();
        rctx.fillStyle = fill;
        rctx.fill();
      }
      rctx.setLineDash(dash || []);
      rctx.strokeStyle = stroke;
      rctx.lineWidth = 2;
      rctx.stroke();
      rctx.setLineDash([]);
    };
    rctx.strokeStyle = "rgba(255,255,255,0.12)";
    rctx.beginPath(); rctx.moveTo(0, h - 5.5); rctx.lineTo(w, h - 5.5); rctx.stroke();
    curve(dens.it, 0, "#5b8dff", "rgba(91,141,255,0.16)");
    curve(dens.dpo, 0, "#2fe3b4", "rgba(47,227,180,0.14)");
    if (state.shown > 0.0005) curve(state.dir === "install" ? dens.it : dens.dpo, shift, "#ffffff", null, [5, 4]);
  }

  // ── controls ──
  const lam = $("lab-lambda"), lamVal = $("lab-lambda-val"), layerIn = $("lab-layer"), layerVal = $("lab-layer-val");
  const ticks = $("lab-ticks"), layerTicks = $("lab-layer-ticks");

  function setFill(input) {
    input.style.setProperty("--p", `${(input.value / input.max) * 100}%`);
  }

  function renderTicks() {
    const star = LAMBDA_STAR[state.pair][state.dir];
    const r = P.meta.layers[state.layer].r;
    const steered = state.layer >= 27;
    const items = [
      { v: 0, label: "0", cls: "" },
      { v: star, label: `λ* ${star.toFixed(3)}${steered ? "" : " ·"}`, cls: "star", title: steered ? "frozen coefficient for this pair and direction" : "frozen coefficient (applied at layers 27–31)" },
      { v: r, label: `full gap ${r.toFixed(3)}`, cls: "full", title: "mean IT→DPO displacement at this layer" },
    ];
    // nudge labels apart when they collide
    if (Math.abs(star - r) < 0.06) items[2].label = `gap ${r.toFixed(2)}`;
    ticks.innerHTML = items
      .filter((t) => t.v <= LMAX)
      .map((t) => `<span class="t ${t.cls}" data-v="${t.v}" title="${t.title || ""}" style="left:${(t.v / LMAX) * 100}%">${t.label}</span>`)
      .join("");
    ticks.querySelectorAll(".t").forEach((el) => el.addEventListener("click", () => setLambda(+el.dataset.v, true)));
  }

  layerTicks.innerHTML =
    `<span class="band-mark" style="left:${(27 / 31) * 100}%;width:${(4 / 31) * 100}%"></span>` +
    [0, 8, 16, 24, 31].map((l) => `<span class="t" style="left:${(l / 31) * 100}%;top:6px">${l}</span>`).join("");

  function setLambda(v, animate) {
    state.touched = true;
    state.lambda = Math.max(0, Math.min(LMAX, v));
    lam.value = Math.round((state.lambda / LMAX) * 1000);
    setFill(lam);
    if (!animate) state.shown = state.lambda;
    updateReadout();
  }

  function updateReadout() {
    const r = P.meta.layers[state.layer].r;
    lamVal.textContent = `${state.dir === "install" ? "+" : "−"}${state.lambda.toFixed(3)}`;
    $("lab-frac").textContent = `${Math.round((state.lambda / r) * 100)}%`;
    $("lab-d").textContent = P.meta.layers[state.layer].cohens_d.toFixed(2);
  }

  function applyDir() {
    const install = state.dir === "install";
    const c = install ? COLORS.dpo : COLORS.rem;
    const tint = new THREE.Color().copy(COLORS.white).lerp(c, 0.22);
    paint(geo, 2 * N, N, tint, 0.15, 9);
    steerArrow.userData.mat.color.copy(c);
    ring.material.color.copy(c);
    lam.style.setProperty("--track-fill", install ? "var(--dpo)" : "var(--rem)");
    $("lab-steer-chip").innerHTML = `<i style="background:${install ? "#bff7e6" : "#ffd0bf"}"></i>steered: ${install ? "IT + λv" : "DPO − λv"}`;
    renderTicks();
    updateReadout();
  }

  function seg(id, key, onChange) {
    $(id).addEventListener("click", async (e) => {
      const b = e.target.closest("button");
      if (!b || b.classList.contains("on")) return;
      $(id).querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      state[key] = b.dataset[key];
      state.touched = true;
      await onChange();
    });
  }
  seg("lab-pair", "pair", async () => {
    P = await loadPair(state.pair);
    setTargets(false);
    applyDir();
  });
  seg("lab-dir", "dir", async () => applyDir());

  lam.addEventListener("input", () => setLambda((lam.value / 1000) * LMAX, true));
  layerIn.addEventListener("input", () => {
    state.layer = +layerIn.value;
    state.touched = true;
    layerVal.innerHTML = state.layer >= 27 ? `${state.layer}<span class="tag-steer">steered</span>` : state.layer;
    setFill(layerIn);
    setTargets(false);
    renderTicks();
    updateReadout();
  });
  $("lab-links").addEventListener("change", (e) => (state.links = e.target.checked));

  setFill(layerIn);
  layerVal.innerHTML = `31<span class="tag-steer">steered</span>`;
  setTargets(true);
  applyDir();
  setLambda(0);
  state.touched = false;

  // ── demo on first view: ease λ up to the frozen coefficient ──
  let demoed = false;
  const io = new IntersectionObserver(([e]) => {
    if (!e.isIntersecting || demoed || state.touched) return;
    demoed = true;
    const target = LAMBDA_STAR[state.pair][state.dir];
    if (reducedMotion) {
      setLambda(target);
      state.touched = false;
      return;
    }
    const t0 = performance.now();
    const go = (now) => {
      if (state.touched && performance.now() - t0 > 50) return;
      const p = Math.min((now - t0) / 2400, 1);
      const e2 = 1 - Math.pow(1 - p, 3);
      state.lambda = target * e2;
      lam.value = Math.round((state.lambda / LMAX) * 1000);
      setFill(lam);
      state.shown = state.lambda;
      updateReadout();
      if (p < 1) requestAnimationFrame(go);
    };
    setTimeout(() => requestAnimationFrame(go), 600);
  }, { threshold: 0.45 });
  io.observe(canvas);

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.fov = w / h < 0.9 ? 52 : 40;
    home.radius = w / h < 0.9 ? 17 : 13.8;
    camera.updateProjectionMatrix();
  }
  resize();
  new ResizeObserver(resize).observe(canvas);

  let running = false;
  whenVisible(canvas, (v) => (running = v));
  const clock = new THREE.Clock();
  const tmpA = new THREE.Vector3(), tmpB = new THREE.Vector3();

  function frame() {
    requestAnimationFrame(frame);
    if (!running || !canvas.clientWidth || !canvas.clientHeight) return;
    const dt = Math.min(clock.getDelta(), 0.05);

    if (morph < 1) {
      morph = Math.min(morph + dt / 0.85, 1);
      const e = morph < 0.5 ? 4 * morph ** 3 : 1 - Math.pow(-2 * morph + 2, 3) / 2;
      for (let k = 0; k < N * 3; k++) {
        base.it[k] = from.it[k] + (to.it[k] - from.it[k]) * e;
        base.dpo[k] = from.dpo[k] + (to.dpo[k] - from.dpo[k]) * e;
      }
    }
    state.shown += (state.lambda - state.shown) * Math.min(dt * 9, 1);

    const install = state.dir === "install";
    const src = install ? base.it : base.dpo;
    const shift = (install ? 1 : -1) * state.shown * WORLD;
    pos.set(base.it, 0);
    pos.set(base.dpo, N * 3);
    for (let i = 0; i < N; i++) {
      const o = (2 * N + i) * 3;
      pos[o] = src[i * 3] + shift;
      pos[o + 1] = src[i * 3 + 1];
      pos[o + 2] = src[i * 3 + 2];
    }
    // the source cloud dims as its steered copy pulls away
    const away = Math.min(state.shown / Math.max(P.meta.layers[state.layer].r, 1e-3), 1);
    const steerA = state.shown > 0.002 ? 0.95 : 0;
    for (let i = 0; i < N; i++) {
      alpha[i] = install ? 1 - 0.55 * away : 0.75;
      alpha[N + i] = install ? 0.75 : 1 - 0.55 * away;
      alpha[2 * N + i] = steerA;
    }
    geo.attributes.position.needsUpdate = true;
    geo.attributes.aAlpha.needsUpdate = true;

    for (let k = 0; k < LINKS; k++) {
      const i = linkIdx[k];
      for (let a = 0; a < 3; a++) {
        linkPos[k * 6 + a] = base.it[i * 3 + a];
        linkPos[k * 6 + 3 + a] = base.dpo[i * 3 + a];
      }
    }
    linkGeo.attributes.position.needsUpdate = true;
    linkMat.opacity += ((state.links ? 0.13 : 0) - linkMat.opacity) * 0.15;
    links.visible = linkMat.opacity > 0.005;

    centroid.it.copy(mean(base.it));
    centroid.dpo.copy(mean(base.dpo));
    mIT.position.copy(centroid.it);
    mDPO.position.copy(centroid.dpo);
    tmpA.copy(install ? centroid.it : centroid.dpo);
    tmpB.copy(tmpA).setX(tmpA.x + shift);
    mSteer.position.copy(tmpB);
    mSteer.visible = ring.visible = state.shown > 0.002;
    ring.position.copy(tmpB);
    ring.lookAt(camera.position);
    setArrow(steerArrow, tmpA.clone().setY(tmpA.y + 2.2), tmpB.clone().setY(tmpB.y + 2.2));
    dIT.position.x = centroid.it.x;
    dDPO.position.x = centroid.dpo.x;
    dSteer.position.x = tmpB.x;
    dSteer.visible = mSteer.visible;

    if (idle) {
      camera.position.setFromSphericalCoords(home.radius, home.phi, home.theta + Math.sin(clock.elapsedTime * 0.22) * 0.38);
      camera.lookAt(controls.target);
    }
    controls.update();
    renderer.render(scene, camera);
    drawRidge();
  }
  frame();
}
