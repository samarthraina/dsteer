import { el, text, glowFilter } from "./svg.js";
import { loadPair, reducedMotion } from "./data.js";

// Four-step method diagram, driven by which step is in view.
const TOWER = { it: 190, dpo: 410, width: 112, top: 150, bottom: 532 };
const LAYERS = 32;
const barY = (l) => TOWER.bottom - (l + 1) * ((TOWER.bottom - TOWER.top) / LAYERS);
const barH = (TOWER.bottom - TOWER.top) / LAYERS - 3;

export async function initMethod(svg, steps) {
  const gsap = window.gsap;
  const defs = el("defs", {}, svg);
  glowFilter(defs, "m-glow", 5);
  glowFilter(defs, "m-glow-s", 2);
  const grad = el("linearGradient", { id: "m-disp", x1: "0", x2: "1" }, defs);
  el("stop", { offset: "0", "stop-color": "#5b8dff" }, grad);
  el("stop", { offset: "1", "stop-color": "#2fe3b4" }, grad);

  // background grid dots
  const bg = el("g", { opacity: 0.35 }, svg);
  for (let x = 20; x < 600; x += 30) for (let y = 20; y < 600; y += 30) el("circle", { cx: x, cy: y, r: 0.9, fill: "#3a4560" }, bg);

  // ── towers (steps 1, 2, 4) ──
  const towers = el("g", {}, svg);
  const bars = { it: [], dpo: [] };
  for (const key of ["it", "dpo"]) {
    const cx = TOWER[key];
    const color = key === "it" ? "#5b8dff" : "#2fe3b4";
    const g = el("g", {}, towers);
    for (let l = 0; l < LAYERS; l++) {
      bars[key].push(el("rect", {
        x: cx - TOWER.width / 2, y: barY(l), width: TOWER.width, height: barH, rx: 2.5,
        fill: color, "fill-opacity": 0.1 + 0.18 * (l / LAYERS), stroke: color, "stroke-opacity": 0.35,
      }, g));
    }
    text(g, cx, 562, key === "it" ? "M_IT" : "M_DPO", { "text-anchor": "middle", fill: color, "font-family": "Space Grotesk", "font-weight": 700, "font-size": 20 });
    text(g, cx, 582, key === "it" ? "instruction-tuned" : "harmlessness DPO", { "text-anchor": "middle", fill: "#7a8399", "font-size": 12.5 });
  }
  const layerLbl = el("g", {}, towers);
  for (const l of [0, 15, 31]) text(layerLbl, 300, barY(l) + barH - 1, `L${l}`, { "text-anchor": "middle", fill: "#5f6880", "font-size": 11, "font-family": "JetBrains Mono" });

  // step 1: shared prompt
  const prompt = el("g", {}, svg);
  el("rect", { x: 170, y: 46, width: 260, height: 46, rx: 23, fill: "rgba(255,255,255,0.05)", stroke: "rgba(255,255,255,0.22)" }, prompt);
  text(prompt, 300, 75, "same prompt · same tokens", { "text-anchor": "middle", fill: "#eef1f7", "font-size": 15, "font-family": "JetBrains Mono" });
  const feed = (x) => el("path", { d: `M 300 92 C 300 120, ${x} 112, ${x} ${TOWER.top - 6}`, fill: "none", stroke: "rgba(255,255,255,0.35)", "stroke-width": 1.5, "stroke-dasharray": "4 5" }, prompt);
  const feeds = [feed(TOWER.it), feed(TOWER.dpo)];

  // step 2: tokens, last one highlighted, read out at every layer
  const tokens = el("g", {}, svg);
  const toks = ["user:", "how", "do", "I", "…", "?", "assistant:"];
  const widths = toks.map((t) => t.length * 8.6 + 22);
  const rowW = widths.reduce((a, b) => a + b, 0) + 6 * (toks.length - 1);
  let tx = (600 - rowW) / 2;
  const tokEls = [];
  let lastMid = 0;
  toks.forEach((t, i) => {
    const w = widths[i];
    const last = i === toks.length - 1;
    // position on the outer group; GSAP animates the inner one, so its y never clobbers the layout
    const g = el("g", {}, el("g", { transform: `translate(${tx} 54)` }, tokens));
    el("rect", { width: w, height: 34, rx: 9, fill: last ? "rgba(255,214,102,0.16)" : "rgba(255,255,255,0.05)", stroke: last ? "#ffd666" : "rgba(255,255,255,0.18)" }, g);
    text(g, w / 2, 22, t, { "text-anchor": "middle", fill: last ? "#ffe6a3" : "#b4bccd", "font-size": 13.5, "font-family": "JetBrains Mono" });
    tokEls.push(g);
    if (last) lastMid = tx + w / 2;
    tx += w + 6;
  });
  text(tokens, lastMid, 40, "final prompt position", { "text-anchor": "middle", fill: "#ffd666", "font-size": 12.5 });
  const beams = [TOWER.it, TOWER.dpo].map((x) =>
    el("path", { d: `M ${lastMid} 90 C ${lastMid} 122, ${x + TOWER.width / 2 - 10} 112, ${x + TOWER.width / 2 - 10} ${TOWER.top - 4}`, fill: "none", stroke: "#ffd666", "stroke-width": 1.6, opacity: 0.8 }, tokens));
  const reads = el("g", {}, tokens);
  for (const key of ["it", "dpo"]) {
    for (let l = 0; l < LAYERS; l++) {
      el("circle", { cx: TOWER[key] + TOWER.width / 2 - 10, cy: barY(l) + barH / 2, r: 2.6, fill: "#ffd666", class: `rd rd-${l}` }, reads);
    }
  }
  text(tokens, (600 - rowW) / 2, 116, "h_ℓ(x) read at all 32 layers", { "text-anchor": "start", fill: "#ffe6a3", "font-size": 12.5, "font-family": "JetBrains Mono" });

  // step 3: real per-prompt displacements at layer 31 (Mistral), and their mean
  const disp = el("g", {}, svg);
  const O = { x: 150, y: 320 };
  el("line", { x1: 40, y1: O.y, x2: 570, y2: O.y, stroke: "rgba(255,255,255,0.14)" }, disp);
  el("line", { x1: O.x, y1: 120, x2: O.x, y2: 520, stroke: "rgba(255,255,255,0.08)" }, disp);
  text(disp, 568, O.y + 22, "v̂ (IT → DPO)", { "text-anchor": "end", fill: "#7a8399", "font-size": 12, "font-family": "JetBrains Mono" });
  text(disp, O.x + 8, 132, "PC2", { fill: "#7a8399", "font-size": 12, "font-family": "JetBrains Mono" });
  const arrows = el("g", {}, disp);
  const meanG = el("g", {}, disp);
  const P = await loadPair("mistral");
  const N = P.N;
  const r = P.meta.layers[31].r;
  const k = 300 / r;
  const o0 = (31 * 2) * N * 3, o1 = (31 * 2 + 1) * N * 3;
  const shown = 90;
  const lines = [];
  for (let j = 0; j < shown; j++) {
    const i = Math.floor((j * N) / shown);
    const dx = (P.data[o1 + i * 3] - P.data[o0 + i * 3]) * k;
    const dy = (P.data[o1 + i * 3 + 1] - P.data[o0 + i * 3 + 1]) * k;
    const len = Math.hypot(dx, dy);
    const s = len > 380 ? 380 / len : 1;
    lines.push(el("line", { x1: O.x, y1: O.y, x2: O.x + dx * s, y2: O.y - dy * s, stroke: "url(#m-disp)", "stroke-width": 1.2, opacity: 0.45, "data-x": O.x + dx * s, "data-y": O.y - dy * s }, arrows));
  }
  const meanLine = el("line", { x1: O.x, y1: O.y, x2: O.x + 300, y2: O.y, stroke: "#2fe3b4", "stroke-width": 6, "stroke-linecap": "round", filter: "url(#m-glow)" }, meanG);
  const meanHead = el("path", { d: "M -2 -13 L 20 0 L -2 13 Z", fill: "#2fe3b4", transform: `translate(${O.x + 300} ${O.y})`, filter: "url(#m-glow-s)" }, meanG);
  text(meanG, O.x + 150, O.y - 18, "d̄₃₁ = mean displacement", { "text-anchor": "middle", fill: "#bff7e6", "font-size": 14, "font-family": "JetBrains Mono" });
  text(disp, 300, 560, `${shown} of 1,378 prompts · Mistral pair · layer 31`, { "text-anchor": "middle", fill: "#5f6880", "font-size": 12 });

  // step 4: inject at 27–31
  const inject = el("g", {}, svg);
  const steerBand = [];
  for (const key of ["it", "dpo"]) {
    const color = key === "it" ? "#2fe3b4" : "#ff6b3d";
    for (let l = 27; l < 32; l++) {
      const y = barY(l) + barH / 2;
      const x0 = key === "it" ? TOWER.it - TOWER.width / 2 - 60 : TOWER.dpo + TOWER.width / 2 + 60;
      const x1 = key === "it" ? TOWER.it - TOWER.width / 2 - 6 : TOWER.dpo + TOWER.width / 2 + 6;
      const g = el("g", {}, inject);
      el("line", { x1: x0, y1: y, x2: x1, y2: y, stroke: color, "stroke-width": 2.2, filter: "url(#m-glow-s)" }, g);
      el("path", { d: key === "it" ? "M 0 -5 L 8 0 L 0 5 Z" : "M 0 -5 L -8 0 L 0 5 Z", fill: color, transform: `translate(${x1 - (key === "it" ? 7 : -7)} ${y})` }, g);
      steerBand.push(g);
    }
    const lx = key === "it" ? TOWER.it - TOWER.width / 2 - 64 : TOWER.dpo + TOWER.width / 2 + 64;
    text(inject, lx, barY(31) - 18, key === "it" ? "+λv" : "−λv", { "text-anchor": key === "it" ? "start" : "end", fill: color, "font-size": 18, "font-family": "JetBrains Mono", "font-weight": 500 });
    text(inject, key === "it" ? TOWER.it : TOWER.dpo, 116, key === "it" ? "install" : "remove", { "text-anchor": "middle", fill: color, "font-family": "Space Grotesk", "font-weight": 600, "font-size": 22 });
  }
  const flow = [TOWER.it, TOWER.dpo].map((x) =>
    el("line", { x1: x, y1: TOWER.bottom, x2: x, y2: TOWER.top - 8, stroke: "rgba(255,255,255,0.55)", "stroke-width": 1.4, "stroke-dasharray": "3 9" }, inject));
  text(inject, 300, barY(29) + 4, "27–31", { "text-anchor": "middle", fill: "#c9b8ff", "font-size": 12, "font-family": "JetBrains Mono" });

  const groups = [prompt, tokens, disp, inject];
  const sets = [[towers, prompt], [towers, tokens], [disp], [towers, inject]];
  const all = [towers, ...groups];
  all.forEach((g) => g.setAttribute("opacity", 0));

  let current = -1;
  function show(i) {
    if (i === current) return;
    current = i;
    const on = sets[i];
    all.forEach((g) => {
      if (!on.includes(g)) gsap.to(g, { opacity: 0, duration: 0.45, ease: "power2.out" });
    });
    on.forEach((g) => gsap.to(g, { opacity: 1, duration: 0.6, delay: 0.15, ease: "power2.out" }));

    // bar styling per step
    bars.it.concat(bars.dpo).forEach((b, idx) => {
      const l = idx % LAYERS;
      const hot = i === 3 && l >= 27;
      gsap.to(b, { attr: { "fill-opacity": hot ? 0.75 : 0.1 + 0.18 * (l / LAYERS), "stroke-opacity": hot ? 1 : 0.35 }, duration: 0.5 });
      b.setAttribute("stroke", hot ? "#a48bff" : idx < LAYERS ? "#5b8dff" : "#2fe3b4");
      b.setAttribute("fill", hot ? "#a48bff" : idx < LAYERS ? "#5b8dff" : "#2fe3b4");
    });
    if (reducedMotion) return;

    if (i === 0) {
      feeds.forEach((f) => gsap.fromTo(f, { attr: { "stroke-dashoffset": 60 } }, { attr: { "stroke-dashoffset": 0 }, duration: 1.2, repeat: -1, ease: "none" }));
      gsap.fromTo([...bars.it, ...bars.dpo], { scaleX: 0.2, transformOrigin: "50% 50%" }, { scaleX: 1, duration: 0.7, stagger: { each: 0.012, from: "start" }, ease: "power3.out" });
    }
    if (i === 1) {
      gsap.fromTo(tokEls, { y: -14, opacity: 0 }, { y: 0, opacity: 1, stagger: 0.06, duration: 0.5, ease: "back.out(2)" });
      beams.forEach((b) => {
        const len = b.getTotalLength();
        gsap.fromTo(b, { attr: { "stroke-dasharray": len, "stroke-dashoffset": len } }, { attr: { "stroke-dashoffset": 0 }, duration: 0.9, delay: 0.4, ease: "power2.inOut" });
      });
      const dots = reads.querySelectorAll(".rd");
      gsap.fromTo(dots, { opacity: 0, scale: 0, transformOrigin: "50% 50%" }, { opacity: 1, scale: 1, duration: 0.3, delay: 0.9, stagger: { each: 0.018, from: "end", grid: [2, LAYERS] } });
    }
    if (i === 2) {
      lines.forEach((ln, j) => {
        gsap.fromTo(ln, { attr: { x2: O.x, y2: O.y }, opacity: 0.8 }, { attr: { x2: +ln.dataset.x, y2: +ln.dataset.y }, opacity: 0.4, duration: 0.9, delay: 0.2 + j * 0.008, ease: "power3.out" });
      });
      gsap.fromTo(meanLine, { attr: { x2: O.x } }, { attr: { x2: O.x + 300 }, duration: 1.1, delay: 1.3, ease: "power3.inOut" });
      gsap.fromTo(meanHead, { opacity: 0 }, { opacity: 1, duration: 0.3, delay: 2.2 });
      gsap.to(arrows, { opacity: 0.45, duration: 0.8, delay: 2.2 });
    } else gsap.set(arrows, { opacity: 1 });
    if (i === 3) {
      gsap.fromTo(steerBand, { opacity: 0, x: (j) => (j < 5 ? -20 : 20) }, { opacity: 1, x: 0, duration: 0.5, stagger: 0.07, delay: 0.3, ease: "power2.out" });
      flow.forEach((f) => gsap.fromTo(f, { attr: { "stroke-dashoffset": 0 } }, { attr: { "stroke-dashoffset": -120 }, duration: 2.4, repeat: -1, ease: "none" }));
    }
  }

  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        steps.forEach((s) => s.classList.toggle("active", s === e.target));
        show(+e.target.dataset.step);
      }
    });
  }, { rootMargin: "-45% 0px -45% 0px" });
  steps.forEach((s) => io.observe(s));
  show(0);
  steps[0].classList.add("active");
}
