import { el, text, scale, niceTicks, fmt, fmtCI, bindTip, showTip, hideTip, glowFilter } from "./svg.js";
import { reducedMotion } from "./data.js";

const C = { install: "#1aa77d", remove: "#e0602d", it: "#3f86ea", dpo: "#1aa77d", rand: "#7d8597", gate: "#f0506e", violet: "#9085e9", ink: "#eef1f7", ink2: "#b4bccd", muted: "#7a8399" };
const CELLS = [
  { key: "mistral_install", pair: "Mistral", dir: "install" },
  { key: "mistral_remove", pair: "Mistral", dir: "remove" },
  { key: "llama_install", pair: "Llama", dir: "install" },
  { key: "llama_remove", pair: "Llama", dir: "remove" },
];
const DIRWORD = { install: "installation", remove: "removal" };

// Charts are laid out at their container's real pixel width so type stays at its CSS size
// on phones. They animate on first view only; later re-renders (resize) draw settled.
let animOK = !reducedMotion;
export function settle() {
  animOK = false;
}
const widthOf = (node, min = 300) => Math.max(min, Math.round(node.clientWidth || node.getBoundingClientRect().width));

function onView(node, fn) {
  const io = new IntersectionObserver(([e]) => {
    if (e.isIntersecting) {
      io.disconnect();
      fn();
    }
  }, { threshold: 0.2 });
  io.observe(node);
}

function animate(svg) {
  const gsap = window.gsap;
  if (!gsap || !animOK) return;
  const ci = svg.querySelectorAll("[data-ci]");
  const pops = svg.querySelectorAll("[data-pop]");
  const draws = svg.querySelectorAll("[data-draw]");
  const fades = svg.querySelectorAll("[data-fade]");
  if (pops.length) gsap.set(pops, { opacity: 0 });
  if (fades.length) gsap.set(fades, { opacity: 0 });
  ci.forEach((l) => gsap.set(l, { attr: { x1: +l.dataset.ci, x2: +l.dataset.ci } }));
  draws.forEach((p) => {
    const len = p.getTotalLength();
    gsap.set(p, { attr: { "stroke-dasharray": len, "stroke-dashoffset": len } });
  });
  onView(svg, () => {
    ci.forEach((l, i) => gsap.to(l, { attr: { x1: +l.dataset.x1, x2: +l.dataset.x2 }, duration: 0.9, delay: 0.15 + i * 0.05, ease: "power3.out" }));
    if (pops.length) gsap.fromTo(pops, { opacity: 0, scale: 0, transformOrigin: "50% 50%" }, { opacity: 1, scale: 1, duration: 0.55, stagger: 0.03, delay: 0.1, ease: "back.out(2.2)" });
    draws.forEach((p, i) => gsap.to(p, { attr: { "stroke-dashoffset": 0 }, duration: 1.4, delay: 0.1 + i * 0.1, ease: "power2.inOut", onComplete: () => p.removeAttribute("stroke-dasharray") }));
    if (fades.length) gsap.to(fades, { opacity: 1, duration: 0.6, delay: 0.7, stagger: 0.03 });
  });
}

function svgRoot(container, w, h, label) {
  container.innerHTML = "";
  return el("svg", { viewBox: `0 0 ${w} ${h}`, width: w, height: h, role: "img", "aria-label": label }, container);
}

function xAxis(svg, x, y0, y1, ticks, f = (v) => v, label) {
  const g = el("g", { class: "axis" }, svg);
  const grid = el("g", { class: "grid" }, svg);
  ticks.forEach((v) => {
    el("line", { x1: x(v), x2: x(v), y1: y0, y2: y1 }, grid);
    text(g, x(v), y1 + 18, f(v), { "text-anchor": "middle" });
  });
  if (label) text(g, (x(x.domain[0]) + x(x.domain[1])) / 2, y1 + 38, label, { "text-anchor": "middle", class: "lbl-muted" });
}

function yAxis(svg, y, x0, x1, ticks, f = (v) => v) {
  const g = el("g", { class: "axis" }, svg);
  const grid = el("g", { class: "grid" }, svg);
  ticks.forEach((v) => {
    el("line", { x1: x0, x2: x1, y1: y(v), y2: y(v) }, grid);
    text(g, x0 - 8, y(v) + 4, f(v), { "text-anchor": "end" });
  });
}

// install = circle, remove = diamond, so direction never rides on colour alone
function marker(parent, dir, x, y, r, fill, stroke) {
  if (dir === "remove") {
    const s = r * 1.15;
    return el("rect", { x: x - s, y: y - s, width: s * 2, height: s * 2, fill, stroke, "stroke-width": 2, transform: `rotate(45 ${x} ${y})` }, parent);
  }
  return el("circle", { cx: x, cy: y, r, fill, stroke, "stroke-width": 2 }, parent);
}

function ciLine(parent, x, lo, hi, y, color, width = 2.5) {
  const x1 = x(lo), x2 = x(hi);
  return el("line", { x1, x2, y1: y, y2: y, stroke: color, "stroke-width": width, "stroke-linecap": "round", "data-ci": (x1 + x2) / 2, "data-x1": x1, "data-x2": x2 }, parent);
}

function hit(parent, x, y, r = 15) {
  return el("circle", { cx: x, cy: y, r, fill: "transparent" }, parent);
}

// ═══════════ depth: relative displacement + coherence by layer ═══════════
export function depthCharts(res, pair, onHover) {
  const rows = res.geometry[pair];
  const draw = (id, key, ciKey, h, yMax, yFmt, label) => {
    const c = document.getElementById(id);
    const W = widthOf(c), m = { l: 44, r: 12, t: 10, b: 30 };
    const svg = svgRoot(c, W, h, label);
    const x = scale(0, 31, m.l, W - m.r);
    const y = scale(0, yMax, h - m.b, m.t);
    el("rect", { x: x(26.5), y: m.t, width: x(31) - x(26.5) + 6, height: h - m.b - m.t, fill: "rgba(164,139,255,0.10)", stroke: "rgba(164,139,255,0.3)", "stroke-dasharray": "3 4" }, svg);
    yAxis(svg, y, m.l, W - m.r, niceTicks(0, yMax, 4), yFmt);
    xAxis(svg, x, m.t, h - m.b, W < 420 ? [0, 8, 16, 24, 31] : [0, 4, 8, 12, 16, 20, 24, 28, 31]);
    const band = rows.map((r, i) => `${i ? "L" : "M"} ${x(r.layer)} ${y(r[ciKey][1])}`).join(" ") + " " +
      rows.slice().reverse().map((r) => `L ${x(r.layer)} ${y(r[ciKey][0])}`).join(" ") + " Z";
    el("path", { d: band, fill: "rgba(164,139,255,0.22)", "data-fade": 1 }, svg);
    el("path", { d: rows.map((r, i) => `${i ? "L" : "M"} ${x(r.layer)} ${y(r[key])}`).join(" "), fill: "none", stroke: C.violet, "stroke-width": 2.2, "stroke-linejoin": "round", "data-draw": 1 }, svg);
    const cross = el("line", { y1: m.t, y2: h - m.b, stroke: "rgba(255,255,255,0.4)", "stroke-dasharray": "2 3", opacity: 0 }, svg);
    const dot = el("circle", { r: 5, fill: C.violet, stroke: "#04060b", "stroke-width": 2, opacity: 0 }, svg);
    const pad = el("rect", { x: m.l, y: 0, width: W - m.l - m.r, height: h, fill: "transparent", style: "touch-action: pan-y" }, svg);
    const at = (e) => {
      const b = svg.getBoundingClientRect();
      return Math.max(0, Math.min(31, Math.round(x.invert(((e.clientX - b.left) / b.width) * W))));
    };
    const set = (l, e) => {
      const r = rows[l];
      cross.setAttribute("x1", x(l)); cross.setAttribute("x2", x(l)); cross.setAttribute("opacity", 1);
      dot.setAttribute("cx", x(l)); dot.setAttribute("cy", y(r[key])); dot.setAttribute("opacity", 1);
      showTip(`<b>Layer ${l}</b>${l >= 27 ? ' <span style="color:#c9b8ff">· steered</span>' : ""}<div class="k"><span>${label}</span><span>${r[key].toFixed(3)}</span></div><div class="k"><span>95% CI</span><span>${r[ciKey][0].toFixed(3)}–${r[ciKey][1].toFixed(3)}</span></div>`, e.clientX, e.clientY);
    };
    pad.addEventListener("pointermove", (e) => { const l = at(e); set(l, e); onHover(l); });
    pad.addEventListener("pointerdown", (e) => { const l = at(e); set(l, e); onHover(l); });
    pad.addEventListener("pointerleave", () => { hideTip(); cross.setAttribute("opacity", 0); dot.setAttribute("opacity", 0); });
    animate(svg);
  };
  const maxR = Math.max(...rows.map((r) => r.r_ci[1]));
  draw("depth-chart", "r", "r_ci", 190, Math.ceil(maxR * 20) / 20, (v) => v.toFixed(2), "relative displacement");
  draw("depth-coh", "coherence", "coherence_ci", 150, 1, (v) => v.toFixed(2), "coherence");
}

// ═══════════ operating window ═══════════
export function windowChart(res, key) {
  const pts = res.dose[key];
  const dir = key.endsWith("install") ? "install" : "remove";
  const color = C[dir];
  // ceiling: last coefficient reached by an unbroken run of usable ones
  let ceil = 0;
  for (const p of pts) { if (!p.usable) break; ceil = p.c; }
  const sel = pts.find((p) => p.c === ceil);
  const fail = pts.find((p) => p.c > ceil);
  const why = !fail ? "" : [fail.invalid > res.gate.invalid && `invalid ${(fail.invalid * 100).toFixed(1)}%`, fail.maxlen > res.gate.maxlen && `max-length ${(fail.maxlen * 100).toFixed(1)}%`, fail.quality_loss > res.gate.quality_loss && `quality loss ${fail.quality_loss.toFixed(3)}`].filter(Boolean).join(", ");
  const [pair] = key.split("_");
  document.getElementById("window-sub").innerHTML =
    `${pair === "mistral" ? "Mistral" : "Llama"} ${DIRWORD[dir]} · ceiling <span class="mono">C = ${ceil.toFixed(3)}</span> · frozen <span class="mono">λ* = ${dir === "install" ? "+" : "−"}${ceil.toFixed(3)}</span> · the next step fails on ${why || "–"}`;

  const xMax = Math.max(...pts.map((p) => p.c)) * 1.04;
  const harmBox = document.getElementById("window-harm");
  const W = widthOf(harmBox);
  // harm panel
  {
    const h = W < 500 ? 240 : 280, m = { l: 46, r: 14, t: 16, b: 34 };
    const svg = svgRoot(harmBox, W, h, "Harmful-enablement movement against coefficient on the calibration panel");
    const defs = el("defs", {}, svg);
    glowFilter(defs, "w-glow", 4);
    const pat = el("pattern", { id: "w-hatch", width: 8, height: 8, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, defs);
    el("line", { x1: 0, y1: 0, x2: 0, y2: 8, stroke: "rgba(240,80,110,0.16)", "stroke-width": 3 }, pat);
    const yMax = Math.max(0.35, Math.ceil(Math.max(...pts.map((p) => p.harm)) * 10) / 10 + 0.02);
    const x = scale(0, xMax, m.l, W - m.r);
    const y = scale(-0.02, yMax, h - m.b, m.t);
    el("rect", { x: x(0), y: m.t, width: x(ceil) - x(0), height: h - m.b - m.t, fill: dir === "install" ? "rgba(26,167,125,0.10)" : "rgba(224,96,45,0.10)", "data-fade": 1 }, svg);
    el("rect", { x: x(ceil), y: m.t, width: x(xMax) - x(ceil), height: h - m.b - m.t, fill: "url(#w-hatch)", "data-fade": 1 }, svg);
    yAxis(svg, y, m.l, W - m.r, niceTicks(0, yMax, 5), (v) => v.toFixed(2));
    xAxis(svg, x, m.t, h - m.b, niceTicks(0, xMax, W < 500 ? 4 : 6), (v) => v.toFixed(2));
    el("line", { x1: m.l, x2: W - m.r, y1: y(0), y2: y(0), class: "zero" }, svg);
    text(svg, x(ceil / 2), m.t + 18, "operating window", { "text-anchor": "middle", class: "lbl", "data-fade": 1 });
    if (x(xMax) - x(ceil) > 70) text(svg, (x(ceil) + x(xMax)) / 2, m.t + 18, "fails the gate", { "text-anchor": "middle", fill: "#ffb3c1", "font-size": 12.5, "data-fade": 1 });
    const inside = pts.filter((p) => p.c <= ceil);
    el("path", { d: pts.map((p, i) => `${i ? "L" : "M"} ${x(p.c)} ${y(p.harm)}`).join(" "), fill: "none", stroke: "rgba(255,255,255,0.25)", "stroke-width": 1.5, "data-draw": 1 }, svg);
    el("path", { d: inside.map((p, i) => `${i ? "L" : "M"} ${x(p.c)} ${y(p.harm)}`).join(" "), fill: "none", stroke: color, "stroke-width": 2.5, filter: "url(#w-glow)", "data-draw": 1 }, svg);
    pts.forEach((p) => {
      const ok = p.c <= ceil;
      const g = el("g", { "data-pop": 1 }, svg);
      marker(g, dir, x(p.c), y(p.harm), 5, ok ? color : "#04060b", ok ? "#04060b" : C.gate);
      hit(g, x(p.c), y(p.harm), 13);
      bindTip(g, `<b>c = ${p.c.toFixed(3)}</b> ${ok ? '<span style="color:#7ff0c9">usable</span>' : '<span style="color:#ffb3c1">outside window</span>'}<div class="k"><span>harm movement</span><span>${fmt(p.harm)}</span></div><div class="k"><span>gap recovery</span><span>${p.gap_recovery.toFixed(2)}</span></div><div class="k"><span>invalid</span><span>${(p.invalid * 100).toFixed(1)}%</span></div><div class="k"><span>max length</span><span>${(p.maxlen * 100).toFixed(1)}%</span></div><div class="k"><span>quality loss</span><span>${p.quality_loss.toFixed(3)}</span></div><div class="k"><span>paired n</span><span>${p.n}</span></div>`);
    });
    const star = el("g", { "data-fade": 1 }, svg);
    el("line", { x1: x(ceil), x2: x(ceil), y1: m.t, y2: h - m.b, stroke: "#fff", "stroke-width": 1.2, "stroke-dasharray": "4 4", opacity: 0.6 }, star);
    el("circle", { cx: x(sel.c), cy: y(sel.harm), r: 11, fill: "none", stroke: "#fff", "stroke-width": 1.5 }, star);
    text(star, x(ceil) - 10, y(sel.harm) - 16, `at λ*: ${fmt(sel.harm)}`, { "text-anchor": "end", class: "val" });
    animate(svg);
  }
  // gate panels: three small multiples sharing the coefficient axis (stacked on narrow screens)
  {
    const box = document.getElementById("window-gates");
    const cols = W < 560 ? 1 : 3;
    const gap = 16, pw = (W - gap * (cols - 1)) / cols, ph = cols === 1 ? 118 : 150, m = { l: 42, r: 10, t: 26, b: 30 };
    const H = ph * Math.ceil(3 / cols);
    const svg = svgRoot(box, W, H, "Integrity and quality against coefficient, with gate thresholds");
    const panels = [
      { k: "invalid", t: "invalid rate", lim: res.gate.invalid, f: (v) => `${Math.round(v * 100)}%` },
      { k: "maxlen", t: "max-length rate", lim: res.gate.maxlen, f: (v) => `${Math.round(v * 100)}%` },
      { k: "quality_loss", t: "quality loss", lim: res.gate.quality_loss, f: (v) => v.toFixed(2) },
    ];
    panels.forEach((pn, j) => {
      const g = el("g", { transform: `translate(${(j % cols) * (pw + gap)} ${Math.floor(j / cols) * ph})` }, svg);
      const vmax = Math.max(pn.lim * 2, ...pts.map((p) => p[pn.k])) * 1.08;
      const x = scale(0, xMax, m.l, pw - m.r);
      const y = scale(Math.min(0, ...pts.map((p) => p[pn.k])), vmax, ph - m.b, m.t);
      text(g, m.l, 14, pn.t, { class: "lbl" });
      el("rect", { x: x(0), y: m.t, width: x(ceil) - x(0), height: ph - m.b - m.t, fill: "rgba(255,255,255,0.03)" }, g);
      yAxis(g, y, m.l, pw - m.r, niceTicks(0, vmax, 3), pn.f);
      const ax = el("g", { class: "axis" }, g);
      niceTicks(0, xMax, cols === 1 ? 5 : 3).forEach((v) => text(ax, x(v), ph - m.b + 16, v.toFixed(2), { "text-anchor": "middle" }));
      el("line", { x1: m.l, x2: pw - m.r, y1: y(pn.lim), y2: y(pn.lim), stroke: C.gate, "stroke-width": 1.4, "stroke-dasharray": "5 4" }, g);
      text(g, m.l + 6, y(pn.lim) - 5, `gate ${pn.f(pn.lim)}`, { fill: "#ffb3c1", "font-size": 11, "font-family": "JetBrains Mono" });
      el("path", { d: pts.map((p, i) => `${i ? "L" : "M"} ${x(p.c)} ${y(p[pn.k])}`).join(" "), fill: "none", stroke: "rgba(238,241,247,0.55)", "stroke-width": 1.6, "data-draw": 1 }, g);
      pts.forEach((p) => {
        const bad = p[pn.k] > pn.lim;
        const d = el("g", { "data-pop": 1 }, g);
        el("circle", { cx: x(p.c), cy: y(p[pn.k]), r: bad ? 4.5 : 3.5, fill: bad ? C.gate : "#d9deea", stroke: "#04060b", "stroke-width": 1.5 }, d);
        hit(d, x(p.c), y(p[pn.k]), 10);
        bindTip(d, `<b>c = ${p.c.toFixed(3)}</b><div class="k"><span>${pn.t}</span><span>${pn.k === "quality_loss" ? p[pn.k].toFixed(3) : `${(p[pn.k] * 100).toFixed(1)}%`}</span></div><div class="k"><span>gate</span><span>${bad ? "fails" : "passes"}</span></div>`);
      });
    });
    animate(svg);
  }
  document.getElementById("window-legend").innerHTML = `
    <span><svg viewBox="0 0 26 14"><rect x="1" y="2" width="24" height="10" rx="2" fill="${dir === "install" ? "rgba(26,167,125,0.25)" : "rgba(224,96,45,0.25)"}"/></svg>operating window</span>
    <span><svg viewBox="0 0 26 14"><circle cx="13" cy="7" r="5" fill="#04060b" stroke="${C.gate}" stroke-width="2"/></svg>coefficient outside the window</span>
    <span><svg viewBox="0 0 26 14"><line x1="1" y1="7" x2="25" y2="7" stroke="${C.gate}" stroke-width="1.6" stroke-dasharray="5 4"/></svg>gate threshold</span>
    <span style="color:var(--muted)">calibration panel · 200 prompts · paired by source ID</span>`;
}

// ═══════════ endpoint tracks ═══════════
export function tracks(res, container) {
  const first = !container.children.length;
  if (first) {
    CELLS.forEach((cell, idx) => {
      const h = res.heldout[cell.key];
      const div = document.createElement("div");
      div.className = `card track-card reveal${idx % 2 ? " d1" : ""}`;
      div.innerHTML = `
        <div class="track-head"><h3>${cell.pair} pair</h3><span class="tag ${cell.dir}">${cell.dir === "install" ? "install +λv" : "remove −λv"} · λ* ${h.lambda > 0 ? "+" : "−"}${Math.abs(h.lambda).toFixed(3)}</span></div>
        <div class="track-big"><span class="num" data-count="${h.harm[0]}" data-dec="3" data-sign="+">+0.000</span><span class="unit">harmful-enablement movement</span></div>
        <div class="track-ci">95% CI ${fmtCI(h.harm)} · n = ${h.n} · gap recovery ${h.gap_recovery[0].toFixed(2)}× ${fmtCI(h.gap_recovery, 2)}</div>
        <div class="track-svg chart"></div>`;
      container.appendChild(div);
    });
  }
  CELLS.forEach((cell, idx) => {
    const pair = cell.key.split("_")[0];
    const IT = res.means[`${pair}_install_baseline`].harm;
    const DPO = res.means[`${pair}_remove_baseline`].harm;
    const steered = res.means[`${cell.key}_real`].harm;
    const src = cell.dir === "install" ? IT : DPO;
    const box = container.children[idx].querySelector(".track-svg");
    const W = widthOf(box, 260), H = 96;
    const svg = svgRoot(box, W, H, `${cell.pair} ${DIRWORD[cell.dir]}: harmful enablement moves from ${src.toFixed(3)} to ${steered.toFixed(3)}`);
    const defs = el("defs", {}, svg);
    glowFilter(defs, `tg${idx}`, 4);
    const x = scale(0, 0.55, 22, W - 22);
    const y = 48;
    const gr = el("linearGradient", { id: `tr${idx}`, x1: 0, x2: 1 }, defs);
    el("stop", { offset: 0, "stop-color": C.dpo, "stop-opacity": 0.7 }, gr);
    el("stop", { offset: 1, "stop-color": C.it, "stop-opacity": 0.7 }, gr);
    el("line", { x1: x(0), x2: x(0.55), y1: y, y2: y, stroke: "rgba(255,255,255,0.1)", "stroke-width": 6, "stroke-linecap": "round" }, svg);
    el("line", { x1: x(DPO), x2: x(IT), y1: y, y2: y, stroke: `url(#tr${idx})`, "stroke-width": 6, "stroke-linecap": "round" }, svg);
    [0, 0.1, 0.2, 0.3, 0.4, 0.5].forEach((v) => text(svg, x(v), H - 2, v.toFixed(1), { "text-anchor": "middle", class: "lbl-muted" }));
    const end = (v, c, lbl, anchor) => {
      el("circle", { cx: x(v), cy: y, r: 7, fill: c, stroke: "#04060b", "stroke-width": 2 }, svg);
      text(svg, x(v), y - 16, `${lbl} ${v.toFixed(3)}`, { "text-anchor": anchor, class: "lbl", "font-size": 12 });
    };
    end(IT, C.it, "IT", "start");
    end(DPO, C.dpo, "DPO", "end");
    const path = el("line", { x1: x(src), x2: x(steered), y1: y, y2: y, stroke: C[cell.dir], "stroke-width": 3, filter: `url(#tg${idx})` }, svg);
    const mk = el("g", { transform: `translate(${x(steered)} ${y})` }, svg);
    el("rect", { x: -7, y: -7, width: 14, height: 14, fill: "#fff", transform: "rotate(45)", filter: `url(#tg${idx})` }, mk);
    hit(mk, 0, 0, 14);
    const lbl = text(svg, x(steered), y + 28, `steered ${steered.toFixed(3)}`, { "text-anchor": "middle", class: "val" });
    bindTip(mk, `<b>${cell.pair} ${DIRWORD[cell.dir]}</b><div class="k"><span>unsteered ${cell.dir === "install" ? "IT" : "DPO"}</span><span>${src.toFixed(3)}</span></div><div class="k"><span>steered</span><span>${steered.toFixed(3)}</span></div><div class="k"><span>${cell.dir === "install" ? "DPO" : "IT"} endpoint</span><span>${(cell.dir === "install" ? DPO : IT).toFixed(3)}</span></div>`);
    if (first && animOK) {
      const gsap = window.gsap;
      mk.setAttribute("transform", `translate(${x(src)} ${y})`);
      path.setAttribute("x2", x(src));
      lbl.setAttribute("opacity", 0);
      onView(svg, () => {
        const o = { v: src };
        gsap.to(o, {
          v: steered, duration: 1.8, delay: 0.3, ease: "power3.inOut",
          onUpdate: () => {
            mk.setAttribute("transform", `translate(${x(o.v)} ${y})`);
            path.setAttribute("x2", x(o.v));
          },
          onComplete: () => gsap.to(lbl, { opacity: 1, duration: 0.4 }),
        });
      });
    }
  });
}

// ═══════════ specificity forest ═══════════
export function forest(res) {
  const box = document.getElementById("forest");
  const W = widthOf(box);
  const narrow = W < 560;
  const rowH = 76, m = { l: narrow ? 96 : 150, r: narrow ? 64 : 176, t: 14, b: 44 };
  const H = m.t + m.b + rowH * CELLS.length;
  const svg = svgRoot(box, W, H, "Real-direction effects with confidence intervals against five random-direction effects for each condition");
  const x = scale(-0.05, 0.4, m.l, W - m.r);
  xAxis(svg, x, m.t, H - m.b, narrow ? [0, 0.2, 0.4] : [-0.05, 0, 0.1, 0.2, 0.3, 0.4], (v) => v.toFixed(2), "harmful-enablement movement, intended direction");
  el("line", { x1: x(0), x2: x(0), y1: m.t, y2: H - m.b, class: "zero" }, svg);
  CELLS.forEach((cell, i) => {
    const s = res.heldout[cell.key].spec;
    const y = m.t + rowH * i + rowH / 2;
    const color = C[cell.dir];
    if (i) el("line", { x1: 0, x2: W, y1: m.t + rowH * i, y2: m.t + rowH * i, stroke: "rgba(255,255,255,0.05)" }, svg);
    text(svg, 0, y - 3, cell.pair, { class: "lbl-strong" });
    text(svg, 0, y + 15, DIRWORD[cell.dir], { class: "lbl", fill: cell.dir === "install" ? "#7ff0c9" : "#ffb59a" });
    Object.entries(s.seeds).forEach(([seed, v], k) => {
      const g = el("g", { "data-pop": 1 }, svg);
      el("circle", { cx: x(v), cy: y + 12 + (k % 2) * 7, r: 4.2, fill: "none", stroke: C.rand, "stroke-width": 1.6 }, g);
      hit(g, x(v), y + 15, 9);
      bindTip(g, `<b>Random direction · seed ${seed}</b><div class="k"><span>effect</span><span>${fmt(v)}</span></div><div class="k"><span>same norm, layers, λ</span><span>yes</span></div>`);
    });
    el("line", { x1: x(s.random_mean), x2: x(s.random_mean), y1: y + 4, y2: y + 26, stroke: C.rand, "stroke-width": 2, "data-fade": 1 }, svg);
    ciLine(svg, x, s.real[1], s.real[2], y - 10, color, 3);
    const g = el("g", { "data-pop": 1 }, svg);
    marker(g, cell.dir, x(s.real[0]), y - 10, 6.5, color, "#04060b");
    hit(g, x(s.real[0]), y - 10, 16);
    bindTip(g, `<b>${cell.pair} ${DIRWORD[cell.dir]} · real direction</b><div class="k"><span>effect</span><span>${fmt(s.real[0])}</span></div><div class="k"><span>95% CI</span><span>${fmtCI(s.real)}</span></div><div class="k"><span>random mean</span><span>${fmt(s.random_mean)}</span></div><div class="k"><span>specificity</span><span>${fmt(s.value[0])}</span></div><div class="k"><span>95% CI</span><span>${fmtCI(s.value)}</span></div><div class="k"><span>n (all arms valid)</span><span>${s.n}</span></div>`);
    const t = el("g", { "data-fade": 1 }, svg);
    text(t, W - m.r + 16, y - 2, fmt(s.value[0]), { class: "lbl-strong", "font-family": "JetBrains Mono", "font-size": narrow ? 14 : 17 });
    text(t, W - m.r + 16, y + 16, narrow ? "spec." : `spec. ${fmtCI(s.value)}`, { class: "lbl-muted" });
  });
  animate(svg);
}

// ═══════════ transfer small multiples ═══════════
export function transfer(res) {
  const sets = [
    { k: "heldout", t: "HarmfulQA (held-out)", get: (c) => res.heldout[c].harm },
    { k: "hh_rlhf", t: "HH-RLHF", get: (c) => res.transfer.hh_rlhf[c].harm },
    { k: "advbench", t: "AdvBench", get: (c) => res.transfer.advbench[c].harm },
  ];
  const box = document.getElementById("transfer");
  const W = widthOf(box);
  const narrow = W < 720;
  const rowH = 40, labW = narrow ? 128 : 150, gapC = 48, top = 30, bottom = 40;
  const panelH = top + bottom + rowH * CELLS.length;
  const colW = narrow ? W - labW : (W - labW - gapC * 2) / 3;
  const H = narrow ? panelH * 3 + 20 : panelH;
  const svg = svgRoot(box, W, H, "Harmful-enablement movement per condition on the held-out, HH-RLHF and AdvBench panels");
  sets.forEach((s, j) => {
    const ox = narrow ? labW : labW + j * (colW + gapC);
    const oy = narrow ? j * (panelH + 10) : 0;
    if (narrow || j === 0) {
      CELLS.forEach((cell, i) => text(svg, 0, oy + top + rowH * i + rowH / 2 + 5, `${cell.pair} ${cell.dir === "install" ? "install" : "remove"}`, { class: "lbl" }));
    }
    const g = el("g", { transform: `translate(${ox} ${oy})` }, svg);
    const x = scale(0, 0.5, 0, colW - 8);
    text(g, 0, 16, s.t, { class: "lbl-strong" });
    xAxis(g, x, top, panelH - bottom, [0, 0.25, 0.5], (v) => v.toFixed(2));
    el("line", { x1: x(0), x2: x(0), y1: top, y2: panelH - bottom, class: "zero" }, g);
    CELLS.forEach((cell, i) => {
      const v = s.get(cell.key);
      const y = top + rowH * i + rowH / 2;
      const color = C[cell.dir];
      ciLine(g, x, v[1], v[2], y, color, 2.5);
      const p = el("g", { "data-pop": 1 }, g);
      marker(p, cell.dir, x(v[0]), y, 5.5, color, "#04060b");
      hit(p, x(v[0]), y, 14);
      const gr = s.k === "heldout" ? res.heldout[cell.key].gap_recovery[0] : res.transfer[s.k][cell.key].gap_recovery;
      bindTip(p, `<b>${cell.pair} ${DIRWORD[cell.dir]} · ${s.t}</b><div class="k"><span>effect</span><span>${fmt(v[0])}</span></div><div class="k"><span>95% CI</span><span>${fmtCI(v)}</span></div><div class="k"><span>gap recovery</span><span>${gr.toFixed(2)}</span></div>`);
      const right = x(v[2]) + 8 > colW - 34;
      text(g, right ? x(v[1]) - 8 : x(v[2]) + 8, y + 4, v[0].toFixed(2), { class: "lbl-muted", "data-fade": 1, "text-anchor": right ? "end" : "start" });
    });
  });
  animate(svg);
}

// ═══════════ raw audit ═══════════
export function rawBars(res) {
  const cells = res.raw_audit.cells;
  const box = document.getElementById("raw-bars");
  const W = widthOf(box);
  const rowH = 40, m = { l: W < 480 ? 128 : 150, r: 58, t: 8, b: 44 };
  const H = m.t + m.b + rowH * cells.length;
  const svg = svgRoot(box, W, H, "Raw-displacement audit effects on the Mistral pair by panel and direction");
  const x = scale(0, 0.6, m.l, W - m.r);
  xAxis(svg, x, m.t, H - m.b, [0, 0.2, 0.4, 0.6], (v) => v.toFixed(1), "harmful-enablement movement");
  el("line", { x1: x(0), x2: x(0), y1: m.t, y2: H - m.b, class: "zero" }, svg);
  cells.forEach((c, i) => {
    const y = m.t + rowH * i + rowH / 2;
    const color = C[c.dir];
    text(svg, 0, y + 4, c.panel, { class: "lbl" });
    text(svg, m.l - 12, y + 4, c.dir, { class: "lbl-muted", "text-anchor": "end", fill: c.dir === "install" ? "#7ff0c9" : "#ffb59a" });
    const bar = el("rect", { x: x(0), y: y - 7, width: x(c.harm[0]) - x(0), height: 14, rx: 4, fill: color, opacity: 0.35 }, svg);
    if (animOK) onView(svg, () => window.gsap.from(bar, { attr: { width: 0 }, duration: 1, delay: 0.1 + i * 0.08, ease: "power3.out" }));
    ciLine(svg, x, c.harm[1], c.harm[2], y, color, 2.2);
    const p = el("g", { "data-pop": 1 }, svg);
    marker(p, c.dir, x(c.harm[0]), y, 5, color, "#04060b");
    hit(p, x(c.harm[0]), y, 14);
    bindTip(p, `<b>${c.panel} · ${c.dir}</b><div class="k"><span>λ_raw</span><span>${c.lambda_raw > 0 ? "+" : ""}${c.lambda_raw}</span></div><div class="k"><span>effect</span><span>${fmt(c.harm[0])}</span></div><div class="k"><span>95% CI</span><span>${fmtCI(c.harm)}</span></div><div class="k"><span>n</span><span>${c.n}</span></div>`);
    text(svg, W - m.r + 10, y + 4, `n=${c.n}`, { class: "lbl-muted" });
  });
  animate(svg);
}

export function rawDepth(res) {
  const d = res.raw_audit.depth_install;
  const box = document.getElementById("raw-depth");
  const W = widthOf(box), H = 250, m = { l: 44, r: 18, t: 24, b: 46 };
  const svg = svgRoot(box, W, H, "Installation effect against number of steered final layers");
  const defs = el("defs", {}, svg);
  glowFilter(defs, "rd-glow", 4);
  const x = scale(0, 11, m.l, W - m.r);
  const y = scale(0, 0.45, H - m.b, m.t);
  yAxis(svg, y, m.l, W - m.r, [0, 0.1, 0.2, 0.3, 0.4], (v) => v.toFixed(1));
  xAxis(svg, x, m.t, H - m.b, [1, 3, 5, 7, 9, 10], (v) => v, "contiguous final layers steered");
  el("path", { d: d.map(([l, v], i) => `${i ? "L" : "M"} ${x(l)} ${y(v)}`).join(" "), fill: "none", stroke: C.install, "stroke-width": 2.5, filter: "url(#rd-glow)", "data-draw": 1 }, svg);
  d.forEach(([l, v], i) => {
    const g = el("g", { "data-pop": 1 }, svg);
    el("circle", { cx: x(l), cy: y(v), r: 5.5, fill: C.install, stroke: "#04060b", "stroke-width": 2 }, g);
    hit(g, x(l), y(v), 14);
    // 9 and 10 sit close together: put the last label underneath
    const below = i === d.length - 1;
    text(svg, x(l), below ? y(v) + 22 : y(v) - 13, v.toFixed(3), { "text-anchor": "middle", class: "val", "data-fade": 1, "font-size": 11.5 });
    bindTip(g, `<b>Last ${l} layer${l > 1 ? "s" : ""}</b><div class="k"><span>installation effect</span><span>${fmt(v)}</span></div>`);
  });
  animate(svg);
}

export function reverse(res) {
  const r = res.reverse.harmfulness;
  const rows = [
    { t: "Reversed-preference training", s: "weights retrained with preferences swapped", v: r.training, c: C.violet, dir: "install" },
    { t: "Steering removal (−λv)", s: "no weight update", v: r.steering, c: C.remove, dir: "remove" },
  ];
  const box = document.getElementById("reverse");
  const W = widthOf(box);
  const rowH = 78, m = { l: 8, r: 64, t: 6, b: 44 };
  const H = m.t + m.b + rowH * rows.length;
  const svg = svgRoot(box, W, H, "Harmfulness increase from reversed-preference training and from steering removal");
  const x = scale(0, 0.14, m.l, W - m.r);
  xAxis(svg, x, m.t, H - m.b, [0, 0.04, 0.08, 0.12], (v) => v.toFixed(2), "increase in harmful enablement vs unsteered DPO");
  rows.forEach((row, i) => {
    const y0 = m.t + rowH * i;
    el("line", { x1: x(0), x2: x(0), y1: y0 + 48, y2: y0 + 72, class: "zero" }, svg);
    text(svg, m.l, y0 + 22, row.t, { class: "lbl-strong", "font-size": 13 });
    text(svg, m.l, y0 + 40, row.s, { class: "lbl-muted" });
    const y = y0 + 60;
    ciLine(svg, x, row.v[1], row.v[2], y, row.c, 3);
    const p = el("g", { "data-pop": 1 }, svg);
    marker(p, row.dir, x(row.v[0]), y, 6.5, row.c, "#04060b");
    hit(p, x(row.v[0]), y, 15);
    text(svg, x(row.v[2]) + 10, y + 4, fmt(row.v[0]), { class: "val", "data-fade": 1 });
    bindTip(p, `<b>${row.t}</b><div class="k"><span>effect</span><span>${fmt(row.v[0])}</span></div><div class="k"><span>95% CI</span><span>${fmtCI(row.v)}</span></div><div class="k"><span>paired n</span><span>${r.n}</span></div>`);
  });
  animate(svg);
}
