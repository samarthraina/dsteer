import { el, text, glowFilter } from "./svg.js";
import { reducedMotion } from "./data.js";

// The two-way street: IT and DPO endpoints joined by an install lane and a remove lane.
export function initStreet(svg, means) {
  const defs = el("defs", {}, svg);
  glowFilter(defs, "st-glow", 6);
  glowFilter(defs, "st-glow-s", 2.5);
  const orb = (id, c0, c1) => {
    const g = el("radialGradient", { id, cx: "38%", cy: "35%", r: "70%" }, defs);
    el("stop", { offset: "0", "stop-color": c0 }, g);
    el("stop", { offset: "1", "stop-color": c1 }, g);
  };
  orb("st-it", "#9db8ff", "#16285a");
  orb("st-dpo", "#a6fbe3", "#0b4a3c");

  const L = { x: 165, y: 210 }, R = { x: 835, y: 210 };
  const upper = el("path", { d: `M ${L.x + 80} ${L.y - 30} C 380 60, 620 60, ${R.x - 80} ${R.y - 30}`, fill: "none", stroke: "#2fe3b4", "stroke-width": 2.5, opacity: 0.85 }, svg);
  const lower = el("path", { d: `M ${R.x - 80} ${R.y + 30} C 620 360, 380 360, ${L.x + 80} ${L.y + 30}`, fill: "none", stroke: "#ff6b3d", "stroke-width": 2.5, opacity: 0.85 }, svg);
  el("path", { d: upper.getAttribute("d"), fill: "none", stroke: "#2fe3b4", "stroke-width": 12, opacity: 0.08 }, svg);
  el("path", { d: lower.getAttribute("d"), fill: "none", stroke: "#ff6b3d", "stroke-width": 12, opacity: 0.08 }, svg);

  // arrowheads at the lane ends
  const head = (x, y, ang, c) => el("path", { d: "M -12 -8 L 4 0 L -12 8 Z", fill: c, transform: `translate(${x} ${y}) rotate(${ang})` }, svg);
  head(R.x - 80, R.y - 30, 32, "#2fe3b4");
  head(L.x + 80, L.y + 30, 212, "#ff6b3d");

  text(svg, 500, 36, "+ λv   install safety", { "text-anchor": "middle", fill: "#bff7e6", "font-family": "Space Grotesk", "font-size": 22, "font-weight": 600 });
  text(svg, 500, 60, "harmful enablement goes down", { "text-anchor": "middle", fill: "#7a8399", "font-size": 14 });
  text(svg, 500, 384, "− λv   remove safety", { "text-anchor": "middle", fill: "#ffd0bf", "font-family": "Space Grotesk", "font-size": 22, "font-weight": 600 });
  text(svg, 500, 406, "harmful enablement goes up", { "text-anchor": "middle", fill: "#7a8399", "font-size": 14 });

  // centre: same vector, opposite signs
  const mid = el("g", { transform: "translate(500 210)" }, svg);
  el("rect", { x: -118, y: -24, width: 236, height: 48, rx: 24, fill: "rgba(255,255,255,0.04)", stroke: "rgba(255,255,255,0.14)" }, mid);
  text(mid, 0, 6, "one vector v, two signs", { "text-anchor": "middle", fill: "#b4bccd", "font-size": 15, "font-family": "JetBrains Mono" });

  const node = (p, grad, glow, title, sub, h) => {
    const g = el("g", { transform: `translate(${p.x} ${p.y})` }, svg);
    el("circle", { r: 92, fill: glow, opacity: 0.1, filter: "url(#st-glow)" }, g);
    el("circle", { r: 72, fill: `url(#${grad})`, stroke: "rgba(255,255,255,0.25)", "stroke-width": 1 }, g);
    text(g, 0, -6, title, { "text-anchor": "middle", fill: "#fff", "font-family": "Space Grotesk", "font-size": 30, "font-weight": 700 });
    text(g, 0, 18, sub, { "text-anchor": "middle", fill: "rgba(255,255,255,0.75)", "font-size": 13 });
    text(g, 0, 112, h, { "text-anchor": "middle", fill: "#b4bccd", "font-size": 13.5, "font-family": "JetBrains Mono" });
    return g;
  };
  node(L, "st-it", "#5b8dff", "IT", "instruction-tuned", `H = ${means.mistral_install_baseline.harm.toFixed(3)}`);
  node(R, "st-dpo", "#2fe3b4", "DPO", "harmlessness DPO", `H = ${means.mistral_remove_baseline.harm.toFixed(3)}`);
  text(svg, 500, 260, "H: mean harmful enablement, Mistral pair, held-out panel", { "text-anchor": "middle", fill: "#5f6880", "font-size": 12 });

  if (reducedMotion) return;
  const lanes = [
    { path: upper, color: "#2fe3b4", n: 12 },
    { path: lower, color: "#ff6b3d", n: 12 },
  ].map((ln) => {
    const len = ln.path.getTotalLength();
    const dots = Array.from({ length: ln.n }, (_, i) => ({
      c: el("circle", { r: 4.5, fill: ln.color, filter: "url(#st-glow-s)" }, svg),
      o: i / ln.n,
    }));
    return { ...ln, len, dots };
  });

  let on = false;
  new IntersectionObserver(([e]) => (on = e.isIntersecting)).observe(svg);
  const t0 = performance.now();
  (function tick(now) {
    requestAnimationFrame(tick);
    if (!on) return;
    const t = (now - t0) / 1000;
    for (const ln of lanes) {
      for (const d of ln.dots) {
        const p = (t * 0.12 + d.o) % 1;
        const pt = ln.path.getPointAtLength(p * ln.len);
        d.c.setAttribute("cx", pt.x);
        d.c.setAttribute("cy", pt.y);
        d.c.setAttribute("opacity", Math.sin(p * Math.PI).toFixed(3));
      }
    }
  })(t0);
}
