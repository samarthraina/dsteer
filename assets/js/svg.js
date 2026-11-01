// Tiny SVG helpers shared by the diagrams and charts.
export const NS = "http://www.w3.org/2000/svg";

export function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}

export function text(parent, x, y, str, attrs = {}) {
  const t = el("text", { x, y, ...attrs }, parent);
  t.textContent = str;
  return t;
}

export function scale(d0, d1, r0, r1) {
  const f = (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
  f.invert = (p) => d0 + ((p - r0) / (r1 - r0)) * (d1 - d0);
  f.domain = [d0, d1];
  return f;
}

export function niceTicks(d0, d1, n = 5) {
  const span = d1 - d0;
  const raw = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(d0 / step) * step; v <= d1 + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

export function fmt(v, d = 3) {
  return (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
}

export function fmtCI(ci, d = 3) {
  return `[${ci[1].toFixed(d)}, ${ci[2].toFixed(d)}]`;
}

// shared tooltip
const tip = () => document.getElementById("tip");
export function showTip(html, x, y) {
  const t = tip();
  t.innerHTML = html;
  const w = t.offsetWidth;
  t.style.left = `${Math.max(w / 2 + 8, Math.min(window.innerWidth - w / 2 - 8, x))}px`;
  t.style.top = `${Math.max(t.offsetHeight + 24, y)}px`;
  t.classList.add("on");
}
export function hideTip() {
  tip().classList.remove("on");
}
export function bindTip(node, html) {
  node.style.cursor = "pointer";
  node.addEventListener("pointerenter", (e) => showTip(typeof html === "function" ? html() : html, e.clientX, e.clientY));
  node.addEventListener("pointermove", (e) => showTip(typeof html === "function" ? html() : html, e.clientX, e.clientY));
  node.addEventListener("pointerleave", hideTip);
}

export function glowFilter(defs, id, std = 4) {
  const f = el("filter", { id, x: "-50%", y: "-50%", width: "200%", height: "200%" }, defs);
  el("feGaussianBlur", { stdDeviation: std, result: "b" }, f);
  const m = el("feMerge", {}, f);
  el("feMergeNode", { in: "b" }, m);
  el("feMergeNode", { in: "SourceGraphic" }, m);
  return f;
}
