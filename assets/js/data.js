// Loads the exported activation projections and result tables.
// points_<A|B>.bin is float32 [layer][model 0=IT,1=DPO][point][x=v̂, y=PC2, z=PC3],
// in units of the layer's mean IT activation norm.

const FILE = { mistral: "B", llama: "A" };
const cache = {};
let results;

// One display scale for both pairs and every layer, so on-screen distances are comparable.
export const WORLD = 13;

export const PAIR_LABEL = { mistral: "Mistral pair", llama: "Llama pair" };
export const LAMBDA_STAR = {
  mistral: { install: 0.135, remove: 0.055 },
  llama: { install: 0.05, remove: 0.05 },
};

export function loadPair(pair) {
  const key = FILE[pair];
  if (!cache[key]) {
    cache[key] = Promise.all([
      fetch(`assets/data/points_${key}.bin`).then((r) => r.arrayBuffer()),
      fetch(`assets/data/points_${key}.json`).then((r) => r.json()),
    ]).then(([buf, meta]) => ({
      pair,
      data: new Float32Array(buf),
      meta,
      N: meta.n_points,
      L: meta.n_layers,
    }));
  }
  return cache[key];
}

export function loadResults() {
  results ??= fetch("assets/data/results.json").then((r) => r.json());
  return results;
}

// Write layer `layer`, model `model` into `out` (Float32Array of N*3) as world coordinates.
export function fillLayer(P, layer, model, out, s = WORLD) {
  const o = (layer * 2 + model) * P.N * 3;
  for (let k = 0; k < P.N * 3; k++) out[k] = P.data[o + k] * s;
  return out;
}

export function point(P, layer, model, i) {
  const o = ((layer * 2 + model) * P.N + i) * 3;
  return [P.data[o], P.data[o + 1], P.data[o + 2]];
}

export const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
