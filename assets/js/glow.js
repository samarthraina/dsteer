import * as THREE from "three";

export const COLORS = {
  it: new THREE.Color("#5b8dff"),
  dpo: new THREE.Color("#2fe3b4"),
  rem: new THREE.Color("#ff6b3d"),
  violet: new THREE.Color("#a48bff"),
  white: new THREE.Color("#ffffff"),
};

// Soft additive sprites. Per-point colour, size and alpha come from attributes so one
// draw call can carry several clouds.
export function glowMaterial({ size = 1, opacity = 1 } = {}) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uSize: { value: size },
      uPR: { value: Math.min(window.devicePixelRatio || 1, 2) },
      uOpacity: { value: opacity },
    },
    vertexShader: /* glsl */ `
      attribute vec3 aColor;
      attribute float aSize;
      attribute float aAlpha;
      uniform float uSize;
      uniform float uPR;
      varying vec3 vColor;
      varying float vAlpha;
      void main() {
        vColor = aColor;
        vAlpha = aAlpha;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = uSize * aSize * uPR * (10.0 / -mv.z);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */ `
      uniform float uOpacity;
      varying vec3 vColor;
      varying float vAlpha;
      void main() {
        float d = length(gl_PointCoord - 0.5);
        if (d > 0.5) discard;
        float a = pow(smoothstep(0.5, 0.0, d), 1.5);
        float core = smoothstep(0.12, 0.0, d);
        gl_FragColor = vec4(vColor * a * 1.25 + vec3(core * 0.14), (a + core * 0.2) * vAlpha * uOpacity);
        #include <colorspace_fragment>
      }`,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
}

export function cloudGeometry(n) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  g.setAttribute("aColor", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  g.setAttribute("aSize", new THREE.BufferAttribute(new Float32Array(n).fill(1), 1));
  g.setAttribute("aAlpha", new THREE.BufferAttribute(new Float32Array(n).fill(1), 1));
  return g;
}

export function paint(g, from, count, color, jitter = 0.18, seed = 1) {
  const c = g.attributes.aColor.array;
  let s = seed;
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  for (let i = from; i < from + count; i++) {
    // slight per-point lightness variation so dense cores read as depth, not a flat blob
    const k = 1 - jitter / 2 + rnd() * jitter;
    c[i * 3] = color.r * k;
    c[i * 3 + 1] = color.g * k;
    c[i * 3 + 2] = color.b * k;
  }
  g.attributes.aColor.needsUpdate = true;
}

// Text label as a camera-facing sprite.
export function textSprite(text, { color = "#eef1f7", size = 0.5, font = "600 52px 'Space Grotesk', Inter, sans-serif", glow } = {}) {
  const pad = 24;
  const cv = document.createElement("canvas");
  const ctx = cv.getContext("2d");
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width) + pad * 2;
  cv.width = w;
  cv.height = 92;
  ctx.font = font;
  ctx.textBaseline = "middle";
  if (glow) {
    ctx.shadowColor = glow;
    ctx.shadowBlur = 18;
  }
  ctx.fillStyle = color;
  ctx.fillText(text, pad, cv.height / 2);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
  sp.scale.set((size * w) / cv.height, size, 1);
  return sp;
}

// Glowing arrow made of a tube and a cone, from a to b.
export function arrow(a, b, color, { radius = 0.05, head = 0.32 } = {}) {
  const group = new THREE.Group();
  const dir = new THREE.Vector3().subVectors(b, a);
  const len = dir.length();
  dir.normalize();
  const mat = new THREE.MeshBasicMaterial({ color, toneMapped: false, transparent: true, opacity: 0.95 });
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, Math.max(len - head, 0.001), 16), mat);
  const cone = new THREE.Mesh(new THREE.ConeGeometry(radius * 3.2, head, 24), mat);
  shaft.position.y = (len - head) / 2;
  cone.position.y = len - head / 2;
  group.add(shaft, cone);
  group.position.copy(a);
  group.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
  group.userData = { shaft, cone, mat, head, radius };
  return group;
}

export function setArrow(g, a, b) {
  const dir = new THREE.Vector3().subVectors(b, a);
  const len = dir.length();
  const { shaft, cone, head } = g.userData;
  g.visible = len > 0.02;
  if (!g.visible) return;
  const h = Math.min(head, len * 0.6);
  shaft.scale.y = Math.max(len - h, 0.001) / shaft.geometry.parameters.height;
  shaft.position.y = (len - h) / 2;
  cone.scale.setScalar(h / head);
  cone.position.y = len - h / 2;
  g.position.copy(a);
  g.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
}

// Pause a render loop when its canvas is off screen.
export function whenVisible(el, onChange) {
  const io = new IntersectionObserver(([e]) => onChange(e.isIntersecting), { rootMargin: "120px" });
  io.observe(el);
  return io;
}
