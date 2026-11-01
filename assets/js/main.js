import { loadResults, reducedMotion } from "./data.js";
import * as charts from "./charts.js";
import { initStreet } from "./street.js";
import { initMethod } from "./method.js";

const gsap = window.gsap;
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

// ── smooth scroll ──
let lenis;
if (!reducedMotion && window.Lenis) {
  lenis = new window.Lenis({ duration: 1.15, smoothWheel: true });
  if (window.ScrollTrigger) {
    gsap.registerPlugin(window.ScrollTrigger);
    lenis.on("scroll", window.ScrollTrigger.update);
  }
  gsap.ticker.add((t) => lenis.raf(t * 1000));
  gsap.ticker.lagSmoothing(0);
}
$$('a[href^="#"]').forEach((a) => {
  a.addEventListener("click", (e) => {
    const id = a.getAttribute("href");
    if (id === "#" || a.getAttribute("aria-disabled") === "true") {
      e.preventDefault();
      return;
    }
    const t = $(id);
    if (!t) return;
    e.preventDefault();
    lenis ? lenis.scrollTo(t, { offset: -20 }) : t.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth" });
  });
});

// ── title: one block, so the gradient text clip survives; wipe it in left to right ──
const title = $("#hero-title");
if (title && !reducedMotion) {
  gsap.fromTo(title,
    { clipPath: "inset(0 100% 0 0)", letterSpacing: "0.04em", y: 30, opacity: 0 },
    { clipPath: "inset(0 -5% 0 0)", letterSpacing: "-0.045em", y: 0, opacity: 1, duration: 1.6, ease: "expo.out", delay: 0.2 });
}

// ── reveals ──
const revealIO = new IntersectionObserver((entries) => {
  entries.forEach((e) => {
    if (e.isIntersecting) {
      e.target.classList.add("in");
      revealIO.unobserve(e.target);
    }
  });
}, { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
const watchReveals = (root = document) => $$(".reveal:not(.in)", root).forEach((n) => revealIO.observe(n));
watchReveals();

// ── counters ──
function countUp(node) {
  const to = parseFloat(node.dataset.count);
  const dec = +(node.dataset.dec || 0);
  const sign = node.dataset.sign || "";
  const set = (v) => (node.textContent = (sign && v >= 0 ? sign : "") + v.toFixed(dec));
  if (reducedMotion) return set(to);
  const o = { v: 0 };
  gsap.to(o, { v: to, duration: 1.8, ease: "power3.out", onUpdate: () => set(o.v) });
}
const countIO = new IntersectionObserver((entries) => {
  entries.forEach((e) => {
    if (e.isIntersecting) {
      countUp(e.target);
      countIO.unobserve(e.target);
    }
  });
}, { threshold: 0.6 });
const watchCounts = (root = document) => $$("[data-count]", root).forEach((n) => countIO.observe(n));
watchCounts();

// ── progress bar, nav hide/show, active section ──
const bar = $(".progress");
const nav = $(".nav");
let lastY = 0;
function onScroll() {
  const y = window.scrollY;
  const max = document.documentElement.scrollHeight - window.innerHeight;
  bar.style.transform = `scaleX(${max > 0 ? y / max : 0})`;
  nav.classList.toggle("hidden", y > window.innerHeight * 0.8 && y > lastY + 2);
  if (y < lastY - 2 || y < window.innerHeight * 0.8) nav.classList.remove("hidden");
  lastY = y;
}
window.addEventListener("scroll", onScroll, { passive: true });
onScroll();
const links = $$(".nav .links a");
const secIO = new IntersectionObserver((entries) => {
  entries.forEach((e) => {
    if (e.isIntersecting) links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${e.target.id}`));
  });
}, { rootMargin: "-45% 0px -50% 0px" });
links.forEach((a) => {
  const s = $(a.getAttribute("href"));
  if (s) secIO.observe(s);
});

// ── maths ──
function renderMath() {
  if (window.renderMathInElement) {
    window.renderMathInElement(document.body, {
      delimiters: [{ left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }],
      throwOnError: false,
    });
  }
}
if (window.renderMathInElement) renderMath();
else window.addEventListener("load", renderMath);

// ── copy bibtex ──
$("#copy-bib")?.addEventListener("click", async (e) => {
  try {
    await navigator.clipboard.writeText($("#bibtex").textContent);
    e.target.textContent = "Copied";
  } catch {
    e.target.textContent = "Select and copy";
  }
  setTimeout(() => (e.target.textContent = "Copy"), 1800);
});

// ── magnetic buttons ──
if (!reducedMotion && matchMedia("(pointer: fine)").matches) {
  $$(".btn").forEach((b) => {
    b.addEventListener("pointermove", (e) => {
      const r = b.getBoundingClientRect();
      gsap.to(b, { x: (e.clientX - r.left - r.width / 2) * 0.18, y: (e.clientY - r.top - r.height / 2) * 0.25, duration: 0.4, ease: "power3.out" });
    });
    b.addEventListener("pointerleave", () => gsap.to(b, { x: 0, y: 0, duration: 0.6, ease: "elastic.out(1, 0.5)" }));
  });
}

// ── lazy initialisers: start each heavy piece shortly before it scrolls in ──
function near(node, fn, margin = "600px") {
  if (!node) return;
  const io = new IntersectionObserver(([e]) => {
    if (e.isIntersecting) {
      io.disconnect();
      fn();
    }
  }, { rootMargin: margin });
  io.observe(node);
}

const hasGL = (() => {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
})();

const noGL = (err) => {
  if (err) console.error(err);
  document.documentElement.classList.add("no-gl");
};
if (hasGL) import("./hero.js").then((m) => m.initHero($("#hero-gl"))).catch(noGL);
else noGL();

// charts lay out at real pixel width, so redraw them (settled, no replayed animation) when it changes
const redraws = [];
let lastW = window.innerWidth, resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (Math.abs(window.innerWidth - lastW) < 24) return;
    lastW = window.innerWidth;
    charts.settle();
    redraws.forEach((f) => f());
  }, 180);
});

loadResults().then((res) => {
  initStreet($("#street-svg"), res.means);
  near($("#method"), () => initMethod($("#method-svg"), $$(".step")));

  if (hasGL) near($("#lab"), () => import("./lab.js").then((m) => m.initLab()).catch(noGL));

  near($("#depth"), async () => {
    let depthApi;
    let pair = "mistral";
    const draw = () => charts.depthCharts(res, pair, (l) => depthApi?.setLayer(l));
    draw();
    redraws.push(draw);
    if (hasGL) depthApi = await import("./depth.js").then((m) => m.initDepth()).catch(noGL);
    $("#depth-pair").addEventListener("click", async (e) => {
      const b = e.target.closest("button");
      if (!b || b.classList.contains("on")) return;
      $$("#depth-pair button").forEach((x) => x.classList.toggle("on", x === b));
      pair = b.dataset.pair;
      draw();
      await depthApi?.setPair(pair);
    });
  });

  near($("#examples"), () => import("./examples.js").then((m) => m.initExamples()).catch((err) => console.error(err)));

  near($("#window"), () => {
    const list = $("#window-cells");
    const cells = [
      ["mistral_install", "Mistral", "install"],
      ["mistral_remove", "Mistral", "remove"],
      ["llama_install", "Llama", "install"],
      ["llama_remove", "Llama", "remove"],
    ];
    list.innerHTML = cells.map(([k, p, d], i) => {
      const lam = res.heldout[k].lambda;
      return `<button class="cell-btn${i ? "" : " on"}" data-k="${k}" data-dir="${d}"><span class="n">${p}</span><span class="d">${d === "install" ? "installation" : "removal"}</span><span class="c">λ* ${lam > 0 ? "+" : "−"}${Math.abs(lam).toFixed(3)}</span></button>`;
    }).join("");
    list.addEventListener("click", (e) => {
      const b = e.target.closest(".cell-btn");
      if (!b || b.classList.contains("on")) return;
      $$(".cell-btn", list).forEach((x) => x.classList.toggle("on", x === b));
      current = b.dataset.k;
      charts.windowChart(res, current);
    });
    let current = "mistral_install";
    charts.windowChart(res, current);
    redraws.push(() => charts.windowChart(res, current));
  });

  near($("#results"), () => {
    charts.tracks(res, $("#tracks"));
    watchReveals($("#tracks"));
    watchCounts($("#tracks"));
    charts.forest(res);
    charts.transfer(res);
    redraws.push(() => {
      charts.tracks(res, $("#tracks"));
      charts.forest(res);
      charts.transfer(res);
    });
  });

  near($("#robustness"), () => {
    const draw = () => {
      charts.rawBars(res);
      charts.rawDepth(res);
      charts.reverse(res);
    };
    draw();
    redraws.push(draw);
  });
});
