import { reducedMotion } from "./data.js";

// Real held-out outputs, already redacted at export time: harmful answers arrive as their
// opening line plus line lengths, never their content.
const ARMS = {
  it: { model: "IT", sub: "unsteered", color: "#5b8dff" },
  it_steer: { model: "IT + λv", sub: "install", color: "#2fe3b4" },
  dpo: { model: "DPO", sub: "unsteered", color: "#2fe3b4" },
  dpo_steer: { model: "DPO − λv", sub: "remove", color: "#ff6b3d" },
};
const PAIR = { mistral: "Mistral-7B", llama: "Llama-3-8B" };
const ICON = {
  safe: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>',
  harm: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 2 21h20zM12 10v5M12 18h.01"/></svg>',
  lock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>',
};

const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

export async function initExamples() {
  const { stats, examples } = await fetch("assets/data/examples.json").then((r) => r.json());
  const tabs = document.getElementById("ex-tabs");
  const cards = [...document.querySelectorAll(".ex-card")];
  let run = 0;

  tabs.innerHTML = examples.map((e, i) =>
    `<button class="ex-tab${i ? "" : " on"}" role="tab" aria-selected="${!i}" data-i="${i}"><span class="pair ${e.pair}">${PAIR[e.pair]}</span>${esc(e.topic)}</button>`).join("");

  function ribbon(pair) {
    const s = stats[pair];
    document.getElementById("ex-ribbon").innerHTML = `
      <div class="ex-stat"><span class="n">${s.install_flips}</span><span class="t">held-out prompts where <b class="dpo">+λv</b> turns a harmful answer (≥ 6/10) into a safe one (≤ 1/10), ${PAIR[pair]}</span></div>
      <div class="ex-stat"><span class="n">${s.remove_flips}</span><span class="t">held-out prompts where <b class="rem">−λv</b> turns the DPO model's safe answer into a harmful one</span></div>
      <div class="ex-stat"><span class="n">${s.flip_both_ways}</span><span class="t">prompts that flip <b>both ways</b>, from one direction vector and its negative</span></div>`;
  }

  function cardHTML(key, a) {
    const meta = ARMS[key];
    const state = a.redacted ? "harm" : "safe";
    const bars = a.redacted
      ? `<div class="ex-bars">${a.bars.slice(0, 7).map((b) => `<div class="ex-bar"><b>${b.step ?? ""}</b><span data-w="${Math.max(28, Math.min(100, (b.len / 150) * 100))}"></span></div>`).join("")}</div>
         <div class="ex-lock">${ICON.lock}${a.bars.filter((b) => b.step).length || a.bars.length} steps redacted</div>`
      : "";
    return `
      <div class="ex-head">
        <div class="ex-model"><i style="background:${meta.color};color:${meta.color}"></i>${meta.model}<small>${meta.sub}</small></div>
        <span class="ex-state">${ICON[state]}${a.redacted ? "complies" : "refuses"}</span>
      </div>
      <div class="ex-text"><span class="typed"></span><span class="caret"></span></div>
      ${bars}
      <div class="ex-foot">
        <div class="ex-meter h"><span>harm</span><span class="track"><span class="fill" data-w="${a.harm * 10}" style="--full:${a.harm ? 1000 / a.harm : 100}%"></span></span><span class="v">${a.harm}/10</span></div>
        <div class="ex-meter r"><span>refusal</span><span class="track"><span class="fill" data-w="${a.refusal * 10}"></span></span><span class="v">${a.refusal}/10</span></div>
        <div class="ex-judge"><b>JUDGE</b>${esc(a.judge)}</div>
      </div>`;
  }

  function type(node, text, delay, id) {
    const caret = node.nextElementSibling;
    if (reducedMotion) {
      node.textContent = text;
      caret.remove();
      return;
    }
    let i = 0;
    const step = () => {
      if (id !== run) return;
      i = Math.min(text.length, i + 2);
      node.textContent = text.slice(0, i);
      if (i < text.length) setTimeout(step, 16);
      else setTimeout(() => caret.remove(), 600);
    };
    setTimeout(step, delay);
  }

  function show(i) {
    const id = ++run;
    const e = examples[i];
    document.getElementById("ex-prompt").textContent = e.prompt;
    document.getElementById("ex-pair").textContent = PAIR[e.pair];
    document.getElementById("ex-l-inst").textContent = `λ* +${e.lambda.install.toFixed(3)}`;
    document.getElementById("ex-l-rem").textContent = `λ* ${e.lambda.remove.toFixed(3).replace("-", "−")}`;
    ribbon(e.pair);
    cards.forEach((card, k) => {
      const a = e.arms[card.dataset.arm];
      card.className = `ex-card ${a.redacted ? "harm" : "safe"}`;
      card.innerHTML = cardHTML(card.dataset.arm, a);
      const order = { it: 0, it_steer: 1, dpo: 2, dpo_steer: 3 }[card.dataset.arm];
      if (!reducedMotion && window.gsap) window.gsap.fromTo(card, { opacity: 0, y: 18 }, { opacity: 1, y: 0, duration: 0.6, delay: order * 0.12, ease: "power3.out" });
      const text = a.text + (a.more && !a.redacted ? " …" : "");
      type(card.querySelector(".typed"), text, 250 + order * 160, id);
      // bars and meters grow once the text has had a head start
      setTimeout(() => {
        if (id !== run) return;
        card.querySelectorAll("[data-w]").forEach((el, j) => setTimeout(() => (el.style.width = `${el.dataset.w}%`), j * 70));
      }, reducedMotion ? 0 : 500 + order * 160);
    });
  }

  tabs.addEventListener("click", (ev) => {
    const b = ev.target.closest(".ex-tab");
    if (!b || b.classList.contains("on")) return;
    tabs.querySelectorAll(".ex-tab").forEach((t) => {
      t.classList.toggle("on", t === b);
      t.setAttribute("aria-selected", t === b);
    });
    show(+b.dataset.i);
  });

  // start typing when the stage scrolls into view
  const stage = document.getElementById("ex-stage");
  const io = new IntersectionObserver(([en]) => {
    if (!en.isIntersecting) return;
    io.disconnect();
    show(0);
  }, { threshold: 0.1 });
  io.observe(stage);
}
