#!/usr/bin/env python3
"""Three summary figures for the rebuttal run, from outputs/analysis/*.csv only.

  fig1_dsteer_vs_comparator   intended harmfulness effect, D-STEER vs matched comparator
  fig2_judge_robustness       D-STEER effect under the original and the independent judge
  fig3_benign_costs           XSTest over-refusal and IFEval invalid-output changes

Style and palette come from the public repo's validated steering.figures (blue/orange
pass the CVD and normal-vision gates). Vector PDF plus PNG.

Run:  .venv/bin/python -I scripts/make_figures.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "GitHub" / "dsteer" / "src"))
from steering.figures import BLUE, INK, INK_MUTED, ORANGE, use_paper_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

A = ROOT / "outputs/analysis"
OUT = ROOT / "figures"
CELLS = [("A", "installation"), ("A", "reversal"), ("B", "installation"), ("B", "reversal")]
CELL_LABEL = {("A", "installation"): "Llama · install", ("A", "reversal"): "Llama · remove",
              ("B", "installation"): "Mistral · install", ("B", "reversal"): "Mistral · remove"}
COND = {("installation", "dsteer"): "dsteer_install", ("reversal", "dsteer"): "dsteer_reverse",
        ("installation", "comp"): "refusal_add_install", ("reversal", "comp"): "refusal_add_reverse"}


def rows(name):
    return [r for r in csv.DictReader((A / f"{name}.csv").open()) if r.get("status") == "complete"]


def dotplot(ax, series, xlabel):
    """series: [(label, colour, {cell: (point, lo, hi)})]; one row per cell, series offset."""
    off = (-0.14, 0.14) if len(series) == 2 else (0.0,)
    for (label, colour, data), dy in zip(series, off):
        for i, cell in enumerate(CELLS):
            if cell not in data:
                continue
            p, lo, hi = data[cell]
            y = i + dy
            ax.plot([lo, hi], [y, y], color=colour, lw=2, solid_capstyle="round")
            ax.plot(p, y, "o", ms=6, color=colour, mec="white", mew=1.0, label=label if i == 0 else None)
    ax.axvline(0, color=INK_MUTED, lw=0.8)
    ax.set_yticks(range(len(CELLS)))
    ax.set_yticklabels([CELL_LABEL[c] for c in CELLS])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def fig1():
    eff = rows("harmful_effects")
    pick = lambda m: {(r["pair"], r["direction"]): (float(r["intended_effect"]), float(r["ci_lo"]), float(r["ci_hi"]))
                      for r in eff if r["method"] == m}
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    dotplot(ax, [("D-STEER", BLUE, pick("dsteer")), ("Matched refusal-direction comparator", ORANGE,
                                                       pick("refusal_add_matched"))],
            "Intended harmfulness change (higher = better), 95% CI")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False)
    ax.set_title("D-STEER vs refusal-direction baseline (300 harmful test prompts)", loc="left", color=INK)
    return fig


def fig2():
    jr = rows("judge_robustness")
    orig = {(r["pair"], r["direction"]): (float(r["effect_original"]), float(r["orig_ci_lo"]), float(r["orig_ci_hi"]))
            for r in jr}
    ind = {(r["pair"], r["direction"]): (float(r["effect_independent"]), float(r["ind_ci_lo"]), float(r["ind_ci_hi"]))
           for r in jr}
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    dotplot(ax, [("Original judge (Qwen3.5-27B)", BLUE, orig), ("Independent judge (phi-4)", ORANGE, ind)],
            "D-STEER intended harmfulness change, 95% CI")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False)
    ax.set_title("Same responses, two judges", loc="left", color=INK)
    return fig


def fig3():
    bu = [r for r in rows("benign_utility") if r["panel"] == "xstest_safe"]
    iv = rows("ifeval_variants")
    # not sharey: each panel inverts its own y-axis, and a shared axis would be inverted twice
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for ax, src, key, lo, hi, title in (
            (a, bu, "full_refusal_diff_vs_endpoint", "diff_ci_lo", "diff_ci_hi", "Over-refusal on 250 safe prompts (XSTest)"),
            (b, iv, "invalid_change", "invalid_ci_lo", "invalid_ci_hi", "Broken outputs on 100 instructions (IFEval)")):
        series = []
        for label, colour, m in (("D-STEER", BLUE, "dsteer"), ("Matched comparator", ORANGE, "comp")):
            data = {}
            for pair, d in CELLS:
                r = next((x for x in src if x["pair"] == pair and x["condition"] == COND[(d, m)]), None)
                if r and r.get(key):
                    data[(pair, d)] = tuple(100 * float(r[k]) for k in (key, lo, hi))
            series.append((label, colour, data))
        dotplot(ax, series, "Change vs. unsteered model (percentage points)")
        ax.set_title(title, loc="left", color=INK)
    b.set_yticklabels([])
    a.legend(loc="upper center", bbox_to_anchor=(1.05, -0.24), ncol=2, frameon=False)
    return fig


def main() -> int:
    use_paper_style(base_size=8.0)
    OUT.mkdir(exist_ok=True)
    for name, make in (("fig1_dsteer_vs_comparator", fig1), ("fig2_judge_robustness", fig2),
                       ("fig3_benign_costs", fig3)):
        fig = make()
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{name}.{ext}")
        plt.close(fig)
        print(f"wrote figures/{name}.pdf/.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
