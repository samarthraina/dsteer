"""Where the steering vector is estimated matters more than the vector itself.

Four curves on the same checkpoint, prompts and grid. Two vectors -- one estimated on
HH-RLHF, one on held-out prompts from the evaluation distribution -- each against a
norm-matched random direction. The random arms are what separate "this direction carries
the behaviour" from "a perturbation this large disturbs it", and without them the two
real curves cannot be compared to anything.

    python scripts/fig_distribution_match.py --data d4_curves.json --output-dir outputs/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steering.figures import BLUE, DOUBLE_COL, INK_MUTED, ORANGE, save, use_paper_style

STYLE = {
    "same-distribution, held out": dict(color=BLUE, marker="o", linestyle="-", zorder=4),
    "same-distribution, random": dict(color=BLUE, marker="^", linestyle=(0, (3, 2)), zorder=3),
    "cross-distribution (HH-RLHF)": dict(color=ORANGE, marker="o", linestyle="-", zorder=4),
    "cross-distribution, random": dict(color=ORANGE, marker="^", linestyle=(0, (3, 2)), zorder=3),
}
SHORT = {
    "same-distribution, held out": "same distribution",
    "same-distribution, random": "same dist., random",
    "cross-distribution (HH-RLHF)": "cross distribution",
    "cross-distribution, random": "cross dist., random",
}


def panel(ax, curves, key, lo, hi, floor, baseline, ylabel, floor_label):
    for name, rows in curves.items():
        if not rows or len(rows) < 2:
            continue
        x = [r["lam"] for r in rows]
        ax.plot(x, [r[key] for r in rows], label=SHORT[name], markersize=3.2, linewidth=1.4,
                **STYLE[name])
        ax.fill_between(x, [r[lo] for r in rows], [r[hi] for r in rows],
                        color=STYLE[name]["color"], alpha=0.10, linewidth=0, zorder=1)

    # The unsteered SFT checkpoint: where complete removal would land.
    ax.axhline(floor, color=INK_MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(floor_label, xy=(0.98, floor), xycoords=("axes fraction", "data"),
                xytext=(0, 4 if floor < baseline else -11), textcoords="offset points",
                ha="right", fontsize=plt.rcParams["font.size"] - 1.5, color=INK_MUTED)

    ax.set_xlabel(r"Steering strength $|\lambda|$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, max(r["lam"] for rows in curves.values() for r in rows))
    ax.set_ylim(0, 1)


def main():
    parser = argparse.ArgumentParser(description="Same- against cross-distribution steering.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="outputs/figures")
    args = parser.parse_args()

    blob = json.loads(Path(args.data).read_text(encoding="utf-8"))
    curves = {k: sorted(v, key=lambda r: r["lam"]) for k, v in blob["curves"].items()}
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    use_paper_style()

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL * 0.9, 2.7))
    base = curves["cross-distribution (HH-RLHF)"][0]

    panel(axes[0], curves, "refusal", "r_lo", "r_hi", blob["floor_refusal"],
          base["refusal"], "Refusal", "unsteered SFT checkpoint")
    panel(axes[1], curves, "harm", "h_lo", "h_hi", blob["floor_harm"],
          base["harm"], "Harmfulness", "unsteered SFT checkpoint")

    axes[0].legend(loc="upper right", fontsize=plt.rcParams["font.size"] - 2,
                   frameon=False, handlelength=2.2)
    fig.tight_layout()
    print(save(fig, out / "distribution_match"))


if __name__ == "__main__":
    main()
