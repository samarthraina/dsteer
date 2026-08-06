"""What a single direction can and cannot remove.

The point of the figure is the distance that is *not* travelled. Steering the aligned
checkpoint back toward its instruction-tuned one moves refusal and harmfulness about a
third of the way and then stops, while generations stay intact -- so the limit is the
direction, not the strength and not degeneration. A curve alone hides that; the reference
line for the unsteered instruction-tuned model is what makes it visible.

Conditioned on prompts where the two checkpoints actually differ. On the full set the
effect is diluted by the two thirds of prompts both checkpoints handle identically, which
is a fact about the benchmark rather than about steering.

    python scripts/fig_dealignment.py --data dealign_curve.json --output-dir outputs/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steering.figures import BLUE, DOUBLE_COL, INK_MUTED, ORANGE, save, use_paper_style


def panel(ax, x, y, lo, hi, floor, colour, ylabel, floor_label):
    ax.plot(x, y, color=colour, marker="o", markersize=3, zorder=3)
    ax.fill_between(x, lo, hi, color=colour, alpha=0.18, linewidth=0, zorder=2)

    # The unsteered instruction-tuned model: where full removal would land.
    ax.axhline(floor, color=INK_MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    ax.annotate(floor_label, xy=(x[-1], floor), xytext=(0, 3 if floor < y[-1] else -11),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=plt.rcParams["font.size"] - 1.5, color=INK_MUTED)

    # Credit the furthest point the curve reaches, not the last one -- harmfulness peaks
    # mid-range and drifts back, and scoring the endpoint would understate the effect.
    best = min(y) if floor < y[0] else max(y)
    reach = (y[0] - best) / (y[0] - floor)

    # Everything between there and the reference line is out of reach at any strength.
    ax.fill_between([x[0], x[-1]], [best] * 2, [floor] * 2,
                    color=INK_MUTED, alpha=0.07, linewidth=0, zorder=0)
    ax.annotate(f"{1 - reach:.0%} out of reach", xy=(0.55, (best + floor) / 2),
                xycoords=("axes fraction", "data"), ha="center", va="center",
                fontsize=plt.rcParams["font.size"] - 1.5, color=INK_MUTED)

    ax.set_xlabel(r"De-alignment strength $|\lambda|$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, max(x))
    ax.set_ylim(0, 1)


def main():
    parser = argparse.ArgumentParser(description="De-alignment reach against steering strength.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="outputs/figures")
    args = parser.parse_args()

    blob = json.loads(Path(args.data).read_text(encoding="utf-8"))
    rows = sorted(blob["rows"], key=lambda r: abs(r["lam"]))
    x = [abs(r["lam"]) for r in rows]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    use_paper_style()

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL * 0.82, 2.5))

    panel(axes[0], x, [r["refusal"] for r in rows], [r["r_lo"] for r in rows],
          [r["r_hi"] for r in rows], blob["it_refusal"], BLUE, "Refusal",
          "unsteered SFT checkpoint")
    panel(axes[1], x, [r["harm"] for r in rows], [r["h_lo"] for r in rows],
          [r["h_hi"] for r in rows], blob["it_harm"], ORANGE, "Harmfulness",
          "unsteered SFT checkpoint")

    # Broken generations stay under 5% everywhere, so the plateau is not degeneration.
    worst = max(r["broken"] for r in rows)
    axes[0].annotate(f"under {worst:.0%} of generations degenerate at every strength",
                     xy=(0.0, 1.06), xycoords="axes fraction", ha="left", va="bottom",
                     fontsize=plt.rcParams["font.size"] - 1.5, color=INK_MUTED)

    fig.tight_layout()
    print(save(fig, out / "dealignment_reach"))


if __name__ == "__main__":
    main()
