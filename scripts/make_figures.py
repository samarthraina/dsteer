"""Build the paper figures from run outputs.

Each figure answers one question, and the question is the caption. Reads the CSVs
written by layer_profile.py and rank_analysis.py, so it needs no GPU and no models --
rerun it freely while the wording of a claim is still moving.

    python scripts/make_figures.py --runs-root outputs --output-dir outputs/figures

Writes PDF (for LaTeX) and PNG (for looking at) side by side.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steering.figures import (
    AQUA, BLUE, DOUBLE_COL, INK_MUTED, ORANGE, SINGLE_COL, VIOLET,
    label_lines, save, use_paper_style,
)
from steering.utils import setup_logging

# Fixed slot order: colour follows the entity, so a figure that drops a series
# must not repaint the others.
PAIR_COLOURS = {
    "tulu3": BLUE,
    "llama3-oh": ORANGE,
    "olmo3": AQUA,
    "tulu3 base->SFT": VIOLET,
}


def _load(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df.set_index("layer") if "layer" in df.columns else df


def fig_displacement_by_depth(runs: Dict[str, pd.DataFrame], out: Path) -> Optional[Path]:
    """Where in the network the displacement actually lives.

    The claim under test is that it is confined to late layers. It is not: it plateaus
    around layer 15 and stays flat.
    """
    if not runs:
        return None
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.3))
    for name, df in runs.items():
        ax.plot(df.index, df["rel_norm"], label=name, color=PAIR_COLOURS[name])

    ax.set_xlabel("Layer")
    ax.set_ylabel(r"$\|\Delta h\| \, / \, \|h\|$")
    ax.set_xlim(0, max(df.index.max() for df in runs.values()))
    ax.set_ylim(bottom=0)
    label_lines(ax, {k: PAIR_COLOURS[k] for k in runs})
    return save(fig, out / "displacement_by_depth")


def fig_effective_rank(runs: Dict[str, pd.DataFrame], out: Path) -> Optional[Path]:
    """How many directions the displacement occupies, against depth.

    Rank-1 would sit on the bottom axis. Nothing does.
    """
    if not runs:
        return None
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.3))
    for name, df in runs.items():
        ax.plot(df.index, df["eff_rank"], label=name, color=PAIR_COLOURS[name])

    ax.axhline(1.0, color=INK_MUTED, linewidth=0.7, linestyle=(0, (3, 3)), zorder=1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Effective rank")
    ax.set_xlim(0, max(df.index.max() for df in runs.values()))
    # llama3-oh spikes to ~35 in the first two layers, which flattens the band where
    # every pair actually sits. Clip and mark it rather than lose the resolution.
    ax.set_ylim(0, 25)

    # Both notes go in the empty band under the curves: "rank 1" hard left where the
    # lines have already climbed away, the clip note to its right.
    ax.annotate("rank 1", xy=(0.04, 1.0), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points",
                fontsize=plt.rcParams["font.size"] - 1.5, color=INK_MUTED)
    ax.annotate("llama3-oh peaks at 35 (layer 0)", xy=(0.42, 0.11), xycoords="axes fraction",
                fontsize=plt.rcParams["font.size"] - 1.5, color=INK_MUTED)
    label_lines(ax, {k: PAIR_COLOURS[k] for k in runs})
    return save(fig, out / "effective_rank_by_depth")


def fig_directional_coherence(runs: Dict[str, pd.DataFrame], out: Path) -> Optional[Path]:
    """How much of each prompt's displacement points along the shared direction.

    This is what bounds a single steering vector: where it is low, no choice of
    lambda reproduces the change.
    """
    if not runs:
        return None
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.3))
    for name, df in runs.items():
        ax.plot(df.index, df["cos_to_mean_mean"], label=name, color=PAIR_COLOURS[name])

    ax.set_xlabel("Layer")
    ax.set_ylabel(r"$\cos(\Delta h_i,\ \bar{v})$")
    ax.set_xlim(0, max(df.index.max() for df in runs.values()))
    ax.set_ylim(0, 1)
    label_lines(ax, {k: PAIR_COLOURS[k] for k in runs})
    return save(fig, out / "directional_coherence")


def fig_readout_position(prompt: pd.DataFrame, response: pd.DataFrame, out: Path) -> Optional[Path]:
    """The same model measured at two readout positions.

    A methods figure: the two disagree enough that a geometry claim is not
    meaningful without saying which token it was read at.
    """
    if prompt is None or response is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL * 0.72, 2.3))

    for ax, col, ylab in [
        (axes[0], "cos_it_dpo_mean", "Cosine similarity"),
        (axes[1], "rel_norm", r"$\|\Delta h\| \, / \, \|h\|$"),
    ]:
        ax.plot(prompt.index, prompt[col], color=BLUE, label="prompt token")
        ax.plot(response.index, response[col], color=ORANGE, label="response token")
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylab)
        ax.set_xlim(0, prompt.index.max())

    axes[0].set_ylim(top=1.0)
    axes[1].set_ylim(bottom=0)
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    return save(fig, out / "readout_position")


def fig_steering_curve(df: pd.DataFrame, out: Path, name: str) -> Optional[Path]:
    """Behaviour against steering strength, with degeneration kept separate.

    Two panels rather than one, because "the model answers differently" and "the model
    stopped producing usable text" are different failures on different units. Folding the
    second into a quality average -- which is what a single quality curve does -- hides
    the collapse inside a number that merely looks lower.
    """
    if df is None or df.empty:
        return None

    fig, (ax, ax_bad) = plt.subplots(
        2, 1, figsize=(SINGLE_COL * 1.35, 3.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
    )

    series = [
        ("steering shift", "steering_shift_mean", BLUE),
        ("helpfulness", "helpfulness_mean", ORANGE),
        ("quality", "quality_mean", AQUA),
        ("refusal", "refusal_mean", VIOLET),
    ]
    for label, col, colour in series:
        if col in df.columns:
            ax.plot(df["lambda"], df[col], label=label, color=colour, marker="o", markersize=2.5)

    # The last strength before generations start breaking, read off the panel below.
    clean = df.loc[df["n_invalid"] <= 0.05 * df["n"], "lambda"]
    if len(clean):
        edge = clean.max()
        ax.axvline(edge, color=INK_MUTED, linewidth=0.7, linestyle=(0, (3, 3)), zorder=1)
        ax.annotate(f"last clean $\\lambda$ = {edge:g}", xy=(edge, 0.04), xytext=(-4, 0),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=plt.rcParams["font.size"] - 1.5, color=INK_MUTED)

    ax.set_ylabel("Judged score")
    ax.set_ylim(0, 1)
    label_lines(ax, {lab: c for lab, _, c in series})

    ax_bad.bar(df["lambda"], df["n_invalid"] / df["n"] * 100,
               width=0.045, color=INK_MUTED, alpha=0.75)
    ax_bad.set_xlabel(r"Steering strength $\lambda$")
    ax_bad.set_ylabel("% broken")
    ax_bad.set_ylim(0, max(5, (df["n_invalid"] / df["n"] * 100).max() * 1.25))

    return save(fig, out / f"steering_curve_{name}")


def main():
    parser = argparse.ArgumentParser(description="Build paper figures from run outputs.")
    parser.add_argument("--runs-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/figures")
    args = parser.parse_args()

    root = Path(args.runs_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out / "make_figures.log")
    use_paper_style()

    # Rank statistics, measured at the response token (the position Eq. 1 specifies).
    rank_dir = root / "analysis" / "response_all4"
    rank_runs = {}
    for name, stem in [
        ("tulu3", "tulu3_dpo"),
        ("llama3-oh", "llama3oh_dpo"),
        ("olmo3", "olmo3_dpo"),
        ("tulu3 base->SFT", "tulu3_basesft"),
    ]:
        df = _load(rank_dir / f"rank_{stem}.csv")
        if df is not None:
            rank_runs[name] = df
        else:
            log.warning(f"missing {rank_dir / f'rank_{stem}.csv'}")

    built = [
        fig_displacement_by_depth(rank_runs, out),
        fig_effective_rank(rank_runs, out),
        fig_directional_coherence(rank_runs, out),
    ]

    # Readout-position comparison, tulu3 only.
    prompt = _load(root / "layer_profile" / "tulu3" / "layer_stats.csv")
    response = _load(root / "layer_profile_response_token" / "tulu3" / "layer_stats.csv")
    for df in (prompt, response):
        if df is not None:
            df["rel_norm"] = df["diff_norm_mean"] / df["it_norm_mean"]
    built.append(fig_readout_position(prompt, response, out))

    # Steering curves, one per scored sweep.
    for summary in sorted(root.glob("steer/*/*/*/scored/summary.csv")):
        parts = summary.parts
        name = f"{parts[-5]}_{parts[-3]}"
        df = pd.read_csv(summary).sort_values("lambda")
        built.append(fig_steering_curve(df, out, name))

    for p in built:
        if p:
            log.info(f"wrote {p}")
    log.info(f"{sum(1 for p in built if p)} figures in {out}")


if __name__ == "__main__":
    main()
