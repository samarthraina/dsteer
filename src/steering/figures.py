"""Publication figure style.

Figures go into a two-column paper, so they are built for that: vector PDF, a serif
face matching the body text, and sizes that stay legible at column width rather than
at screen size.

Colour. The four-series palette below was validated rather than chosen by eye, using
the OKLab deltaE checks (Machado-Oliveira-Fernandes severity-1.0 protan/deutan
simulation) against a white page:

    worst adjacent CVD deltaE   9.2   (target 8.0)
    worst normal-vision deltaE 16.3   (floor 15.0, hard gate)

Two alternatives were rejected on those numbers: swapping violet for red puts orange
and red at deltaE 7.1 under normal vision, and magenta lands at 12.9 -- both below the
floor, i.e. readers cannot reliably tell the pair apart. Okabe-Ito passes the floor but
sits in the CVD warn band at 7.6.

AQUA measures 2.82:1 against white, under the 3.0 contrast gate. The mitigation is
direct labelling, which `label_lines` does, so identity never rests on colour alone.

Do not add a fifth series colour. Past four the all-pairs floors cannot be met; fold
into "other", facet, or use small multiples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical slots, in fixed order. Assign by position, never cycle.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
SERIES = (BLUE, ORANGE, AQUA, VIOLET)

# Chrome. Recessive by design: the marks carry the data, the axes stay out of the way.
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Column widths for a two-column paper, in inches.
SINGLE_COL = 3.35
DOUBLE_COL = 7.0


def use_paper_style(base_size: float = 8.0) -> None:
    """Set rcParams for figures embedded in a two-column paper."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,          # embed TrueType, so the PDF is not bitmapped
        "ps.fonttype": 42,

        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base_size,
        "axes.titlesize": base_size + 1,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 1,

        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,      # grid behind the marks, never over them
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 1.0,

        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,

        "lines.linewidth": 1.4,
        "lines.markersize": 3.0,

        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.2,
        "legend.labelspacing": 0.3,
    })


def label_lines(
    ax,
    labels: Dict[str, str],
    x_frac: float = 1.01,
    size: Optional[float] = None,
    leader: bool = True,
) -> None:
    """Direct-label each series at the right edge of the axes.

    Carries identity without a legend box, and satisfies the relief rule for the
    palette slots that sit under 3:1 against a white page.

    Series that end at similar values would otherwise print on top of each other --
    which happens whenever curves converge, so it cannot be left to chance. Labels
    are pushed apart to a readable minimum gap, and any label moved off its line
    gets a hairline leader so it still reads as belonging to that curve.

    labels: {series label: colour}.
    """
    entries = []
    for line in ax.get_lines():
        name = line.get_label()
        if name not in labels:
            continue
        ys = line.get_ydata()
        if len(ys):
            entries.append([name, float(ys[-1]), labels[name]])
    if not entries:
        return

    fontsize = size or plt.rcParams["font.size"] - 1
    entries.sort(key=lambda e: e[1])
    placed = [e[1] for e in entries]

    # Work out the minimum readable gap in data units, via the rendered axes height.
    ax.figure.canvas.draw()
    y0, y1 = ax.get_ylim()
    height_px = ax.get_window_extent().height
    if height_px > 0 and y1 != y0:
        gap_px = fontsize * 1.35 * ax.figure.dpi / 72.0
        min_gap = gap_px * (y1 - y0) / height_px

        # One upward pass, then clamp back inside the axes if the stack overflowed.
        for i in range(1, len(placed)):
            if placed[i] - placed[i - 1] < min_gap:
                placed[i] = placed[i - 1] + min_gap
        overflow = placed[-1] - y1
        if overflow > 0:
            placed = [p - overflow for p in placed]

    for (name, y_line, colour), y_label in zip(entries, placed):
        ax.annotate(
            name,
            xy=(x_frac, y_label), xycoords=("axes fraction", "data"),
            va="center", ha="left", color=colour, fontsize=fontsize,
            annotation_clip=False,
        )
        # Only draw a leader when the label had to move off its own line.
        if leader and abs(y_label - y_line) > 1e-9:
            ax.annotate(
                "", xy=(x_frac - 0.005, y_label), xytext=(1.0, y_line),
                xycoords=("axes fraction", "data"), textcoords=("axes fraction", "data"),
                arrowprops=dict(arrowstyle="-", color=colour, linewidth=0.5, alpha=0.55),
                annotation_clip=False,
            )


def save(fig, path: Union[str, Path], also_png: bool = True) -> Path:
    """Write vector PDF for the paper, and a PNG for quick viewing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = path.with_suffix(".pdf")
    fig.savefig(pdf)
    if also_png:
        fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return pdf
