"""STEP 22 - Figure 3 with two parts (no new experiment).

(a) Share of agents holding a poisoned entry over time, meshes of 25, 50, 75 and 100 agents
    (results/logs/recall_scale_curves.csv from step 12). Colour = mesh size; dotted = without recall,
    solid = with recall (measured detector), dashed = with recall (weak detector, 0.25). Mean of 100 runs x 5 seeds.
(b) Peak share of infected agents per mesh size (results/tables/recall_scale.csv from step 12);
    number on each bar = steps to clear the mesh after detection.
Output: results/figures/fig3_recall_column.pdf/.png (parts stacked, one column)
        results/figures/fig3_recall_wide.pdf/.png   (parts side by side, full page width)
Run:    python s22_fig3_recall.py      or      python s22_fig3_recall.py curves.csv scale.csv outdir
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

if len(sys.argv) == 4:
    curves, scale, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
else:
    from config import LOGS, TABLES, FIGS
    curves, scale, out = LOGS / "recall_scale_curves.csv", TABLES / "recall_scale.csv", FIGS
MAXSTEP = 20

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
GRID, BLUE, GREEN = "#e4e3df", "#2a78d6", "#1baf7a"

rc = pd.read_csv(curves)
rc = rc[rc.step <= MAXSTEP].copy()
rc["pct"] = 100 * rc.holders / rc.agents
c = rc.groupby(["agents", "mode", "step"]).pct.mean().reset_index()
sizes = sorted(c.agents.unique())
SHADE = dict(zip(sizes, ["#9ecae1", "#4292c6", "#2171b5", "#08306b"]))     # light -> dark = small -> large
modes = list(dict.fromkeys(rc["mode"]))                                       # without, measured, weak
sc = pd.read_csv(scale)
good = sc[sc["mode"].str.contains("0.88")].set_index("agents").loc[sizes]
weak = sc[sc["mode"].str.contains("0.25")].set_index("agents").loc[sizes]


def part_a(ax):
    for n in sizes:
        for mode, ls, lw in zip(modes, [":", "-", "--"], [1.2, 1.7, 1.3]):
            d = c[(c.agents == n) & (c["mode"] == mode)]
            ax.plot(d.step, d.pct, color=SHADE[n], ls=ls, lw=lw)
    ax.set_xlabel("Time step"); ax.set_ylabel("Agents holding the entry (%)")
    ax.set_xlim(0, MAXSTEP); ax.set_ylim(0, 102); ax.set_xticks(range(0, MAXSTEP + 1, 5))
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    leg = ax.legend([Line2D([], [], color=SHADE[n], lw=2.5) for n in sizes], [f"{n} agents" for n in sizes],
                    frameon=False, fontsize=6.5, loc="lower right", bbox_to_anchor=(1.0, 0.36),
                    title="Mesh size", title_fontsize=6.5)
    ax.add_artist(leg)
    ax.legend([Line2D([], [], color="black", ls=ls, lw=1.4) for ls in [":", "-", "--"]],
              ["Without recall", "With recall (detection 0.88)", "With recall (detection 0.25)"],
              frameon=False, fontsize=6.5, loc="lower right", bbox_to_anchor=(1.0, 0.07))
    ax.set_title("(a) Spread over time", fontsize=8, loc="left")


def part_b(ax):
    x, w = np.arange(len(sizes)), 0.36
    for dx, df, col, lab in [(-w / 2, good, BLUE, "With recall (detection 0.88)"),
                             (w / 2, weak, GREEN, "With recall (detection 0.25)")]:
        ax.bar(x + dx, df.peak_pct_of_mesh, w, color=col, label=lab)
        for xi, p, s in zip(x + dx, df.peak_pct_of_mesh, df.steps_to_clear):
            ax.text(xi, p + 0.8, f"{s:.1f}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlabel("Mesh size (agents)"); ax.set_ylabel("Peak infected agents (%)")
    ax.set_ylim(0, 40); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.set_title("(b) Peak infection (number on bar = steps to clear)", fontsize=8, loc="left")


out.mkdir(parents=True, exist_ok=True)
for name, size, grid in [("column", (3.5, 4.6), (2, 1)), ("wide", (7.16, 2.5), (1, 2))]:
    fig, axes = plt.subplots(*grid, figsize=size)
    part_a(axes[0]); part_b(axes[1])
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig3_recall_{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
print("saved to", out)
