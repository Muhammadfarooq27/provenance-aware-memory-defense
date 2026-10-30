"""STEP 9 - Turn the saved results into paper-ready tables and figures.

Reads:  results/logs/*.csv, results/tables/*.csv
Writes: results/tables/table2.tex   rows for TABLE 2
            columns: Model & Accuracy & F1 & AUC & FPR & F1 (memshield)
        results/tables/table3.tex   rows for TABLE 3
            columns: Device & ms/entry & MB & bytes/msg & sign+verify ms & recall steps
        results/figures/fig2_detection.pdf/.png   memshield-only F1 by overlap level
        results/figures/fig3_recall.pdf/.png      agents holding poison over time
Run:    python s09_make_tables_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import LOGS, TABLES, FIGS, MAIN_OVERLAP, N_AGENTS

plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = ["#2a78d6", "#eb6834", "#1baf7a"]            # colour-blind safe order
GRID = "#e4e3df"
SHORT = {"D1: Content only": "D1: Content", "D2: Content + provenance": "D2: + Provenance",
         "D3: Content + prov. + network": "D3: + Network"}


def pm(r, k, d=3):
    return f"{r[(k, 'mean')]:.{d}f} $\\pm$ {r[(k, 'std')]:.{d}f}"


# ---------------- TABLE 2 ----------------
runs = pd.read_csv(LOGS / "detectors_all_runs.csv")
variants = list(dict.fromkeys(runs.variant))
main = runs[runs.overlap == MAIN_OVERLAP].groupby("variant", sort=False)
agg = main[["accuracy", "f1", "auc", "fpr", "f1_memshield_only"]].agg(["mean", "std"])
lines = [f"{v} & {pm(r,'accuracy')} & {pm(r,'f1')} & {pm(r,'auc')} & {r[('fpr','mean')]:.3f} & "
         f"{pm(r,'f1_memshield_only')} \\\\" for v, r in agg.iterrows()]
base = TABLES / "baselines.csv"
if base.exists():
    for _, r in pd.read_csv(base).iterrows():
        auc = f"{r['auc']:.3f}" if "auc" in r and pd.notna(r["auc"]) else "--"
        lines.append(f"{r['method']} & {r['accuracy']:.3f} & {r['f1']:.3f} & {auc} & {r['fpr']:.3f} & -- \\\\")
(TABLES / "table2.tex").write_text("\n".join(lines) + "\n")
print("TABLE 2 rows:\n" + "\n".join(lines))

# ---------------- TABLE 3 ----------------
rec = pd.read_csv(TABLES / "recall_summary.csv", index_col=0)
measured_mode = [m for m in rec.index if m.startswith("With recall")][0]   # first = measured detector
clear = rec.loc[measured_mode, "steps_to_clear"]
edge = TABLES / "table3_edge.csv"
if edge.exists():
    e = pd.read_csv(edge).drop_duplicates("device", keep="last")
    t3 = [f"{r.device} & {r.detector_ms_median:.1f} & {r.model_mb:.0f} & "
          f"{r.envelope_bytes:.0f} & {r.sign_ms + r.verify_ms:.2f} & {clear:.1f} \\\\" for r in e.itertuples()]
    (TABLES / "table3.tex").write_text("\n".join(t3) + "\n")
    print("TABLE 3 rows:\n" + "\n".join(t3))
print("\nRecall summary (mean over runs):\n", rec.round(2))

# ---------------- FIGURE 2: memshield-only F1 by overlap level ----------
order = ["low", "medium", "high"]
labels = {"low": "Low overlap\n(20%)", "medium": "Medium overlap\n(50%)", "high": "High overlap\n(80%)"}
g = runs.groupby(["overlap", "variant"]).f1_memshield_only.agg(["mean", "std"])
fig, ax = plt.subplots(figsize=(3.5, 2.3))
x, w = np.arange(len(order)), 0.26
for i, v in enumerate(variants):
    m = [g.loc[(o, v), "mean"] for o in order]; s = [g.loc[(o, v), "std"] for o in order]
    ax.bar(x + (i - 1) * w, m, w - 0.03, yerr=s, color=COL[i], label=SHORT.get(v, v),
           capsize=2, error_kw={"elinewidth": 0.7, "ecolor": "#52514e"}, zorder=3)
ax.set_xticks(x); ax.set_xticklabels([labels[o] for o in order])
lo = max(0.0, np.floor((g["mean"].min() - 0.08) * 20) / 20)
hi = min(1.0, np.ceil((g["mean"].max() + 0.06) * 20) / 20)
ax.set_ylim(lo, hi); ax.set_ylabel("F1 on memshield-bench entries")
ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=3, fontsize=6.3, loc="upper center",
          bbox_to_anchor=(0.5, 1.17), handlelength=1.0, columnspacing=0.8)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIGS / f"fig2_detection.{ext}", dpi=300, bbox_inches="tight")

# ---------------- FIGURE 3: recall curves (mean + std band) ---------------
rc = pd.read_csv(LOGS / "recall_curves.csv")
modes = list(dict.fromkeys(rc["mode"]))                 # without, measured, weak
c = rc.groupby(["mode", "step"]).holders.agg(["mean", "std"]).reset_index()
style = [(COL[1], "o", "-"), (COL[0], "s", "-"), (COL[2], "^", "--")]
fig, ax = plt.subplots(figsize=(3.5, 2.2))
for (mode, (col, mk, ls)) in zip(modes, style):
    d = c[c["mode"] == mode]
    ax.plot(d.step, d["mean"], color=col, lw=1.8, ls=ls, marker=mk, ms=2.5, label=mode)
    ax.fill_between(d.step, (d["mean"] - d["std"]).clip(lower=0), d["mean"] + d["std"],
                    color=col, alpha=0.12, lw=0)
ax.set_xlabel("Time step"); ax.set_ylabel(f"Agents holding poison (of {N_AGENTS})")
ax.set_ylim(0, N_AGENTS + 1)
ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=6.3, loc="center right")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIGS / f"fig3_recall.{ext}", dpi=300, bbox_inches="tight")
print("figures saved to", FIGS)
