"""STEP 10 - Test the two thresholds (accept / quarantine / recall).

The paper's Layer 3 uses TWO thresholds instead of one cut-off:
    risk < t_low            -> ACCEPT      (stored)
    t_low <= risk < t_high  -> QUARANTINE  (held for review)
    risk >= t_high          -> RECALL      (purged + recall message)
Question: compared with a single 0.5 cut-off, do fewer entries end up
in the WRONG final place (poison accepted, or clean entry recalled)?

Uses D3 (and D1 for comparison) at medium overlap, 5 seeds, test split.
Nothing is changed in earlier results.

Output: results/tables/two_thresholds.csv
Run:    python s10_two_thresholds.py      (CPU: about 5 minutes)
"""
import numpy as np
import pandas as pd

from common import VARIANTS, FusionDetector, fit_stage1, load
from config import SEEDS, MAIN_OVERLAP, TABLES

PAIRS = [(0.5, 0.5), (0.3, 0.7), (0.2, 0.8)]       # (0.5, 0.5) = single cut-off
USE = [k for k in VARIANTS if k.startswith(("D1", "D3"))]

rows = []
for seed in SEEDS:
    df, emb = load(MAIN_OVERLAP, seed)
    tr, te = df.split.values == "train", df.split.values == "test"
    y = df.label.values
    st1 = fit_stage1(emb[tr], y[tr], seed)
    for name in USE:
        det = FusionDetector(VARIANTS[name], seed).fit(emb[tr], df[tr], st1)
        p = det.predict_proba(emb[te], df[te])
        yt = y[te]
        n_clean, n_poison = (yt == 0).sum(), (yt == 1).sum()
        for lo, hi in PAIRS:
            acc_z = p < lo
            rec_z = p >= hi
            qua_z = ~acc_z & ~rec_z
            rows.append(dict(
                variant=name, seed=seed, t_low=lo, t_high=hi,
                poison_accepted_pct=100 * (acc_z & (yt == 1)).sum() / n_poison,   # missed poison
                clean_recalled_pct=100 * (rec_z & (yt == 0)).sum() / n_clean,     # false recall
                poison_quarantined_pct=100 * (qua_z & (yt == 1)).sum() / n_poison,
                clean_quarantined_pct=100 * (qua_z & (yt == 0)).sum() / n_clean,
                poison_recalled_pct=100 * (rec_z & (yt == 1)).sum() / n_poison,
                clean_accepted_pct=100 * (acc_z & (yt == 0)).sum() / n_clean,
                quarantine_share_pct=100 * qua_z.mean(),                        # review load
            ))
        print(seed, name, "done")

r = pd.DataFrame(rows)
cols = [c for c in r.columns if c.endswith("_pct")]
summary = r.groupby(["variant", "t_low", "t_high"], sort=False)[cols].mean().round(2)
summary.to_csv(TABLES / "two_thresholds.csv")
pd.set_option("display.width", 220)
print("\nTWO-THRESHOLD RESULT (mean over 5 seeds, % of test entries)\n", summary)
