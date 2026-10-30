"""STEP 4 - Train detectors D1, D2, D3 at every overlap level, 5 seeds.

Output:
  results/logs/detectors_all_runs.csv   (one row per variant x level x seed)
  results/tables/table2_detectors.csv   (mean +/- std at MAIN_OVERLAP)
  models/d3_<MAIN_OVERLAP>_seed0.joblib (one saved model for Step 7)
Run:    python s04_train_detectors.py   (CPU: about 5-15 minutes)
"""
import joblib
import pandas as pd

from common import VARIANTS, FusionDetector, fit_stage1, load, metrics
from config import SEEDS, OVERLAP_LEVELS, MAIN_OVERLAP, LOGS, TABLES, MODELS

runs, stage1 = [], {}
for level in OVERLAP_LEVELS:
    for seed in SEEDS:
        df, emb = load(level, seed)
        tr, te = df.split.values == "train", df.split.values == "test"
        y = df.label.values
        if seed not in stage1:                 # text model: once per seed
            stage1[seed] = fit_stage1(emb[tr], y[tr], seed)
        for name, cols in VARIANTS.items():
            det = FusionDetector(cols, seed).fit(emb[tr], df[tr], stage1[seed])
            p = det.predict_proba(emb[te], df[te])
            m = metrics(y[te], (p >= 0.5).astype(int), p)
            # detection rate on the hardest subset: signed-but-poisoned
            sp = df[te].signed_poison.values == 1
            m["signed_poison_recall"] = (p[sp] >= 0.5).mean() if sp.any() else float("nan")
            # sanity check on memshield only (clean and poison from the same domains)
            ms = df[te].source.values == "memshield"
            m["f1_memshield_only"] = metrics(y[te][ms], (p[ms] >= 0.5).astype(int))["f1"] if ms.any() else float("nan")
            m["auc_memshield_only"] = metrics(y[te][ms], (p[ms] >= 0.5).astype(int), p[ms]).get("auc", float("nan")) if ms.any() else float("nan")
            runs.append(dict(variant=name, overlap=level, seed=seed, **m))
            print(level, seed, name, {k: round(v, 3) for k, v in m.items()})
            if level == MAIN_OVERLAP and seed == 0 and name.startswith("D3"):
                joblib.dump(det, MODELS / f"d3_{MAIN_OVERLAP}_seed0.joblib")

runs = pd.DataFrame(runs)
runs.to_csv(LOGS / "detectors_all_runs.csv", index=False)

cols = ["accuracy", "f1", "auc", "fpr", "signed_poison_recall", "f1_memshield_only", "auc_memshield_only"]
main = runs[runs.overlap == MAIN_OVERLAP]
table = main.groupby("variant", sort=False)[cols].agg(["mean", "std"])
table.to_csv(TABLES / "table2_detectors.csv")
pd.set_option("display.width", 200)
print("\nTABLE 2 (overlap = %s)\n" % MAIN_OVERLAP, table.round(3))
print("\nF1 and AUC by overlap level (mean over seeds)\n",
      runs.groupby(["overlap", "variant"], sort=False)[["f1", "auc", "f1_memshield_only"]].mean().round(3))
