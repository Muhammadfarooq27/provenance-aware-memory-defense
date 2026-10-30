"""STEP 16 - Embedding-threshold baseline on the memshield-bench subset (re-scoring only).

Reviewer request: Table 2 shows F1 on the memshield-bench subset (column F1-dagger)
for D1-D3 but not for the embedding-threshold baseline.

This script repeats EXACTLY the threshold rule of step 5 (same centroid, same
threshold chosen on the validation split) and then scores the test entries that
come from memshield-bench, using the same subset rule as step 4
(df.source == "memshield").

No training, no LLM, no new experiment: it only reads existing files.
It first re-prints the full-test numbers so they can be checked against Table 2
(expected: accuracy 0.833, F1 0.782, AUC 0.893, FPR 0.194).

Output: results/tables/threshold_memshield.csv  (baselines.csv is NOT touched)
Run:    python s16_threshold_memshield.py        (a few seconds)
"""
import numpy as np
import pandas as pd

from common import metrics
from config import PROC, TABLES

df = pd.read_csv(PROC / "dataset.csv")
emb = np.load(PROC / "embeddings.npy")
tr, va, te = (df.split.values == s for s in ("train", "val", "test"))
y = df.label.values

# same rule as s05_baselines.py
centroid = emb[tr & (y == 0)].mean(0)
centroid /= np.linalg.norm(centroid)
score = 1 - emb @ centroid
best_t = max(np.quantile(score[va], np.linspace(0.05, 0.95, 91)),
             key=lambda t: metrics(y[va], (score[va] > t).astype(int))["f1"])

full = metrics(y[te], (score[te] > best_t).astype(int), score[te])
ms = te & (df.source.values == "memshield")
sub = metrics(y[ms], (score[ms] > best_t).astype(int), score[ms])

print(f"threshold chosen on validation: {best_t:.4f}")
print("FULL TEST   (check vs Table 2):", {k: round(v, 3) for k, v in full.items()})
print(f"MEMSHIELD SUBSET (n={int(ms.sum())}, poisoned={int(y[ms].sum())}):",
      {k: round(v, 3) for k, v in sub.items()})

pd.DataFrame([dict(part="full_test", n=int(te.sum()), **full),
              dict(part="memshield_subset", n=int(ms.sum()), **sub)]
             ).to_csv(TABLES / "threshold_memshield.csv", index=False)
print("saved results/tables/threshold_memshield.csv")
