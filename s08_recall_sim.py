"""STEP 8 - Network-wide recall: does the RECALL message clean the mesh?

One poisoned entry starts at one agent and spreads to neighbours.
  * WITHOUT recall: it keeps spreading.
  * WITH recall: an agent only checks the entry when it USES it (chance
    P_USE per step). The check flags it with the detection recall of D3 from
    Step 4. The flagging agent then sends RECALL(entry, lineage).
    Each step the recall travels one hop; every agent it reaches deletes the
    entry AND its derived copies, and refuses the entry afterwards.
We repeat 100 runs per seed and average. As a sensitivity check we also
run recall with a weak detector (detection chance 0.25).

Output: results/logs/recall_curves.csv, results/tables/recall_summary.csv
Run:    python s08_recall_sim.py   (seconds)
"""
import networkx as nx
import numpy as np
import pandas as pd

from config import LOGS, TABLES, SEEDS, N_AGENTS, MAIN_OVERLAP

T, SHARE, P_USE, RUNS = 20, 0.35, 0.15, 100
runs = pd.read_csv(LOGS / "detectors_all_runs.csv")
MEASURED = runs[(runs.overlap == MAIN_OVERLAP) & runs.variant.str.startswith("D3")].recall.mean()
print(f"per-agent detection chance (D3 recall) = {MEASURED:.3f}")
# Sensitivity check: the measured recall is high partly because PoisonedRAG
# texts are easy to spot. We also run a deliberately weak detector (0.25) to
# show recall still contains the poison when detection is much harder.
DETECT_LEVELS = {"measured": MEASURED, "weak": 0.25}


def simulate(G, rng, use_recall, DETECT=MEASURED):
    holders, purged, recall_front = {int(rng.integers(N_AGENTS))}, set(), set()
    curve, detected_at, cleared_at = [], None, None
    for t in range(T):
        # 1) spread
        new = {b for a in holders for b in G.neighbors(a)
               if b not in holders and b not in purged and rng.random() < SHARE}
        holders |= new
        if not use_recall:
            curve.append(len(holders))
            continue
        # 2) an agent uses the entry and its detector flags it
        if detected_at is None:
            flagged = [a for a in holders if rng.random() < P_USE * DETECT]
            if flagged:
                detected_at = t
                recall_front = {flagged[0]}            # recall starts at that agent
        # 3) recall travels one hop per step along links
        if detected_at is not None:
            reached = recall_front | {b for a in recall_front for b in G.neighbors(a)}
            purged |= reached
            holders -= purged
            recall_front = reached
            if not holders and cleared_at is None:
                cleared_at = t
        curve.append(len(holders))
    return [1] + curve, detected_at, cleared_at


MODES = [("Without recall", False, MEASURED)] + [
    (f"With recall (detection {d:.2f})", True, d) for d in DETECT_LEVELS.values()]
rows, summ = [], []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    G = nx.connected_watts_strogatz_graph(N_AGENTS, k=4, p=0.2, seed=seed)
    for mode, use, d in MODES:
        for r in range(RUNS):
            curve, det, clr = simulate(G, rng, use, d)
            rows += [dict(mode=mode, seed=seed, run=r, step=t, holders=h) for t, h in enumerate(curve)]
            summ.append(dict(mode=mode, seed=seed, detected_at=det, final_holders=curve[-1],
                             steps_to_clear=(clr - det) if clr is not None and det is not None else np.nan,
                             peak_holders=max(curve)))

pd.DataFrame(rows).to_csv(LOGS / "recall_curves.csv", index=False)
s = pd.DataFrame(summ)
summary = s.groupby("mode", sort=False)[["detected_at", "steps_to_clear", "peak_holders", "final_holders"]].mean()
summary.to_csv(TABLES / "recall_summary.csv")
print(summary.round(2))
