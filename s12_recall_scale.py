"""STEP 12 - Does network-wide recall scale to bigger meshes?

Same recall simulation as step 8 (same spread, use and detection rules),
repeated for meshes of 25, 50 and 100 agents. We also count the RECALL
messages: every agent that receives the recall forwards it once to each
of its neighbours.

The 25-agent row should be close to step 8 (small differences come only
from the random-number order).

Output: results/tables/recall_scale.csv
Run:    python s12_recall_scale.py      (seconds to a few minutes)
"""
import networkx as nx
import numpy as np
import pandas as pd

from config import LOGS, TABLES, SEEDS, MAIN_OVERLAP

SIZES = [25, 50, 100]
T, SHARE, P_USE, RUNS = 30, 0.35, 0.15, 100        # T longer than step 8 so big meshes can finish
runs_csv = pd.read_csv(LOGS / "detectors_all_runs.csv")
MEASURED = runs_csv[(runs_csv.overlap == MAIN_OVERLAP) & runs_csv.variant.str.startswith("D3")].recall.mean()
DETECT_LEVELS = {"measured": MEASURED, "weak": 0.25}
print(f"detection chances: measured={MEASURED:.3f}, weak=0.25")


def simulate(G, n, rng, use_recall, detect):
    holders, purged, front = {int(rng.integers(n))}, set(), set()
    detected_at = cleared_at = None
    peak = 1
    for t in range(T):
        holders |= {b for a in holders for b in G.neighbors(a)
                    if b not in holders and b not in purged and rng.random() < SHARE}
        if use_recall:
            if detected_at is None:
                flagged = [a for a in holders if rng.random() < P_USE * detect]
                if flagged:
                    detected_at, front = t, {flagged[0]}
            if detected_at is not None:
                reached = front | {b for a in front for b in G.neighbors(a)}
                purged |= reached
                holders -= purged
                front = reached
                if not holders and cleared_at is None:
                    cleared_at = t
        peak = max(peak, len(holders))
    messages = sum(G.degree(a) for a in purged)           # each reached agent forwards once
    return dict(detected_at=detected_at, peak_holders=peak, final_holders=len(holders),
                steps_to_clear=(cleared_at - detected_at) if cleared_at is not None and detected_at is not None else np.nan,
                recall_messages=messages if use_recall else 0)


rows = []
for n in SIZES:
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        G = nx.connected_watts_strogatz_graph(n, k=4, p=0.2, seed=seed)
        modes = [("Without recall", False, MEASURED)] + [
            (f"With recall (detection {d:.2f})", True, d) for d in DETECT_LEVELS.values()]
        for mode, use, d in modes:
            for _ in range(RUNS):
                rows.append(dict(agents=n, mode=mode, seed=seed, **simulate(G, n, rng, use, d)))
    print("done", n, "agents")

r = pd.DataFrame(rows)
summ = r.groupby(["agents", "mode"], sort=False)[
    ["detected_at", "steps_to_clear", "peak_holders", "final_holders", "recall_messages"]].mean()
summ["peak_pct_of_mesh"] = 100 * summ.peak_holders / summ.index.get_level_values("agents")
summ["messages_per_agent"] = summ.recall_messages / summ.index.get_level_values("agents")
summ = summ.round(2)
summ.to_csv(TABLES / "recall_scale.csv")
pd.set_option("display.width", 220)
print("\nRECALL AT SCALE (mean of 100 runs x 5 seeds)\n", summ)
