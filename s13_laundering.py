"""STEP 13 - Laundering test: what if the poison's ORIGIN is hidden?

Idea (from TMA-NM, arXiv:2606.24322): an attacker can "launder" a poisoned
entry. The agent rewrites the poison as its OWN note, so the entry now looks
like a self-written note from an honest, trusted agent, with a normal lineage
and normal spread. Provenance and network signals then look benign.

We test three conditions on the POISONED test entries (clean entries are
never changed, so the false-alarm rate on clean entries stays the same):
  none : no laundering (the normal test, for reference)
  prov : provenance + network features laundered, original text
  full : provenance + network laundered AND text rewritten by the LLM
         (rewrites taken from step 11's cache, data/processed/paraphrase_cache.csv)

How the features are laundered: for every poisoned test entry we copy the
9 provenance/network values of a randomly chosen CLEAN test entry and set the
channel to 'self' (the agent wrote it). So the side features carry no signal.

Detectors D1-D3 are trained on the ORIGINAL data only (as in steps 6 and 11).
We also report one fixed rule decided BEFORE running, taken from Layer 3:
  "content veto": an entry is flagged if the fused score >= 0.5 OR the
  content score alone >= 0.7 (the paper's upper threshold).
It is applied to every condition, including clean entries, so its cost in
false alarms is visible.

Output: results/logs/laundering_runs.csv, results/tables/laundering_summary.csv
Run:    python s13_laundering.py      (CPU: about 5-10 minutes)
"""
import numpy as np
import pandas as pd

from common import VARIANTS, FusionDetector, fit_stage1, load, metrics
from config import PROC, LOGS, TABLES, SEEDS, MAIN_OVERLAP, EMBED_MODEL, PROV_COLS, NET_COLS

SIDE = PROV_COLS + NET_COLS
VETO = 0.7                                   # Layer-3 upper threshold
CACHE = PROC / "paraphrase_cache.csv"

base = pd.read_csv(PROC / "dataset.csv")
target = ((base.split == "test") & (base.label == 1)).values

# ---- rewritten poison from step 11 (optional: 'full' is skipped if missing) ----
emb_full = None
if CACHE.exists():
    from sentence_transformers import SentenceTransformer
    cache = pd.read_csv(CACHE).drop_duplicates("uid").set_index("uid")
    uids = base.loc[target, "uid"]
    missing = int((~uids.isin(cache.index)).sum())
    texts = [cache.at[u, "paraphrase"] if u in cache.index else t
             for u, t in zip(uids, base.loc[target, "text"])]
    print(f"rewrites found for {len(uids) - missing}/{len(uids)} poisoned test entries")
    emb_full = np.load(PROC / "embeddings.npy").copy()
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    emb_full[target] = model.encode(texts, batch_size=64, normalize_embeddings=True)
else:
    print("WARNING: paraphrase_cache.csv not found -> condition 'full' is skipped")


def launder(df_te, rng):
    """Copy side features of random clean test entries onto the poisoned ones."""
    out = df_te.copy()
    pois = out.label.values == 1
    donors = out.loc[~pois, SIDE].to_numpy()
    pick = donors[rng.integers(len(donors), size=int(pois.sum()))]
    out.loc[pois, SIDE] = pick
    out.loc[pois, ["ch_user", "ch_tool", "ch_peer"]] = 0
    out.loc[pois, "ch_self"] = 1
    return out


runs = []
for seed in SEEDS:
    df_o, emb_o = load(MAIN_OVERLAP, seed)
    tr, te = df_o.split.values == "train", df_o.split.values == "test"
    y = df_o.label.values
    st1 = fit_stage1(emb_o[tr], y[tr], seed)
    df_l = launder(df_o[te], np.random.default_rng(100 + seed))
    conds = {"none": (emb_o[te], df_o[te]), "prov": (emb_o[te], df_l)}
    if emb_full is not None:
        conds["full"] = (emb_full[te], df_l)
    for name, cols in VARIANTS.items():
        det = FusionDetector(cols, seed).fit(emb_o[tr], df_o[tr], st1)
        for cond, (E, D) in conds.items():
            p = det.predict_proba(E, D)
            content = det.content.predict_proba(E)[:, 1]
            for rule, pred in (("fused", p >= 0.5), ("content veto", (p >= 0.5) | (content >= VETO))):
                m = metrics(y[te], pred.astype(int), p if rule == "fused" else None)
                runs.append(dict(variant=name, condition=cond, rule=rule, seed=seed, **m))
        print(seed, name, "done")

runs = pd.DataFrame(runs)
runs.to_csv(LOGS / "laundering_runs.csv", index=False)
summ = runs.groupby(["condition", "rule", "variant"], sort=False)[["f1", "recall", "fpr"]] \
           .agg(["mean", "std"]).round(3)
summ.to_csv(TABLES / "laundering_summary.csv")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
print("\nLAUNDERING RESULT (trained on original data; recall = share of poison caught)\n", summ)
