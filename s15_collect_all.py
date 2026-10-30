"""STEP 15 - Print every final number in one place (nothing is computed here).

Run:    python s15_collect_all.py > results/FINAL_NUMBERS.txt
Then send results/FINAL_NUMBERS.txt.
"""
import pandas as pd

from config import PROC, TABLES, LOGS

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)


def show(title, path, **kw):
    print("\n" + "=" * 100 + f"\n{title}\n  ({path.name})\n" + "=" * 100)
    if not path.exists():
        print("MISSING"); return
    print(pd.read_csv(path, **kw).to_string())


df = pd.read_csv(PROC / "dataset.csv")
print("DATASET:", len(df), "entries |", int(df.label.sum()), "poisoned")
print(df.groupby(["source", "split", "label"]).size().unstack(fill_value=0).to_string())
if "group" in df:
    leak = df[df.group.notna()].groupby("group").split.nunique().gt(1).sum()
    print("PoisonedRAG questions split across splits:", int(leak))

show("TABLE 2 - detectors at medium overlap (mean/std over 5 seeds)", TABLES / "table2_detectors.csv", header=[0, 1], index_col=0)
runs = pd.read_csv(LOGS / "detectors_all_runs.csv")
print("\nF1 / AUC / FPR / recall / memshield F1 by overlap (mean over seeds)")
print(runs.groupby(["overlap", "variant"], sort=False)[["f1", "auc", "fpr", "recall", "f1_memshield_only", "signed_poison_recall"]].mean().round(3).to_string())
show("BASELINES (threshold, LLM judge, D3 on same 200)", TABLES / "baselines.csv")
syn = pd.read_csv(LOGS / "synonym_runs.csv")
print("\nSYNONYM ATTACK (mean/std)\n", syn.groupby("variant", sort=False)[["f1", "auc", "recall", "fpr"]].agg(["mean", "std"]).round(3).to_string())
show("LLM PARAPHRASE", TABLES / "paraphrase_summary.csv", header=[0, 1], index_col=0)
show("TWO THRESHOLDS", TABLES / "two_thresholds.csv")
show("RECALL (25 agents)", TABLES / "recall_summary.csv")
show("RECALL AT SCALE", TABLES / "recall_scale.csv")
show("LAUNDERING", TABLES / "laundering_summary.csv", header=[0, 1], index_col=[0, 1, 2])
show("END-TO-END", TABLES / "e2e_summary.csv")
show("EDGE COST (not rerun)", TABLES / "table3_edge.csv")
