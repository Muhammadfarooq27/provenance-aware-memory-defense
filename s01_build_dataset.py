"""STEP 1 - Build one labelled dataset from public sources.

Output: data/processed/dataset.csv with columns
    uid, text, label (1 = poisoned, 0 = clean), source, split, group

Split rule (updated 19 Oct 2026): PoisonedRAG gives 5 near-identical poisoned
passages per question. All passages of one question share a 'group' and are
put in the SAME split, so the test set never contains a sibling of a training
passage. memshield-bench keeps its official splits; other rows are split one
by one (70/15/15, class ratio kept).
Run:    python s01_build_dataset.py
"""
import json
import random

import pandas as pd

from config import RAW, PROC

random.seed(42)
rows = []


def add(text, label, source, split=None, group=None):
    text = " ".join(str(text).split())          # tidy whitespace
    if 30 <= len(text) <= 1200:                 # drop very short / very long
        rows.append({"text": text, "label": label, "source": source, "split": split, "group": group})


# --- A. memshield-bench (main source: labelled, official splits) ----------
try:
    from datasets import load_dataset
    ms = load_dataset("npow/memshield-bench")
    for split in ("train", "validation", "test"):
        for e in ms[split]:
            add(e["content"], 1 if e["label"] == "poisoned" else 0,
                "memshield", {"validation": "val"}.get(split, split))
    print("memshield-bench loaded")
except Exception as err:                         # no internet, wrong name, ...
    print("WARNING memshield-bench not loaded:", err)

# --- B. PoisonedRAG poisoned texts (extra poison) --------------------------
for name in ("nq", "hotpotqa", "msmarco"):
    f = RAW / "poisonedrag" / f"{name}.json"
    if not f.exists():
        print("WARNING missing", f); continue
    for qid, item in json.load(open(f)).items():
        for t in item["adv_texts"]:
            add(t, 1, f"poisonedrag_{name}", group=f"{name}_{qid}")

# --- C. LongMemEval (clean background memory: user turns) ------------------
f = RAW / "longmemeval" / "longmemeval_oracle.json"
if f.exists():
    n = 0
    for item in json.load(open(f)):
        for session in item.get("haystack_sessions", []):
            for turn in session:
                if turn.get("role") == "user" and n < 3000:
                    add(turn["content"], 0, "longmemeval"); n += 1
else:
    print("WARNING missing", f)

df = pd.DataFrame(rows).drop_duplicates("text").reset_index(drop=True)

# Grouped rows (PoisonedRAG questions): whole groups go to one split, 70/15/15.
need = df["split"].isna()
groups = sorted(df.loc[need & df.group.notna(), "group"].unique())
random.shuffle(groups)
a, b = int(0.70 * len(groups)), int(0.85 * len(groups))
gsplit = {g: s for s, gs in (("train", groups[:a]), ("val", groups[a:b]), ("test", groups[b:])) for g in gs}
df.loc[need & df.group.notna(), "split"] = df.loc[need & df.group.notna(), "group"].map(gsplit)

# Remaining rows without an official split: stratified 70/15/15, one by one.
need = df["split"].isna()
for lab in (0, 1):
    idx = df[need & (df.label == lab)].index.tolist()
    random.shuffle(idx)
    a, b = int(0.70 * len(idx)), int(0.85 * len(idx))
    df.loc[idx[:a], "split"] = "train"
    df.loc[idx[a:b], "split"] = "val"
    df.loc[idx[b:], "split"] = "test"

# Check: no PoisonedRAG question appears in more than one split.
leak = df[df.group.notna()].groupby("group").split.nunique().gt(1).sum()
print("PoisonedRAG questions split across train/val/test:", int(leak), "(must be 0)")
assert leak == 0

df.insert(0, "uid", range(len(df)))
df.to_csv(PROC / "dataset.csv", index=False)

summary = df.groupby(["source", "split", "label"]).size().unstack(fill_value=0)
summary.columns = ["clean", "poisoned"]
print(summary)
print("TOTAL:", len(df), "| poisoned:", int(df.label.sum()), "| clean:", int((df.label == 0).sum()))
summary.to_csv(PROC / "dataset_summary.csv")
