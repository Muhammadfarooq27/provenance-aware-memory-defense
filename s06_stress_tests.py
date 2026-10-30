"""STEP 6 - Synonym attack: can the attacker fool the detector by swapping
words for synonyms in the poisoned texts?

  1. Take every POISONED TEST text and replace ~25% of its words with a
     WordNet synonym (meaning stays, wording changes).
  2. Re-embed only those texts with MiniLM.
  3. Train D1-D3 on the ORIGINAL training data; test on the changed texts.

(The overlap-level results for Figure 2 already come from Step 4, and the
signed-but-poisoned results are the 'signed_poison_recall' column there.)

Before running once:  python -c "import nltk; nltk.download('wordnet')"
Output: data/processed/embeddings_synonym.npy, results/logs/synonym_runs.csv
Run:    python s06_stress_tests.py
"""
import random

import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer

from common import VARIANTS, FusionDetector, fit_stage1, load, metrics
from config import PROC, LOGS, SEEDS, MAIN_OVERLAP, EMBED_MODEL

random.seed(0)


def swap_synonyms(text, rate=0.25):
    words = text.split()
    for i, w in enumerate(words):
        core = w.strip(".,;:!?\"'()")
        # skip names/capitalised words and short words, so facts stay the same
        if core.islower() and len(core) > 3 and random.random() < rate:
            syns = {l.name().replace("_", " ") for s in wn.synsets(core) for l in s.lemmas()}
            syns.discard(core)
            if syns:
                words[i] = w.replace(w.strip(".,;:!?\"'()"), random.choice(sorted(syns)))
    return " ".join(words)


df = pd.read_csv(PROC / "dataset.csv")
target = (df.split == "test") & (df.label == 1)
changed = df.loc[target, "text"].map(swap_synonyms)
pd.DataFrame({"original": df.loc[target, "text"], "synonym": changed}).head(20) \
  .to_csv(LOGS / "synonym_examples.csv", index=False)    # keep examples for the paper

emb_syn = np.load(PROC / "embeddings.npy").copy()
model = SentenceTransformer(EMBED_MODEL, device="cpu")
emb_syn[target.values] = model.encode(changed.tolist(), batch_size=64, normalize_embeddings=True)
np.save(PROC / "embeddings_synonym.npy", emb_syn)

runs = []
for seed in SEEDS:
    df_o, emb_o = load(MAIN_OVERLAP, seed)
    tr, te = df_o.split.values == "train", df_o.split.values == "test"
    y = df_o.label.values
    st1 = fit_stage1(emb_o[tr], y[tr], seed)            # trained on ORIGINAL texts
    for name, cols in VARIANTS.items():
        det = FusionDetector(cols, seed).fit(emb_o[tr], df_o[tr], st1)
        p = det.predict_proba(emb_syn[te], df_o[te])     # tested on CHANGED texts
        m = metrics(y[te], (p >= 0.5).astype(int), p)
        runs.append(dict(variant=name, overlap="synonym", seed=seed, **m))
        print(seed, name, round(m["f1"], 3))
pd.DataFrame(runs).to_csv(LOGS / "synonym_runs.csv", index=False)
print("saved synonym_runs.csv")
