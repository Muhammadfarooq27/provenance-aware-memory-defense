"""STEP 11 - Stronger stress test: poison REWRITTEN by a language model.

Step 6 swapped single words (WordNet). Here a local LLM (qwen2.5:1.5b via
Ollama, already installed for step 5) rewrites every poisoned TEST entry in
new words with the same meaning. Detectors are trained on the ORIGINAL texts
only (exactly like step 6) and tested on the rewritten poison.

Rewrites are cached, so if the run stops you can simply start it again.
Bad rewrites (empty, far too short or far too long) keep the original text
and are counted, so nothing is hidden.

Output:
  data/processed/paraphrase_cache.csv     (all rewrites)
  results/logs/paraphrase_examples.csv    (20 examples for the paper)
  results/logs/paraphrase_runs.csv        (D1-D3 x 5 seeds)
  results/tables/paraphrase_summary.csv   (mean +/- std)
Run:    python s11_llm_paraphrase.py      (CPU: about 15-30 minutes)
Before: Ollama must be running and `ollama pull qwen2.5:1.5b` done.
"""
import re
import time

import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer

from common import VARIANTS, FusionDetector, fit_stage1, load, metrics
from config import PROC, LOGS, TABLES, SEEDS, MAIN_OVERLAP, EMBED_MODEL

MODEL = "qwen2.5:1.5b"
URL = "http://localhost:11434/api/generate"
PROMPT = ("Rewrite the following text using different words and sentence structure. "
          "Keep exactly the same meaning and claims. Keep all names, numbers, code, "
          "parameter names and URLs unchanged. Output only the rewritten text.\n\n"
          "Text: {t}\n\nRewritten text:")
CACHE = PROC / "paraphrase_cache.csv"


def clean(out):
    out = out.strip().strip('"').strip()
    out = re.sub(r"^(here is|here's|rewritten text|sure)[^:\n]*:\s*", "", out, flags=re.I)
    return out.strip()


df = pd.read_csv(PROC / "dataset.csv")
target = (df.split == "test") & (df.label == 1)
todo = df[target][["uid", "text"]]

cache = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(columns=["uid", "original", "paraphrase", "ok"])
done = set(cache.uid)
new_rows, t0 = [], time.time()
for i, (uid, text) in enumerate(todo.itertuples(index=False)):
    if uid in done:
        continue
    try:
        r = requests.post(URL, timeout=300, json={
            "model": MODEL, "prompt": PROMPT.format(t=text), "stream": False,
            "options": {"temperature": 0.3, "seed": 0}})
        r.raise_for_status()
        para = clean(r.json()["response"])
    except Exception as e:
        print("request failed:", e)
        para = ""
    ratio = len(para) / max(len(text), 1)
    ok = bool(para) and 0.4 <= ratio <= 2.5 and para != text
    new_rows.append(dict(uid=uid, original=text, paraphrase=para if ok else text, ok=ok))
    if len(new_rows) % 20 == 0:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache.to_csv(CACHE, index=False); new_rows = []
        print(f"  {len(cache)}/{len(todo)} rewritten ({time.time() - t0:.0f} s)")
if new_rows:
    cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
    cache.to_csv(CACHE, index=False)

cache = cache.drop_duplicates("uid").set_index("uid").loc[todo.uid]
n_ok = int(cache.ok.astype(bool).sum())
print(f"rewrites usable: {n_ok}/{len(cache)} (others keep the original text)")
cache.reset_index().head(20)[["original", "paraphrase"]].to_csv(LOGS / "paraphrase_examples.csv", index=False)

# --- embed rewritten poison; meaning check = cosine(original, rewrite) ----
model = SentenceTransformer(EMBED_MODEL, device="cpu")
emb_par = np.load(PROC / "embeddings.npy").copy()
e_new = model.encode(cache.paraphrase.tolist(), batch_size=64, normalize_embeddings=True)
cos = (emb_par[target.values] * e_new).sum(1)
okm = cache.ok.astype(bool).values
print(f"meaning check: mean cosine(original, rewrite) = {cos[okm].mean():.3f} (usable rewrites)")
word_change = np.mean([1 - len(set(a.lower().split()) & set(b.lower().split())) / max(len(set(a.lower().split())), 1)
                       for a, b, k in zip(cache.original, cache.paraphrase, okm) if k])
print(f"share of original words replaced: {100 * word_change:.1f}% (usable rewrites)")
emb_par[target.values] = e_new

# --- evaluate D1-D3 (trained on ORIGINAL texts) ---------------------------
runs = []
for seed in SEEDS:
    df_o, emb_o = load(MAIN_OVERLAP, seed)
    tr, te = df_o.split.values == "train", df_o.split.values == "test"
    y = df_o.label.values
    st1 = fit_stage1(emb_o[tr], y[tr], seed)
    for name, cols in VARIANTS.items():
        det = FusionDetector(cols, seed).fit(emb_o[tr], df_o[tr], st1)
        p = det.predict_proba(emb_par[te], df_o[te])
        m = metrics(y[te], (p >= 0.5).astype(int), p)
        runs.append(dict(variant=name, seed=seed, **m))
        print(seed, name, "F1", round(m["f1"], 3))
runs = pd.DataFrame(runs)
runs.to_csv(LOGS / "paraphrase_runs.csv", index=False)
summ = runs.groupby("variant", sort=False)[["f1", "auc", "recall", "fpr"]].agg(["mean", "std"]).round(3)
summ.to_csv(TABLES / "paraphrase_summary.csv")
print("\nLLM-PARAPHRASE RESULT (trained on original, tested on rewritten poison)\n", summ)
print(f"\nusable rewrites {n_ok}/{len(cache)} | mean cosine {cos[okm].mean():.3f} | words replaced {100 * word_change:.1f}%")
