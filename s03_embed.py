"""STEP 3 - Turn every text into 384 numbers with MiniLM (run ONCE).

Output: data/processed/embeddings.npy  (row i = dataset row i)
Run:    python s03_embed.py          (CPU: about 1-5 minutes)
"""
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import PROC, EMBED_MODEL

df = pd.read_csv(PROC / "dataset.csv")
model = SentenceTransformer(EMBED_MODEL, device="cpu")
t0 = time.time()
emb = model.encode(df.text.tolist(), batch_size=64, show_progress_bar=True,
                   normalize_embeddings=True)
np.save(PROC / "embeddings.npy", emb.astype("float32"))
print(f"saved {emb.shape} in {time.time() - t0:.1f} s")
