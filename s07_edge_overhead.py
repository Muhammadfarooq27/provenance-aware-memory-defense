"""STEP 7 - Edge cost: is the defense cheap enough for IoT devices?

Measures, one entry at a time, on ONE CPU core:
  * detector latency (MiniLM encoding + MLP decision), ms per entry
  * model size on disk (MB)
  * Ed25519 signing and verification time of the memory envelope (ms)
  * envelope size: bytes added to every shared memory entry
Run the same script on each device you have (laptop, Raspberry Pi 5, Jetson)
and give the device a name:
    python s07_edge_overhead.py --device "Laptop CPU (1 core)"
Output: one row appended to results/tables/table3_edge.csv
"""
import argparse
import json
import math
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from sentence_transformers import SentenceTransformer

from config import PROC, MODELS, TABLES, EMBED_MODEL, MAIN_OVERLAP

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="Laptop CPU (1 core)")
ap.add_argument("--n", type=int, default=200, help="entries to time")
ap.add_argument("--threads", type=int, default=1, help="CPU cores to use (1 = small device)")
args = ap.parse_args()

torch.set_num_threads(args.threads)             # 1 = behave like a small device
os.environ["OMP_NUM_THREADS"] = str(args.threads)

df = pd.read_csv(PROC / "dataset.csv")
texts = df[df.split == "test"].text.sample(args.n, random_state=0, replace=True).tolist()
feats = pd.read_csv(PROC / f"features_{MAIN_OVERLAP}_seed0.csv")
d3 = joblib.load(MODELS / f"d3_{MAIN_OVERLAP}_seed0.joblib")
row1 = feats.iloc[:1]                            # one entry's provenance/network features
encoder = SentenceTransformer(EMBED_MODEL, device="cpu")
encoder.encode(["warm up"])                     # first call is always slow

lat = []
for t in texts:
    t0 = time.perf_counter()
    e = encoder.encode([t], normalize_embeddings=True)
    d3.predict_proba(e, row1)
    lat.append((time.perf_counter() - t0) * 1000)

# model size: MiniLM files in the local cache + the saved MLP
mlp_mb = os.path.getsize(MODELS / f"d3_{MAIN_OVERLAP}_seed0.joblib") / 1e6
mini_mb = sum(p.numel() * p.element_size() for p in encoder.parameters()) / 1e6

# signed memory envelope (Layer 1)
key = Ed25519PrivateKey.generate()
envelope = {"writer": "agent-07", "channel": "peer", "parents": ["a1f3", "9bc2"],
            "trust": 0.82, "expiry": 1767225600, "content_hash": "0" * 64}
payload = json.dumps(envelope, separators=(",", ":")).encode()
sign_ms, verify_ms = [], []
for _ in range(1000):
    t0 = time.perf_counter(); sig = key.sign(payload); sign_ms.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter(); key.public_key().verify(sig, payload); verify_ms.append((time.perf_counter() - t0) * 1000)
merkle_proof = 32 * math.ceil(math.log2(10_000))     # proof for a 10k-entry log
env_bytes = len(payload) + 64 + 32 + merkle_proof    # fields + signature + public key + proof

row = dict(device=args.device, threads=args.threads,
           detector_ms_median=np.median(lat), detector_ms_p95=np.percentile(lat, 95),
           model_mb=mini_mb + mlp_mb,
           sign_ms=np.median(sign_ms), verify_ms=np.median(verify_ms),
           envelope_bytes=env_bytes)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
out = TABLES / "table3_edge.csv"
pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
print("appended to", out.name)
