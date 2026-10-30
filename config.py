"""Shared settings for every step. Change values here only."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"
LOGS = ROOT / "results" / "logs"
MODELS = ROOT / "models"
for p in (RAW, PROC, TABLES, FIGS, LOGS, MODELS):
    p.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]               # 5 repeated runs -> mean +/- std
N_AGENTS = 25                         # size of the simulated IoT agent mesh
SIM_STEPS = 10                        # time steps an entry may spread
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim, ~90 MB
# Overlap = share of poisoned entries whose provenance/network behaviour is
# drawn from the BENIGN profile (i.e. looks normal). Higher = harder.
OVERLAP_LEVELS = {"low": 0.2, "medium": 0.5, "high": 0.8}
MAIN_OVERLAP = "medium"               # the level reported in Table 2

# Feature groups made in Step 2
PROV_COLS = ["ch_user", "ch_tool", "ch_peer", "ch_self", "writer_rep", "lineage_depth"]
NET_COLS = ["hop_count", "n_holders", "spread_rate"]
