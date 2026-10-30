"""STEP 2 - Simulate a 25-agent IoT mesh and give every entry
provenance features and network features.

How it works (plain words):
  * 25 agents are connected in a small-world graph (like IoT devices that
    talk to nearby devices plus a few far ones).
  * Every entry from the dataset is "written" into one agent through one of
    four channels: user, tool, peer, self.
  * The entry may then be shared to neighbours for SIM_STEPS time steps.
  * We record where it came from (provenance) and how it spread (network).

Poisoned entries follow an ATTACKER profile, clean entries a BENIGN profile.
To stay honest, a share of poisoned entries (the OVERLAP) uses the benign
profile, so they look normal. We repeat for 3 overlap levels x 5 seeds.

Output: data/processed/features_<level>_seed<k>.csv
Run:    python s02_simulate_mesh.py
"""
import networkx as nx
import numpy as np
import pandas as pd

from config import PROC, SEEDS, N_AGENTS, SIM_STEPS, OVERLAP_LEVELS

CHANNELS = ["user", "tool", "peer", "self"]
PROFILES = {
    #            channel probabilities        writer reputation   share prob.
    "benign":   dict(ch=[.30, .30, .25, .15], rep=(5, 2),         share=.15),
    "attacker": dict(ch=[.45, .35, .15, .05], rep=(2, 3),         share=.35),
}
# "Signed-but-poisoned" (MINJA-style): an honest, high-reputation agent is
# tricked and writes the poison itself through the 'self' channel.
SIGNED_POISON_SHARE = 0.10

df = pd.read_csv(PROC / "dataset.csv")

for level, overlap in OVERLAP_LEVELS.items():
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        G = nx.connected_watts_strogatz_graph(N_AGENTS, k=4, p=0.2, seed=seed)
        reputation = rng.beta(5, 2, N_AGENTS)        # trust score of each agent
        recs = []
        for _, r in df.iterrows():
            poisoned = r.label == 1
            signed_poison = poisoned and rng.random() < SIGNED_POISON_SHARE
            looks_benign = (not poisoned) or signed_poison or rng.random() < overlap
            prof = PROFILES["benign" if looks_benign else "attacker"]

            channel = "self" if signed_poison else rng.choice(CHANNELS, p=prof["ch"])
            writer = int(rng.integers(N_AGENTS))
            writer_rep = reputation[writer] if looks_benign else rng.beta(*prof["rep"])
            lineage_depth = int(rng.poisson(2.0 if looks_benign else 0.7))

            # ---- spread the entry over the graph (independent cascade) ----
            holders, frontier, first_hop = {writer}, {writer}, {writer: 0}
            for step in range(1, SIM_STEPS + 1):
                new = set()
                for a in frontier:
                    for b in G.neighbors(a):
                        if b not in holders and rng.random() < prof["share"]:
                            new.add(b); first_hop[b] = step
                holders |= new; frontier = new
                if not frontier:
                    break
            recs.append(dict(
                uid=r.uid,
                ch_user=int(channel == "user"), ch_tool=int(channel == "tool"),
                ch_peer=int(channel == "peer"), ch_self=int(channel == "self"),
                writer_rep=round(float(writer_rep), 4),
                lineage_depth=lineage_depth,
                hop_count=max(first_hop.values()),
                n_holders=len(holders),
                spread_rate=round(len(holders) / SIM_STEPS, 4),
                signed_poison=int(signed_poison),
            ))
        out = PROC / f"features_{level}_seed{seed}.csv"
        pd.DataFrame(recs).to_csv(out, index=False)
        print("saved", out.name)
