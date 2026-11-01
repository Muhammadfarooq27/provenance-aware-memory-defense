"""STEP 19b - Energy per entry on the laptop from the HWiNFO power log (no new measurement).

Step 19 records when each IDLE and ACTIVE window started and ended and how many entries were
checked (results/edge/laptop/laptop_power_windows.csv). HWiNFO64 logged "CPU Package Power"
every 2 s during the same time (results/edge/laptop/laptop_cpu_package_power.csv, the two
needed columns of the HWiNFO log). For each run:
    energy per entry = (mean ACTIVE power - mean IDLE power) x ACTIVE seconds / entries
4 s at both ends of each window are ignored. Output: results/tables/laptop_energy.csv
Run:  python s19b_laptop_energy.py
"""
import statistics as st
from datetime import datetime, timedelta

import pandas as pd

from config import ROOT, TABLES

D = ROOT / "results" / "edge" / "laptop"
log = pd.read_csv(D / "laptop_cpu_package_power.csv")


def stamp(d, t):
    day, month, year = map(int, d.split("."))
    h, m, s = t.split(":")
    return datetime(year, month, day, int(h), int(m)) + timedelta(seconds=float(s))


log["ts"] = [stamp(d, t) for d, t in zip(log.date, log.time)]


def mean_power(a, b, trim=4):
    a = datetime.fromisoformat(a) + timedelta(seconds=trim)
    b = datetime.fromisoformat(b) - timedelta(seconds=trim)
    return log[(log.ts >= a) & (log.ts <= b)].cpu_package_power_W.mean()


rows = []
for r in pd.read_csv(D / "laptop_power_windows.csv").itertuples():
    p_idle, p_act = mean_power(r.idle_start, r.idle_end), mean_power(r.active_start, r.active_end)
    rows.append(dict(method=r.method, idle_power_W=round(p_idle, 3), active_power_W=round(p_act, 3),
                     active_seconds=r.active_seconds, entries=r.entries,
                     energy_per_entry_J=round((p_act - p_idle) * r.active_seconds / r.entries, 4)))
out = pd.DataFrame(rows)
out.to_csv(TABLES / "laptop_energy.csv", index=False)
print(out.to_string(index=False))
for m, g in out.groupby("method"):
    e = g.energy_per_entry_J
    print(f"{m}: {st.mean(e):.3f} +- {st.stdev(e):.3f} J per entry over {len(e)} runs")
