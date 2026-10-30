"""FINAL RERUN with the question-grouped split (19 Oct 2026).

What it does, in order:
  1. Backs up data/processed and results to backup_entrysplit/ (copy, nothing deleted)
  2. Deletes only results/logs/e2e_cache.csv (old agent answers must not be reused)
  3. Runs steps 1-6, 8-15 one after another and stops at the first error.
     Step 7 (edge speed) and the Raspberry Pi test are NOT rerun: speed and size
     do not depend on the data split.

Run:      python run_final.py
Resume:   python run_final.py --from s11      (starts again at that step)
Needs:    Ollama running with qwen2.5:1.5b (steps 5, 11, 14)
Time:     about 2.5-3 hours on the laptop CPU
Log:      results/final_run_log.txt
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ["s01_build_dataset.py"], ["s02_simulate_mesh.py"], ["s03_embed.py"],
    ["s04_train_detectors.py"], ["s05_baselines.py", "--llm"], ["s06_stress_tests.py"],
    ["s08_recall_sim.py"], ["s09_make_tables_figures.py"], ["s10_two_thresholds.py"],
    ["s11_llm_paraphrase.py"], ["s12_recall_scale.py"], ["s13_laundering.py"],
    ["s14_end_to_end.py"], ["s15_collect_all.py"],
]
start = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else None

if start is None:
    bk = ROOT / "backup_entrysplit"
    if bk.exists():
        print("backup_entrysplit already exists - not overwritten")
    else:
        for d in ("data/processed", "results"):
            shutil.copytree(ROOT / d, bk / d)
        print("backup saved to", bk)
    old = ROOT / "results" / "logs" / "e2e_cache.csv"
    if old.exists():
        old.unlink(); print("deleted old e2e_cache.csv (a copy is in the backup)")

log = open(ROOT / "results" / "final_run_log.txt", "a", encoding="utf-8")
skipping = start is not None
for step in STEPS:
    if skipping and not step[0].startswith(start):
        continue
    skipping = False
    t0 = time.time()
    print(f"\n===== {' '.join(step)} =====", flush=True)
    log.write(f"\n===== {' '.join(step)} =====\n")
    out_path = ROOT / "results" / "FINAL_NUMBERS.txt" if step[0].startswith("s15") else None
    p = subprocess.run([sys.executable, *step], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log.write(p.stdout + p.stderr); log.flush()
    print(p.stdout[-3000:])
    if out_path:
        out_path.write_text(p.stdout, encoding="utf-8")
    if p.returncode != 0:
        print(p.stderr[-3000:])
        print(f"FAILED at {step[0]} - fix the problem, then run:  python run_final.py --from {step[0][:3]}")
        sys.exit(1)
    print(f"done in {time.time() - t0:.0f} s")
print("\nALL DONE. Send results/FINAL_NUMBERS.txt and results/final_run_log.txt")
