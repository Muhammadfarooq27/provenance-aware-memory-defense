#!/bin/bash
# Latency, peak memory and energy on an edge device, one CPU core (about 16 minutes; do not touch the device).
# Run from the repository folder:  bash edge/run_device_tests.sh pi      (Raspberry Pi 5)
#                                  bash edge/run_device_tests.sh jetson  (Jetson Orin Nano)
# Jetson: the INA3221 power sensor is readable by root only; run once before:
#   sudo chmod a+r /sys/class/hwmon/hwmon*/*   (resets at reboot)
set -e -o pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
case "$1" in
  pi)     DEV="Raspberry Pi 5 (1 core)";    POWER=s20_pi_power_window.py ;;
  jetson) DEV="Jetson Orin Nano (1 core)";  POWER=s21_jetson_power_window.py ;;
  *) echo "usage: bash edge/run_device_tests.sh pi|jetson"; exit 1 ;;
esac
mkdir -p results
{
echo "=== 1/3 LATENCY ===";     python s07_edge_overhead.py --device "$DEV"
echo "=== 2/3 PEAK MEMORY ==="; python s18_peak_memory.py --device "$DEV"
echo "=== 3/3 ENERGY ===";      python "$POWER" --device "$DEV"
echo "=== ALL DONE ==="
} 2>&1 | tee "results/edge_output_$1.txt"
