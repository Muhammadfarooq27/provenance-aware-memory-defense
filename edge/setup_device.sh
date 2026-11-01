#!/bin/bash
# One-time setup on an edge device (Raspberry Pi 5 or Jetson). Needs internet.
# Run from the repository folder:  bash edge/setup_device.sh
# Needs models/d3_params.npz (from: python make_portable_model.py --export on the training machine)
# and data/processed/dataset.csv + data/processed/features_medium_seed0.csv.
set -e
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
if python3 -c 'import sys; sys.exit(0 if sys.version_info < (3, 9) else 1)'; then
  # Python 3.8 (e.g. Jetson with JetPack 5): last versions with prebuilt wheels
  pip install torch==2.1.2 "sentence-transformers==2.7.0" "transformers<4.46" scikit-learn pandas joblib cryptography psutil requests
else
  pip install torch sentence-transformers scikit-learn pandas joblib cryptography psutil requests
fi
python make_portable_model.py --build
python -c "import config; from sentence_transformers import SentenceTransformer as S; S(config.EMBED_MODEL); print('MINILM DOWNLOADED OK')"
echo "SETUP FINISHED"
