"""Make the trained D3 detector loadable on small devices with older Python/numpy.

The detector saved by step 4 (models/d3_medium_seed0.joblib) is a scikit-learn pickle and only
loads with the same scikit-learn/numpy versions. Edge devices often have older ones (e.g.
Jetson with Python 3.8). This script moves the weights through a version-independent .npz file
and rebuilds the same detector as portable.PortableD3 (same weights, same math; outputs match
the original to about 1e-7).

  1) On the training machine:   python make_portable_model.py --export
     -> models/d3_params.npz
  2) On the edge device (copy models/d3_params.npz there first):
                                python make_portable_model.py --build
     -> models/d3_medium_seed0.joblib (portable version, used by steps 7, 18, 20, 21)
"""
import argparse

import joblib
import numpy as np

from config import MODELS, MAIN_OVERLAP

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--export", action="store_true", help="write models/d3_params.npz from the trained detector")
g.add_argument("--build", action="store_true", help="rebuild the portable detector from models/d3_params.npz")
args = ap.parse_args()

src = MODELS / f"d3_{MAIN_OVERLAP}_seed0.joblib"
npz = MODELS / "d3_params.npz"

if args.export:
    d3 = joblib.load(src)
    m = d3.content
    np.savez(npz, W0=m.coefs_[0], W1=m.coefs_[1], b0=m.intercepts_[0], b1=m.intercepts_[1],
             mean=d3.scaler.mean_, scale=d3.scaler.scale_, coef=d3.fuser.coef_,
             intercept=d3.fuser.intercept_, cols=np.array(d3.cols))
    print("written", npz)
else:
    from portable import PortableD3
    joblib.dump(PortableD3(dict(np.load(npz))), src)
    print("written", src, "(portable)")
