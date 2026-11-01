"""Portable copy of the trained EngraMesh D3 detector for older Python/numpy (e.g. Jetson, Python 3.8).

Same weights and the same math as common.FusionDetector.predict_proba:
  stage 1: MLP (ReLU hidden layer, logistic output) on the 384-dim MiniLM embedding -> content score
  stage 2: StandardScaler + logistic regression on [content score, 9 provenance/network features]
The weights were exported from models/d3_medium_seed0.joblib; outputs match it to ~1e-7.
"""
import numpy as np


class PortableD3:
    def __init__(self, p):
        self.W = [p["W0"], p["W1"]]
        self.b = [p["b0"], p["b1"]]
        self.mean, self.scale = p["mean"], p["scale"]
        self.coef, self.intercept = p["coef"], p["intercept"]
        self.cols = [str(c) for c in p["cols"]]

    def predict_proba(self, emb_rows, df_rows):
        h = np.maximum(np.asarray(emb_rows, dtype=np.float64) @ self.W[0] + self.b[0], 0.0)
        s = 1.0 / (1.0 + np.exp(-(h @ self.W[1] + self.b[1])[:, 0]))
        side = df_rows[self.cols].to_numpy("float32") if self.cols else np.zeros((len(df_rows), 0), "float32")
        Z = (np.column_stack([s, side]) - self.mean) / self.scale
        return 1.0 / (1.0 + np.exp(-(Z @ self.coef[0] + self.intercept[0])))
