"""Helpers shared by steps 4-7.

Detector design (two-stage "late fusion"):
  Stage 1: an MLP reads only the MiniLM text embedding and gives a content
           score between 0 and 1 (how poisoned the WORDS look).
  Stage 2: a small logistic regression combines that content score with the
           provenance / network features and makes the final decision.
D1 uses the content score only; D2 adds provenance; D3 adds network too.
Why two stages: if the 9 side features are simply glued to the 384 embedding
numbers, one MLP mostly ignores them. Training stage 2 on OUT-OF-FOLD content
scores (5-fold on the training split) avoids leaking training labels.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from config import PROC, PROV_COLS, NET_COLS

VARIANTS = {
    "D1: Content only": [],
    "D2: Content + provenance": PROV_COLS,
    "D3: Content + prov. + network": PROV_COLS + NET_COLS,
}


def load(level, seed, emb_file="embeddings.npy"):
    df = pd.read_csv(PROC / "dataset.csv")
    feats = pd.read_csv(PROC / f"features_{level}_seed{seed}.csv")
    df = df.merge(feats, on="uid", how="left")
    emb = np.load(PROC / emb_file)
    return df, emb


def make_mlp(seed):
    return MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-3, max_iter=300,
                         early_stopping=True, validation_fraction=0.15,
                         random_state=seed)


def fit_stage1(emb_tr, y_tr, seed):
    """Content MLP + its out-of-fold scores on the training split.
    It depends only on the text, labels and seed, so it can be shared by
    D1, D2, D3 and by all overlap levels."""
    oof = cross_val_predict(make_mlp(seed), emb_tr, y_tr, cv=5, method="predict_proba")[:, 1]
    return make_mlp(seed).fit(emb_tr, y_tr), oof


class FusionDetector:
    """Stage 1 = content MLP, Stage 2 = logistic regression on [score + features]."""

    def __init__(self, cols, seed):
        self.cols, self.seed = list(cols), seed

    def _side(self, df_rows):
        return df_rows[self.cols].to_numpy("float32") if self.cols else np.zeros((len(df_rows), 0), "float32")

    def fit(self, emb_tr, df_tr, stage1=None):
        y = df_tr.label.values
        self.content, oof = stage1 if stage1 is not None else fit_stage1(emb_tr, y, self.seed)
        Z = np.column_stack([oof, self._side(df_tr)])
        self.scaler = StandardScaler().fit(Z)
        self.fuser = LogisticRegression(max_iter=2000).fit(self.scaler.transform(Z), y)
        return self

    def predict_proba(self, emb_rows, df_rows):
        s = self.content.predict_proba(emb_rows)[:, 1]
        Z = np.column_stack([s, self._side(df_rows)])
        return self.fuser.predict_proba(self.scaler.transform(Z))[:, 1]

    def predict(self, emb_rows, df_rows):
        return (self.predict_proba(emb_rows, df_rows) >= 0.5).astype(int)


def metrics(y, pred, proba=None):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = dict(accuracy=accuracy_score(y, pred), f1=f1_score(y, pred, zero_division=0),
             fpr=fp / max(fp + tn, 1), recall=tp / max(tp + fn, 1))
    if proba is not None and len(set(y)) == 2:
        m["auc"] = roc_auc_score(y, proba)
    return m
