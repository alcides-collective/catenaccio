#!/usr/bin/env python3
"""Calibrate P(player scores >4 pts) on realized MD1+MD2 data.
Replaces the unfit linear clip p4=clip((proj-1.5)/6,0,.85) (Codex audit issue #3).
Honest eval: 5-fold CV Brier + log-loss vs the old clip baseline."""
import json, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
rows = json.load(open(ROOT / "engine/calibration_data.json"))
POS = ["GK", "DEF", "MID", "FWD"]

def feats(r):
    proj = r["proj"] or 0.0
    own = r["own"] or 0.0
    onehot = [1.0 if r["pos"] == p else 0.0 for p in POS]
    return [proj, np.log1p(own), proj*own/100.0] + onehot

X = np.array([feats(r) for r in rows])
y = np.array([1 if r["real"] > 4 else 0 for r in rows])
n = len(y)

def clip(x, a, b): return max(a, min(b, x))
old = np.array([clip((r["proj"]-1.5)/6, 0, .85) for r in rows])

clf = LogisticRegression(max_iter=2000, C=1.0)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
new_cv = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

print(f"n={n}  base rate={y.mean():.3f}")
print(f"{'model':16}{'Brier':>9}{'logloss':>10}{'mean pred':>11}")
for name, p in [("old linear clip", old), ("logistic (5-fold CV)", new_cv)]:
    print(f"{name:16}{brier_score_loss(y,p):>9.4f}{log_loss(y,p):>10.4f}{p.mean():>11.3f}")

# calibration by ownership bucket (where the old clip is suspected biased)
print("\nCalibration by ownership bucket (pred vs actual P>4):")
buckets = [(0,5,"sub-5%"),(5,20,"5-20%"),(20,100,"20%+")]
for lo,hi,lab in buckets:
    idx=[i for i,r in enumerate(rows) if lo<=(r["own"] or 0)<hi]
    if not idx: continue
    a=y[idx].mean(); o=old[idx].mean(); nw=new_cv[idx].mean()
    print(f"  {lab:8} n={len(idx):4}  actual {a:.3f} | old {o:.3f} ({o-a:+.3f}) | new {nw:.3f} ({nw-a:+.3f})")

# --- honest prospective validation (Codex round-2): random CV is optimistic ---
from sklearn.model_selection import GroupKFold
gid = np.array([r["id"] for r in rows])
gkf_pred = cross_val_predict(LogisticRegression(max_iter=2000), X, y,
                             cv=GroupKFold(5), groups=gid, method="predict_proba")[:, 1]
rnd = np.array([r["rnd"] for r in rows])
loro = np.zeros(len(y))
for te in (1, 2):
    tr = rnd != te
    m = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
    loro[rnd == te] = m.predict_proba(X[rnd == te])[:, 1]
print("\nProspective validation (more honest than random CV):")
print(f"{'GroupKFold(player)':22}Brier {brier_score_loss(y,gkf_pred):.4f}  logloss {log_loss(y,gkf_pred):.4f}")
print(f"{'leave-one-round-out':22}Brier {brier_score_loss(y,loro):.4f}  logloss {log_loss(y,loro):.4f}")
print(f"{'old clip (reference)':22}Brier {brier_score_loss(y,old):.4f}  logloss {log_loss(y,old):.4f}")
print("NOTE: under LORO the Brier edge vanishes; the logistic wins on LOG-LOSS (calibration),")
print("not on Brier. And for sub-5% & proj>=5 players the clip overestimates P>4 by ~+0.33 —")
print("so do NOT over-trust scouting EV for high-projection differentials.")

# fit on ALL data, save coefficients for the engine
clf.fit(X, y)
model = {"coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
         "feature_order": ["proj","log1p_own","proj*own/100"]+[f"is_{p}" for p in POS],
         "pos_order": POS}
json.dump(model, open(ROOT / "engine/p4_model.json","w"), indent=1)
print("\nsaved engine/p4_model.json")
