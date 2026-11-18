"""Record a REAL epoch-wise training and validation loss history for the MLP.

sklearn's MLPClassifier.fit() only exposes training loss and an accuracy-type validation score, not a
validation loss. partial_fit() runs exactly one epoch per call and keeps the Adam state, so we call it in a
loop and, after every epoch, measure log loss on (a) the training trips and (b) held-out validation TRIPS.
The validation trips are whole trips (never rows of a trip used for fitting), taken from the 80 train trips;
student-test and evaluator data are not used.

Usage (repo root, candidate tables built):  python scripts/learning_curve_mlp.py [--epochs 60]
Output: results/logs/mlp_learning_curve_history.json  (keys: loss_curve, val_loss)
        results/figures/mlp_learning_curve_training_history.png
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from pathsnap.config import CACHE_DIR, RESULTS_DIR, SEED
from pathsnap.plots import plot_mlp_loss_curve

FEATURES = ["dist_m", "along_frac", "arc_len_m", "speed_mps", "cos_heading",
            "dist_rank", "dist_minus_min", "n_cand"]

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--val_fraction", type=float, default=0.2, help="share of TRAIN TRIPS held out for validation")
args = ap.parse_args()
warnings.filterwarnings("ignore")

df = pd.read_parquet(CACHE_DIR / "candidates_train.parquet")
df = df[df.has_ref].reset_index(drop=True)

rng = np.random.default_rng(SEED)
trips = np.array(sorted(df.trip_id.unique()))
val_trips = set(rng.choice(trips, size=int(round(len(trips) * args.val_fraction)), replace=False))
is_val = df.trip_id.isin(val_trips).to_numpy()
print(f"{len(trips) - len(val_trips)} fitting trips, {len(val_trips)} validation trips "
      f"({(~is_val).sum():,} / {is_val.sum():,} rows)")

scaler = StandardScaler().fit(df.loc[~is_val, FEATURES])   # fitted on fitting trips only
X_fit, y_fit = scaler.transform(df.loc[~is_val, FEATURES]), df.loc[~is_val, "label"].to_numpy()
X_val, y_val = scaler.transform(df.loc[is_val, FEATURES]), df.loc[is_val, "label"].to_numpy()

# Same architecture and settings as scripts/train_mlp.py (without early stopping: we want the full curve).
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), random_state=SEED)
classes = np.array([0, 1])

train_loss, val_loss = [], []
t0 = time.time()
for epoch in range(1, args.epochs + 1):
    mlp.partial_fit(X_fit, y_fit, classes=classes)               # exactly one epoch
    train_loss.append(float(mlp.loss_))                          # sklearn's running training loss of this epoch
    val_loss.append(float(log_loss(y_val, mlp.predict_proba(X_val)[:, 1], labels=[0, 1])))
    print(f"epoch {epoch:3d}  train loss {train_loss[-1]:.5f}  val loss {val_loss[-1]:.5f}  ({time.time() - t0:.0f}s)", flush=True)

(RESULTS_DIR / "logs").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "logs" / "mlp_learning_curve_history.json").write_text(
    json.dumps({"loss_curve": train_loss, "val_loss": val_loss, "val_trips": sorted(val_trips), "seed": SEED}))
plot_mlp_loss_curve(train_loss, RESULTS_DIR / "figures" / "mlp_learning_curve_training_history.png", val_loss=val_loss)
