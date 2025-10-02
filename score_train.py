#!/usr/bin/env python3
"""
score_train_and_plot.py

Train a text-to-score regressor on merged battle logs, add optional Gaussian
noise to train embeddings, show MAE/R² bar charts, and save the model.

Usage
-----
python score_train_and_plot.py score_evaluation_merged.json \
       --noise-sd 0.02 \
       --model-out score_regressor.pkl
"""

from __future__ import annotations
import argparse, json, joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost.callback import EarlyStopping

# ───────────────────────────── CLI ──────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("json_file",      help="merged score_evaluation file")
parser.add_argument("--val-size",     type=float, default=0.15,
                    help="validation fraction (default 0.15)")
parser.add_argument("--test-size",    type=float, default=0.15,
                    help="test fraction (default 0.15)")
parser.add_argument("--noise-sd",     type=float, default=0.0,
                    help="σ of Gaussian noise added to TRAIN embeddings")
parser.add_argument("--model-out",    default="score_regressor.pkl",
                    help="where to save encoder+regressor pickle")
args = parser.parse_args()

# ─────────────────────── load merged JSON ──────────────────────────
json_path = Path(args.json_file)
if not json_path.is_file():
    raise FileNotFoundError(json_path)

with json_path.open() as f:
    raw = json.load(f)

texts, scores = [], []
for battle, turns in raw.items():
    for turn, entry in turns.items():
        concat = " | ".join(
            [entry["Current Game State"],
             entry["Player Next Action"],
             entry["Opponent Next Action"],
             entry["Rationale"]]
        )
        texts.append(concat)
        scores.append(float(entry["Score"]))

print(f"✓ Loaded {len(texts):,} turns")

# ───────────────────────── embeddings ───────────────────────────────
print("→ Encoding text with MiniLM-L6-v2 …")
encoder = SentenceTransformer("all-MiniLM-L6-v2")
X_all = encoder.encode(texts, batch_size=64, show_progress_bar=True)
y_all = np.array(scores, dtype=float)

# ─────────────────────────── splits ─────────────────────────────────
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X_all, y_all, test_size=args.test_size, random_state=42
)
val_frac = args.val_size / (1 - args.test_size)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=val_frac, random_state=42
)

print(f"Split → train {len(y_train)}, val {len(y_val)}, test {len(y_test)}")

# ───────────────────── add Gaussian noise ───────────────────────────
if args.noise_sd > 0:
    noise = np.random.normal(0.0, args.noise_sd, X_train.shape)
    X_train_noisy = X_train + noise
    print(f"Added N(0,{args.noise_sd}) noise to train embeddings")
else:
    X_train_noisy = X_train

# ─────────────────────────── train ──────────────────────────────────
# reg = RidgeCV(alphas=(0.1, 1.0, 10.0)).fit(X_train_noisy, y_train)

reg = XGBRegressor(
    n_estimators=2000, learning_rate=0.03, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42, reg_lambda=1.0, eval_metric="mae", objective="reg:squarederror"
).fit(X_train_noisy, y_train,eval_set=[(X_train_noisy,y_train),(X_val,y_val)])


# ───────────────────────── evaluate ─────────────────────────────────
def metrics(name, y_true, X):
    pred = reg.predict(X)
    mae, r2 = mean_absolute_error(y_true, pred), r2_score(y_true, pred)
    return name, mae, r2

splits = [
    metrics("Train", y_train, X_train_noisy),
    metrics("Val",   y_val,   X_val),
    metrics("Test",  y_test,  X_test),
]

labels, mae_vals, r2_vals = zip(*splits)

# ────────────────────────── visualize ───────────────────────────────
fig, ax = plt.subplots()
ax.bar(labels, mae_vals, color="#4C72B0")
ax.set_ylabel("MAE")
ax.set_title("Mean Absolute Error")
for i, v in enumerate(mae_vals):
    ax.text(i, v + 0.05, f"{v:.2f}", ha="center")
plt.tight_layout()

fig2, ax2 = plt.subplots()
ax2.bar(labels, r2_vals, color="#55A868")
ax2.set_ylabel("R²")
ax2.set_title("R² Score")
for i, v in enumerate(r2_vals):
    ax2.text(i, v + 0.02, f"{v:.3f}", ha="center")
plt.tight_layout()
plt.show()

# ────────────────────────── save model ──────────────────────────────
joblib.dump({"encoder": encoder, "regressor": reg}, args.model_out)
print(f"✓ Model saved → {args.model_out}")
