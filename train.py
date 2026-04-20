"""
MoA Prediction: Nystroem features + pure-NumPy vectorized ensemble + OOF weights

Key design choices:
  - Load from features_cache_v2.npz (472-dim: Nystroem RBF + PCA + interactions)
  - Pure-NumPy full-batch vec_logreg (Adam): avoids sklearn loky/multiprocess overhead
  - Pure-NumPy full-batch vec_2layer (Adam): 10x faster than sklearn MLPClassifier
  - OOF weight optimization: scipy L-BFGS-B on 15% holdout minimizes log loss directly
  - OMP_NUM_THREADS=4 for single-process BLAS (no loky workers → no oversubscription)
  - Warm bias init: logit(base_rate) for fast convergence on 99.7%-sparse targets
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

t_start = time.time()

# ============================================================
# Load features from cache
# ============================================================
CACHE_FILE = "features_cache_v2.npz"

print(f"Loading cached features...")
cache = np.load(CACHE_FILE)
X_train = cache['X_train'].astype(np.float32)
X_test  = cache['X_test'].astype(np.float32)
y_active = cache['y_active'].astype(np.float32)
active_idx = cache['active_idx']
base_rates = cache['base_rates'].astype(np.float32)
test_trt_positions = cache['test_trt_positions'].tolist()
n_targets = int(cache['n_targets'])
print(f"Cache loaded: {time.time() - t_start:.1f}s  {X_train.shape=}")

# Load submission metadata
test_features = pd.read_csv("data/test_features.csv")
target_cols = [c for c in pd.read_csv("data/train_targets.csv", nrows=0).columns if c != "sig_id"]
n_targets = len(target_cols)
n_active = y_active.shape[1]

# ============================================================
# OOF split for blend weight optimization
# ============================================================
VAL_FRAC = 0.12
np.random.seed(42)
perm = np.random.permutation(len(X_train))
n_val = int(len(X_train) * VAL_FRAC)
val_idx = perm[:n_val]
tr_idx  = perm[n_val:]

X_tr  = X_train[tr_idx]
X_val = X_train[val_idx]
y_tr  = y_active[tr_idx]
y_val = y_active[val_idx]
print(f"OOF split: {len(X_tr)} train / {len(X_val)} val  t={time.time()-t_start:.1f}s")

# ============================================================
# Vectorized multi-output LogReg with Adam (warm bias init)
# ============================================================
def vec_logreg(X, y, C=0.1, n_iter=80, lr=0.01):
    """Full-batch Adam logistic regression, all targets simultaneously."""
    n, p = X.shape; m = y.shape[1]
    Xb = np.hstack([X, np.ones((n, 1), dtype=np.float32)])
    base = y.mean(0).clip(1e-4, 1 - 1e-4).astype(np.float32)
    W = np.zeros((p + 1, m), dtype=np.float32)
    W[-1] = np.log(base / (1.0 - base))  # warm bias
    XtY = (Xb.T @ y).astype(np.float32)
    reg = np.float32(1.0 / (C * n))
    b1, b2, eps = np.float32(0.9), np.float32(0.999), np.float32(1e-8)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    for it in range(1, n_iter + 1):
        pred = 1.0 / (1.0 + np.exp(-np.clip(Xb @ W, -50, 50)))
        g = (Xb.T @ pred - XtY) / n
        g[:-1] += reg * W[:-1]
        mW = b1 * mW + (1 - b1) * g
        vW = b2 * vW + (1 - b2) * (g * g)
        W -= lr * (mW / (1 - b1 ** it)) / (np.sqrt(vW / (1 - b2 ** it)) + eps)
    def predict(Xnew):
        Xb2 = np.hstack([Xnew, np.ones((len(Xnew), 1), dtype=np.float32)])
        return 1.0 / (1.0 + np.exp(-np.clip(Xb2 @ W, -50, 50)))
    return predict

# ============================================================
# Vectorized 2-layer MLP with Adam (warm bias init)
# ============================================================
def vec_2layer(X, y, C=0.1, hidden=64, n_iter=80, lr=0.005, seed=0):
    """Full-batch 2-layer MLP. X → hidden → ReLU → 205 sigmoid outputs.
    Warm bias = logit(base_rate) for fast convergence on sparse targets."""
    n, p = X.shape; m = y.shape[1]
    rng = np.random.default_rng(seed)
    W1 = (rng.standard_normal((p, hidden)) * np.sqrt(2.0 / p)).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    W2 = (rng.standard_normal((hidden, m)) * np.sqrt(2.0 / hidden)).astype(np.float32)
    base = y.mean(0).clip(1e-4, 1 - 1e-4).astype(np.float32)
    b2 = np.log(base / (1.0 - base))
    reg = np.float32(1.0 / (C * n))
    a1, a2, eps = np.float32(0.9), np.float32(0.999), np.float32(1e-8)
    mW1, vW1 = np.zeros_like(W1), np.zeros_like(W1)
    mb1, vb1 = np.zeros_like(b1), np.zeros_like(b1)
    mW2, vW2 = np.zeros_like(W2), np.zeros_like(W2)
    mb2, vb2 = np.zeros_like(b2), np.zeros_like(b2)
    for it in range(1, n_iter + 1):
        bc, bv = 1.0 - a1 ** it, 1.0 - a2 ** it
        h    = np.maximum(0.0, X @ W1 + b1)
        pred = 1.0 / (1.0 + np.exp(-np.clip(h @ W2 + b2, -50, 50)))
        err  = (pred - y) / n
        gW2 = h.T @ err;   gW2 += reg * W2
        gb2 = err.sum(0)
        dh  = (err @ W2.T) * (h > 0)
        gW1 = X.T @ dh;    gW1 += reg * W1
        gb1 = dh.sum(0)
        for param, g, mP, vP in [(W2, gW2, mW2, vW2), (b2, gb2, mb2, vb2),
                                   (W1, gW1, mW1, vW1), (b1, gb1, mb1, vb1)]:
            mP[:] = a1 * mP + (1 - a1) * g
            vP[:] = a2 * vP + (1 - a2) * (g * g)
            param -= lr * (mP / bc) / (np.sqrt(vP / bv) + eps)
    def predict(Xnew):
        h = np.maximum(0.0, Xnew @ W1 + b1)
        return 1.0 / (1.0 + np.exp(-np.clip(h @ W2 + b2, -50, 50)))
    return predict

# ============================================================
# Helpers: full-target predictions + submission
# ============================================================
def full_pred(p_active):
    """Expand n_active predictions to n_targets."""
    out = np.full((len(p_active), n_targets), 0.001, dtype=np.float32)
    out[:, active_idx] = np.clip(p_active, 0, 1)
    return out


def adaptive_alpha(br):
    if   br < 0.001: return 0.10
    elif br < 0.003: return 0.07
    elif br < 0.007: return 0.05
    elif br < 0.020: return 0.03
    else:            return 0.01


alphas = np.array([adaptive_alpha(br) for br in base_rates], dtype=np.float32)


def save_submission(ensemble_active):
    """Calibrate + write submission.csv."""
    ens = np.clip(ensemble_active, 1e-6, 1 - 1e-6).astype(np.float32)
    # Bayesian shrinkage toward base rate
    ens = (1.0 - alphas) * ens + alphas * base_rates
    ens = np.clip(ens, 1e-6, 1 - 1e-6)

    preds_full = np.full((len(test_features), n_targets), 0.001, dtype=np.float32)
    for i, orig_pos in enumerate(test_trt_positions):
        row = np.full(n_targets, 0.001, dtype=np.float32)
        row[active_idx] = ens[i]
        preds_full[orig_pos] = row
    sub = pd.DataFrame(preds_full, columns=target_cols)
    sub.insert(0, "sig_id", test_features["sig_id"].values)
    sub.to_csv("submission.csv", index=False)


# ============================================================
# Training: fill budget with LogReg then 2-layer MLP
# BUDGET: 260s for models, 30s for OOF opt + I/O
# ============================================================
BUDGET = 260.0

all_test  = []   # test set predictions per model
all_val   = []   # val set predictions per model


def time_left():
    return BUDGET - (time.time() - t_start)


def add_model(fn):
    all_test.append(full_pred(fn(X_test)))
    all_val.append(full_pred(fn(X_val)))


# --- LogReg at multiple C values ---
print("LogReg ensemble...")
for C in [0.03, 0.05, 0.07, 0.10, 0.13, 0.18, 0.25]:
    if time_left() < 8:
        print(f"  Budget exhausted at {len(all_test)} LogReg models")
        break
    t0 = time.time()
    fn = vec_logreg(X_tr, y_tr, C=C, n_iter=80, lr=0.01)
    add_model(fn)
    print(f"  LR C={C}: {time.time()-t0:.1f}s  total={time.time()-t_start:.1f}s")

# --- Modality LogReg: gene-only (0:50) and cell-only (50:80) PCA components ---
print("Modality LogReg...")
# Gene PCA is cols 0:50, cell PCA is cols 50:80 in the cached feature matrix
for slc, label in [(slice(0, 50), "gene"), (slice(50, 80), "cell")]:
    X_mod_tr  = X_tr[:, slc]
    X_mod_val = X_val[:, slc]
    X_mod_te  = X_test[:, slc]
    for C in [0.1, 0.3]:
        if time_left() < 8:
            print(f"  Budget exhausted, skipping {label} C={C}")
            break
        t0 = time.time()
        fn = vec_logreg(X_mod_tr, y_tr, C=C, n_iter=80, lr=0.01)
        all_test.append(full_pred(fn(X_mod_te)))
        all_val.append(full_pred(fn(X_mod_val)))
        print(f"  {label} C={C}: {time.time()-t0:.1f}s  total={time.time()-t_start:.1f}s")

# --- 2-layer MLP ensemble at multiple configs ---
print("2-layer MLP ensemble...")
mlp_configs = [
    # (seed, hidden, n_iter, lr, C)
    (0,   64,  80, 0.005, 0.10),
    (1,   64,  80, 0.005, 0.10),
    (2,   64,  80, 0.005, 0.10),
    (3,   64,  80, 0.005, 0.10),
    (4,   64,  80, 0.005, 0.10),
    (5,   64,  80, 0.005, 0.10),
    (0,   64,  80, 0.005, 0.20),
    (1,   64,  80, 0.005, 0.20),
    (2,   64,  80, 0.005, 0.20),
    (0,  128,  60, 0.003, 0.10),
    (1,  128,  60, 0.003, 0.10),
    (2,  128,  60, 0.003, 0.10),
    (3,  128,  60, 0.003, 0.10),
    (0,  128,  60, 0.003, 0.20),
    (1,  128,  60, 0.003, 0.20),
    (0,   64, 100, 0.003, 0.10),
    (1,   64, 100, 0.003, 0.10),
    (2,   64, 100, 0.003, 0.10),
    (0,  128,  80, 0.003, 0.10),
    (1,  128,  80, 0.003, 0.10),
    (0,  256,  40, 0.002, 0.10),
    (1,  256,  40, 0.002, 0.10),
    (2,  256,  40, 0.002, 0.10),
]
for seed, hidden, n_iter, lr, C in mlp_configs:
    remaining = time_left()
    if remaining < 20:
        print(f"  Stopping MLP at {len(all_test)} models, {remaining:.0f}s remain")
        break
    if hidden >= 256 and remaining < 30:
        continue
    if hidden >= 128 and remaining < 25:
        continue
    t0 = time.time()
    fn = vec_2layer(X_tr, y_tr, C=C, hidden=hidden, n_iter=n_iter, lr=lr, seed=seed)
    add_model(fn)
    dt = time.time() - t0
    print(f"  MLP h={hidden} s={seed} C={C}: {dt:.1f}s  total={time.time()-t_start:.1f}s")

# ============================================================
# OOF weight optimization with scipy L-BFGS-B
# ============================================================
print(f"\nOOF weight optimization ({len(all_test)} models, {time_left():.0f}s remain)...")
from scipy.optimize import minimize as scipy_minimize

val_stack  = np.array(all_val,  dtype=np.float64)   # (n_models, n_val, n_targets)
test_stack = np.array(all_test, dtype=np.float64)
n_m = len(all_test)

# Build full val targets
y_val_full = np.zeros((len(X_val), n_targets), dtype=np.float64)
y_val_full[:, active_idx] = y_val.astype(np.float64)


def oof_loss(log_w):
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    blend = np.einsum('m,mjt->jt', w, val_stack)
    blend = np.clip(blend, 1e-7, 1 - 1e-7)
    ll = -(y_val_full * np.log(blend) + (1 - y_val_full) * np.log(1 - blend))
    return ll.mean()


t_opt = time.time()
res = scipy_minimize(oof_loss, np.zeros(n_m), method='L-BFGS-B',
                     options={'maxiter': 500, 'ftol': 1e-9})
opt_w = np.exp(res.x - res.x.max()); opt_w /= opt_w.sum()
print(f"  OOF val loss: {res.fun:.6f}  ({time.time()-t_opt:.1f}s)")
print(f"  Top weights: {sorted(opt_w, reverse=True)[:5]}")

# ============================================================
# Weighted ensemble + calibration + submission
# ============================================================
print(f"Ensembling {n_m} models...")
w = opt_w.astype(np.float32)
ensemble_full = np.einsum('m,mjt->jt', w.astype(np.float64),
                           test_stack).astype(np.float32)

# Extract active predictions for calibration
ens_active = ensemble_full[:, active_idx]
save_submission(ens_active)

print(f"\nFinal ensemble: {n_m} models")
print(f"Total time: {time.time() - t_start:.1f}s")
print(f"Submission: {len(test_features)} rows x {n_targets} targets")
