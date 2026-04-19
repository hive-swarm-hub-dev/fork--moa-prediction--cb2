"""
MoA Prediction: Transductive PCA + Nystroem kernel features + LogReg ensemble

Strategy: MLPClassifier multiprocessing doesn't parallelize on this node (1075s).
LogisticRegression LBFGS via MultiOutputClassifier(n_jobs=-1) IS fast (~15s each).
Maximise richness of features + diversity of LogReg ensemble + adaptive calibration.

Features:
  - Transductive PCA: gene (50 comps), cell (30 comps) — fit on train+test combined
  - Top-50 raw high-variance gene features alongside PCA
  - cp_time/cp_dose interactions with top gene PCs
  - Cross-PCA (gene × cell) interactions
  - Nystroem RBF kernel features (~200) to add non-linearity for LogReg
"""
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import Nystroem
from sklearn.multioutput import MultiOutputClassifier

t_start = time.time()

# ============================================================
# Load data
# ============================================================
train_features = pd.read_csv("data/train_features.csv")
train_targets   = pd.read_csv("data/train_targets.csv")
test_features   = pd.read_csv("data/test_features.csv")

target_cols = [c for c in train_targets.columns if c != "sig_id"]
gene_cols   = [c for c in train_features.columns if c.startswith('g-')]
cell_cols   = [c for c in train_features.columns if c.startswith('c-')]

# ============================================================
# Separate controls (all-zero targets by definition)
# ============================================================
train_ctrl_mask = train_features['cp_type'] == 'ctl_vehicle'
test_ctrl_mask  = test_features['cp_type']  == 'ctl_vehicle'

train_trt = train_features[~train_ctrl_mask].reset_index(drop=True)
test_trt  = test_features[~test_ctrl_mask].copy()
test_trt_positions = test_features[~test_ctrl_mask].index.tolist()

y_train = (
    train_targets
    .set_index('sig_id')
    .loc[train_trt['sig_id'].tolist(), target_cols]
    .values
)

# ============================================================
# Encode cp_time / cp_dose
# ============================================================
train_time = train_trt['cp_time'].map({24: 0.0, 48: 0.5, 72: 1.0}).values
train_dose = train_trt['cp_dose'].map({'D1': 0.0, 'D2': 1.0}).values
test_time  = test_trt['cp_time'].map({24: 0.0, 48: 0.5, 72: 1.0}).values
test_dose  = test_trt['cp_dose'].map({'D1': 0.0, 'D2': 1.0}).values

# ============================================================
# Transductive PCA — randomized solver, fit on train+test combined
# ============================================================
def trans_pca(train_arr, test_arr, n_components):
    all_data = np.vstack([train_arr, test_arr])
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    pca.fit(all_data)
    return pca.transform(train_arr), pca.transform(test_arr)

gene_tr, gene_te = trans_pca(
    train_trt[gene_cols].values, test_trt[gene_cols].values, n_components=50)
cell_tr, cell_te = trans_pca(
    train_trt[cell_cols].values, test_trt[cell_cols].values, n_components=30)

print(f"Transductive PCA done: {time.time() - t_start:.1f}s")

# ============================================================
# Top-50 high-variance raw gene features (captures signal PCA may compress away)
# ============================================================
gene_var = train_trt[gene_cols].var().values
top50_idx = np.argsort(gene_var)[-50:]
raw_gene_tr = train_trt[gene_cols].values[:, top50_idx]
raw_gene_te = test_trt[gene_cols].values[:, top50_idx]

# ============================================================
# Interaction features
# ============================================================
N_INTERACT = 15
t_tr = train_time.reshape(-1, 1)
d_tr = train_dose.reshape(-1, 1)
t_te = test_time.reshape(-1, 1)
d_te = test_dose.reshape(-1, 1)

interact_tr = np.hstack([
    gene_tr[:, :N_INTERACT] * t_tr,
    gene_tr[:, :N_INTERACT] * d_tr,
    gene_tr[:, :10] * cell_tr[:, :10],    # cross-modality
])
interact_te = np.hstack([
    gene_te[:, :N_INTERACT] * t_te,
    gene_te[:, :N_INTERACT] * d_te,
    gene_te[:, :10] * cell_te[:, :10],
])
cp_tr = np.column_stack([train_time, train_dose])
cp_te = np.column_stack([test_time,  test_dose])

# ============================================================
# Assemble base features
# ============================================================
X_base_tr = np.hstack([gene_tr, cell_tr, raw_gene_tr, interact_tr, cp_tr])
X_base_te = np.hstack([gene_te, cell_te, raw_gene_te, interact_te, cp_te])

scaler = StandardScaler()
X_base_tr = scaler.fit_transform(X_base_tr)
X_base_te = scaler.transform(X_base_te)

print(f"Base features: {X_base_tr.shape}")

# ============================================================
# Nystroem RBF kernel features — adds non-linearity that LogReg can exploit
# ============================================================
nys = Nystroem(kernel='rbf', n_components=200, gamma=0.1, random_state=42)
nys.fit(X_base_tr)
X_nys_tr = nys.transform(X_base_tr)
X_nys_te = nys.transform(X_base_te)

# Concatenate base + kernel features
X_train = np.hstack([X_base_tr, X_nys_tr])
X_test  = np.hstack([X_base_te, X_nys_te])

print(f"Full feature matrix: {X_train.shape}")
print(f"Feature engineering done: {time.time() - t_start:.1f}s")

# ============================================================
# Target bookkeeping
# ============================================================
active_mask = y_train.sum(axis=0) > 0
active_idx  = np.where(active_mask)[0]
y_active    = y_train[:, active_idx]

# ============================================================
# LogReg ensemble — LBFGS parallelizes across targets via n_jobs=-1
# ============================================================
def get_proba(mo_clf, X):
    out = np.zeros((X.shape[0], len(mo_clf.estimators_)))
    for i, est in enumerate(mo_clf.estimators_):
        if not hasattr(est, 'classes_') or len(est.classes_) == 1:
            out[:, i] = float(est.classes_[0]) if hasattr(est, 'classes_') else 0.0
        else:
            pos = np.where(est.classes_ == 1)[0][0]
            out[:, i] = est.predict_proba(X)[:, pos]
    return out

lr_preds = []
for C in [0.03, 0.07, 0.15, 0.30]:
    t0 = time.time()
    mo = MultiOutputClassifier(
        LogisticRegression(C=C, solver='lbfgs', max_iter=300),
        n_jobs=-1,
    )
    mo.fit(X_train, y_active)
    lr_preds.append(get_proba(mo, X_test))
    print(f"  LogReg C={C}: {time.time()-t0:.1f}s")

lr_ens = np.mean(lr_preds, axis=0)
print(f"LogReg ensemble done: {time.time() - t_start:.1f}s")

# ============================================================
# Adaptive Bayesian calibration
# ============================================================
base_rates = y_active.mean(axis=0)

def adaptive_alpha(br):
    if   br < 0.001: return 0.80
    elif br < 0.003: return 0.85
    elif br < 0.007: return 0.90
    elif br < 0.015: return 0.93
    else:            return 0.95

alphas = np.array([adaptive_alpha(br) for br in base_rates])
blend  = lr_ens * alphas + base_rates * (1.0 - alphas)
blend  = np.clip(blend, 1e-6, 1.0 - 1e-6)

# ============================================================
# Assemble submission
# ============================================================
preds_full = np.full((len(test_features), len(target_cols)), 0.001)
for i, orig_pos in enumerate(test_trt_positions):
    row = np.full(len(target_cols), 0.001)
    row[active_idx] = blend[i]
    preds_full[orig_pos] = row

submission = pd.DataFrame(preds_full, columns=target_cols)
submission.insert(0, "sig_id", test_features["sig_id"].values)
submission.to_csv("submission.csv", index=False)
print(f"Submission saved: {len(submission)} rows x {len(target_cols)} targets")
print(f"Total time: {time.time() - t_start:.1f}s")
