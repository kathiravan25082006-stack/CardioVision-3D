import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from xgboost import XGBClassifier
import shap

print("--- Step 1: Loading Dataset ---")
excel_files = glob.glob("*.xlsx") + glob.glob("*.xls")
if not excel_files:
    raise FileNotFoundError("Could not find the Excel dataset file in this folder.")

dataset_path = excel_files[0]
print(f"Loading: {dataset_path}")
df = pd.read_excel(dataset_path)

# Normalize column headers
df.columns = [str(c).strip() for c in df.columns]
col_map = {c.lower(): c for c in df.columns}

# Identify LAD, LCX, RCA, Cath, CAD
lad_col = col_map.get("lad")
lcx_col = col_map.get("lcx")
rca_col = col_map.get("rca")
cath_col = col_map.get("cath")
cad_col = col_map.get("cad")

# In this dataset, 'Cath' represents overall CAD status if 'CAD' column is absent
overall_col = cad_col if cad_col else cath_col

# Guardrail: Mandatory leakage exclusion (IIT Mandi guidelines)
leakage_cols = [c for c in [cath_col, cad_col, lad_col, lcx_col, rca_col] if c is not None]
print(f"Target and Leakage columns to isolate: {leakage_cols}")

# Construct binary targets cleanly:
# If value contains 'sten' (stenotic/stenosis) or 'cad' or '1' -> 1 (Positive), else 0 (Normal)
def to_binary(series):
    s = series.astype(str).str.strip().str.lower()
    return s.apply(lambda val: 1 if ('sten' in val or 'cad' in val or val in ['1', 'yes', 'true', 'positive']) else 0)

Y_dict = {}
if overall_col:
    Y_dict["cad"] = to_binary(df[overall_col])
if lad_col:
    Y_dict["lad"] = to_binary(df[lad_col])
if lcx_col:
    Y_dict["lcx"] = to_binary(df[lcx_col])
if rca_col:
    Y_dict["rca"] = to_binary(df[rca_col])

for k, v in Y_dict.items():
    pos_count = v.sum()
    print(f"Target [{k.upper()}]: Total={len(v)} | Positive (Stenosis/CAD)={pos_count} | Normal={len(v)-pos_count}")

# Feature matrix X (Drop all leakage columns)
X = df.drop(columns=leakage_cols).copy()

# Encode features
for col in X.columns:
    if X[col].dtype == 'object' or str(X[col].dtype).startswith('string') or str(X[col].dtype) == 'category':
        X[col] = X[col].astype('category').cat.codes.astype(float)
    else:
        X[col] = pd.to_numeric(X[col], errors='coerce').astype(float)
    
    med = X[col].median()
    if pd.isna(med):
        med = 0.0
    X[col] = X[col].fillna(med)

# Split 80/20 train/test with fixed seed
X_train, X_test, idx_train, idx_test = train_test_split(
    X, np.arange(len(X)), test_size=0.20, random_state=42
)

os.makedirs("saved_models", exist_ok=True)

# Export feature metadata for the UI
feature_info = {
    "columns": list(X.columns),
    "sample_row": X.iloc[0].to_dict(),
    "feature_defaults": X.median().to_dict(),
    "feature_mins": X.min().to_dict(),
    "feature_maxs": X.max().to_dict()
}
with open("saved_models/feature_info.pkl", "wb") as f:
    pickle.dump(feature_info, f)

print("\n--- Step 2: Training Models with XGBoost ---")
models = {}
eval_metrics = {}

for k in ["cad", "lad", "lcx", "rca"]:
    if k not in Y_dict:
        continue
    y = Y_dict[k]
    y_train = y.iloc[idx_train]
    y_test = y.iloc[idx_test]

    # Calculate class weighting to handle mild class imbalances
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos = (neg / pos) if pos > 0 else 1.0

    clf = XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.04,
        scale_pos_weight=scale_pos,
        random_state=42,
        eval_metric='logloss'
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, probs)
    except:
        auc = 0.5

    eval_metrics[k.upper()] = {"Accuracy": acc, "F1": f1, "ROC-AUC": auc}
    models[k] = clf

    with open(f"saved_models/{k}_model.pkl", "wb") as f:
        pickle.dump(clf, f)

print("\n=== Validated Performance Metrics ===")
for k, m in eval_metrics.items():
    print(f"Target [{k}]: Accuracy={m['Accuracy']*100:.2f}% | F1={m['F1']:.3f} | ROC-AUC={m['ROC-AUC']:.3f}")

print("\n--- Step 3: Computing SHAP Explainer ---")
cad_model = models.get("cad")
if cad_model:
    explainer = shap.TreeExplainer(cad_model)
    with open("saved_models/shap_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)
    print("SHAP TreeExplainer saved successfully.")

print("\n[SUCCESS] Engine ready. Models and metadata exported.")