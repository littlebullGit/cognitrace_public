"""
EXP-003: Italian PD eGeMAPS features + LightGBM classification
Labels from S3 folder structure: 28 PD / 22 Elderly HC / 15 Young HC
Subject-level stratified CV to prevent data leakage (multiple recordings per subject)
"""
import os, sys, json, time, gc, io, subprocess, re, warnings
warnings.filterwarnings("ignore")

subprocess.run(["apt-get", "update", "-qq"], capture_output=True, timeout=60)
subprocess.run(["apt-get", "install", "-y", "-qq", "libgomp1"], capture_output=True, timeout=60)

import logging; logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s'); log = logging.getLogger()
OUTPUTS_DIR = "/work/outputs"; os.makedirs(OUTPUTS_DIR, exist_ok=True)
SEED = 42

import boto3, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report)

np.random.seed(SEED)
S3_BUCKET = os.environ.get("DATA_S3_BUCKET", "YOUR_DATA_S3_BUCKET")
s3 = boto3.client("s3")
start_time = time.time()

# ── 1. Load feature CSV from EXP-002 ────────────────────────────────────────
log.info("Loading EXP-002 features...")
obj = s3.get_object(Bucket=S3_BUCKET,
    Key="competitions/cognitrace-pd/features/exp-002/italian_pd_egemaps_features.csv")
df = pd.read_csv(io.BytesIO(obj["Body"].read()))
log.info(f"Features loaded: {df.shape}")

# ── 2. Assign labels from S3 key folder structure ───────────────────────────
def get_label(s3_key):
    key_lower = s3_key.lower()
    if "parkinson" in key_lower or "/28 " in s3_key:
        return "PD"
    elif "elderly" in key_lower or "/22 " in s3_key:
        return "elderly_HC"
    elif "young" in key_lower or "/15 " in s3_key:
        return "young_HC"
    return "unknown"

def get_subject_id(s3_key):
    """Extract subject folder name from S3 key as subject ID."""
    parts = s3_key.split("/")
    # Structure: .../15 Young Healthy Control/SubjectName/file.wav
    for i, p in enumerate(parts):
        if any(x in p for x in ["Young", "Elderly", "Parkinson", "15 ", "22 ", "28 "]):
            if i + 1 < len(parts) and not parts[i+1].endswith(".wav"):
                return parts[i+1]
    # Fallback: extract from filename encoding
    fname = parts[-1]
    # Filenames encode subject info after task prefix (e.g., B1LBULCAAS94M1001...)
    # Use chars 2-12 as subject signature
    if len(fname) > 12:
        return fname[2:14]
    return fname

df["label"] = df["s3_key"].apply(get_label)
df["subject_id"] = df["s3_key"].apply(get_subject_id)

log.info(f"Label distribution:\n{df['label'].value_counts()}")
log.info(f"Unique subjects: {df['subject_id'].nunique()}")
log.info(f"Subject distribution per label:")
for label in df["label"].unique():
    n_subj = df[df["label"]==label]["subject_id"].nunique()
    n_rec = len(df[df["label"]==label])
    log.info(f"  {label}: {n_subj} subjects, {n_rec} recordings")

unknowns = df[df["label"] == "unknown"]
if len(unknowns) > 0:
    log.warning(f"Unknown labels: {len(unknowns)} files")
    log.warning(f"  Sample keys: {unknowns['s3_key'].head().tolist()}")
    df = df[df["label"] != "unknown"]

# ── 3. Feature columns ──────────────────────────────────────────────────────
meta_cols = ["filename", "s3_key", "sample_rate", "n_samples", "has_pd_in_name",
             "has_ctrl_in_name", "label", "subject_id"]
feat_cols = [c for c in df.columns if c not in meta_cols]
log.info(f"Feature columns ({len(feat_cols)}): {feat_cols[:10]}...")

X = df[feat_cols].values
# Replace inf/nan
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# ── 4. Binary classification (PD vs HC) ─────────────────────────────────────
df["binary_label"] = (df["label"] == "PD").astype(int)
y_bin = df["binary_label"].values
groups = df["subject_id"].values

log.info(f"\n{'='*60}")
log.info(f"BINARY CLASSIFICATION: PD vs HC")
log.info(f"PD={sum(y_bin==1)}, HC={sum(y_bin==0)}")

# Subject-level stratified group k-fold
try:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(X, y_bin, groups))
    log.info("Using StratifiedGroupKFold (subject-level)")
except Exception as e:
    log.warning(f"StratifiedGroupKFold failed: {e}, falling back to StratifiedKFold")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(skf.split(X, y_bin))

yt_all, yp_all, yprob_all = [], [], []
fi_total = np.zeros(X.shape[1])
fold_stats = []

for fold_idx, (tri, vai) in enumerate(splits):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X[tri]); Xva = sc.transform(X[vai])
    ytr, yva = y_bin[tri], y_bin[vai]

    m = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=6,
        num_leaves=31, min_child_samples=5, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        class_weight="balanced", random_state=SEED, verbose=-1, n_jobs=-1
    )
    m.fit(Xtr, ytr)
    pred = m.predict(Xva); prob = m.predict_proba(Xva)[:, 1]
    yt_all.extend(yva); yp_all.extend(pred); yprob_all.extend(prob)
    fi_total += m.feature_importances_

    acc = accuracy_score(yva, pred)
    auc = roc_auc_score(yva, prob) if len(np.unique(yva)) > 1 else 0
    log.info(f"  Fold {fold_idx+1}: acc={acc:.4f} auc={auc:.4f} (train={len(tri)}, val={len(vai)})")
    fold_stats.append({"acc": acc, "auc": auc})
    del m, sc; gc.collect()

yt_all = np.array(yt_all); yp_all = np.array(yp_all); yprob_all = np.array(yprob_all)
bin_acc = accuracy_score(yt_all, yp_all)
bin_f1 = f1_score(yt_all, yp_all, average="macro")
bin_sens = recall_score(yt_all, yp_all, pos_label=1)
tn, fp, fn, tp = confusion_matrix(yt_all, yp_all).ravel()
bin_spec = tn/(tn+fp) if (tn+fp) > 0 else 0
bin_auc = roc_auc_score(yt_all, yprob_all)

fi_total /= 5
top10i = np.argsort(fi_total)[-10:][::-1]
top10 = [(feat_cols[j], float(fi_total[j])) for j in top10i]

cv_accs = [s["acc"] for s in fold_stats]; cv_aucs = [s["auc"] for s in fold_stats]

log.info(f"\nBINARY RESULTS:")
log.info(f"  Accuracy:    {bin_acc:.4f}")
log.info(f"  F1 macro:    {bin_f1:.4f}")
log.info(f"  Sensitivity: {bin_sens:.4f}")
log.info(f"  Specificity: {bin_spec:.4f}")
log.info(f"  AUC-ROC:     {bin_auc:.4f}")
log.info(f"  CV acc: {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")
log.info(f"  CV auc: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
log.info(f"  Top 10 features: {top10}")
log.info(f"  Confusion: TP={tp} TN={tn} FP={fp} FN={fn}")

# ── 5. 3-class classification ───────────────────────────────────────────────
log.info(f"\n{'='*60}")
log.info(f"3-CLASS CLASSIFICATION: PD vs Elderly HC vs Young HC")

le = LabelEncoder()
y_3c = le.fit_transform(df["label"].values)
log.info(f"Classes: {le.classes_}, distribution: {np.unique(y_3c, return_counts=True)}")

try:
    sgkf3 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits3 = list(sgkf3.split(X, y_3c, groups))
except:
    skf3 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits3 = list(skf3.split(X, y_3c))

yt3, yp3, yprob3 = [], [], []
fold3_stats = []

for fold_idx, (tri, vai) in enumerate(splits3):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X[tri]); Xva = sc.transform(X[vai])
    ytr, yva = y_3c[tri], y_3c[vai]

    m = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=6,
        num_leaves=31, min_child_samples=5, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        class_weight="balanced", random_state=SEED, verbose=-1, n_jobs=-1
    )
    m.fit(Xtr, ytr)
    pred = m.predict(Xva); prob = m.predict_proba(Xva)
    yt3.extend(yva); yp3.extend(pred); yprob3.append(prob)

    acc = accuracy_score(yva, pred)
    log.info(f"  Fold {fold_idx+1}: acc={acc:.4f}")
    fold3_stats.append(acc)
    del m, sc; gc.collect()

yt3 = np.array(yt3); yp3 = np.array(yp3); yprob3 = np.vstack(yprob3)
mc_acc = accuracy_score(yt3, yp3)
mc_f1 = f1_score(yt3, yp3, average="macro")
try:
    mc_auc = roc_auc_score(yt3, yprob3, multi_class="ovr", average="macro")
except:
    mc_auc = 0.0

log.info(f"\n3-CLASS RESULTS:")
log.info(f"  Accuracy:  {mc_acc:.4f}")
log.info(f"  F1 macro:  {mc_f1:.4f}")
log.info(f"  AUC-ROC:   {mc_auc:.4f}")
log.info(f"  CV acc:    {np.mean(fold3_stats):.4f} ± {np.std(fold3_stats):.4f}")
log.info(f"\n{classification_report(yt3, yp3, target_names=le.classes_)}")

# ── 6. Write metrics ────────────────────────────────────────────────────────
elapsed = round(time.time() - start_time, 1)

metrics = [
    {"metric_name": "accuracy", "metric_value": round(bin_acc,4), "split_name": "cv_binary", "is_primary": True},
    {"metric_name": "f1_macro", "metric_value": round(bin_f1,4), "split_name": "cv_binary", "is_primary": False},
    {"metric_name": "sensitivity", "metric_value": round(bin_sens,4), "split_name": "cv_binary", "is_primary": False},
    {"metric_name": "specificity", "metric_value": round(bin_spec,4), "split_name": "cv_binary", "is_primary": False},
    {"metric_name": "auc_roc", "metric_value": round(bin_auc,4), "split_name": "cv_binary", "is_primary": False},
    {"metric_name": "accuracy_3class", "metric_value": round(mc_acc,4), "split_name": "cv_multiclass", "is_primary": False},
    {"metric_name": "f1_macro_3class", "metric_value": round(mc_f1,4), "split_name": "cv_multiclass", "is_primary": False},
    {"metric_name": "auc_roc_3class", "metric_value": round(mc_auc,4), "split_name": "cv_multiclass", "is_primary": False},
    {"metric_name": "pipeline_latency_seconds", "metric_value": elapsed, "split_name": "overall", "is_primary": False},
]

detailed = {
    "binary": {
        "accuracy": round(bin_acc,4), "f1_macro": round(bin_f1,4),
        "sensitivity": round(bin_sens,4), "specificity": round(bin_spec,4),
        "auc_roc": round(bin_auc,4),
        "cv_mean_acc": round(np.mean(cv_accs),4), "cv_std_acc": round(np.std(cv_accs),4),
        "cv_mean_auc": round(np.mean(cv_aucs),4), "cv_std_auc": round(np.std(cv_aucs),4),
        "top10_features": top10,
        "confusion": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "n_pd": int(sum(y_bin==1)), "n_hc": int(sum(y_bin==0))
    },
    "multiclass": {
        "accuracy": round(mc_acc,4), "f1_macro": round(mc_f1,4), "auc_roc": round(mc_auc,4),
        "classes": list(le.classes_),
        "cv_mean_acc": round(np.mean(fold3_stats),4), "cv_std_acc": round(np.std(fold3_stats),4),
    },
    "dataset_info": {
        "n_samples": len(df), "n_features": len(feat_cols),
        "n_subjects": int(df["subject_id"].nunique()),
        "label_dist": df["label"].value_counts().to_dict(),
    }
}

with open(os.path.join(OUTPUTS_DIR, "metrics.json"), "w") as f: json.dump(metrics, f, indent=2)
with open(os.path.join(OUTPUTS_DIR, "detailed_results.json"), "w") as f: json.dump(detailed, f, indent=2)

log.info(f"\nEXP-003 COMPLETE in {elapsed}s")
log.info(f"Binary:  acc={bin_acc:.4f} f1={bin_f1:.4f} sens={bin_sens:.4f} spec={bin_spec:.4f} auc={bin_auc:.4f}")
log.info(f"3-class: acc={mc_acc:.4f} f1={mc_f1:.4f} auc={mc_auc:.4f}")
