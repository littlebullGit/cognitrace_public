"""
EXP-012: Optimized ensemble with Optuna hyperparameter tuning + feature selection
Hypothesis: Optuna-tuned LGB+XGB+CB ensemble with feature selection will achieve >=0.98
            accuracy (up from 0.976 in EXP-007) and AUC-ROC >=0.998 on subject-stratified
            5-fold CV on Italian PD eGeMAPS features.
Profile: cpu-large
"""

# ── 0. SYSTEM DEPS ─────────────────────────────────────────────────────────
import subprocess as _sp
_sp.run(["apt-get", "update", "-qq"], capture_output=True, timeout=60)
_sp.run(["apt-get", "install", "-y", "-qq", "libgomp1"], capture_output=True, timeout=60)

# ── 1. STDLIB ──────────────────────────────────────────────────────────────
import gc, json, math, os, random, sys, time
from pathlib import Path

# ── 2. THIRD-PARTY ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import io
import boto3
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 3. CONFIG ──────────────────────────────────────────────────────────────
class CFG:
    seed         = 42
    debug        = bool(os.environ.get("DEBUG", ""))
    n_folds      = 5
    n_optuna     = 50   # Optuna trials per model
    top_k_feat   = None # Set after feature importance analysis (None = all)

# ── 4. ENVIRONMENT ─────────────────────────────────────────────────────────
BUCKET     = os.environ.get("DATA_S3_BUCKET", "YOUR_DATA_S3_BUCKET")
RUN_ID     = os.environ.get("RUN_ID", "local")
S3_PREFIX  = "competitions/cognitrace-pd"
OUTPUT_DIR = Path("/work/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 5. REPRODUCIBILITY ────────────────────────────────────────────────────
def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

# ── 6. DATA ───────────────────────────────────────────────────────────────
def load_data():
    """Load Italian PD eGeMAPS features from S3."""
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET,
                        Key=f"{S3_PREFIX}/features/exp-002/italian_pd_egemaps_features.csv")
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))

    def get_label(s3_key):
        key_lower = s3_key.lower()
        if "parkinson" in key_lower or "/28 " in s3_key:
            return 1
        return 0

    def get_subject(s3_key):
        parts = s3_key.split("/")
        return parts[-2] if len(parts) >= 5 else "unknown"

    df["label"] = df["s3_key"].apply(get_label)
    df["subject"] = df["s3_key"].apply(get_subject)

    meta_cols = ["filename", "s3_key", "label", "subject"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    return df, feature_cols

# ── 7. FEATURE IMPORTANCE + SELECTION ──────────────────────────────────────
def get_top_features(X, y, groups, feature_cols, top_k=30):
    """Use LightGBM importance to select top-k features."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        num_leaves=31, min_child_samples=5,
        random_state=CFG.seed, verbose=-1, n_jobs=1,
    )
    model.fit(X_s, y)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    top_feats = importance.head(top_k)["feature"].tolist()
    print(f"Top {top_k} features selected:")
    for _, row in importance.head(top_k).iterrows():
        print(f"  {row['feature']}: {row['importance']}")

    return top_feats

# ── 8. OPTUNA OBJECTIVE ───────────────────────────────────────────────────
def make_objective(X, y, groups, model_type):
    """Return an Optuna objective function for the given model type."""
    sgkf = StratifiedGroupKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)

    def objective(trial):
        if model_type == "lgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "min_child_samples": trial.suggest_int("min_child_samples", 3, 30),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            }
        elif model_type == "xgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            }
        elif model_type == "cb":
            params = {
                "iterations": trial.suggest_int("iterations", 100, 500),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            }

        accs = []
        for train_idx, val_idx in sgkf.split(X, y, groups):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)

            if model_type == "lgb":
                model = lgb.LGBMClassifier(**params, random_state=CFG.seed, verbose=-1, n_jobs=1)
            elif model_type == "xgb":
                model = xgb.XGBClassifier(**params, random_state=CFG.seed, verbosity=0,
                                          n_jobs=1, eval_metric="logloss")
            elif model_type == "cb":
                model = cb.CatBoostClassifier(**params, random_seed=CFG.seed, verbose=0, thread_count=1)

            model.fit(X_tr_s, y_tr)
            pred = model.predict(X_va_s)
            accs.append(accuracy_score(y_va, pred))

            del model
            gc.collect()

        return np.mean(accs)

    return objective

# ── 9. MAIN ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    seed_everything(CFG.seed)
    t0 = time.time()

    print("=== EXP-012: Optimized Ensemble with Optuna ===")

    # Load data
    df, feature_cols = load_data()
    X_all = df[feature_cols].values
    y = df["label"].values
    groups = df["subject"].values

    print(f"Loaded: {len(df)} samples, {len(feature_cols)} features")
    print(f"PD: {y.sum()}, HC: {(y==0).sum()}, Subjects: {len(np.unique(groups))}")

    # Feature selection: top 30
    top_feats = get_top_features(X_all, y, groups, feature_cols, top_k=30)
    feat_idx = [feature_cols.index(f) for f in top_feats]
    X = X_all[:, feat_idx]
    print(f"\nUsing {len(top_feats)} features for optimization")

    # Optuna tuning for each model
    best_params = {}
    for model_type in ["lgb", "xgb", "cb"]:
        print(f"\n--- Tuning {model_type.upper()} ({CFG.n_optuna} trials) ---")
        study = optuna.create_study(direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=CFG.seed))
        study.optimize(make_objective(X, y, groups, model_type),
                       n_trials=CFG.n_optuna, show_progress_bar=False)
        best_params[model_type] = study.best_params
        print(f"  Best {model_type}: acc={study.best_value:.4f}")
        print(f"  Params: {study.best_params}")

    # Final evaluation with tuned params
    print(f"\n{'='*60}")
    print("FINAL EVALUATION: Tuned ensemble on 5-fold CV")
    print(f"{'='*60}")

    sgkf = StratifiedGroupKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
    all_y_true, all_y_pred, all_y_prob = [], [], []
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        seed_everything(CFG.seed + fold)
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        probs = []

        # LightGBM
        lgb_p = best_params["lgb"].copy()
        m = lgb.LGBMClassifier(**lgb_p, random_state=CFG.seed, verbose=-1, n_jobs=1)
        m.fit(X_tr_s, y_tr)
        probs.append(m.predict_proba(X_va_s)[:, 1])
        del m

        # XGBoost
        xgb_p = best_params["xgb"].copy()
        m = xgb.XGBClassifier(**xgb_p, random_state=CFG.seed, verbosity=0,
                              n_jobs=1, eval_metric="logloss")
        m.fit(X_tr_s, y_tr)
        probs.append(m.predict_proba(X_va_s)[:, 1])
        del m

        # CatBoost
        cb_p = best_params["cb"].copy()
        m = cb.CatBoostClassifier(**cb_p, random_seed=CFG.seed, verbose=0, thread_count=1)
        m.fit(X_tr_s, y_tr)
        probs.append(m.predict_proba(X_va_s)[:, 1])
        del m

        avg_prob = np.mean(probs, axis=0)
        avg_pred = (avg_prob >= 0.5).astype(int)

        fold_acc = accuracy_score(y_va, avg_pred)
        fold_scores.append(fold_acc)
        print(f"  Fold {fold+1}: acc={fold_acc:.4f}")

        all_y_true.extend(y_va.tolist())
        all_y_pred.extend(avg_pred.tolist())
        all_y_prob.extend(avg_prob.tolist())
        gc.collect()

    # Overall metrics
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)

    acc = float(accuracy_score(all_y_true, all_y_pred))
    f1 = float(f1_score(all_y_true, all_y_pred, average="macro", zero_division=0))
    auc = float(roc_auc_score(all_y_true, all_y_prob))
    cm = confusion_matrix(all_y_true, all_y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print("OVERALL RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:    {acc:.4f}")
    print(f"F1 macro:    {f1:.4f}")
    print(f"AUC-ROC:     {auc:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"Fold std:    {np.std(fold_scores):.4f}")
    print(f"Confusion:\n{cm}")
    print(f"\n{classification_report(all_y_true, all_y_pred, target_names=['HC', 'PD'])}")

    # ── 10. METRICS ────────────────────────────────────────────────────────
    metrics = [
        {"metric_name": "accuracy", "metric_value": acc, "split_name": "cv", "is_primary": True},
        {"metric_name": "f1_macro", "metric_value": f1, "split_name": "cv"},
        {"metric_name": "auc_roc", "metric_value": auc, "split_name": "cv"},
        {"metric_name": "sensitivity", "metric_value": sens, "split_name": "cv"},
        {"metric_name": "specificity", "metric_value": spec, "split_name": "cv"},
    ]
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    detailed = {
        "metrics": {k["metric_name"]: k["metric_value"] for k in metrics},
        "best_params": best_params,
        "top_features": top_feats,
        "fold_scores": fold_scores,
        "confusion_matrix": cm.tolist(),
        "pipeline_latency_seconds": round(elapsed, 1),
    }
    (OUTPUT_DIR / "detailed_results.json").write_text(json.dumps(detailed, indent=2, default=str))

    print("\n---")
    print(f"accuracy:    {acc:.4f}")
    print(f"f1_macro:    {f1:.4f}")
    print(f"auc_roc:     {auc:.4f}")
    print(f"sensitivity: {sens:.4f}")
    print(f"specificity: {spec:.4f}")
    print(f"fold_std:    {np.std(fold_scores):.4f}")
    print(f"elapsed:     {elapsed:.1f}s")

    if math.isnan(acc):
        print("FAIL: NaN metric")
        sys.exit(1)

    print(f"\n=== EXP-012 DONE in {elapsed:.1f}s ===")
