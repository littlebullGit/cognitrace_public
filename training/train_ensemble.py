"""
CogniTrace ML Training Pipeline

Trains a 3-model ensemble (LightGBM, XGBoost, CatBoost) on all 56 acoustic
biomarkers extracted by the Swift feature extractor. Exports ONNX models and
StandardScaler params for on-device inference.

Dataset: Italian PD Voice & Speech (IEEE DataPort, open access)
         65 subjects aged 50+, 831 WAV recordings
         Both PD patients and age-matched healthy controls

Evaluation: 5-fold StratifiedGroupKFold (stratified by label, grouped by
            subject -- no data leakage between train/val splits)

Ensemble: Simple probability averaging. With 65 subjects, there is not
          enough data to reliably estimate combination weights or train a
          stacking meta-learner without overfitting. All three models achieve
          similar individual AUC, so equal weighting is appropriate.

Usage:
    python train_ensemble.py data/swift_features_831.csv output/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import numpy as np
import onnxmltools
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

SEED = 42

# Columns that are metadata, not features.
META_COLUMNS = {
    "audio_path", "label", "subject", "filename", "s3_key", "sample_rate",
    "normalize_ms", "f0_track_ms", "f0_stats_ms", "jitter_ms", "shimmer_ms",
    "hnr_ms", "spectral_ms", "mfcc_ms", "energy_ms", "feature_total_ms",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLUMNS]


def make_models():
    shared = dict(random_state=SEED)
    return [
        ("lgb", lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.03,
            num_leaves=31, min_child_samples=5, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            verbose=-1, n_jobs=1, **shared,
        )),
        ("xgb", xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=0.1, verbosity=0, n_jobs=1,
            eval_metric="logloss", **shared,
        )),
        ("cb", cb.CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.03,
            random_seed=SEED, verbose=0, thread_count=1,
        )),
    ]


def evaluate(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    X = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y = (df["label"] == "PD").astype(int).to_numpy()
    groups = df["subject"].to_numpy()

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    all_true, all_pred, all_prob = [], [], []

    for train_idx, val_idx in cv.split(X, y, groups):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        probs = []
        for _, model in make_models():
            model.fit(X_train, y[train_idx])
            probs.append(model.predict_proba(X_val)[:, 1])

        avg_prob = np.mean(probs, axis=0)
        all_true.extend(y[val_idx].tolist())
        all_pred.extend((avg_prob >= 0.5).astype(int).tolist())
        all_prob.extend(avg_prob.tolist())

    return {
        "accuracy": round(accuracy_score(all_true, all_pred), 4),
        "f1_macro": round(f1_score(all_true, all_pred, average="macro"), 4),
        "auc_roc": round(roc_auc_score(all_true, all_prob), 4),
        "n_features": len(feature_cols),
        "n_samples": len(df),
        "n_subjects": df["subject"].nunique(),
        "cv_folds": 5,
    }


def train_and_export(
    df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
) -> None:
    X = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y = (df["label"] == "PD").astype(int).to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {}
    for name, model in make_models():
        model.fit(X_scaled, y)
        models[name] = model

    output_dir.mkdir(parents=True, exist_ok=True)

    # Scaler params (read by the Flutter app at runtime).
    (output_dir / "scaler_params.json").write_text(json.dumps({
        "feature_names": feature_cols,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }, indent=2))

    # ONNX export.
    n = len(feature_cols)
    float_type = onnxmltools.convert.common.data_types.FloatTensorType([None, n])

    lgb_onnx = onnxmltools.convert_lightgbm(
        models["lgb"], initial_types=[("features", float_type)],
    )
    xgb_onnx = onnxmltools.convert_xgboost(
        models["xgb"], initial_types=[("features", float_type)],
    )

    onnxmltools.utils.save_model(lgb_onnx, str(output_dir / "lgb_model.onnx"))
    onnxmltools.utils.save_model(xgb_onnx, str(output_dir / "xgb_model.onnx"))
    models["cb"].save_model(str(output_dir / "cb_model.onnx"), format="onnx")

    # Mobile-safe patched versions (remove ZipMap nodes for iOS ONNX Runtime).
    try:
        from patch_mobile_safe_onnx import patch_zipmap_model
        patch_zipmap_model(
            output_dir / "lgb_model.onnx",
            output_dir / "lgb_model_mobile.onnx",
        )
        patch_zipmap_model(
            output_dir / "cb_model.onnx",
            output_dir / "cb_model_mobile.onnx",
        )
    except ImportError:
        print("Warning: patch_mobile_safe_onnx not found, skipping mobile patches")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the CogniTrace 3-model ensemble on all 56 features.",
    )
    parser.add_argument("csv", type=Path, help="Swift-extracted features CSV")
    parser.add_argument("output", type=Path, help="Output directory for models")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    features = get_feature_columns(df)
    print(f"Dataset: {len(df)} recordings, {df['subject'].nunique()} subjects")
    print(f"Features: {len(features)}")

    metrics = evaluate(df, features)
    print(f"\n5-fold CV results:")
    print(f"  Accuracy: {metrics['accuracy']}")
    print(f"  F1 Macro: {metrics['f1_macro']}")
    print(f"  AUC-ROC:  {metrics['auc_roc']}")

    train_and_export(df, features, args.output)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nModels saved to {args.output}/")


if __name__ == "__main__":
    main()
