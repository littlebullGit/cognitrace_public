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

from patch_mobile_safe_onnx import patch_zipmap_model


SEED = 42
TOP_K = 30


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    meta = {
        "audio_path",
        "label",
        "subject",
        "filename",
        "s3_key",
        "sample_rate",
        "normalize_ms",
        "f0_track_ms",
        "f0_stats_ms",
        "jitter_ms",
        "shimmer_ms",
        "hnr_ms",
        "spectral_ms",
        "mfcc_ms",
        "energy_ms",
        "feature_total_ms",
    }
    return [c for c in df.columns if c not in meta]


def select_top_features(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    X = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y = (df["label"] == "PD").astype(int).to_numpy()

    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,
        random_state=SEED,
        verbose=-1,
        n_jobs=1,
    )
    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    return importance.sort_values(ascending=False).head(TOP_K).index.tolist()


def evaluate_models(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    X = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y = (df["label"] == "PD").astype(int).to_numpy()
    groups = df["subject"].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    all_true: list[int] = []
    all_pred: list[int] = []
    all_prob: list[float] = []

    for train_idx, val_idx in sgkf.split(X, y, groups):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        y_train = y[train_idx]
        y_val = y[val_idx]

        models = [
            lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.03,
                num_leaves=31,
                min_child_samples=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=SEED,
                verbose=-1,
                n_jobs=1,
            ),
            xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=SEED,
                verbosity=0,
                n_jobs=1,
                eval_metric="logloss",
            ),
            cb.CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.03,
                random_seed=SEED,
                verbose=0,
                thread_count=1,
            ),
        ]

        probs = []
        for model in models:
            model.fit(X_train, y_train)
            probs.append(model.predict_proba(X_val)[:, 1])

        avg_prob = np.mean(probs, axis=0)
        avg_pred = (avg_prob >= 0.5).astype(int)
        all_true.extend(y_val.tolist())
        all_pred.extend(avg_pred.tolist())
        all_prob.extend(avg_prob.tolist())

    all_true_np = np.array(all_true)
    all_pred_np = np.array(all_pred)
    all_prob_np = np.array(all_prob)
    return {
        "accuracy": float(accuracy_score(all_true_np, all_pred_np)),
        "f1_macro": float(f1_score(all_true_np, all_pred_np, average="macro")),
        "auc_roc": float(roc_auc_score(all_true_np, all_prob_np)),
    }


def train_final_models(df: pd.DataFrame, feature_cols: list[str], output_dir: Path) -> None:
    X = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y = (df["label"] == "PD").astype(int).to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "lgb": lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=SEED,
            verbose=-1,
            n_jobs=1,
        ),
        "xgb": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=SEED,
            verbosity=0,
            n_jobs=1,
            eval_metric="logloss",
        ),
        "cb": cb.CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.03,
            random_seed=SEED,
            verbose=0,
            thread_count=1,
        ),
    }

    for model in models.values():
        model.fit(X_scaled, y)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scaler_params.json").write_text(
        json.dumps(
            {
                "feature_names": feature_cols,
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            },
            indent=2,
        )
    )

    lgb_onnx = onnxmltools.convert_lightgbm(models["lgb"], initial_types=[("features", onnxmltools.convert.common.data_types.FloatTensorType([None, len(feature_cols)]))])
    xgb_onnx = onnxmltools.convert_xgboost(models["xgb"], initial_types=[("features", onnxmltools.convert.common.data_types.FloatTensorType([None, len(feature_cols)]))])

    raw_lgb = output_dir / "lgb_model.onnx"
    raw_xgb = output_dir / "xgb_model.onnx"
    raw_cb = output_dir / "cb_model.onnx"
    onnxmltools.utils.save_model(lgb_onnx, str(raw_lgb))
    onnxmltools.utils.save_model(xgb_onnx, str(raw_xgb))
    models["cb"].save_model(str(raw_cb), format="onnx")

    patch_zipmap_model(raw_lgb, output_dir / "lgb_model_mobile.onnx")
    patch_zipmap_model(raw_cb, output_dir / "cb_model_mobile.onnx")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain the CogniTrace ensemble on Swift-extracted features.")
    parser.add_argument("swift_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.swift_csv)
    feature_cols = get_feature_columns(df)
    top_features = select_top_features(df, feature_cols)
    metrics = evaluate_models(df, top_features)
    train_final_models(df, top_features, args.output_dir)

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.output_dir / "selected_features.json").write_text(json.dumps(top_features, indent=2))
    print(json.dumps({"metrics": metrics, "selected_features": top_features}, indent=2))


if __name__ == "__main__":
    main()
