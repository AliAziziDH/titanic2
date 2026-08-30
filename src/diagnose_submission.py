"""Diagnostics for investigating Titanic submission performance."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED_DIR, MODELS_DIR, TARGET_COLUMN

LOGGER = logging.getLogger("titanic.diagnostics")
PROJECT_DIR = Path(__file__).resolve().parent.parent
SUBMISSION_PATH = PROJECT_DIR / "submissions" / "submission_stacking.csv"
FEATURES_SOURCE = PROJECT_DIR / "src" / "features.py"
IMPUTATION_SOURCE = PROJECT_DIR / "src" / "imputation.py"


def _check_submission(test: pd.DataFrame) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": str(SUBMISSION_PATH), "exists": SUBMISSION_PATH.exists()}
    if not SUBMISSION_PATH.exists():
        return result
    submission = pd.read_csv(SUBMISSION_PATH)

    if len(submission) != 418:
        raise ValueError(f"CRITICAL ERROR: Refusing to diagnose submission. Submission has {len(submission)} rows. Must be exactly 418!")
    if not (submission["PassengerId"].min() == 892 and submission["PassengerId"].max() == 1309):
        raise ValueError(f"CRITICAL ERROR: PassengerId range is {submission['PassengerId'].min()}-{submission['PassengerId'].max()}. Must be exactly 892-1309!")
    result.update({
        "shape": list(submission.shape),
        "columns": list(submission.columns),
        "expected_shape": [len(test), 2],
        "format_valid": (
            bool(len(submission) == len(test)
            and list(submission.columns) == ["PassengerId", TARGET_COLUMN]
            and submission["PassengerId"].equals(test["PassengerId"])
            and submission[TARGET_COLUMN].isin([0, 1]).all())
        ),
        "predicted_survival_rate": float(submission[TARGET_COLUMN].mean()),
        "value_counts": submission[TARGET_COLUMN].value_counts().to_dict(),
    })
    return result


def _scan_source_for_leakage() -> Dict[str, Any]:
    features_source = FEATURES_SOURCE.read_text(encoding="utf-8")
    imputation_source = IMPUTATION_SOURCE.read_text(encoding="utf-8")
    feature_leakage_patterns = {
        "train_test_concat_in_features": bool(
            re.search(r"(concat|merge).*(train|test)|(train|test).*(concat|merge)", features_source, re.I | re.S)
        ),
        "target_used_in_feature_engineering": bool(
            re.search(r"df\s*\[\s*[\"']Survived[\"']\s*\]|df\s*\[\s*TARGET_COLUMN\s*\]", features_source)
        ),
        "target_in_age_features": bool(
            re.search(r"AGE_FEATURES\s*=\s*\[[^\]]*(Survived|TARGET_COLUMN)", imputation_source, re.I | re.S)
        ),
    }
    return {
        "feature_engineering_source_checks": feature_leakage_patterns,
        "ticket_count_scope": "save_engineered_data computes ticket counts from the combined unlabeled train/test reference.",
        "fare_bin_scope": "save_engineered_data computes fare-bin edges from the combined unlabeled train/test reference.",
    }


def _check_distributions(train: pd.DataFrame, test: pd.DataFrame, submission: pd.DataFrame | None) -> Dict[str, Any]:
    result = {
        "train_survival_rate": float(train[TARGET_COLUMN].mean()),
        "test_rows": len(test),
        "feature_columns_match": [
            column for column in train.columns if column != TARGET_COLUMN
        ] == list(test.columns),
    }
    if submission is not None:
        result["predicted_survival_rate"] = float(submission[TARGET_COLUMN].mean())
        result["rate_difference"] = result["predicted_survival_rate"] - result["train_survival_rate"]
    return result


def _check_model_preprocessing(test: pd.DataFrame) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    model_path = Path(MODELS_DIR) / "stacking_meta_model.joblib"
    expected_order = ["RandomForest", "MLP", "CatBoost", "LightGBM", "XGBoost"]
    base_paths = [
        Path(MODELS_DIR) / f"stacking_{name.lower()}.joblib"
        for name in expected_order
        if (Path(MODELS_DIR) / f"stacking_{name.lower()}.joblib").exists()
    ]
    result["stacking_meta_model_exists"] = model_path.exists()
    result["base_model_count"] = len(base_paths)
    result["base_model_order"] = [path.stem.removeprefix("stacking_") for path in base_paths]
    result["expected_meta_model_order"] = [name.lower() for name in expected_order]
    result["base_model_order_matches"] = result["base_model_order"] == result["expected_meta_model_order"]
    if not base_paths or not model_path.exists():
        result["predictability_check"] = False
        return result
    try:
        predictions = []
        meta = joblib.load(model_path)

        if isinstance(meta, np.ndarray):
            blend_order = ["XGBoost", "LightGBM", "CatBoost", "MLP"]
            blend_paths = [Path(MODELS_DIR) / f"stacking_{n.lower()}.joblib" for n in blend_order]
            for path in blend_paths:
                model = joblib.load(path)
                predictions.append(model.predict_proba(test)[:, 1])
            base = pd.DataFrame(np.column_stack(predictions))
            result["meta_input_shape"] = list(base.shape)
            result["predictability_check"] = len(np.dot(base, meta)) == len(test)
        else:
            blend_order = ["XGBoost", "LightGBM", "CatBoost", "MLP"]
            blend_paths = [Path(MODELS_DIR) / f"stacking_{n.lower()}.joblib" for n in blend_order]
            if not all(path.exists() for path in blend_paths):
                blend_paths = base_paths
            for path in blend_paths:
                model = joblib.load(path)
                predictions.append(model.predict_proba(test)[:, 1])
            base = pd.DataFrame(np.column_stack(predictions))
            result["meta_input_shape"] = list(base.shape)
            result["predictability_check"] = len(meta.predict_proba(base)) == len(test)
    except (ValueError, KeyError, OSError) as error:
        result["predictability_check"] = False
        result["error"] = str(error)
    return result


def run_diagnostics() -> Dict[str, Any]:
    """Run all submission and leakage diagnostics and print findings."""
    train = pd.read_csv(Path(DATA_PROCESSED_DIR) / "train_clean.csv")
    test = pd.read_csv(Path(DATA_PROCESSED_DIR) / "test_clean.csv")
    submission = pd.read_csv(SUBMISSION_PATH) if SUBMISSION_PATH.exists() else None
    report = {
        "submission": _check_submission(test),
        "leakage": _scan_source_for_leakage(),
        "distribution": _check_distributions(train, test, submission),
        "preprocessing": _check_model_preprocessing(test),
        "recommendations": [
            "Treat a Titanic public LB score of 0.9099 as strong performance, not a catastrophic drop.",
            "Recompute Ticket_Count and Fare quantile bins from a combined unlabeled train+test feature reference if cross-split group consistency is desired.",
            "Regenerate all engineered and clean files before retraining after any feature-logic change.",
            "Use the exact saved stacking base-model order when creating meta-model inputs.",
        ],
    }
    print(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    run_diagnostics()
