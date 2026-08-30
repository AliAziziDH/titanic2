"""OOF stacking ensemble for the Titanic project."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.config import (
    DATA_PROCESSED_DIR,
    EXPERIMENTS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    SUBMISSIONS_DIR,
    TARGET_COLUMN,
)
from src.modeling import build_preprocessor, load_modeling_data, WCGSurvivalEncoder, AgeImputer, compute_ipw_weights, FoldLocalGroupFeatures
from src.csa_allocator import ConfidentSinkhornAllocator

LOGGER = logging.getLogger("titanic.stacking")
FEATURE_EXCLUSIONS = {TARGET_COLUMN, "PassengerId"}


def _base_models(seed: int = RANDOM_STATE) -> Dict[str, Any]:
    """Build base estimators with safe CPU defaults."""
    models: Dict[str, Any] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=seed, n_jobs=1
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(32,), activation="relu", solver="adam",
            alpha=0.1, batch_size=32, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=10,
            random_state=seed, max_iter=500,
        ),
    }
    try:
        from catboost import CatBoostClassifier

        models["CatBoost"] = CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=4,
            l2_leaf_reg=50.0, subsample=0.7,
            random_seed=seed, verbose=False, task_type="CPU",
        )
    except ImportError:
        LOGGER.warning("CatBoost is unavailable; skipping it")
    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            max_depth=4, reg_lambda=50.0,
            subsample=0.7, colsample_bytree=0.7,
            random_state=seed, device="cpu", verbosity=-1,
            n_jobs=1,
        )
    except ImportError:
        LOGGER.warning("LightGBM is unavailable; skipping it")

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            reg_lambda=50.0, subsample=0.7, colsample_bytree=0.7,
            random_state=seed, tree_method="hist", eval_metric="logloss",
            n_jobs=1,
        )
    except ImportError:
        LOGGER.warning("XGBoost is unavailable; skipping it")

    return models


def _pipeline(model: Any, frame: pd.DataFrame, seed: int = RANDOM_STATE) -> Pipeline:
    """Build a leakage-safe pipeline with FoldLocalGroupFeatures as step 0.

    frame must be the fold-train slice so that the probe for build_preprocessor
    reflects the correct post-transform column set.
    """
    _probe = FoldLocalGroupFeatures().fit(frame).transform(frame.copy())
    return ImbPipeline([
        ("fold_features", FoldLocalGroupFeatures()),   # step 0: fold-local, no leakage
        ("wcg_encoder", WCGSurvivalEncoder()),
        ("age_imputer", AgeImputer(random_state=seed)),
        ("preprocessor", build_preprocessor(_probe)),
        ("model", model)
    ])


def _metrics(y_true: pd.Series, probabilities: np.ndarray) -> Dict[str, float]:
    labels = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, labels)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "f1_macro": float(f1_score(y_true, labels, average="macro")),
    }


def generate_meta_oof_predictions(
    blend_oof: pd.DataFrame, y: pd.Series, seed: int = RANDOM_STATE
) -> tuple[np.ndarray, list[Dict[str, float]], LogisticRegression]:
    """Generate second-level OOF predictions for the logistic meta-model."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
    meta_oof = np.full(len(blend_oof), np.nan, dtype=float)
    fold_scores: list[Dict[str, float]] = []

    for fold, (fit_idx, valid_idx) in enumerate(cv.split(blend_oof, y), start=1):
        meta_model = LogisticRegression(C=0.01, penalty='l2', max_iter=1000, random_state=seed)
        meta_model.fit(blend_oof.iloc[fit_idx], y.iloc[fit_idx])
        probabilities = meta_model.predict_proba(blend_oof.iloc[valid_idx])[:, 1]
        meta_oof[valid_idx] = probabilities
        fold_scores.append(_metrics(y.iloc[valid_idx], probabilities))
        LOGGER.info("Generated meta OOF predictions for fold %d/5", fold)

    if np.isnan(meta_oof).any():
        raise RuntimeError("Meta OOF prediction matrix contains missing values")

    final_meta_model = LogisticRegression(C=0.01, penalty='l2', max_iter=1000, random_state=seed)
    final_meta_model.fit(blend_oof, y)
    return meta_oof, fold_scores, final_meta_model


def generate_oof_predictions(
    X: pd.DataFrame, y: pd.Series, models: Dict[str, Any], seed: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], Dict[str, np.ndarray]]:
    """Generate one probability per row from each base model using OOF folds."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
    oof = pd.DataFrame(index=X.index, columns=models.keys(), dtype=float)
    fold_scores: Dict[str, list[Dict[str, float]]] = {name: [] for name in models}
    for fold, (fit_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        for name, estimator in models.items():
            fitted = _pipeline(clone(estimator), X.iloc[fit_idx], seed=seed)
            fitted.fit(X.iloc[fit_idx], y.iloc[fit_idx])
            probabilities = fitted.predict_proba(X.iloc[valid_idx])[:, 1]
            oof.iloc[valid_idx, oof.columns.get_loc(name)] = probabilities
            fold_scores[name].append(_metrics(y.iloc[valid_idx], probabilities))
        LOGGER.info("Generated OOF predictions for fold %d/5", fold)

    full_models: Dict[str, Pipeline] = {}
    for name, estimator in models.items():
        fitted = _pipeline(clone(estimator), X, seed=seed)
        fitted.fit(X, y)
        full_models[name] = fitted
        LOGGER.info(
            "%s OOF metrics: %s",
            name,
            {key: round(float(np.mean([score[key] for score in values])), 4)
             for key in ["accuracy", "roc_auc", "f1_macro"]
             for values in [fold_scores[name]]},
        )
    return oof, full_models, fold_scores


def run_stacking_pipeline() -> pd.DataFrame:
    """Train the OOF stacker and create the stacking submission."""
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(EXPERIMENTS_DIR).mkdir(parents=True, exist_ok=True)
    train, test = load_modeling_data()
    features = [column for column in train.columns if column not in FEATURE_EXCLUSIONS]
    X_train, y_train = train[features], train[TARGET_COLUMN].astype(int)
    X_test = test[features]

    SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    oof_meta_probs_list = []
    test_predictions_list = []
    all_fold_scores = {}
    # Streamlined, structurally diverse blend: XGBoost (numerical), CatBoost (categorical), MLP (neural smooth)
    blend_models = ["XGBoost", "CatBoost", "MLP"]

    for seed in SEEDS:
        LOGGER.info(f"=== Running Stacking Pipeline with Seed {seed} ===")
        models = _base_models(seed=seed)
        oof, full_models, fold_scores = generate_oof_predictions(X_train, y_train, models, seed=seed)
        if oof.isna().any().any():
            raise RuntimeError(f"OOF prediction matrix for seed {seed} contains missing values")

        # Accumulate fold scores
        for name, scores in fold_scores.items():
            if name not in all_fold_scores:
                all_fold_scores[name] = []
            all_fold_scores[name].extend(scores)

        curr_blend_models = [m for m in blend_models if m in oof.columns]
        if not curr_blend_models:
            blend_oof = oof
            curr_blend_models = list(oof.columns)
        else:
            blend_oof = oof[curr_blend_models]

        meta_probabilities, meta_fold_scores, meta_model = generate_meta_oof_predictions(
            blend_oof, y_train, seed=seed
        )
        oof_meta_probs_list.append(meta_probabilities)
        all_fold_scores[f"StackMeta_seed_{seed}"] = meta_fold_scores

        test_base_initial = pd.DataFrame({
            name: full_models[name].predict_proba(X_test)[:, 1] for name in curr_blend_models
        })
        test_predictions_initial = meta_model.predict_proba(test_base_initial)[:, 1]

        # Semi-Supervised Pseudo-Labeling via Confident Sinkhorn Allocation
        allocator = ConfidentSinkhornAllocator()
        high_conf_idx, pseudo_labels = allocator.fit_allocate(test_predictions_initial, top_pct=0.25)

        X_train_aug = pd.concat([X_train, X_test.iloc[high_conf_idx]], ignore_index=True)
        y_train_aug = pd.concat([y_train, pd.Series(pseudo_labels[high_conf_idx])], ignore_index=True)

        retrained_models: Dict[str, Pipeline] = {}
        supported_models = ['CatBoost', 'XGBoost', 'RandomForest', 'LightGBM']

        for name, estimator in models.items():
            fitted = _pipeline(clone(estimator), X_train_aug, seed=seed)
            if name in supported_models:
                weights = compute_ipw_weights(X_train_aug)
                fitted.fit(X_train_aug, y_train_aug, model__sample_weight=weights)
            else:
                fitted.fit(X_train_aug, y_train_aug)
            retrained_models[name] = fitted

        test_base_retrained = pd.DataFrame({
            name: retrained_models[name].predict_proba(X_test)[:, 1] for name in curr_blend_models
        })
        test_predictions_final = meta_model.predict_proba(test_base_retrained)[:, 1]
        test_predictions_list.append(test_predictions_final)

        # Dump models with seed suffix
        for name, model in retrained_models.items():
            joblib.dump(model, Path(MODELS_DIR) / f"stacking_{name.lower()}_seed_{seed}.joblib")
        joblib.dump(meta_model, Path(MODELS_DIR) / f"stacking_meta_model_seed_{seed}.joblib")

        # Save seed 42 models without suffix for backward compatibility
        if seed == 42:
            for name, model in retrained_models.items():
                joblib.dump(model, Path(MODELS_DIR) / f"stacking_{name.lower()}.joblib")
            joblib.dump(meta_model, Path(MODELS_DIR) / "stacking_meta_model.joblib")

    # Average OOF and test probabilities over all seeds
    avg_meta_oof_probs = np.mean(oof_meta_probs_list, axis=0)
    avg_test_predictions = np.mean(test_predictions_list, axis=0)

    # Save averaged OOF predictions
    oof_path = Path(EXPERIMENTS_DIR) / "oof_predictions.npz"
    np.savez(oof_path, predictions=avg_meta_oof_probs, target=y_train.to_numpy())

    def optimize_threshold(oof_probs, y_true):
        """Find the accuracy-optimal threshold subject to an OOF positive-rate guard.

        The meta-model (LogisticRegression C=0.01, very heavy L2) produces a
        bimodal probability distribution with a sparse valley around [0.40, 0.50).
        Pure accuracy maximisation anchors the threshold inside that valley at
        ~0.4687, yielding only ~29% OOF predicted-positive rate even though the
        true rate is 38.4%.  That valley-locked threshold is a bimodal artefact,
        not a calibrated decision boundary.

        Fix: search thresholds in [0.30, 0.60] and require the OOF predicted-
        positive rate to fall within [32%, 43%] (a loose bracket around the
        true 38.4% rate, computed purely from OOF data — no test data involved).
        If no threshold satisfies the constraint, fall back to the unconstrained
        accuracy maximum.
        """
        n = len(y_true)
        lo_guard = int(0.32 * n)  # 32% floor — 285 for n=891
        hi_guard = int(0.43 * n)  # 43% ceiling — 383 for n=891

        best_threshold = 0.50
        best_score = 0.0
        best_unconstrained_t = 0.50
        best_unconstrained_score = 0.0

        for t in np.linspace(0.30, 0.60, 300):
            preds = (oof_probs >= t).astype(int)
            score = accuracy_score(y_true, preds)
            n_pos = int(preds.sum())

            # Track unconstrained best as fallback
            if score > best_unconstrained_score:
                best_unconstrained_score = score
                best_unconstrained_t = t

            # Accept only if OOF positive rate is within the calibration guard
            if lo_guard <= n_pos <= hi_guard and score > best_score:
                best_score = score
                best_threshold = t

        if best_score == 0.0:
            # No threshold satisfied the constraint; use unconstrained fallback
            LOGGER.warning(
                "No threshold in [0.30, 0.60] satisfied OOF-rate guard [%d, %d]; "
                "falling back to unconstrained best %.4f",
                lo_guard, hi_guard, best_unconstrained_t,
            )
            best_threshold = best_unconstrained_t
            best_score = best_unconstrained_score

        LOGGER.info(
            "🏆 Optimal decision threshold (OOF-rate constrained) found on averaged OOF: "
            "%.4f (Accuracy: %.4f, OOF survivors: %d / %d = %.1f%%)",
            best_threshold, best_score,
            int((oof_probs >= best_threshold).sum()), n,
            100.0 * (oof_probs >= best_threshold).mean(),
        )
        return best_threshold


    optimal_threshold = optimize_threshold(avg_meta_oof_probs, y_train)
    stack_metrics = _metrics(y_train, avg_meta_oof_probs)
    stack_metrics["training_metrics"] = _metrics(y_train, avg_meta_oof_probs)
    LOGGER.info("Meta-Model Bagged OOF metrics: %s", stack_metrics)

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        TARGET_COLUMN: (avg_test_predictions >= optimal_threshold).astype(int),
    })
    submission_path = Path(SUBMISSIONS_DIR) / "submission_stacking.csv"
    submission.to_csv(submission_path, index=False)

    submission_prob = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        TARGET_COLUMN: avg_test_predictions,
    })
    submission_prob_path = Path(SUBMISSIONS_DIR) / "submission_blend_probabilities.csv"
    submission_prob.to_csv(submission_prob_path, index=False)

    results: Dict[str, Any] = {
        "base_models": {
            name: {
                "accuracy": float(np.mean([score["accuracy"] for score in scores])),
                "roc_auc": float(np.mean([score["roc_auc"] for score in scores])),
                "f1_macro": float(np.mean([score["f1_macro"] for score in scores])),
            }
            for name, scores in all_fold_scores.items()
        },
        "stacker": stack_metrics,
        "submission": {"path": str(submission_path), "rows": len(submission)},
    }
    (Path(EXPERIMENTS_DIR) / "stacking_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    # Save the optimal threshold so final_submission can use it
    summary_path = Path(SUBMISSIONS_DIR) / "submission_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
        except Exception:
            summary = {}
    else:
        summary = {}
    summary["optimal_threshold"] = optimal_threshold
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    LOGGER.info("Multi-Seed Stacking submission saved to %s", submission_path)
    return submission


if __name__ == "__main__":
    run_stacking_pipeline()
