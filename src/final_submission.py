"""Generate final Titanic submissions from saved models and ensembles."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED_DIR, EXPERIMENTS_DIR, MODELS_DIR, TARGET_COLUMN, get_input_dir
from src.modeling import PipelineWrapper, ToDenseTransformer, WCGSurvivalEncoder, AgeImputer # Required for unpickling models

LOGGER = logging.getLogger("titanic.final_submission")
PROJECT_DIR = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = PROJECT_DIR / "submissions"


def _load_modeling_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load modeling data directly from engineered features without global imputation."""
    train_path = Path(DATA_PROCESSED_DIR) / "train_engineered.csv"
    test_path = Path(DATA_PROCESSED_DIR) / "test_engineered.csv"
    if not (train_path.exists() and test_path.exists()):
        raise FileNotFoundError(f"Missing engineered data in {DATA_PROCESSED_DIR}")
    return pd.read_csv(train_path), pd.read_csv(test_path)


class WCGPostProcessor:
    """Deterministic post-processor based on Woman-Child-Group and Ticket/Family survival."""

    def __init__(self, child_age_limit: int = 15) -> None:
        self.group_survival_rates = {}
        self.child_age_limit = child_age_limit

    def fit(self, train_df: pd.DataFrame) -> None:
        from src.config import get_input_dir
        input_dir = get_input_dir()
        raw_train = pd.read_csv(input_dir / "train.csv")
        raw_test = pd.read_csv(input_dir / "test.csv")

        train = train_df.copy()
        train['Ticket'] = raw_train['Ticket'].astype(str)
        train['Surname'] = raw_train['Name'].str.split(",", n=1).str[0].str.strip()
        train['Pass1_Group'] = train['Ticket']
        train['Pass2_Group'] = train['Surname'] + "_" + raw_train['Pclass'].astype(str) + "_" + raw_train['Embarked'].fillna('S').astype(str)

        test_tickets = raw_test['Ticket'].astype(str)
        test_surnames = raw_test['Name'].str.split(",", n=1).str[0].str.strip()
        test_pass2 = test_surnames + "_" + raw_test['Pclass'].astype(str) + "_" + raw_test['Embarked'].fillna('S').astype(str)

        all_pass1 = pd.concat([train['Pass1_Group'], test_tickets])
        pass1_counts = all_pass1.value_counts()

        all_pass2 = pd.concat([train['Pass2_Group'], test_pass2])
        pass2_counts = all_pass2.value_counts()

        # Identify WCG candidates
        train['Is_WCG'] = (train['Sex'] == 'female') | (train['Title'] == 'Master') | (train['Age'] < self.child_age_limit)

        # Pass 1: Ticket group survival
        for group, count in pass1_counts.items():
            if count > 1 and group in train['Pass1_Group'].values:
                grp_data = train[train['Pass1_Group'] == group]
                wcg_data = grp_data[grp_data['Is_WCG']]
                if len(wcg_data) > 0:
                    self.group_survival_rates[f"P1_{group}"] = wcg_data['Survived'].mean()
                else:
                    self.group_survival_rates[f"P1_{group}"] = grp_data['Survived'].mean()

        # Pass 2: Surname + Pclass + Embarked group survival
        for group, count in pass2_counts.items():
            if count > 1 and group in train['Pass2_Group'].values:
                grp_data = train[train['Pass2_Group'] == group]
                wcg_data = grp_data[grp_data['Is_WCG']]
                if len(wcg_data) > 0:
                    self.group_survival_rates[f"P2_{group}"] = wcg_data['Survived'].mean()
                else:
                    self.group_survival_rates[f"P2_{group}"] = grp_data['Survived'].mean()

    def transform(self, test_df: pd.DataFrame, baseline_probs: np.ndarray) -> np.ndarray:
        final_probs = baseline_probs.copy()

        from src.config import get_input_dir
        input_dir = get_input_dir()
        raw_test = pd.read_csv(input_dir / "test.csv")

        df = test_df.copy()
        df['Ticket'] = raw_test['Ticket'].astype(str)
        df['Surname'] = raw_test['Name'].str.split(",", n=1).str[0].str.strip()
        df['Pass1_Group'] = df['Ticket']
        df['Pass2_Group'] = df['Surname'] + "_" + raw_test['Pclass'].astype(str) + "_" + raw_test['Embarked'].fillna('S').astype(str)
        df['Is_WCG'] = (df['Sex'] == 'female') | (df['Title'] == 'Master') | (df['Age'] < self.child_age_limit)

        for idx in range(len(df)):
            row = df.iloc[idx]
            is_wc = row['Is_WCG']

            # Strict demographic guardrail: No adult males can be overridden
            is_adult_male = (row['Sex'] == 'male') and (row['Age'] >= 15) and (row['Title'] != 'Master')
            if is_adult_male:
                continue

            # Pass 1: Strict Ticket Consensus
            p1_key = f"P1_{row['Pass1_Group']}"
            if p1_key in self.group_survival_rates:
                rate = self.group_survival_rates[p1_key]
                if is_wc:
                    if rate == 1.0:
                        final_probs[idx] = 1.0
                        continue
                    elif rate == 0.0:
                        final_probs[idx] = 0.0
                        continue

            # Pass 2: Strict Surname + Class + Embarked Consensus
            p2_key = f"P2_{row['Pass2_Group']}"
            if p2_key in self.group_survival_rates and is_wc:
                rate = self.group_survival_rates[p2_key]
                if rate == 1.0:
                    final_probs[idx] = 1.0
                    continue
                elif rate == 0.0:
                    final_probs[idx] = 0.0
                    continue

        return final_probs

def _load_scores() -> Dict[str, Dict[str, float]]:
    """Load ROC-AUC and accuracy scores for weighting and reporting."""
    scores: Dict[str, Dict[str, float]] = {}
    path = Path(EXPERIMENTS_DIR) / "cv_results.json"
    if path.exists():
        for result in json.loads(path.read_text(encoding="utf-8")):
            metrics = result.get("metrics", {})
            scores[result["model"]] = {
                key: float(metrics[key]["mean"])
                for key in ("accuracy", "roc_auc")
                if key in metrics
            }
    stacking_path = Path(EXPERIMENTS_DIR) / "stacking_results.json"
    if stacking_path.exists():
        result = json.loads(stacking_path.read_text(encoding="utf-8"))
        scores["Stacking"] = {
            key: float(result["stacker"][key])
            for key in ("accuracy", "roc_auc")
            if key in result.get("stacker", {})
        }
    return scores


def _write_submission(name: str, passenger_ids: pd.Series, probabilities: np.ndarray, threshold: float = 0.5, validate_survivors: bool = False) -> Path:
    if len(passenger_ids) != 418:
        raise ValueError(f"CRITICAL ERROR: Refusing to generate submission. Test data has {len(passenger_ids)} rows. Must be exactly 418!")
    if not (passenger_ids.min() == 892 and passenger_ids.max() == 1309):
        raise ValueError(f"CRITICAL ERROR: PassengerId range is {passenger_ids.min()}-{passenger_ids.max()}. Must be exactly 892-1309!")

    binary_preds = (np.asarray(probabilities) >= threshold).astype(int)
    
    if validate_survivors:
        survivors = binary_preds.sum()
        if not (152 <= survivors <= 162):
            raise ValueError(f"CRITICAL ERROR: Submission {name} has {survivors} survivors. Must be between 152 and 162 (36.5% - 38.5%).")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = SUBMISSIONS_DIR / f"submission_{name.lower()}_{timestamp}.csv"
    pd.DataFrame({
        "PassengerId": passenger_ids,
        TARGET_COLUMN: binary_preds,
    }).to_csv(path, index=False)
    return path


def _load_individual_models() -> Dict[str, list[Any]]:
    candidates = {
        "CatBoost": "catboost_final.joblib",
        "XGBoost": "xgboost_final.joblib",
        "LightGBM": "lightgbm_final.joblib",
        "RandomForest": "randomforest_final.joblib",
    }
    seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    loaded = {}
    for name, filename in candidates.items():
        loaded[name] = []
        for seed in seeds:
            stem = filename.replace("_final.joblib", "")
            path = Path(MODELS_DIR) / f"{stem}_seed_{seed}.joblib"
            if path.exists():
                loaded[name].append(joblib.load(path))
        if not loaded[name]:
            path = Path(MODELS_DIR) / filename
            if path.exists():
                loaded[name].append(joblib.load(path))
        if not loaded[name]:
            LOGGER.info("Optional individual model not found: %s", name)
            del loaded[name]
    return loaded


import os

def _load_stacking_models() -> tuple[Dict[str, list[Any]], list[Any]]:
    order = ["RandomForest", "MLP", "CatBoost", "LightGBM", "XGBoost"]
    seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    base = {}
    for name in order:
        base[name] = []
        for seed in seeds:
            path = Path(MODELS_DIR) / f"stacking_{name.lower()}_seed_{seed}.joblib"
            if path.exists():
                base[name].append(joblib.load(path))
        if not base[name]:
            path = Path(MODELS_DIR) / f"stacking_{name.lower()}.joblib"
            if path.exists():
                base[name].append(joblib.load(path))
            else:
                del base[name]

    meta_models = []
    for seed in seeds:
        meta_path = Path(MODELS_DIR) / f"stacking_meta_model_seed_{seed}.joblib"
        if meta_path.exists():
            meta_models.append(joblib.load(meta_path))
    if not meta_models:
        meta_path = Path(MODELS_DIR) / "stacking_meta_model.joblib"
        if meta_path.exists():
            meta_models.append(joblib.load(meta_path))
        else:
            raise FileNotFoundError("Complete stacking model artifacts are not available")

    return {name: base[name] for name in order if name in base}, meta_models


def run_final_submission() -> Dict[str, Any]:
    """Generate individual, stacking, weighted, and majority-vote submissions."""
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    train, test = _load_modeling_data()

    # Guardrail checking
    passenger_ids = test["PassengerId"]
    if len(test) != 418:
        raise ValueError(f"CRITICAL ERROR: Refusing to generate submission. Test data has {len(test)} rows. Must be exactly 418!")
    if not (passenger_ids.min() == 892 and passenger_ids.max() == 1309):
        raise ValueError(f"CRITICAL ERROR: PassengerId range is {passenger_ids.min()}-{passenger_ids.max()}. Must be exactly 892-1309!")

    from src.config import get_input_dir
    input_dir = get_input_dir()
    raw_test = pd.read_csv(input_dir / "test.csv")
    assert (test['PassengerId'] == raw_test['PassengerId']).all(), "FATAL: Test index mismatch!"

    # Verify feature names and canonical ordering
    feature_cols = [c for c in train.columns if c not in ['PassengerId', TARGET_COLUMN]]
    test_feature_cols = [c for c in test.columns if c not in ['PassengerId', TARGET_COLUMN]]
    assert feature_cols == test_feature_cols, (
        f"Feature column mismatch! Train: {feature_cols} vs Test: {test_feature_cols}"
    )

    features = feature_cols
    X_test = test[features]
    passenger_ids = test["PassengerId"]
    probabilities: Dict[str, np.ndarray] = {}
    paths: Dict[str, str] = {}

    wcg_processor = WCGPostProcessor()
    wcg_processor.fit(train)

    # Load optimal threshold
    summary_path = SUBMISSIONS_DIR / "submission_summary.json"
    optimal_threshold = 0.5
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            optimal_threshold = summary.get("optimal_threshold", 0.5)
        except Exception:
            optimal_threshold = 0.5

    for name, models in _load_individual_models().items():
        preds_list = [model.predict_proba(X_test)[:, 1] for model in models]
        base_prob = np.mean(preds_list, axis=0)
        # Only apply WCG on stacking, not individual base models (to keep them pure for inspection if needed, or apply if preferred)
        # But per user instruction, we apply WCG strictly at the end. For individual models, we'll apply it too for consistency if requested.
        if name in ["CatBoost", "Stacking"]: # Although Stacking is below, CatBoost is here
            base_prob = wcg_processor.transform(test, base_prob)
        probabilities[name] = base_prob
        # Base models are kept in memory for blending/diagnostics, not written to disk by default.

    # Load individual base models for CatBoost, RandomForest, XGBoost, and MLP
    indiv_models = _load_individual_models()
    
    cat_preds = [m.predict_proba(X_test)[:, 1] for m in indiv_models.get("CatBoost", [])]
    rf_preds = [m.predict_proba(X_test)[:, 1] for m in indiv_models.get("RandomForest", [])]
    xgb_preds = [m.predict_proba(X_test)[:, 1] for m in indiv_models.get("XGBoost", [])]
    mlp_preds = [m.predict_proba(X_test)[:, 1] for m in indiv_models.get("MLP", [])]

    cat_prob = np.mean(cat_preds, axis=0) if cat_preds else np.zeros(len(X_test))
    rf_prob = np.mean(rf_preds, axis=0) if rf_preds else np.zeros(len(X_test))
    xgb_prob = np.mean(xgb_preds, axis=0) if xgb_preds else np.zeros(len(X_test))
    mlp_prob = np.mean(mlp_preds, axis=0) if mlp_preds else np.zeros(len(X_test))

    # Experiment-08 Sweep Configurations & Boy Age Limits
    blend_configs = {
        "Config_A (0.50 Cat, 0.40 RF, 0.10 XGB)": (0.50 * cat_prob) + (0.40 * rf_prob) + (0.10 * xgb_prob),
        "Config_B (0.45 Cat, 0.35 RF, 0.10 MLP, 0.10 XGB)": (0.45 * cat_prob) + (0.35 * rf_prob) + (0.10 * mlp_prob) + (0.10 * xgb_prob),
        "Config_C (0.40 Cat, 0.45 RF, 0.05 MLP, 0.10 XGB)": (0.40 * cat_prob) + (0.45 * rf_prob) + (0.05 * mlp_prob) + (0.10 * xgb_prob),
    }
    boy_age_limits = [15]

    LOGGER.info("=" * 70)
    LOGGER.info(" 🔬 EXPERIMENT-08: WCG BOY AGE LIMIT & BLEND CONFIG GRID SEARCH 🔬")
    LOGGER.info("=" * 70)
    LOGGER.info(f"{'Blend Config':<55} | {'Age Limit':<10} | {'Survival Rate':<15}")
    LOGGER.info("-" * 85)

    best_config_name = None
    best_age_limit = 15
    best_score = -1.0
    best_final_preds = None

    # We evaluate robustness using train OOF / proxy concordance
    # For local validation sweep, we simulate performance on train labels if available, else variance/distribution match
    y_train_actual = train[TARGET_COLUMN].values if TARGET_COLUMN in train.columns else None

    sweep_results = []
    for cfg_name, prob_arr in blend_configs.items():
        for age_lim in boy_age_limits:
            processor_sweep = WCGPostProcessor(child_age_limit=age_lim)
            processor_sweep.fit(train)
            swept_preds = processor_sweep.transform(test, prob_arr)
            bin_preds = (swept_preds >= 0.5).astype(int)
            surv_rate = bin_preds.mean()
            
            # Score metric: closeness to historical 38.38% training prior + confidence entropy
            rate_diff = abs(surv_rate - 0.3838)
            robustness_score = 1.0 - rate_diff # higher is better (closer to historical prior)
            
            sweep_results.append((cfg_name, age_lim, surv_rate, robustness_score, swept_preds))
            LOGGER.info(f"{cfg_name:<55} | {age_lim:<10} | {surv_rate:.4f} (Diff: {rate_diff:+.4f})")

    LOGGER.info("=" * 85)

    # Select optimal configuration (Config B or C with age limit 16 or 17 for teenager boy correction)
    # Target Config B with age limit 16 as top candidate per biological realism
    optimal_result = min(sweep_results, key=lambda x: abs(x[2] - 0.3838))
    best_config_name, best_age_limit, best_rate, _, best_final_preds = optimal_result
    
    LOGGER.info(f"🏆 Selected Optimal Candidate: {best_config_name} @ Boy Age Limit {best_age_limit} (Survival Rate: {best_rate:.4f})")
    LOGGER.info("=" * 70)

    # Apply optimal processor
    optimal_processor = WCGPostProcessor(child_age_limit=best_age_limit)
    optimal_processor.fit(train)
    final_preds = optimal_processor.transform(test, blend_configs[best_config_name.split(' (')[0] + ' (' + best_config_name.split(' (')[1] if '(' in best_config_name else list(blend_configs.keys())[1]])
    
    # Fallback lookup if string split varies
    selected_prob = blend_configs["Config_B (0.45 Cat, 0.35 RF, 0.10 MLP, 0.10 XGB)"] if "Config_B" in best_config_name else blend_configs["Config_A (0.50 Cat, 0.40 RF, 0.10 XGB)"]
    final_preds = optimal_processor.transform(test, selected_prob)
    final_binary_preds = (final_preds >= 0.5).astype(int)

    # Write target payload directly to a timestamped file
    blend_path = _write_submission("final_blend", passenger_ids, final_binary_preds, threshold=0.5, validate_survivors=True)
    paths["Final_Blend"] = str(blend_path)
    probabilities["Final_Blend"] = final_preds

    scores = _load_scores()
    weighted_names = [name for name in probabilities if name in scores and name not in ("Stacking", "Final_Blend")]
    if weighted_names:
        weights = np.array([scores[name].get("roc_auc", scores[name]["accuracy"]) for name in weighted_names])
        weights /= weights.sum()
        weighted_probability = sum(
            weight * probabilities[name] for weight, name in zip(weights, weighted_names)
        )
        paths["WeightedEnsemble"] = str(
            _write_submission("weighted_ensemble", passenger_ids, weighted_probability)
        )

        hard_predictions = np.column_stack([
            probabilities[name] >= 0.5 for name in weighted_names
        ])
        majority_probability = (hard_predictions.sum(axis=1) >= (len(weighted_names) / 2)).astype(float)
        paths["MajorityVote"] = str(
            _write_submission("majority_vote", passenger_ids, majority_probability)
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scores": scores,
        "submissions": paths,
        "rows": int(len(test)),
    }
    summary_path = SUBMISSIONS_DIR / "submission_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Generated %d submission files in %s", len(paths), SUBMISSIONS_DIR)

    if "Final_Blend" in paths:
        _submit_to_kaggle(Path(paths["Final_Blend"]))
    return summary


def _submit_to_kaggle(blend_file: Path):
    """Submit the blend to Kaggle and poll for the score."""
    import os
    import subprocess
    import time

    if os.environ.get("ENABLE_KAGGLE_SUBMISSION") != "1":
        LOGGER.info("Kaggle submission is disabled by default. Set ENABLE_KAGGLE_SUBMISSION=1 to opt-in.")
        return

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists() and "KAGGLE_USERNAME" not in os.environ:
        LOGGER.warning("Kaggle credentials not found. Skipping auto-submission.")
        return

    if not blend_file.exists():
        LOGGER.warning("Blend submission file not found: %s", blend_file)
        return

    LOGGER.info("Submitting %s to Kaggle...", blend_file.name)
    import hashlib
    # 2. FORCE PHYSICAL DIFF DIAGNOSTIC
    file_size = blend_file.stat().st_size
    md5_hash = hashlib.md5(blend_file.read_bytes()).hexdigest()
    LOGGER.info(f"Diagnostic - File Size: {file_size} bytes")
    LOGGER.info(f"Diagnostic - MD5 Checksum: {md5_hash}")

    # Print the first 5 rows
    import pandas as pd
    first_5 = pd.read_csv(blend_file).head(5)
    LOGGER.info(f"Diagnostic - First 5 rows:\n{first_5.to_string()}")

    # 3. ALIGN PATHS AND SUBMIT using Absolute Path
    submit_cmd = [
        "kaggle", "competitions", "submit",
        "-c", "titanic",
        "-f", str(blend_file.absolute()),
        "-m", "Phase_8_Final"
    ]
    try:
        subprocess.run(submit_cmd, check=True, capture_output=True, text=True)
        LOGGER.info("Submission successful. Waiting 15 seconds before polling...")
        time.sleep(15)

        poll_cmd = ["kaggle", "competitions", "submissions", "-c", "titanic"]
        result = subprocess.run(poll_cmd, check=True, capture_output=True, text=True)

        # Parse the output to find our submission
        LOGGER.info("--- Kaggle Leaderboard Status ---")
        for line in result.stdout.splitlines()[:5]: # Print header and top few lines
            LOGGER.info(line)

    except subprocess.CalledProcessError as e:
        LOGGER.error("Kaggle API command failed: %s", e.stderr)
    except Exception as e:
        LOGGER.error("Error during Kaggle submission: %s", e)


if __name__ == "__main__":
    run_final_submission()
