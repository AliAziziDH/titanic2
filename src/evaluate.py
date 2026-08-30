"""Evaluation harness for the Titanic stacking pipeline.

Runs stacking, compares new Stratified 5-Fold OOF CV against baseline (stored in models/baseline.json,
defaulting to 0.8406), checks prediction distribution drift (~37.8% survival rate),
saves updated baseline upon improvement, and exits with appropriate status code.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, SUBMISSIONS_DIR, TARGET_COLUMN
from src.stacking import run_stacking_pipeline

LOGGER = logging.getLogger("titanic.evaluate")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

BASELINE_FILE = Path(MODELS_DIR) / "baseline.json"
DEFAULT_BASELINE_SCORE = 0.8180
TARGET_SURVIVAL_RATE = 0.378
SURVIVAL_RATE_TOLERANCE = 0.05  # ±5% acceptable drift tolerance (~32.8% - 42.8%)


def load_baseline() -> Dict[str, Any]:
    """Load baseline metrics from models/baseline.json or return default baseline."""
    if BASELINE_FILE.exists():
        try:
            with open(BASELINE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                LOGGER.info("Loaded baseline from %s: %s", BASELINE_FILE, data)
                return data
        except Exception as e:
            LOGGER.warning("Could not read baseline from %s: %s. Using default.", BASELINE_FILE, e)

    return {
        "cv_score": DEFAULT_BASELINE_SCORE,
        "metric": "accuracy",
        "description": "Default baseline score"
    }


def save_baseline(data: Dict[str, Any]) -> None:
    """Save updated baseline score to models/baseline.json."""
    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    LOGGER.info("Saved updated baseline to %s", BASELINE_FILE)


def evaluate() -> None:
    """Execute evaluation harness."""
    LOGGER.info("Starting Evaluation Harness...")

    baseline_data = load_baseline()
    baseline_score = float(baseline_data.get("cv_score", DEFAULT_BASELINE_SCORE))
    LOGGER.info("Target Baseline CV Accuracy: %.4f", baseline_score)

    # 1. Run Stacking Pipeline
    LOGGER.info("Running stacking pipeline...")
    submission = run_stacking_pipeline()

    # Load stacking results JSON
    from src.config import EXPERIMENTS_DIR
    results_path = Path(EXPERIMENTS_DIR) / "stacking_results.json"
    if not results_path.exists():
        LOGGER.error("Stacking results not found at %s", results_path)
        sys.exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        stacking_results = json.load(f)

    # Get CV Accuracy (from stacker metrics)
    new_score = float(stacking_results.get("stacker", {}).get("accuracy", 0.0))
    LOGGER.info("Evaluated New 5-Fold OOF CV Accuracy: %.4f (Baseline: %.4f)", new_score, baseline_score)

    # 2. Check Prediction Distribution / Drift
    if TARGET_COLUMN not in submission.columns:
        LOGGER.error("Submission does not contain column %s", TARGET_COLUMN)
        sys.exit(1)

    survival_rate = float(submission[TARGET_COLUMN].mean())
    LOGGER.info(
        "Predicted Survival Rate: %.4f (Expected: ~%.4f ± %.4f)",
        survival_rate,
        TARGET_SURVIVAL_RATE,
        SURVIVAL_RATE_TOLERANCE
    )

    drift_error = abs(survival_rate - TARGET_SURVIVAL_RATE)
    if drift_error > SURVIVAL_RATE_TOLERANCE:
        LOGGER.error(
            "CRITICAL: Prediction distribution drift detected! Survival rate is %.4f (expected ~%.4f, diff %.4f > tolerance %.4f)",
            survival_rate,
            TARGET_SURVIVAL_RATE,
            drift_error,
            SURVIVAL_RATE_TOLERANCE
        )
        sys.exit(1)

    # 3. Check Performance Degradation
    # sys.exit(1) if score degrades against baseline
    if new_score < baseline_score:
        LOGGER.error(
            "CRITICAL: Performance degraded! New OOF CV Accuracy (%.4f) < Baseline (%.4f)",
            new_score,
            baseline_score
        )
        sys.exit(1)

    # 4. Save new baseline if improved or maintained
    LOGGER.info("Evaluation PASSED! New score %.4f >= Baseline %.4f without drift.", new_score, baseline_score)
    updated_baseline = {
        "cv_score": new_score,
        "metric": "accuracy",
        "predicted_survival_rate": survival_rate,
        "stacker_metrics": stacking_results.get("stacker", {})
    }
    save_baseline(updated_baseline)
    sys.exit(0)


if __name__ == "__main__":
    evaluate()
