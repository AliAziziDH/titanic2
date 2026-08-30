"""Central configuration for the Titanic project."""

import os
# Prevent macOS OpenMP / Intel MKL multi-runtime collision segmentation faults
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import logging
from pathlib import Path
from typing import Optional


# Keep this seed fixed for reproducible comparisons throughout the project.
RANDOM_SEED = 42
RANDOM_STATE = RANDOM_SEED


# Project paths.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR = DATA_PROCESSED_DIR
MODELS_DIR = BASE_DIR / "models"
SUBMISSIONS_DIR = BASE_DIR / "submissions"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
import os
KAGGLE_INPUT_DIR = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input/competitions/titanic"))


# Cross-validation settings.
N_FOLDS = 5
N_REPEATS = 3
CV_STRATEGY = "RepeatedStratifiedKFold"


# Dataset and feature definitions.
TARGET_COLUMN = "Survived"
NUMERICAL_FEATURES = ["Age", "Fare", "SibSp", "Parch"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]
TEXT_FEATURES = ["Name", "Ticket", "Cabin"]
DROP_FEATURES = ["PassengerId"]
ENGINEERED_FEATURES = []


# GPU settings.
USE_GPU = True
try:
    import torch
except ImportError:
    torch = None


def is_gpu_available() -> bool:
    """Return whether a CUDA GPU is available through PyTorch."""
    return torch is not None and torch.cuda.is_available()


DEVICE = "cuda" if is_gpu_available() else "cpu"


def get_input_dir() -> Path:
    """Return local raw data when present, otherwise the Kaggle input directory."""
    local_files = [DATA_RAW_DIR / "train.csv", DATA_RAW_DIR / "test.csv"]
    if all(path.exists() for path in local_files):
        return DATA_RAW_DIR
    return KAGGLE_INPUT_DIR


# Baseline model parameters.
CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "random_seed": RANDOM_STATE,
    "verbose": False,
    "task_type": "GPU",
    "devices": "0:1",
}

XGBOOST_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "random_state": RANDOM_STATE,
    "tree_method": "gpu_hist",
    "predictor": "gpu_predictor",
}

LIGHTGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": RANDOM_STATE,
    "device": "gpu",
    "verbose": -1,
}


# Centralized logging configuration.
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOGGER = logging.getLogger("titanic")


def setup_directories() -> None:
    """Create project directories required by data processing and experiments."""
    for directory in (
        DATA_DIR,
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        MODELS_DIR,
        SUBMISSIONS_DIR,
        NOTEBOOKS_DIR,
        EXPERIMENTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def setup_file_logging(log_file: Optional[Path] = None) -> None:
    """Add optional file logging under the experiments directory."""
    if log_file is None:
        log_file = EXPERIMENTS_DIR / "titanic.log"
    setup_directories()
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    LOGGER.addHandler(file_handler)
