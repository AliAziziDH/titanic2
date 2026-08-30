"""Missing-value imputation for engineered Titanic features."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold

from src.config import DATA_PROCESSED_DIR, N_FOLDS, N_REPEATS, RANDOM_STATE, TARGET_COLUMN

import logging


LOGGER = logging.getLogger("titanic.imputation")

AGE_FEATURES = [
    "Title_Num",
    "Title_Encoded",
    "Pclass",
    "Sex",
    "SibSp",
    "Parch",
    "Fare",
    "Is_Mother",
    "Is_Alone",
    "Ticket_Frequency",
]


def _age_feature_frame(
    frame: pd.DataFrame, medians: pd.Series | None = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build numeric age-model features without using the target column."""
    available = [column for column in AGE_FEATURES if column in frame.columns]
    if not available:
        raise ValueError("No age-imputation features are available")
    features = frame[available].copy()
    if "Title_Num" in features and "Title_Encoded" in features:
        features = features.drop(columns=["Title_Num"])
    if "Sex" in features:
        features["Sex"] = features["Sex"].map({"female": 1, "male": 0})
    features = pd.get_dummies(features, columns=[c for c in ["Sex"] if c in features], dtype=float)
    features = features.apply(pd.to_numeric, errors="coerce")
    if medians is None:
        medians = features.median().fillna(0)
    features = features.reindex(columns=medians.index, fill_value=0).fillna(medians)
    return features.astype(float), medians


def _age_strata(frame: pd.DataFrame) -> pd.Series:
    """Return stratification labels while keeping Survived out of model inputs."""
    if TARGET_COLUMN not in frame:
        return pd.Series(np.zeros(len(frame), dtype=int), index=frame.index)
    return frame[TARGET_COLUMN].fillna(0).astype(int)


def impute_age_with_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_folds: int = N_FOLDS,
    n_repeats: int = N_REPEATS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Impute Age using a RandomForestRegressor and repeated stratified CV."""
    train_clean, test_clean = train_df.copy(), test_df.copy()
    if "Age" not in train_clean or "Age" not in test_clean:
        raise ValueError("Both train and test must contain an Age column")

    known = train_clean["Age"].notna()
    missing_train = int((~known).sum())
    if not known.any():
        raise ValueError("Training data has no known Age values")
    x_known, medians = _age_feature_frame(train_clean.loc[known])
    y_known = train_clean.loc[known, "Age"].astype(float)
    model_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    min_class_count = _age_strata(train_clean.loc[known]).value_counts().min()
    splits = min(n_folds, int(min_class_count)) if min_class_count else 0
    if splits >= 2:
        cv = RepeatedStratifiedKFold(
            n_splits=splits, n_repeats=n_repeats, random_state=RANDOM_STATE
        )
        predictions = np.empty(len(y_known), dtype=float)
        for fold, (fit_idx, validation_idx) in enumerate(
            cv.split(x_known, _age_strata(train_clean.loc[known]).to_numpy()), start=1
        ):
            model = RandomForestRegressor(**model_params)
            model.fit(x_known.iloc[fit_idx], y_known.iloc[fit_idx])
            predictions[validation_idx] = model.predict(x_known.iloc[validation_idx])
            LOGGER.debug("Completed Age CV fold %d", fold)
        rmse = mean_squared_error(y_known, predictions) ** 0.5
        LOGGER.info("Age CV performance: RMSE=%.4f, R2=%.4f", rmse, r2_score(y_known, predictions))
    else:
        LOGGER.warning("Insufficient target-class samples for stratified Age CV")

    final_model = RandomForestRegressor(**model_params)
    final_model.fit(x_known, y_known)
    if missing_train:
        x_missing, _ = _age_feature_frame(train_clean.loc[~known], medians)
        train_clean.loc[~known, "Age"] = final_model.predict(x_missing)
    missing_test = int(test_clean["Age"].isna().sum())
    if missing_test:
        x_test, _ = _age_feature_frame(test_clean.loc[test_clean["Age"].isna()], medians)
        test_clean.loc[test_clean["Age"].isna(), "Age"] = final_model.predict(x_test)

    # A group median is the deterministic fallback for any unusual prediction failure.
    if train_clean["Age"].isna().any():
        group_medians = train_clean.groupby(
            "Title_Encoded" if "Title_Encoded" in train_clean else "Title"
        )["Age"].transform("median")
        train_clean["Age"] = train_clean["Age"].fillna(group_medians)
    train_clean["Age"] = train_clean["Age"].fillna(y_known.median())
    test_clean["Age"] = test_clean["Age"].fillna(train_clean["Age"].median())
    LOGGER.info("Age missing values filled: train=%d, test=%d", missing_train, missing_test)
    return train_clean, test_clean


def impute_embarked(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Impute Embarked from Pclass and nearest raw Fare, with mode fallback."""
    train_clean, test_clean = train_df.copy(), test_df.copy()
    overall_mode = train_clean["Embarked"].dropna().mode()
    fallback = overall_mode.iloc[0] if not overall_mode.empty else "S"
    for frame in (train_clean, test_clean):
        for index in frame.index[frame["Embarked"].isna()]:
            row = frame.loc[index]
            candidates = train_clean[
                train_clean["Embarked"].notna() & train_clean["Pclass"].eq(row["Pclass"])
            ]
            if not candidates.empty and pd.notna(row["Fare"]):
                nearest = (candidates["Fare"] - row["Fare"]).abs().idxmin()
                frame.loc[index, "Embarked"] = train_clean.loc[nearest, "Embarked"]
            elif not candidates.empty:
                frame.loc[index, "Embarked"] = candidates["Embarked"].mode().iloc[0]
            else:
                frame.loc[index, "Embarked"] = fallback
    LOGGER.info("Embarked missing values filled; fallback=%s", fallback)
    return train_clean, test_clean


def impute_fare(test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Impute test Fare using train-only Pclass and Embarked medians.

    AdjFare is intentionally not imputed here; it is derived CV-locally by
    FoldLocalGroupFeatures from the imputed raw Fare.
    """
    test_clean = test_df.copy()
    global_median = train_df["Fare"].median()
    for index in test_clean.index[test_clean["Fare"].isna()]:
        row = test_clean.loc[index]
        candidates = train_df[
            train_df["Pclass"].eq(row["Pclass"])
            & train_df["Embarked"].eq(row["Embarked"])
        ]["Fare"].dropna()
        if candidates.empty:
            candidates = train_df[train_df["Pclass"].eq(row["Pclass"])]["Fare"].dropna()
        test_clean.loc[index, "Fare"] = candidates.median() if not candidates.empty else global_median

    return test_clean


def impute_missing_values(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run Age, Embarked, and Fare imputation using train-derived parameters."""
    train_clean, test_clean = impute_age_with_cv(train_df, test_df)
    train_clean, test_clean = impute_embarked(train_clean, test_clean)
    test_clean = impute_fare(test_clean, train_clean)
    for column in ["Age", "Embarked", "Fare"]:
        LOGGER.info(
            "%s missing after imputation: train=%d, test=%d",
            column,
            int(train_clean[column].isna().sum() if column in train_clean else 0),
            int(test_clean[column].isna().sum() if column in test_clean else 0),
        )
    return train_clean, test_clean


def save_clean_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path = DATA_PROCESSED_DIR,
) -> Tuple[Path, Path]:
    """Save clean train and test data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path, test_path = output_dir / "train_clean.csv", output_dir / "test_clean.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    LOGGER.info("Saved clean data to %s and %s", train_path, test_path)
    return train_path, test_path


if __name__ == "__main__":
    train = pd.read_csv(DATA_PROCESSED_DIR / "train_engineered.csv")
    test = pd.read_csv(DATA_PROCESSED_DIR / "test_engineered.csv")
    clean_train, clean_test = impute_missing_values(train, test)
    save_clean_data(clean_train, clean_test)
