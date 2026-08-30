"""Model comparison, validation, and submission generation for Titanic."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.combine import SMOTEENN
except ImportError:
    from sklearn.pipeline import Pipeline as ImbPipeline
    SMOTEENN = None

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_predict

try:
    from gplearn.genetic import SymbolicTransformer
    from gplearn.functions import make_function
    gp_tanh = make_function(function=np.tanh, name='tanh', arity=1)
except ImportError:
    SymbolicTransformer = None
    make_function = None
    gp_tanh = None
# from tabpfn import TabPFNClassifier

from src.config import (
    DATA_PROCESSED_DIR,
    EXPERIMENTS_DIR,
    get_input_dir,
    LIGHTGBM_PARAMS,
    MODELS_DIR,
    RANDOM_STATE,
    SUBMISSIONS_DIR,
    TARGET_COLUMN,
)
from src.features import save_engineered_data, FoldLocalGroupFeatures

LOGGER = logging.getLogger("titanic.modeling")
MODEL_DIR = Path(MODELS_DIR)
CV_RESULTS_PATH = Path(EXPERIMENTS_DIR) / "cv_results.json"

NUMERICAL_FEATURES = [
    "Age", "SibSp", "Parch", "Family_Size", "Ticket_Frequency", "AdjFare", "Title_Encoded", "Deck_Encoded",
    "Demographic_Prior", "GP_LogFare_Per_Class", "GP_Family_Vulnerability", "GP_Fare_Class_Synergy",
]
CATEGORICAL_FEATURES = [
    "Sex", "Embarked", "Deck", "Deck_Group", "Family_Name", "Last_Name", "Family_Size_Category", "Ticket_Prefix",
    "AdjFare_Bin", "Age_Band", "Sex_Pclass", "Title_Sex", "Group_ID",
]
BINARY_FEATURES = ["Has_Cabin", "Is_Alone", "Is_Group", "Is_Mother", "WCG_Member"]


class WCGSurvivalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.default_rate = 0.5
        self.pass1_groups = {}
        self.pass2_groups = {}

    def fit(self, X, y=None):
        if y is None or 'Last_Name' not in X or 'AdjFare' not in X or 'Ticket' not in X:
            return self
        if not X.index.is_unique:
            raise ValueError("WCGSurvivalEncoder requires a unique index for leakage-safe self-exclusion")

        df = X.copy()
        df['Survived'] = y

        # We store lists of (index, survival) for Pass 1 and Pass 2

        df['Pass1_Group'] = df['Last_Name'].astype(str) + "_" + df['AdjFare'].astype(str)
        df['Pass2_Group'] = df['Ticket'].astype(str)

        self.pass1_groups = df.groupby('Pass1_Group').apply(
            lambda g: list(zip(g.index, g['Survived'])), include_groups=False
        ).to_dict()

        self.pass2_groups = df.groupby('Pass2_Group').apply(
            lambda g: list(zip(g.index, g['Survived'])), include_groups=False
        ).to_dict()

        return self

    def transform(self, X):
        df = X.copy()
        if 'Last_Name' not in df or 'AdjFare' not in df or 'Ticket' not in df:
            df['WCG_Survival'] = self.default_rate
            return df

        df['Pass1_Group'] = df['Last_Name'].astype(str) + "_" + df['AdjFare'].astype(str)
        df['Pass2_Group'] = df['Ticket'].astype(str)

        def calculate_survival(row):
            # Pass 1
            pass1_members = self.pass1_groups.get(row['Pass1_Group'], [])
            # Exclude self
            other_pass1 = [surv for idx, surv in pass1_members if idx != row.name]

            if len(other_pass1) > 0:
                max_surv = max(other_pass1)
                min_surv = min(other_pass1)
                if max_surv == 1.0:
                    return 1.0
                if min_surv == 0.0:
                    return 0.0

            # Pass 2
            pass2_members = self.pass2_groups.get(row['Pass2_Group'], [])
            # Exclude self
            other_pass2 = [surv for idx, surv in pass2_members if idx != row.name]

            if len(other_pass2) > 0:
                max_surv = max(other_pass2)
                min_surv = min(other_pass2)
                if max_surv == 1.0:
                    return 1.0
                if min_surv == 0.0:
                    return 0.0

            return self.default_rate

        df['WCG_Survival'] = df.apply(calculate_survival, axis=1)

        # Clean up temporary columns
        df.drop(columns=['Pass1_Group', 'Pass2_Group'], inplace=True, errors='ignore')

        return df


class AgeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.available_cols = []
        self.global_median = 28.0

    def fit(self, X, y=None):
        from sklearn.linear_model import BayesianRidge

        # Features to use for imputing
        cols_to_use = ['Title_Encoded', 'Pclass', 'Family_Size']

        self.available_cols = [c for c in cols_to_use if c in X.columns]

        if 'Age' in X.columns:
            self.global_median = X['Age'].median()

            # Filter rows where Age is known and all predictor columns are available
            known_mask = X['Age'].notna() & X[self.available_cols].notna().all(axis=1)

            if known_mask.sum() > 0 and len(self.available_cols) > 0:
                self.model = BayesianRidge()
                self.model.fit(X.loc[known_mask, self.available_cols], X.loc[known_mask, 'Age'])
            else:
                self.model = None
        else:
            self.model = None

        return self

    def transform(self, X):
        df = X.copy()
        if 'Age' in df.columns:
            missing_mask = df['Age'].isna()

            if missing_mask.sum() > 0:
                if self.model is not None:
                    # Impute missing values with BayesianRidge if predictors are valid
                    pred_mask = missing_mask & df[self.available_cols].notna().all(axis=1)
                    if pred_mask.sum() > 0:
                        df.loc[pred_mask, 'Age'] = self.model.predict(df.loc[pred_mask, self.available_cols])

                # Fallback to global median for any remaining missing values
                df['Age'] = df['Age'].fillna(self.global_median)
        return df


# class TabPFNFeatureExtractor(BaseEstimator, TransformerMixin):
#     def __init__(self, random_state=42):
#         self.random_state = random_state
#         self.tabpfn = TabPFNClassifier(n_estimators=1, ignore_pretraining_limits=True,
#             model_path="models/tabpfn/tabpfn-v3-classifier-v3_default.ckpt",
#             device='cpu'
#         )
#         self.fitted_ = False
#         self.train_probs_ = None

#     def fit(self, X, y=None):
#         if y is None:
#             return self

#         from sklearn.model_selection import StratifiedKFold
#         cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

#         X_imputed = np.nan_to_num(X, nan=-999.0)

#         self.train_probs_ = cross_val_predict(
#             self.tabpfn, X_imputed, y, cv=cv, method='predict_proba', n_jobs=-1
#         )[:, 1].reshape(-1, 1)

#         self.tabpfn.fit(X_imputed, y)
#         self.fitted_ = True
#         return self

#     def fit_transform(self, X, y=None):
#         self.fit(X, y)
#         return self.train_probs_

#     def transform(self, X):
#         if not self.fitted_:
#             return np.zeros((X.shape[0], 1))

#         X_imputed = np.nan_to_num(X, nan=-999.0)
#         probs = self.tabpfn.predict_proba(X_imputed)[:, 1].reshape(-1, 1)
#         return probs


class ToDenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        import pandas as pd
        from scipy import sparse
        if sparse.issparse(X):
            arr = X.toarray()
            # If X has feature names (e.g. from set_output), try to restore them
            if hasattr(X, "columns"):
                return pd.DataFrame(arr, columns=X.columns, index=X.index)
            # Actually sparse matrices don't have columns. Scikit-learn outputs a CSR matrix.
            # But wait, with set_output(transform="pandas"), Scikit-learn's ColumnTransformer
            # outputs a DataFrame, not a sparse matrix, as long as it's possible.
            # OneHotEncoder with sparse_output=False (or when outputting pandas) returns dense DataFrame!
        if isinstance(X, pd.DataFrame):
            return X
        return np.array(X)


class PipelineWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessor, union):
        self.preprocessor = preprocessor
        self.union = union
        from sklearn.pipeline import Pipeline
        self.pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("meta_union", union)
        ])

    def fit(self, X, y=None):
        self.pipe.fit(X, y)
        return self

    def transform(self, X):
        return self.pipe.transform(X)

    def fit_transform(self, X, y=None):
        return self.pipe.fit_transform(X, y)


def build_meta_features(preprocessor, seed=RANDOM_STATE):
    symbolic = SymbolicTransformer(
        population_size=100,
        hall_of_fame=20,
        n_components=10,
        generations=5,
        tournament_size=10,
        stopping_criteria=1.0,
        const_range=(-1.0, 1.0),
        init_depth=(2, 4),
        init_method='half and half',
        function_set=['add', 'sub', 'mul', 'div', 'sqrt', gp_tanh],
        metric='pearson',
        parsimony_coefficient=0.05,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=1.0,
        feature_names=None,
        warm_start=False,
        low_memory=False,
        n_jobs=1,
        verbose=0,
        random_state=seed
    )

    from sklearn.compose import ColumnTransformer
    # Age is 0, Family_Size is 3, AdjFare is 5 based on numeric pipeline output
    # Also pass only these features to TabPFN
    core_features = ColumnTransformer(
        [("num_features", "passthrough", ["numeric__Age", "numeric__Family_Size", "numeric__AdjFare"])],
        remainder="drop"
    )
    from sklearn.pipeline import make_pipeline
    union = FeatureUnion([
        ("original", FunctionTransformer()),
        ("symbolic", make_pipeline(core_features, symbolic)),
        # Disabled TabPFN due to licensing checkpoint download issues in headless mode
        # ("tabpfn", make_pipeline(core_features, TabPFNFeatureExtractor(random_state=RANDOM_STATE)))
    ])

    union = make_pipeline(ToDenseTransformer(), union)
    return PipelineWrapper(preprocessor, union)


def load_modeling_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load engineered data so every fold imputes from its own training rows."""
    train_path = Path(DATA_PROCESSED_DIR) / "train_engineered.csv"
    test_path = Path(DATA_PROCESSED_DIR) / "test_engineered.csv"
    if train_path.exists() and test_path.exists():
        return pd.read_csv(train_path), pd.read_csv(test_path)

    input_dir = get_input_dir()
    raw_train = pd.read_csv(input_dir / "train.csv")
    raw_test = pd.read_csv(input_dir / "test.csv")
    train_engineered_path, test_engineered_path = save_engineered_data(
        raw_train, raw_test
    )
    return pd.read_csv(train_engineered_path), pd.read_csv(test_engineered_path)


def _available_columns(frame: pd.DataFrame) -> Tuple[list[str], list[str], list[str]]:
    numeric = [column for column in NUMERICAL_FEATURES if column in frame]
    categorical = [column for column in CATEGORICAL_FEATURES if column in frame]
    binary = [column for column in BINARY_FEATURES if column in frame]
    return numeric, categorical, binary


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing that is fitted independently within each CV fold."""
    numeric, categorical, binary = _available_columns(frame)
    numeric_pipeline = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = ImbPipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical + binary),
        ],
        remainder="drop",
    ).set_output(transform="pandas")


def _optional_models(seed=None) -> Dict[str, Any]:
    """Return available optional gradient-boosting estimators with safe CPU defaults."""
    if seed is None:
        seed = RANDOM_STATE
    models: Dict[str, Any] = {}
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
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            reg_lambda=50.0, subsample=0.7, colsample_bytree=0.7,
            random_state=seed, tree_method="hist", eval_metric="logloss",
        )
    except ImportError:
        LOGGER.warning("XGBoost is unavailable; skipping it")
    try:
        from lightgbm import LGBMClassifier

        params = dict(LIGHTGBM_PARAMS)
        params.update({
            "device": "cpu", "verbosity": -1,
            "max_depth": 4, "reg_lambda": 50.0,
            "subsample": 0.7, "colsample_bytree": 0.7,
            "n_jobs": 1,
            "random_state": seed,
        })
        models["LightGBM"] = LGBMClassifier(**params)
    except ImportError:
        LOGGER.warning("LightGBM is unavailable; skipping it")
    return models


def default_models(seed=None) -> Dict[str, Any]:
    """Return the configured model candidates."""
    if seed is None:
        seed = RANDOM_STATE
    models = _optional_models(seed=seed)
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=500, max_depth=10, random_state=seed, n_jobs=-1,
    )
    models["MLP"] = MLPClassifier(
        hidden_layer_sizes=(32,), alpha=0.1, early_stopping=True,
        random_state=seed, max_iter=500,
    )
    return models


def compute_ipw_weights(X_fold: pd.DataFrame) -> np.ndarray:
    """Compute Inverse Probability Weights (IPW) for a training fold."""
    if 'Pclass' not in X_fold.columns:
        return np.ones(len(X_fold))

    # Treatment T_i = 1 if Pclass in {1, 2} (Upper/Middle), else 0 (Lower)
    T = X_fold['Pclass'].isin([1, 2]).astype(int).to_numpy()

    # Baseline confounders for Propensity Score Model
    confounders = ['Sex', 'Age', 'Embarked', 'Family_Size']
    available = [c for c in confounders if c in X_fold.columns]

    if not available:
        return np.ones(len(X_fold))

    X_propensity = X_fold[available].copy()

    # Simple imputation and encoding for Logistic Regression
    if 'Sex' in X_propensity.columns:
        X_propensity['Sex'] = X_propensity['Sex'].map({'male': 0, 'female': 1}).fillna(0)

    if 'Age' in X_propensity.columns:
        X_propensity['Age'] = X_propensity['Age'].fillna(X_propensity['Age'].median())

    if 'Embarked' in X_propensity.columns:
        X_propensity['Embarked'] = X_propensity['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).fillna(0)

    if 'Family_Size' in X_propensity.columns:
        X_propensity['Family_Size'] = X_propensity['Family_Size'].fillna(X_propensity['Family_Size'].median())

    from sklearn.linear_model import LogisticRegression
    # Fit Propensity Score model
    ps_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    ps_model.fit(X_propensity, T)

    # Estimate propensity scores e(X_i) = P(T_i = 1 | X_i)
    e = ps_model.predict_proba(X_propensity)[:, 1]

    # Trim to stabilize weights
    e = np.clip(e, 0.05, 0.95)

    # Calculate IPW
    w = (T / e) + ((1 - T) / (1 - e))
    return w


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_strategy: RepeatedStratifiedKFold,
    model_name: str,
) -> Dict[str, Any]:
    """Evaluate a model with preprocessing fitted separately on each fold."""
    scores = {"accuracy": [], "roc_auc": [], "f1_macro": []}
    for fold, (fit_idx, validation_idx) in enumerate(cv_strategy.split(X_train, y_train), start=1):
        X_fold_train = X_train.iloc[fit_idx]
        y_fold_train = y_train.iloc[fit_idx]

        # Build a temporary fold-local feature frame to derive the correct
        # column set for build_preprocessor (so ColumnTransformer column lists
        # reflect the post-transform schema rather than the raw CSV schema).
        _probe = FoldLocalGroupFeatures().fit(X_fold_train).transform(X_fold_train.copy())

        pipeline = ImbPipeline([
            ("fold_features", FoldLocalGroupFeatures()),  # step 0: fold-local, no leakage
            ("wcg_encoder", WCGSurvivalEncoder()),
            ("age_imputer", AgeImputer(random_state=RANDOM_STATE)),
            ("meta_features", build_meta_features(build_preprocessor(_probe))),
            ("model", clone(model)),
        ])

        # Determine if model supports sample weights and apply IPW
        # CatBoost, XGBoost, LightGBM, and RandomForest generally do.
        supported_models = ['CatBoost', 'XGBoost', 'RandomForest', 'LightGBM']
        if model_name in supported_models:
            weights = compute_ipw_weights(X_fold_train)
            pipeline.fit(X_fold_train, y_fold_train, model__sample_weight=weights)
        else:
            pipeline.fit(X_fold_train, y_fold_train)

        predictions = pipeline.predict(X_train.iloc[validation_idx])
        probabilities = pipeline.predict_proba(X_train.iloc[validation_idx])[:, 1]
        scores["accuracy"].append(accuracy_score(y_train.iloc[validation_idx], predictions))
        scores["roc_auc"].append(roc_auc_score(y_train.iloc[validation_idx], probabilities))
        scores["f1_macro"].append(f1_score(y_train.iloc[validation_idx], predictions, average="macro"))
        LOGGER.debug("%s fold %d complete", model_name, fold)

    result = {
        "model": model_name,
        "metrics": {
            metric: {"mean": float(np.mean(values)), "std": float(np.std(values)), "folds": values}
            for metric, values in scores.items()
        },
    }
    LOGGER.info(
        "%s: accuracy=%.4f (+/- %.4f), roc_auc=%.4f (+/- %.4f), f1_macro=%.4f (+/- %.4f)",
        model_name,
        result["metrics"]["accuracy"]["mean"], result["metrics"]["accuracy"]["std"],
        result["metrics"]["roc_auc"]["mean"], result["metrics"]["roc_auc"]["std"],
        result["metrics"]["f1_macro"]["mean"], result["metrics"]["f1_macro"]["std"],
    )
    return result


def _select_best(results: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Select by ROC-AUC, with accuracy and macro-F1 as tie-breakers."""
    return max(
        results,
        key=lambda result: (
            result["metrics"]["roc_auc"]["mean"],
            result["metrics"]["accuracy"]["mean"],
            result["metrics"]["f1_macro"]["mean"],
        ),
    )


def run_modeling_pipeline() -> pd.DataFrame:
    """Train candidate models, select the best, and write a Kaggle submission."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    Path(EXPERIMENTS_DIR).mkdir(parents=True, exist_ok=True)
    train, test = load_modeling_data()
    if TARGET_COLUMN not in train:
        raise ValueError(f"{TARGET_COLUMN} is missing from training data")
    feature_columns = [column for column in train.columns if column not in {TARGET_COLUMN, "PassengerId"}]
    X_train, y_train = train[feature_columns], train[TARGET_COLUMN].astype(int)
    X_test = test[feature_columns]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)

    results = [
        evaluate_model(model, X_train, y_train, cv, name)
        for name, model in default_models().items()
    ]
    CV_RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    best = _select_best(results)

    # Train and save ALL individual models across 10 seeds so final_submission.py can load them
    supported_models = ['CatBoost', 'XGBoost', 'RandomForest', 'LightGBM']
    SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]

    # Probe frame: apply FoldLocalGroupFeatures to full X_train once to derive
    # correct column set for build_preprocessor in final (non-CV) pipelines.
    _full_probe = FoldLocalGroupFeatures().fit(X_train).transform(X_train.copy())

    for seed in SEEDS:
        models_dict = default_models(seed=seed)
        for name, model in models_dict.items():
            pipeline = ImbPipeline([
                ("fold_features", FoldLocalGroupFeatures()),  # step 0: fold-local
                ("wcg_encoder", WCGSurvivalEncoder()),
                ("age_imputer", AgeImputer(random_state=seed)),
                ("meta_features", build_meta_features(build_preprocessor(_full_probe), seed=seed)),
                ("model", model),
            ])

            if name in supported_models:
                weights = compute_ipw_weights(X_train)
                pipeline.fit(X_train, y_train, model__sample_weight=weights)
            else:
                pipeline.fit(X_train, y_train)

            joblib.dump(pipeline, MODEL_DIR / f"{name.lower()}_seed_{seed}.joblib")
            if seed == 42:
                joblib.dump(pipeline, MODEL_DIR / f"{name.lower()}_final.joblib")

    # The rest proceeds as before for the "best" model logic using seed 42
    best_models_dict = default_models(seed=RANDOM_STATE)
    best_model = best_models_dict[best["model"]]
    final_pipeline = ImbPipeline([
        ("fold_features", FoldLocalGroupFeatures()),  # step 0: fold-local
        ("wcg_encoder", WCGSurvivalEncoder()),
        ("age_imputer", AgeImputer(random_state=RANDOM_STATE)),
        ("meta_features", build_meta_features(build_preprocessor(_full_probe), seed=RANDOM_STATE)),
        ("model", best_model),
    ])

    if best["model"] in supported_models:
        weights = compute_ipw_weights(X_train)
        final_pipeline.fit(X_train, y_train, model__sample_weight=weights)
    else:
        final_pipeline.fit(X_train, y_train)
    test_predictions = final_pipeline.predict(X_test).astype(int)

    submission = pd.DataFrame({"PassengerId": test["PassengerId"], TARGET_COLUMN: test_predictions})
    submission_path = Path(SUBMISSIONS_DIR) / "submission_modeling.csv"
    submission.to_csv(submission_path, index=False)
    (MODEL_DIR / "best_model.json").write_text(
        json.dumps({"model": best["model"], "metrics": best["metrics"]}, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Selected %s; submission saved to %s", best["model"], submission_path)
    return submission


if __name__ == "__main__":
    run_modeling_pipeline()
