"""Feature engineering utilities for the Titanic project."""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import DATA_PROCESSED_DIR, KAGGLE_INPUT_DIR, TARGET_COLUMN


LOGGER = logging.getLogger("titanic.features")

TITLE_MAPPING = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
DECK_MAPPING = {"U": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}
ESSENTIAL_COLUMNS = [
    "PassengerId",
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Embarked",
]

# Columns produced by FoldLocalGroupFeatures — must NOT be pre-computed in engineer_features.
FOLD_LOCAL_COLUMNS = [
    "Ticket_Frequency",
    "Is_Group",
    "AdjFare",
    "AdjFare_Bin",
    "Deck",
    "Deck_Encoded",
]


class FoldLocalGroupFeatures(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer that derives fold-sensitive features.

    fit(X) must be called only on fold-train rows (never on validation or
    Kaggle test rows).  transform(X) applies the frozen fit-state to any
    subset, preserving the DataFrame index exactly.

    Produced columns
    ----------------
    Ticket_Frequency : int   — ticket share count from fit rows only
    Is_Group         : int8  — 1 if Ticket_Frequency > 1
    AdjFare          : float — Fare / Ticket_Frequency (per-person fare)
    AdjFare_Bin      : cat   — quintile bin from fit-fold AdjFare distribution
    Deck             : str   — cabin deck, propagated via fit-fold donors only
    Deck_Encoded     : int8  — numeric mapping of Deck via DECK_MAPPING

    Required input columns: Ticket, Fare, Cabin (or Raw_Deck if Cabin absent).
    """

    def fit(self, X: pd.DataFrame, y=None) -> "FoldLocalGroupFeatures":
        """Learn ticket counts, deck-donor map, and fare-bin edges from X."""
        # --- Ticket counts (fold-train only) ---
        self.ticket_counts_: dict = X["Ticket"].value_counts().to_dict()

        # --- Ticket → deck donor map (fold-train only) ---
        if "Raw_Deck" in X.columns:
            raw = X[["Ticket", "Raw_Deck"]].copy()
        elif "Cabin" in X.columns:
            cabin = X["Cabin"].fillna("").astype(str)
            raw_deck = cabin.str.extract(r"^([A-Za-z])", expand=False).str.upper()
            raw_deck = raw_deck.where(raw_deck.isin(DECK_MAPPING), np.nan)
            raw = pd.DataFrame({"Ticket": X["Ticket"], "Raw_Deck": raw_deck})
        else:
            raw = pd.DataFrame({"Ticket": X["Ticket"],
                                "Raw_Deck": pd.Series(np.nan, index=X.index)})

        self.ticket_deck_map_: dict = (
            raw.dropna(subset=["Raw_Deck"])
            .groupby("Ticket")["Raw_Deck"]
            .first()
            .to_dict()
        )

        # --- AdjFare quintile bin edges (fold-train only) ---
        ticket_freq = X["Ticket"].map(self.ticket_counts_).fillna(1).replace(0, 1)
        adj_fare = (X["Fare"] / ticket_freq).dropna()
        try:
            quantiles = adj_fare.quantile([0, 0.2, 0.4, 0.6, 0.8, 1]).to_numpy()
            edges = np.unique(quantiles)
            if len(edges) < 2:
                raise ValueError("Insufficient unique quantile edges")
            self.fare_bin_edges_: np.ndarray = edges
            self.fare_bin_labels_: list = [
                "Very Low", "Low", "Medium", "High", "Very High"
            ][: len(edges) - 1]
        except (ValueError, IndexError):
            self.fare_bin_edges_ = np.array([])
            self.fare_bin_labels_ = []

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply frozen fit-state to X, returning X with fold-local columns added."""
        df = X.copy()

        # --- Ticket_Frequency and Is_Group ---
        ticket_freq = df["Ticket"].map(self.ticket_counts_).fillna(1).replace(0, 1)
        df["Ticket_Frequency"] = ticket_freq.astype("int64")
        df["Is_Group"] = (ticket_freq > 1).astype("int8")

        # --- AdjFare ---
        df["AdjFare"] = df["Fare"] / ticket_freq

        # --- AdjFare_Bin using frozen edges ---
        if len(self.fare_bin_edges_) >= 2:
            df["AdjFare_Bin"] = pd.cut(
                df["AdjFare"],
                bins=self.fare_bin_edges_,
                labels=self.fare_bin_labels_,
                include_lowest=True,
            )
        else:
            df["AdjFare_Bin"] = pd.Series(pd.NA, index=df.index, dtype="string")

        # --- Deck propagation using fold-train donor map only ---
        if "Raw_Deck" in df.columns:
            raw_deck = df["Raw_Deck"]
        elif "Cabin" in df.columns:
            cabin = df["Cabin"].fillna("").astype(str)
            raw_deck = cabin.str.extract(r"^([A-Za-z])", expand=False).str.upper()
            raw_deck = raw_deck.where(raw_deck.isin(DECK_MAPPING), np.nan)
        else:
            raw_deck = pd.Series(np.nan, index=df.index)

        # Fill missing via fold-train donor map, then default unknown → 'U'
        propagated = raw_deck.fillna(df["Ticket"].map(self.ticket_deck_map_))
        final_deck = propagated.fillna("U")
        final_deck = final_deck.where(final_deck.isin(DECK_MAPPING), "U")

        df["Deck"] = final_deck
        df["Deck_Encoded"] = df["Deck"].map(DECK_MAPPING).fillna(0).astype("int8")

        # --- Deck_Group (derived from fold-local Deck) ---
        def _deck_group(deck: str) -> str:
            if deck in {"A", "B", "C"}:
                return "High"
            if deck in {"D", "E", "F", "G"}:
                return "Low"
            return "U"

        df["Deck_Group"] = df["Deck"].map(_deck_group)

        # --- GP mathematical features (require fold-local AdjFare) ---
        # Guards handle minimal frames (e.g. unit-test fixtures) that lack
        # Pclass / Age / Family_Size; real passenger data always has them.
        adj_fare_safe = np.maximum(df["AdjFare"].fillna(0).to_numpy(), 0)
        pclass_safe = (
            df["Pclass"].fillna(3).astype(float).to_numpy()
            if "Pclass" in df.columns
            else np.full(len(df), 3.0)
        )
        df["GP_LogFare_Per_Class"] = np.log1p(adj_fare_safe) / pclass_safe

        age_safe = np.maximum(
            df["Age"].fillna(28.0).to_numpy() if "Age" in df.columns else np.full(len(df), 28.0),
            0,
        )
        family_size_safe = (
            df["Family_Size"].fillna(1).astype(float).to_numpy()
            if "Family_Size" in df.columns
            else np.ones(len(df))
        )
        df["GP_Family_Vulnerability"] = family_size_safe / (pclass_safe * (age_safe + 1.0))

        df["GP_Fare_Class_Synergy"] = np.sqrt(adj_fare_safe) * (4.0 - pclass_safe)

        return df




def _extract_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and numerically encode passenger titles."""
    titles = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False).fillna("Rare")
    titles = titles.where(titles.isin(TITLE_MAPPING), "Rare")
    df["Title"] = titles
    df["Title_Encoded"] = titles.map(TITLE_MAPPING).astype("int8")
    return df



def _extract_deck(df: pd.DataFrame) -> pd.DataFrame:
    """Extract raw cabin deck letter and Has_Cabin flag.

    Ticket-based co-traveler deck propagation and final Deck/Deck_Encoded
    derivation have been moved to FoldLocalGroupFeatures.transform so they
    are computed exclusively from fold-train donors during CV.
    """
    cabin = df["Cabin"].fillna("").astype(str)
    # Extract raw deck letter only — no cross-row propagation here
    raw_deck = cabin.str.extract(r"^([A-Za-z])", expand=False).str.upper()
    raw_deck = raw_deck.where(raw_deck.isin(DECK_MAPPING), np.nan)
    df["Raw_Deck"] = raw_deck

    # Binary Has_Cabin feature (row-local, no leakage)
    df["Has_Cabin"] = df["Cabin"].notnull().astype("int8")
    return df




def _extract_family_name(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the surname portion of each passenger name."""
    df["Family_Name"] = df["Name"].str.split(",", n=1).str[0].str.strip()
    df["Last_Name"] = df["Family_Name"]
    return df


def _extract_ticket_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract alphabetic and punctuation ticket prefixes."""
    tickets = df["Ticket"].fillna("").astype(str).str.strip()
    prefixes = tickets.str.extract(r"^([A-Za-z/.]+)", expand=False)
    df["Ticket_Prefix"] = prefixes.fillna("NUMERIC").str.replace(".", "", regex=False).str.upper()
    return df


def _create_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create family size and family-size category features."""
    df["Family_Size"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    df["Is_Alone"] = (df["Family_Size"] == 1).astype("int8")
    df["Family_Size_Category"] = pd.cut(
        df["Family_Size"],
        bins=[0, 1, 4, np.inf],
        labels=["Alone", "Small", "Large"],
        include_lowest=True,
    )
    return df


def _create_ticket_features(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create passenger count and group indicators for each ticket."""
    ticket_counts = (reference if reference is not None else df)["Ticket"].value_counts(dropna=False)
    df["Ticket_Frequency"] = df["Ticket"].map(ticket_counts)
    df["Ticket_Frequency"] = df["Ticket_Frequency"].fillna(1).astype("int64")
    df["Is_Group"] = (df["Ticket_Frequency"] > 1).astype("int8")
    return df


def _create_fare_features(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create per-person fare and quantile-based fare category."""
    # Compute Ticket_Count strictly based on exact Ticket string matches
    if reference is not None:
        ticket_counts = reference["Ticket"].value_counts(dropna=False)
    else:
        ticket_counts = df["Ticket"].value_counts(dropna=False)

    ticket_freq = df["Ticket"].map(ticket_counts).fillna(1).replace(0, 1)
    df["AdjFare"] = df["Fare"] / ticket_freq

    try:
        if reference is not None and "AdjFare" not in reference.columns:
            ref_ticket_freq = reference["Ticket"].map(ticket_counts).fillna(1).replace(0, 1)
            reference["AdjFare"] = reference["Fare"] / ref_ticket_freq

        reference_fare = (reference if reference is not None else df)["AdjFare"].dropna()
        # 5 quantile bins (quintiles)
        quantiles = reference_fare.quantile([0, 0.2, 0.4, 0.6, 0.8, 1]).to_numpy()
        edges = np.unique(quantiles)
        labels = ["Very Low", "Low", "Medium", "High", "Very High"][: len(edges) - 1]

        df["AdjFare_Bin"] = pd.cut(df["AdjFare"], bins=edges, labels=labels, include_lowest=True)
    except ValueError:
        df["AdjFare_Bin"] = pd.Series(pd.NA, index=df.index, dtype="string")

    return df


def _create_gp_mathematical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create non-leaking mathematical causal features using robust GP-style operators.
    
    Note: Under SPEC-04, we restrict all downstream symbolic GP operators to smooth 
    functions (tanh, sqrt, remove sin/cos) and enforce high parsimony pressure 
    (parsimony_coefficient=0.05) to control tree complexity and prevent bloat.
    """
    # 1. Log-transformed Socioeconomic Burden: log(1 + AdjFare) / Pclass
    adj_fare_safe = np.maximum(df["AdjFare"].fillna(0).to_numpy(), 0)
    pclass_safe = df["Pclass"].fillna(3).astype(float).to_numpy()
    df["GP_LogFare_Per_Class"] = np.log1p(adj_fare_safe) / pclass_safe

    # 2. Family Vulnerability Index: Family_Size / (Pclass * (Age + 1))
    age_safe = np.maximum(df["Age"].fillna(28.0).to_numpy(), 0)
    family_size_safe = df["Family_Size"].fillna(1).astype(float).to_numpy()
    df["GP_Family_Vulnerability"] = family_size_safe / (pclass_safe * (age_safe + 1.0))

    # 3. Class-Scaled Ticket Wealth: np.sqrt(AdjFare) * (4 - Pclass)
    df["GP_Fare_Class_Synergy"] = np.sqrt(adj_fare_safe) * (4.0 - pclass_safe)

    return df


def _create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interactions between sex, class, title, and sex."""
    df["Sex_Pclass"] = df["Sex"].astype("string") + "_" + df["Pclass"].astype("string")
    df["Title_Sex"] = df["Title"].astype("string") + "_" + df["Sex"].astype("string")
    return df


def _create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create mother, age band, and grouped deck features."""
    df["Is_Mother"] = (
        df["Sex"].eq("female") & df["Title"].eq("Mrs") & df["Parch"].fillna(0).gt(0)
    ).astype("int8")
    age = df["Age"].fillna(-1)
    df["Age_Band"] = pd.cut(
        age,
        bins=[-np.inf, -0.5, 12, 18, 49.999999, np.inf],
        labels=["Missing", "Child", "Teen", "Adult", "Senior"],
    )
    # NOTE: Deck_Group is derived from Deck, which is fold-local.
    # It is computed in FoldLocalGroupFeatures.transform after Deck is produced.


    # WCG Features
    df["WCG_Member"] = (
        df["Sex"].eq("female") |
        (df["Sex"].eq("male") & df["Age"].le(18)) |
        df["Title"].eq("Master")
    ).astype("int8")

    # Generate Group_ID string correctly
    df["Group_ID"] = (
        df["Last_Name"].astype(str) + "_" +
        df["Ticket"].astype(str) + "_" +
        df["Fare"].fillna(-1).astype(str) + "_" +
        df["Embarked"].fillna("").astype(str)
    )

    # Demographic Survival Prior (Bayesian Laplace-smoothed priors across major demographic frontiers)
    def compute_demographic_prior(row):
        sex = row["Sex"]
        pclass = row["Pclass"]
        title = row.get("Title", "Mr")
        age = row.get("Age", 28.0)
        is_child = (title == "Master") or (pd.notna(age) and age <= 14.0)
        if sex == "female" or is_child:
            return 0.91 if pclass in [1, 2] else 0.49
        else:
            if pclass == 1:
                return 0.37
            elif pclass == 2:
                return 0.15
            else:
                return 0.13

    df["Demographic_Prior"] = df.apply(compute_demographic_prior, axis=1).astype(float)
    return df


def engineer_features(df: pd.DataFrame, reference_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a new DataFrame with row-local Titanic feature engineering applied.

    Fold-sensitive features (Ticket_Frequency, Is_Group, AdjFare, AdjFare_Bin,
    Deck, Deck_Encoded, Deck_Group, GP_*) are NOT produced here.  They are
    computed by FoldLocalGroupFeatures inside each CV fold pipeline.

    Raw input columns Ticket, Fare, Cabin, and Raw_Deck are preserved in the
    output so that FoldLocalGroupFeatures.fit() can operate on them.

    The reference_df parameter is retained for API compatibility but is no
    longer used; fold-sensitive derivations that previously relied on it now
    live in FoldLocalGroupFeatures.
    """
    required = set(ESSENTIAL_COLUMNS + ["Name", "Ticket", "Cabin", "Fare"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    LOGGER.info("Engineering features for %d rows", len(df))
    engineered = df.copy(deep=True)
    _extract_title(engineered)
    _extract_family_name(engineered)
    _extract_ticket_prefix(engineered)
    _create_family_features(engineered)
    # NOTE: _create_ticket_features and _create_fare_features removed —
    # their outputs are now fold-local (produced by FoldLocalGroupFeatures).
    _extract_deck(engineered)           # produces Raw_Deck and Has_Cabin only
    _create_interaction_features(engineered)
    _create_advanced_features(engineered)
    # NOTE: _create_gp_mathematical_features removed — GP features depend on
    # AdjFare which is fold-local (produced by FoldLocalGroupFeatures).

    # Row-local columns only; fold-local columns are absent by design.
    columns = ESSENTIAL_COLUMNS + [
        "Title",
        "Title_Encoded",
        "Family_Name",
        "Last_Name",
        "Has_Cabin",
        "Raw_Deck",        # required by FoldLocalGroupFeatures.fit()
        "Family_Size",
        "Is_Alone",
        "Family_Size_Category",
        "Ticket_Prefix",
        "Sex_Pclass",
        "Title_Sex",
        "Is_Mother",
        "Age_Band",
        "WCG_Member",
        "Group_ID",
        "Demographic_Prior",
        # Raw passthrough columns required by FoldLocalGroupFeatures
        "Ticket",
        "Fare",
        "Cabin",
    ]
    if TARGET_COLUMN in engineered.columns:
        columns.insert(1, TARGET_COLUMN)
    result = engineered.loc[:, columns].copy()
    LOGGER.info("Created %d engineered columns (fold-local columns deferred)", len(result.columns))
    return result


def save_engineered_data(
    train: pd.DataFrame, test: pd.DataFrame, output_dir: Path = DATA_PROCESSED_DIR
) -> Tuple[Path, Path]:
    """Engineer and save train/test datasets to the processed-data directory.

    Train and test are engineered separately so no test-set rows influence
    any computation in the stateless engineering phase.  Fold-sensitive columns
    (Ticket_Frequency, AdjFare, Deck, etc.) are intentionally absent from the
    saved CSVs; they are produced by FoldLocalGroupFeatures inside each CV fold.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_engineered.csv"
    test_path = output_dir / "test_engineered.csv"
    # Engineer each split independently — no combined concat
    engineer_train = engineer_features(train)
    engineer_test = engineer_features(test)
    engineer_train.to_csv(train_path, index=False)
    engineer_test.drop(columns=[TARGET_COLUMN], errors="ignore").to_csv(test_path, index=False)
    LOGGER.info("Saved engineered data to %s and %s", train_path, test_path)
    return train_path, test_path


if __name__ == "__main__":
    from src.config import get_input_dir

    input_dir = get_input_dir()
    train_data = pd.read_csv(input_dir / "train.csv")
    test_data = pd.read_csv(input_dir / "test.csv")
    save_engineered_data(train_data, test_data)

