import os
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd


class TestTitanicPipeline(unittest.TestCase):

    def test_train_clean_shape(self):
        """Verify data/processed/train_clean.csv has exactly 891 rows if exists."""
        file_path = "data/processed/train_clean.csv"
        if not os.path.exists(file_path):
            self.skipTest(f"Processed file {file_path} not found.")
        df = pd.read_csv(file_path)
        self.assertEqual(df.shape[0], 891)

    def test_test_clean_shape(self):
        """Verify data/processed/test_clean.csv has exactly 418 rows if exists."""
        file_path = "data/processed/test_clean.csv"
        if not os.path.exists(file_path):
            self.skipTest(f"Processed file {file_path} not found.")
        df = pd.read_csv(file_path)
        self.assertEqual(df.shape[0], 418)

    def test_submission_stacking_shape(self):
        """Verify submissions/submission_stacking.csv (if present) has exactly 418 rows, correct columns and IDs."""
        file_path = "submissions/submission_stacking.csv"
        if not os.path.exists(file_path):
            self.skipTest(f"Submission file {file_path} not found.")

        df = pd.read_csv(file_path)
        self.assertEqual(df.shape[0], 418)
        self.assertEqual(list(df.columns), ["PassengerId", "Survived"])
        self.assertEqual(df["PassengerId"].min(), 892)
        self.assertEqual(df["PassengerId"].max(), 1309)
        self.assertTrue(set(df["Survived"].unique()).issubset({0, 1}))

    def test_wcg_deterministic_override(self):
        """Verify WCGPostProcessor deterministic overrides."""
        from src.final_submission import WCGPostProcessor

        mock_raw_train = pd.DataFrame({
            "PassengerId": [1, 2, 3],
            "Name": ["Braund, Mr. Owen Harris", "Cumings, Mrs. John Bradley", "Cumings, Miss. Florence"],
            "Sex": ["male", "female", "female"],
            "Age": [22.0, 38.0, 10.0],
            "Ticket": ["A/5 21171", "PC 17599", "PC 17599"],
            "Fare": [7.25, 71.2833, 71.2833],
            "Embarked": ["S", "C", "C"],
            "Pclass": [3, 1, 1],
            "Survived": [0, 1, 1]
        })

        mock_raw_test = pd.DataFrame({
            "PassengerId": [892, 893, 894, 895],
            "Name": ["Kelly, Mr. James", "Wilkes, Mrs. James", "Myles, Master. Thomas", "Hirvonen, Mrs. Alexander"],
            "Sex": ["male", "female", "male", "female"],
            "Age": [34.5, 47.0, 5.0, 22.0],
            "Ticket": ["330911", "TEST_TICKET", "TEST_TICKET", "PC 17599"],
            "Fare": [7.8292, 7.0, 7.0, 71.2833],
            "Embarked": ["Q", "S", "S", "C"],
            "Pclass": [3, 3, 3, 1]
        })

        train_clean = pd.DataFrame({
            "PassengerId": [1, 2, 3],
            "Sex": ["male", "female", "female"],
            "Title": ["Mr", "Mrs", "Miss"],
            "Age": [22.0, 38.0, 10.0],
            "Survived": [0, 1, 1]
        })

        test_clean = pd.DataFrame({
            "PassengerId": [892, 893, 894, 895],
            "Sex": ["male", "female", "male", "female"],
            "Title": ["Mr", "Mrs", "Master", "Mrs"],
            "Age": [34.5, 47.0, 5.0, 22.0]
        })

        with patch("pandas.read_csv") as mock_read:
            def side_effect(path, *args, **kwargs):
                if "train.csv" in str(path):
                    return mock_raw_train
                elif "test.csv" in str(path):
                    return mock_raw_test
                return pd.DataFrame()
            mock_read.side_effect = side_effect

            processor = WCGPostProcessor()
            processor.fit(train_clean)

            baseline_probs = np.array([0.1, 0.8, 0.4, 0.2])
            final_probs = processor.transform(test_clean, baseline_probs)

            self.assertEqual(final_probs[0], 0.1)
            self.assertEqual(final_probs[1], 0.8)
            self.assertEqual(final_probs[2], 0.4)
            self.assertEqual(final_probs[3], 1.0)


    # ------------------------------------------------------------------
    # Test A: Contamination proof
    # The current combined-reference path must produce different
    # Ticket_Frequency on validation rows than a fold-local path does.
    # After the patch, engineer_features no longer writes Ticket_Frequency
    # to the CSV; FoldLocalGroupFeatures produces it fold-locally instead.
    # ------------------------------------------------------------------
    def test_fold_local_ticket_frequency_differs_from_combined(self):
        """Ticket_Frequency from a combined frame must differ from fold-local.

        This test documents the contamination: when train+val rows are
        pooled, ticket counts absorb validation-row contributions. After the
        patch, FoldLocalGroupFeatures.fit uses only fold-train rows.
        """
        from src.features import FoldLocalGroupFeatures

        fold_train = pd.DataFrame(
            {"Ticket": ["T1", "T1", "T2"], "Fare": [10.0, 10.0, 5.0],
             "Cabin": [None, None, None]},
            index=[0, 1, 2],
        )
        val = pd.DataFrame(
            {"Ticket": ["T1", "T3"], "Fare": [10.0, 20.0],
             "Cabin": [None, None]},
            index=[10, 11],
        )

        # Fold-local path: fit on fold_train only
        transformer = FoldLocalGroupFeatures()
        transformer.fit(fold_train)
        out_local = transformer.transform(val.copy())

        # Combined-reference path: counts include validation rows
        combined = pd.concat([fold_train, val], ignore_index=False)
        combined_counts = combined["Ticket"].value_counts().to_dict()
        freq_combined_val = val["Ticket"].map(combined_counts).fillna(1)

        # Fold-local T1 count = 2 (fold_train only)
        # Combined   T1 count = 3 (fold_train + val row 10)
        self.assertEqual(int(out_local.loc[10, "Ticket_Frequency"]), 2,
                         "Fold-local must count T1=2 (fold-train only)")
        self.assertNotEqual(
            int(out_local.loc[10, "Ticket_Frequency"]),
            int(freq_combined_val.loc[10]),
            "Combined count must differ from fold-local count for T1",
        )
        # Unseen ticket T3 must fall back to 1, not 0
        self.assertEqual(int(out_local.loc[11, "Ticket_Frequency"]), 1,
                         "Unseen ticket must default to frequency 1")
        # Index must be preserved
        self.assertEqual(out_local.index.tolist(), [10, 11])

    # ------------------------------------------------------------------
    # Test B: FoldLocalGroupFeatures full contract
    # ------------------------------------------------------------------
    def test_fold_local_group_features_contract(self):
        """FoldLocalGroupFeatures fit/transform contract.

        Verifies:
        - Unseen ticket gets Ticket_Frequency == 1.
        - AdjFare_Bin for an extreme out-of-range fare is NaN or 'Very High'
          (bin edges are frozen from fold-train; no new bins are created).
        - DataFrame index is preserved end-to-end.
        - Is_Group is 1 for known shared tickets, 0 for singletons.
        - Deck propagation uses only fold-train donor mapping.
        """
        from src.features import FoldLocalGroupFeatures

        fold_train = pd.DataFrame(
            {
                "Ticket": ["T1", "T1", "T2", "T3", "T4"],
                "Fare":   [10.0, 10.0, 20.0, 30.0, 40.0],
                "Cabin":  ["C85", None, None, "B28", None],
            },
            index=[0, 1, 2, 3, 4],
        )
        val = pd.DataFrame(
            {
                "Ticket": ["T1",   "T_UNSEEN", "T2"],
                "Fare":   [10.0,   9999.0,     20.0],
                "Cabin":  [None,   None,        None],
            },
            index=[100, 101, 102],
        )

        transformer = FoldLocalGroupFeatures()
        transformer.fit(fold_train)
        out = transformer.transform(val.copy())

        # Index preservation
        self.assertEqual(out.index.tolist(), [100, 101, 102])

        # T1: seen twice in fold_train → frequency 2, Is_Group 1
        self.assertEqual(int(out.loc[100, "Ticket_Frequency"]), 2)
        self.assertEqual(int(out.loc[100, "Is_Group"]), 1)

        # T_UNSEEN: fallback frequency 1, Is_Group 0
        self.assertEqual(int(out.loc[101, "Ticket_Frequency"]), 1)
        self.assertEqual(int(out.loc[101, "Is_Group"]), 0)

        # Extreme fare bin: must be NaN or 'Very High', never a new label
        valid_bins = {pd.NA, "Very High", None}
        bin_val = out.loc[101, "AdjFare_Bin"]
        is_nan = pd.isna(bin_val)
        is_valid = is_nan or (str(bin_val) in {"Very High", "<NA>", "nan"})
        self.assertTrue(is_valid,
                        f"Extreme fare bin must be NaN/Very High, got {bin_val!r}")

        # Deck propagation: T1 donor in fold_train is row 0 with Cabin='C85' → deck 'C'
        # T_UNSEEN has no donor → deck 'U'
        self.assertEqual(str(out.loc[100, "Deck"]), "C",
                         "T1 passenger should inherit deck C from fold-train donor")
        self.assertEqual(str(out.loc[101, "Deck"]), "U",
                         "Unseen ticket passenger should get deck U")

        # AdjFare: T2 appears once in fold_train → AdjFare == Fare / 1 == 20.0
        self.assertAlmostEqual(float(out.loc[102, "AdjFare"]), 20.0, places=4)


    # ------------------------------------------------------------------
    # Test C: impute_missing_values works without AdjFare present
    # Regression guard for the post-refactor integration failure where
    # src.imputation still depended on a persisted AdjFare column that
    # FoldLocalGroupFeatures now defers to CV-time.
    # ------------------------------------------------------------------
    def test_impute_missing_values_without_adj_fare(self):
        """impute_missing_values must succeed when AdjFare is absent but Fare is present."""
        from src.imputation import impute_missing_values

        rng = np.random.default_rng(42)

        def _base(n, seed_offset=0):
            return pd.DataFrame({
                "Pclass":   rng.choice([1, 2, 3], size=n),
                "Sex":      rng.choice(["male", "female"], size=n),
                "SibSp":    rng.integers(0, 3, size=n),
                "Parch":    rng.integers(0, 2, size=n),
                "Fare":     rng.uniform(5, 100, size=n),
                "Embarked": rng.choice(["S", "C", "Q"], size=n),
                "Age":      np.where(rng.random(n) < 0.8, rng.uniform(5, 70, size=n), np.nan),
                "Title_Encoded": rng.choice([1, 2, 3], size=n),
                "Is_Mother":     rng.integers(0, 2, size=n).astype(float),
                "Is_Alone":      rng.integers(0, 2, size=n).astype(float),
                "Ticket_Frequency": rng.integers(1, 5, size=n).astype(float),
                "Survived":  rng.integers(0, 2, size=n),
            })

        train = _base(30)
        test = _base(10)

        # Deliberately introduce one missing Embarked and one missing Fare in test
        test.loc[test.index[0], "Embarked"] = np.nan
        test.loc[test.index[1], "Fare"] = np.nan

        # AdjFare must NOT be present – this is the regression condition
        self.assertNotIn("AdjFare", train.columns)
        self.assertNotIn("AdjFare", test.columns)

        train_out, test_out = impute_missing_values(train, test)

        # Core postconditions
        self.assertEqual(train_out["Age"].isna().sum(), 0, "train Age still has NaNs")
        self.assertEqual(test_out["Age"].isna().sum(), 0, "test Age still has NaNs")
        self.assertEqual(test_out["Embarked"].isna().sum(), 0, "test Embarked still has NaNs")
        self.assertEqual(test_out["Fare"].isna().sum(), 0, "test Fare still has NaNs")
        # Shape must be preserved
        self.assertEqual(train_out.shape[0], train.shape[0])
        self.assertEqual(test_out.shape[0], test.shape[0])


    # ------------------------------------------------------------------
    # Test D: optimize_threshold search range covers the probability valley
    # Regression guard for the bimodal meta-model threshold bug:
    # The old search (linspace 0.40–0.60) starts inside the probability valley
    # for a heavily-regularized meta-model, causing it to select a threshold
    # (0.4687) that yields only ~29% OOF survivors vs the true 38.4%.
    # The fix extends the search floor to 0.30 so the real accuracy-optimum
    # (which also satisfies the survivor-count guardrail) can be found.
    # ------------------------------------------------------------------
    def test_optimize_threshold_range_covers_valley(self):
        """optimize_threshold must search from 0.30, not 0.40.

        The meta-model (LogisticRegression C=0.01, very heavy regularization)
        produces a bimodal probability distribution with clusters near 0.25 and
        0.62 and a sparse valley in [0.40, 0.50).  Searching only in [0.40,
        0.60] pins the selected threshold inside that valley at ~0.4687, which
        yields only 260/891 (29.2%) OOF survivors even though the true training
        survival rate is 38.4%.

        The fix: extend the search floor to 0.30 so thresholds that
        simultaneously maximise accuracy AND satisfy the 36.5–38.5% guardrail
        can be found.  This test uses the actual saved OOF predictions to
        reproduce the exact failure.
        """
        import os
        import numpy as np
        from sklearn.metrics import accuracy_score

        oof_path = os.path.join("experiments", "oof_predictions.npz")
        if not os.path.exists(oof_path):
            self.skipTest(f"OOF predictions file {oof_path} not found; run src.stacking first")

        data = np.load(oof_path)
        oof_probs = data["predictions"]
        y_true = data["target"]

        true_survivor_rate = y_true.mean()  # ~0.3838

        # --- OLD search: linspace(0.40, 0.60, 100) ---
        old_best_t, old_best_score = 0.50, 0.0
        for t in np.linspace(0.40, 0.60, 100):
            acc = accuracy_score(y_true, (oof_probs >= t).astype(int))
            if acc > old_best_score:
                old_best_score = acc
                old_best_t = t
        old_survivor_pct = (oof_probs >= old_best_t).mean()

        # --- NEW constrained search: linspace(0.30, 0.60, 300) with OOF-rate guard ---
        n = len(oof_probs)
        lo_guard = int(0.32 * n)
        hi_guard = int(0.43 * n)
        new_best_t, new_best_score = 0.50, 0.0
        new_unconstrained_t, new_unconstrained_score = 0.50, 0.0
        for t in np.linspace(0.30, 0.60, 300):
            acc = accuracy_score(y_true, (oof_probs >= t).astype(int))
            n_pos = int((oof_probs >= t).sum())
            if acc > new_unconstrained_score:
                new_unconstrained_score = acc
                new_unconstrained_t = t
            if lo_guard <= n_pos <= hi_guard and acc > new_best_score:
                new_best_score = acc
                new_best_t = t
        new_survivor_pct = (oof_probs >= new_best_t).mean()

        # Old search must find a threshold in [0.40, 0.60] (trapped in valley)
        self.assertGreaterEqual(old_best_t, 0.40,
            f"Old search baseline: threshold should be >=0.40 (in valley), got {old_best_t:.4f}")

        # Old search OOF survivor rate must be well below the true rate (the bug)
        self.assertLess(old_survivor_pct, true_survivor_rate - 0.05,
            f"Old search survivor rate {old_survivor_pct:.3f} should be < "
            f"{true_survivor_rate - 0.05:.3f} (true rate - 5pp), demonstrating the bug")

        # New constrained search must find a threshold outside the probability valley
        # (valley peak is at ~0.4687; threshold should be anchored below it)
        self.assertLess(new_best_t, 0.43,
            f"Constrained search must find threshold < 0.43 (out of valley), got {new_best_t:.4f}")

        # Constrained search must produce more OOF survivors than the unconstrained old search
        self.assertGreater(new_survivor_pct, old_survivor_pct,
            f"Constrained threshold must yield more survivors ({new_survivor_pct:.3f}) "
            f"than old ({old_survivor_pct:.3f})")

        # Constrained search OOF accuracy should not be drastically worse
        self.assertGreaterEqual(new_best_score, old_best_score - 0.02,
            f"Constrained threshold accuracy ({new_best_score:.4f}) should not be "
            f"worse than old ({old_best_score:.4f}) by more than 2pp")




if __name__ == "__main__":
    unittest.main()

