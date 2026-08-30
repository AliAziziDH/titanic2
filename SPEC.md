# SPEC-07: Causal Recalibration & Multi-Seed Bagging (Experiment-07)

```yaml
schema_version: "2.0.0"
spec_type: "Spec-Driven Development (SDD)"
experiment_id: "EXP-07"
title: "Causal Recalibration & Multi-Seed Bagging"
status: "ACTIVE"
created_at: "2026-08-22"
```

## 1. Objective
The goal of this experiment is to implement Causal Recalibration and Multi-Seed Bagged Stacking to break the 77% leaderboard ceiling. By training our models across 10 random seeds and averaging their predictions, we stabilize the decision boundary against individual seed variances. Concurrently, we lower the strict local evaluation validation check to 0.8180 to prevent compensatory overfitting and roll back the rigid test-only consensus probability pooling.

- **Multi-Seed Bagging**: Train the stacking pipeline (including base CatBoost, XGBoost, and MLP models) across 10 distinct random seeds: `[42, 101, 202, 303, 404, 505, 606, 707, 808, 909]`.
- **Averaging Predictions**: Average Out-of-Fold (OOF) and test predictions using probabilistic soft voting to stabilize the decision boundary.
- **Causal Recalibration**: Lower the local validation accuracy check in `src/evaluate.py` from 0.8406 to 0.8180 to align with a generalized, leakage-free CV accuracy.
- **Rollback Test-Only Consensus**: Remove test-only passenger group consensus probability pooling in `WCGPostProcessor`. Retain deterministic training set overrides on overlap groups.

---

## 2. Acceptance Criteria

- **Multi-Seed Configuration**:
  - The list of 10 seeds must be explicitly defined: `[42, 101, 202, 303, 404, 505, 606, 707, 808, 909]`.
  - All base models and meta-models in `src/stacking.py` must be trained using these 10 distinct seeds.
  - All models must be saved with a `_seed_{seed}.joblib` suffix.
  
- **Soft Voting Averaging**:
  - The final stacking OOF probabilities and test probabilities must be the simple average of the OOF and test probabilities predicted across all 10 seed stacking runs.
  - The optimal decision threshold must be evaluated on the averaged OOF probabilities.

- **Post-Processor Calibrated Overrides**:
  - `WCGPostProcessor` must NOT pool or average the predicted probabilities of test-only families/groups.
  - Standard deterministic training set overrides (0.0 or 1.0 survival rates) must still apply to overlapping groups.

- **Evaluation Harness & Survival Rate Stability**:
  - The local validation accuracy threshold in `src/evaluate.py` must be updated to 0.8180.
  - Running `python -m src.evaluate` must pass with stable CV accuracy $\ge 0.818$ and final predicted survival rate within $[32.8\%, 42.8\%]$ (target $37.8\% \pm 5\%$).

---

## 3. Verification Protocol

1. **Pipeline Execution**:
   Run all pipeline stages to train models and generate final outputs:
   ```bash
   python -m src.features
   python -m src.imputation
   python -m src.modeling
   python -m src.stacking
   python -m src.final_submission
   ```

2. **Evaluation Harness**:
   Verify local performance and prediction distribution sanity:
   ```bash
   python -m src.evaluate
   ```

3. **Automated Regression Testing**:
   Ensure all unit tests pass:
   ```bash
   PYTHONPATH=. pytest tests/ -v
   ```

4. **Submission Diagnostics**:
   Ensure no structural leakage, correct ID ranges (892-1309), and exactly 418 rows:
   ```bash
   python -m src.diagnose_submission
   ```

---

## 4. Agent Brakes (Circuit Breaker)

- **Halt Conditions**:
  - If the average survival probability computation results in any `NaN` or infinite values.
  - If the final predicted survival rate falls outside the $[32.8\%, 42.8\%]$ range.
  - If the local pipeline evaluation or unit tests fail.
