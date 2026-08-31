# 🚢 Titanic Survival Prediction: A Leakage-Safe Stratified Ensemble Pipeline with Causal Post-Processing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Kaggle Public Benchmark](https://img.shields.io/badge/Kaggle%20Benchmark-0.79186%20(Top%208%25)-success.svg)](https://www.kaggle.com/c/titanic)
[![CI/CD: Pytest & Guardrails](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

> An end-to-end, production-hardened Machine Learning pipeline on the classic Kaggle Titanic benchmark (\\(N_{\text{train}}=891, N_{\text{test}}=418\\)). Designed with strict cross-validation containment to prevent data leakage, regularized gradient boosting ensembles, and causal family-group heuristics (Woman-Child-Group) backed by automated CI/CD guardrails.

---

## 📌 Executive Summary

Most public solutions to the Kaggle Titanic challenge fall into one of two failure modes:
1. **The Trivial Plateau:** Default estimators that quickly stall at the demographic prior / gender baseline (\\(\approx 0.76555\\)).
2. **The Leakage Illusion:** Notebooks that achieve artificially high local cross-validation scores (\\(>0.84\\)) by computing target encodings, global median imputations, and ticket frequency aggregations across the combined train/test set before splitting.

This repository presents a **strictly leak-free, modular ML architecture** that achieves a verified **0.79186 on the Kaggle Public Leaderboard (Top ~8–10% bracket)** without external data snooping or post-hoc label scraping.

+---------------------------------------------------------------------------------------------------+ |                                      END-TO-END ARCHITECTURE                                      | +---------------------------------------------------------------------------------------------------+ Raw Data (Train / Test) │ ▼ [Stage 1: Deterministic Feature Extraction] ──► Titles, Surnames, Group Sizes, Ticket Cleanse │ ▼ [Stage 2: 5-Fold Stratified CV Boundary] ────► Fold-Local Imputation & Scalers (Zero Leakage) │ ▼ [Stage 3: Multi-Model Ensemble] ─────────────► CatBoost + LightGBM + Random Forest │ ▼ [Stage 4: Calibrated Probability Blend] ─────► Out-of-Fold Constrained Optimization │ ▼ [Stage 5: Two-Pass Causal WCG Override] ─────► Family/Ticket Fate Gating (Strict Child/Female scope) │ ▼ [Stage 6: Pre-Submission CI/CD Guardrails] ──► Strict Index Match (892–1309) & Hash Diagnostics

---

## 🔬 Key Architectural Components

### 1. Fold-Local Preprocessing & Leakage Elimination
* **Independent Transformations:** Global frequency mapping and target-dependent encodings create subtle variance leaks across fold boundaries. Here, imputation transformers (median age conditioned on `Pclass` \\(\times\\) `Title`) and group frequency calculations are fitted **strictly on training partitions** and transformed across evaluation splits.
* **Invariant Deck & Adjusted Fare Features:** Ticket fares in the original dataset represent aggregate prices paid for entire traveling parties. We derive the unit-cost feature:
  \\[\text{AdjFare} = \frac{\text{Fare}}{\text{Ticket Frequency}}\\]
  This isolates individual purchasing power from party size, allowing tree-based estimators to reliably discriminate economic tier.

### 2. Causal Two-Pass Woman-Child-Group (WCG) Heuristic
The evacuation of the RMS Titanic followed strict social coordination ("women and children first"), causing survival outcomes within travelling parties to be heavily correlated. We model this as a causal post-processing filter:
* **Pass 1 (Shared Ticket ID):** Identify travel cohorts sharing identical ticket numbers.
* **Pass 2 (Surname + Class + Port):** Cluster biological families who purchased separate tickets but embarked together.
* **Strict Demographic Scoping:**
  \\[\text{Override Condition} = (\text{Sex} == \text{'female'} \lor \text{Age} < 15 \lor \text{Title} == \text{'Master'})\\]
  Adult males (\\(\ge 15\\) years, non-Master) are **strictly excluded** from group survival overrides and receive their raw, soft-calibrated ensemble probabilities. Solo travelers (\\(N_{\text{group}}=1\\)) bypass heuristic overrides entirely.

### 3. Constrained Gradient-Boosted Ensemble
* **Diversity across Model Families:** Combines symmetric decision trees (**CatBoost**), leaf-wise tree growth (**LightGBM**), and bootstrap aggregated bagging (**Random Forest**).
* **Heavy Structural Regularization:** Due to the small sample size (\\(N=891\\)), gradient boosting hyperparameters enforce high L2 regularization (`l2_leaf_reg=50.0`, `reg_lambda=50.0`, `min_child_samples=20`) to prevent leaf memorization of outlier passenger profiles.

---

## 📊 Exploratory & Diagnostic Visualizations

The pipeline automatically compiles publication-quality visual diagnostics using Seaborn:

### Empirical Survival Distribution by Socio-Demographic Strata
The figure below demonstrates the empirical class and gender divide observed in the historical training partition against the base survival prior (\\(\approx 38.38\%\\)):

<p align="center">
  <img src="figures/survival_distribution_by_demographic.png" alt="Empirical Survival Rate by Class & Sex" width="850"/>
</p>

*Key Takeaway:* First and second-class females exhibit survival probabilities exceeding \\(90\%\\), while third-class adult males drop below \\(15\%\\). The primary model challenge resides on the margin: **Third-class females in large families** and **First-class males traveling solo**.

### Feature Correlation Matrix (Raw Numerical Covariates)
Multivariate dependencies across numerical features and the survival target:

<p align="center">
  <img src="figures/correlation_matrix.png" alt="Feature Correlation Matrix" width="750"/>
</p>

---

## 📈 Leaderboard Progression & Experimental Benchmarks

Every pipeline iteration was logged, version-controlled, and validated against the public evaluation engine:

| Experiment | Pipeline Configuration | CV Accuracy | Public LB Score | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline** | Naive Gender Heuristic (`Sex == 'female'`) | 0.7867 | 0.76555 | Rejected |
| **Exp 02** | Uncalibrated Random Forest (Default HP) | 0.8124 | 0.77511 | Overfitted |
| **Exp 05** | Stacking Classifier + Global Imputation | 0.8249 | 0.77751 | Marginal |
| **Exp 08** | **Leak-Free Ensemble + Stratified CV + WCG Heuristic** | **0.8328** | **0.79186** | 🏆 **Top Benchmark** |
| **Exp 10** | Zero-Noise Feature Ablation (Dropped `Deck`/`Cabin`) | 0.8215 | 0.78468 | Underfitted |
| **Exp 12** | Forced Threshold Shift (Fixed 157-Survivor Quota) | 0.8294 | 0.75837 | Distorted Margin |

---

## 🛠️ Engineering Post-Mortem & Technical Takeaways

Building production machine learning pipelines on constrained tabular datasets yields critical lessons that benchmark metrics alone obscure:

1. **The Threshold Forcing Trap:**
   In an attempt to align the test predictions with the expected historical survival rate (\\(\approx 37.5\%\\)), shifting the decision threshold to enforce an exact quota of 157 survivors caused a severe drop to **0.75837**. Artificially forcing global prediction quotas damages classification performance on margin-adjacent probabilities. Model calibration must be driven by proper scoring rules (Brier score, Log Loss) rather than post-hoc threshold slicing.

2. **Feature Ablation vs. Regularization:**
   Pruning high-cardinality features (`Cabin`, `Deck`) to eliminate noise stripped tree models of crucial spatial information that separated affluent first-class cabins from lower-deck steerage. Rather than deleting informative but sparse features, the superior engineering choice was **retaining the structural indicators while penalizing tree complexity via heavy L2 regularization**.

3. **Silent Index Inversion in Group Transforms:**
   When performing grouped transformations (e.g., ticket clustering and WCG matching), Pandas operations that sort or re-group data can silently alter row order. In testing, this created a subtle index misalignment that briefly scrambled predictions down to **0.72966**. Adding deterministic index assertions before writing submissions permanently resolved the defect:
   ```python
   assert (test_df['PassengerId'] == raw_test['PassengerId']).all(), "FATAL: Index mismatch"
📁 Repository Structure
.
├── data/
│   ├── raw/                       # Ground-truth raw datasets (train.csv, test.csv)
│   └── processed/                 # Leakage-free engineered partitions
├── figures/                       # High-resolution generated analytical plots
│   ├── correlation_matrix.png
│   └── survival_distribution_by_demographic.png
├── models/                        # Serialized pipeline estimators (git-ignored)
├── src/
│   ├── __init__.py
│   ├── features.py                # Deterministic feature engineering & parsing
│   ├── imputation.py              # Fold-contained Bayesian & median imputation
│   ├── modeling.py                # Stratified multi-model training & CV harness
│   ├── final_submission.py        # Ensemble probability blending & WCG overrides
│   ├── diagnose_submission.py     # Pre-commit CI submission validation suite
│   └── generate_visuals.py        # Matplotlib / Seaborn visualization generator
├── submissions/                   # Audit summaries and submission payloads
│   ├── run_summary.txt            # Timestamped diagnostic logs & MD5 hashes
│   └── submission_summary.json
├── tests/                         # Pytest test suite for data integrity & guardrails
│   ├── test_features.py
│   └── test_final_submission_guardrails.py
├── .gitignore
├── requirements.txt
└── README.md
🚀 Quickstart & Reproducibility
1. Environment Setup
Clone the repository and install all locked dependencies:
git clone https://github.com/AliAziziDH/titanic2.git
cd titanic2
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
2. Run the End-to-End Pipeline
Execute each pipeline phase sequentially:
# 1. Feature Engineering
python -m src.features

# 2. Model Training & Cross-Validation
python -m src.modeling

# 3. Final Inference & Causal Post-Processing
python -m src.final_submission

# 4. Generate Visualizations
python -m src.generate_visuals
3. Verify Guardrails & Integrity Tests
Run the test harness to confirm zero index drift and strict WCG boundary conditions:
pytest tests/ -v
python -m src.diagnose_submission
📜 Citation & Acknowledgements
Developed as an open-source machine learning case study exploring causal inference, out-of-fold feature containment, and disciplined MLOps practices on small-sample tabular benchmarks.
Primary Author: Ali Azizi (GitHub)
Dataset: Kaggle Titanic: Machine Learning from Disaster
License: MIT License
