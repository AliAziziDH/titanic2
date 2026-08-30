# AGENTS.md - Pure ML / Kaggle Titanic

```yaml
schema_version: "2.0.0"
spec_type: "Spec-Driven Development (SDD)"
architecture_pattern: "Progressive Disclosure"

execution_sequence:
  phase_1_to_5: "Clean & Advanced Structural Feature Engineering (AdjFare, Dynamic Title, Name Length)"
  phase_6: "Target-Free Bayesian Ridge Imputation & Causal Inverse Probability Weighting (IPW)"
  phase_7: "Symbolic Genetic GP Feature Synthesis"
  phase_8: "L2-Regularized Stacking Meta-Learner & Two-Pass WCG Decision Override"

pipeline_lifecycle:
  execution_protocol:
    step_1: "python -m src.features"
    step_2: "python -m src.imputation"
    step_3: "python -m src.modeling"
    step_4: "python -m src.stacking"
    step_5: "python -m src.final_submission"
  evaluation_gates:
    pytest_suite: "pytest tests/ -v"
    diagnostic_suite: "python -m src.diagnose_submission"

  local_brain_cloud_muscle:
    philosophy: "Zero local SSD/RAM footprint. Local machine is strictly for authoring and 5-row synthetic mock validation. Heavy compute is dispatched to cloud accelerators."
    stage_1_authoring: "Modular code under src/ with strict directory isolation."
    stage_2_mock_validation: "python3 .agents/skills/local-brain-cloud-muscle/scripts/mock_validator.py"
    stage_3_cloud_dispatch: "python3 .agents/skills/local-brain-cloud-muscle/scripts/cloud_dispatcher.py"
    stage_4_metric_ingestion: "python3 .agents/skills/local-brain-cloud-muscle/scripts/pull_metrics.py"

hard_rules:
  local_resource_guardrails:
    zero_local_training: "Never train heavy models or load full multi-gigabyte datasets locally. Validate pipeline logic against synthetic 5-row fixtures before cloud dispatch."
    checksum_verification: "All downloaded remote outputs must have MD5 checksums computed and verified."
  target_leakage_and_oof_guardrails:
    fold_isolation: "All target encodings, group calculations, and complex imputations must be strictly computed within local CV fold boundaries."
    stacking_protocol: "Stacking strictly uses Out-Of-Fold (OOF) predictions."
    wcg_anti_leakage: "For group-level target encoding (like WCG Two-Pass grouping) within a CV pipeline, store a mapping of Group_ID -> list of (index, target) during fit. In transform, use the passenger's DataFrame index to exclude their own target value from the group calculation to perfectly prevent target leakage."
    bayesian_imputation_rules: "Missing ages are defined as latent variables and imputed via fold-local Bayesian Ridge regression conditioned purely on non-target demographic confounders."
  strict_directory_isolation:
    working_directory: "Every command (python, pytest, git, etc.) and every file write/edit must be strictly targeted and executed from within the v2/ subdirectory."
  timestamped_isolated_payloads:
    payload_versioning: "Never use or overwrite static submission files. Programmatically generate a unique timestamped/checksum-tagged directory under v2/submissions/ (e.g., v2/submissions/exp08_softblend_age15_<timestamp>/) and output the MD5 checksum of every generated CSV payload."
  surgical_execution_policy:
    surgical_scope: "Edits must be strictly confined to the isolated root cause of a failure. Speculative variable renaming or unprompted refactoring of adjacent helper functions is prohibited."
    failing_test_first: "The agent must verify or write a localized failing test in the sandbox environment to reproduce and isolate the error before executing any self-repair modifications."

pre_submission_checklist:
  data_shape_hygiene: "891 train rows, 418 test rows."
  passenger_ids: "Exact Passenger IDs ranging from 892 to 1309."
  prediction_format: "Submission CSV must strictly output binary predictions (0 or 1) in the 'Survived' column."
  distribution_guardrail: "Predicted test survival rate must strictly fall within 36.5% - 38.5% (approx. 152-162 survivors). Any threshold producing > 165 or < 150 survivors must be rejected due to threshold overfitting."
  evaluation_protocol: "After every iteration, explicitly compute and report: (1) baseline CV score, (2) delta vs previous best, (3) test survivor count, and (4) root-cause analysis before recommending submission."

skills_discovery_catalog:
  location: ".agents/skills/"
  budget_constraint: "< 50 tokens per skill for activation metadata"
  policy: "Skills (such as 'self-repair', 'feature-engineering', 'local-brain-cloud-muscle') must register only their activation metadata to prevent active Context Rot, loading the full SKILL.md rules dynamically and on-demand only."
```

## Operational Overview
This document serves as the primary operational blueprint for automated agents and developers working on this repository.

### Directory Structure
- `data/`: Contains raw and processed dataset layers.
- `src/`: Contains the sequential execution scripts.
- `models/`: Stores trained models and pipelines.
- `submissions/`: Output directory for Kaggle predictions.
- `tests/`: Automated test suite for data pipeline validation.
- `experiments/`: Scratchpad for ad-hoc exploration.
- `.agents/skills/`: Dynamic, on-demand skill definitions.

### Sequential Execution Protocol
The pipeline relies on a specific sequence to prevent out-of-order execution anomalies. Always execute scripts sequentially in the following order:
1. `python -m src.features`
2. `python -m src.imputation`
3. `python -m src.modeling`
4. `python -m src.stacking`
5. `python -m src.final_submission`

### Validation & Diagnostics Suite
Before committing changes or generating submission artifacts:
- **Pytest Suite**: `pytest tests/ -v` (for functional regression testing)
- **Diagnostic Suite**: `python -m src.diagnose_submission` (for validating submission dimensions, schema drift, and target leakage anomalies)

### Target Leakage & Out-Of-Fold (OOF) Guardrails
To prevent target leakage and maintain a strict CV validation setup:
- All target encodings, group calculations, and complex imputations must be strictly computed within local CV fold boundaries.
- Stacking uses OOF predictions.
- For group-level target encoding (like WCG Two-Pass grouping) within a CV pipeline, store a mapping of `Group_ID -> list of (index, target)` during `fit`. In `transform`, use the passenger's DataFrame index to exclude their own target value from the group calculation to perfectly prevent target leakage.

### Surgical Edits & Self-Repair Protocols
- **Surgical Scope**: Edits must be strictly confined to the isolated root cause of a failure. Speculative variable renaming or unprompted refactoring of adjacent helper functions is prohibited.
- **Failing Test First**: The agent must verify or write a localized failing test in the sandbox environment to reproduce and isolate the error before executing any self-repair modifications.

### Pre-submission Checklist
- Ensure data shape hygiene (891 train rows, 418 test rows).
- Ensure final submission has exact Passenger IDs from 892 to 1309.
- Submission CSV must strictly output binary predictions (0 or 1) in the 'Survived' column.

### Skills Discovery Catalog
Located at `.agents/skills/`. Skills must expose lightweight activation metadata (< 50 tokens) to avoid context rot, deferring full `SKILL.md` loading until active task invocation.
