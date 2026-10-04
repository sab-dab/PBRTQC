# PBRTQC

## Machine learning and multianalyte patient-based real-time quality control

This repository contains the reproducible R analysis supporting the revised manuscript evaluating machine-learning models and multianalyte patient-based real-time quality control (PBRTQC) for detection of simulated clinical chemistry disturbances.

## Primary analysis script

**`PBRTQC_revision_analysis.R`** is the primary analysis script supporting the revised manuscript.

The script includes:

- Independent synthetic training and test cohorts
- Explicit simulation of systematic shift, progressive drift, hemolysis-associated disturbance, and delayed-processing scenarios
- Logistic regression, Random Forest, and XGBoost models
- Five-fold cross-validation and hyperparameter tuning
- Training-only selection of classification thresholds
- Tuning of multianalyte PBRTQC window size and control limit
- Overall and perturbation-specific performance analysis
- ROC-AUC, sensitivity, specificity, accuracy, and 95% confidence intervals
- Pairwise DeLong comparisons of machine-learning ROC-AUCs
- Random Forest permutation importance and XGBoost gain importance
- NHANES-based transportability analysis
- Participant-level cluster bootstrap for NHANES confidence intervals
- Generation of manuscript Figures 1–4

## NHANES transportability analysis

The transportability analysis uses 400 unique participants from the NHANES August 2021–August 2023 cycle with complete glucose, sodium, potassium, and creatinine measurements.

Each participant is represented under five computational scenarios: normal, systematic shift, progressive drift, hemolysis-associated disturbance, and mixed perturbation.

Therefore, the resulting 2,000 scenario observations represent **400 unique participants**, not 2,000 independent participants.

The NHANES analysis is intended as a **transportability and sensitivity analysis using independently sourced patient chemistry measurements with computationally imposed disturbances**. It should not be interpreted as validation against documented real-world laboratory quality-control failures.

The NHANES component requires the derived input file:

`NHANES_validation_400.csv`

The source laboratory data were obtained from the publicly available NHANES August 2021–August 2023 laboratory files.

## Repository files

- `PBRTQC_revision_analysis.R` — primary reproducible analysis supporting the revised manuscript.
- `PBRTQC.R` — original analysis script associated with the initial manuscript submission; retained for transparency and version history.
- `README.md` — repository documentation.

## Reproducibility

The revised script sets random seeds for reproducibility and records the R session information after analysis.

Major R packages used include:

- tidymodels
- pROC
- ranger
- xgboost
- dplyr
- tidyr
- purrr
- ggplot2
- readr

The script creates a `revision_results` directory containing analysis outputs and manuscript figures.

Precomputed outputs from the revised analysis are provided in the `revision_results2` directory. When `PBRTQC_revision_analysis.R` is executed, the script generates outputs locally in the `revision_results` directory.

## Important note

The disturbances evaluated in this study are computationally simulated. The NHANES component uses real participant chemistry measurements as baseline data, but the analytical and pre-analytical disturbances are computationally imposed.
