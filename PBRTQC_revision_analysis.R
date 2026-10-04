# =============================================================================
# PBRTQC REVISED ANALYSIS
# Reproducible analysis supporting the revised manuscript
#
# Final synthetic-cohort analysis, model tuning, training-only threshold
# selection, tuned multianalyte PBRTQC, NHANES-based transportability,
# patient-level cluster bootstrap, and manuscript Figures 1-4.
#
# NHANES analysis requires: NHANES_validation_400.csv
# =============================================================================

set.seed(123)

required_packages <- c(
  "dplyr", "tidyr", "purrr", "ggplot2", "tidymodels",
  "pROC", "ranger", "xgboost", "vip", "readr", "haven"
)

to_install <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(to_install) > 0) install.packages(to_install)

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(ggplot2)
  library(tidymodels)
  library(pROC)
  library(ranger)
  library(xgboost)
  library(vip)
  library(readr)
  library(haven)
})

tidymodels_prefer()
dir.create("revision_results", showWarnings = FALSE)

# -----------------------------------------------------------------------------
# 1. Reproducible synthetic baseline generation
# -----------------------------------------------------------------------------
# These distributions reproduce the ORIGINAL submitted analysis.
# They are now stated explicitly for reproducibility.
#
# glucose:    N(100, 15), lower-bounded at 40
# sodium:     N(140, 3)
# potassium:  N(4.2, 0.4)
# creatinine: N(1.0, 0.2), lower-bounded at 0.3
#
# Training and test cohorts are generated INDEPENDENTLY. The original script
# generated train and test by perturbing the same 10,000 baseline rows, which
# could leak patient-specific baseline values across train and test.

generate_baseline <- function(n, seed, id_prefix) {
  set.seed(seed)
  tibble(
    patient_id = paste0(id_prefix, seq_len(n)),
    order_id = seq_len(n),
    glucose = pmax(rnorm(n, 100, 15), 40),
    sodium = rnorm(n, 140, 3),
    potassium = rnorm(n, 4.2, 0.4),
    creatinine = pmax(rnorm(n, 1.0, 0.2), 0.3)
  )
}

# Keep 10,000 observations in each cohort so the revised analysis remains
# comparable in scale to the submitted analysis while ensuring independence.
train_base <- generate_baseline(10000, seed = 1001, id_prefix = "TR_")
test_base  <- generate_baseline(10000, seed = 2001, id_prefix = "TE_")

# -----------------------------------------------------------------------------
# 2. Explicit perturbation functions
# -----------------------------------------------------------------------------
# These magnitudes reproduce the ORIGINAL internal perturbation magnitudes:
#   shift:      glucose +30 mg/dL
#   drift:      sodium increases progressively up to approximately +6 mmol/L
#   hemolysis:  potassium +1.5 mmol/L
#   delay:      glucose -20 mg/dL
#
# The revised code records scenario and perturbation magnitude for every
# error-positive sample. "error = 1" means the sample received a simulated
# perturbation; "error = 0" means it did not.

apply_one_error <- function(df, idx, scenario) {
  if (length(idx) == 0) return(df)

  if (scenario == "shift") {
    df$glucose[idx] <- df$glucose[idx] + 30
    df$perturbation_value[idx] <- 30
  }

  if (scenario == "drift") {
    ord <- order(df$order_id[idx])
    drift_values <- seq(0.5, 6.0, length.out = length(idx))
    idx_ord <- idx[ord]
    df$sodium[idx_ord] <- df$sodium[idx_ord] + drift_values
    df$perturbation_value[idx_ord] <- drift_values
  }

  if (scenario == "hemolysis") {
    df$potassium[idx] <- df$potassium[idx] + 1.5
    df$perturbation_value[idx] <- 1.5
  }

  if (scenario == "delay") {
    df$glucose[idx] <- pmax(df$glucose[idx] - 20, 0.01)
    df$perturbation_value[idx] <- -20
  }

  df
}

inject_errors <- function(base_df, rate, seed) {
  set.seed(seed)

  n_err <- round(rate * nrow(base_df))
  idx <- sample(seq_len(nrow(base_df)), n_err, replace = FALSE)

  scenario_pool <- rep(
    c("shift", "drift", "hemolysis", "delay"),
    length.out = n_err
  )
  scenario_pool <- sample(scenario_pool, length(scenario_pool), replace = FALSE)

  out <- base_df %>%
    mutate(
      error = 0L,
      scenario = "normal",
      perturbation_value = 0
    )

  out$error[idx] <- 1L
  out$scenario[idx] <- scenario_pool

  for (sc in c("shift", "drift", "hemolysis", "delay")) {
    sc_idx <- idx[scenario_pool == sc]
    out <- apply_one_error(out, sc_idx, sc)
  }

  out %>%
    mutate(
      error = factor(error, levels = c(0, 1)),
      scenario = factor(
        scenario,
        levels = c("normal", "shift", "drift", "hemolysis", "delay")
      )
    )
}

train <- inject_errors(train_base, rate = 0.10, seed = 3001)
test  <- inject_errors(test_base,  rate = 0.05, seed = 4001)

write_csv(
  train %>% count(scenario, error, name = "n"),
  "revision_results/internal_train_scenario_counts.csv"
)
write_csv(
  test %>% count(scenario, error, name = "n"),
  "revision_results/internal_test_scenario_counts.csv"
)

# -----------------------------------------------------------------------------
# 3. ML data and preprocessing
# -----------------------------------------------------------------------------
features <- c("glucose", "sodium", "potassium", "creatinine")

train_ml <- train %>% select(all_of(features), error, scenario, patient_id, order_id)
test_ml  <- test  %>% select(all_of(features), error, scenario, patient_id, order_id)

rec <- recipe(
  error ~ glucose + sodium + potassium + creatinine,
  data = train_ml
) %>%
  step_normalize(all_predictors())

set.seed(5001)
cv <- vfold_cv(train_ml, v = 5, strata = error)

# -----------------------------------------------------------------------------
# 4. Logistic regression: CV predictions -> training-only threshold
# -----------------------------------------------------------------------------
log_mod <- logistic_reg() %>% set_engine("glm")

log_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(log_mod)

set.seed(5002)
log_cv <- fit_resamples(
  log_wf,
  resamples = cv,
  metrics = metric_set(roc_auc),
  control = control_resamples(save_pred = TRUE)
)

log_oof <- collect_predictions(log_cv)

youden_threshold <- function(truth, prob) {
  truth_num <- as.integer(as.character(truth))
  
  r <- pROC::roc(
    response = truth_num,
    predictor = prob,
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )
  
  best <- pROC::coords(
    r,
    x = "best",
    best.method = "youden",
    ret = c("threshold", "sensitivity", "specificity"),
    transpose = FALSE
  )
  
  threshold <- as.numeric(unlist(best$threshold))[1]
  
  return(threshold)
}

log_threshold <- youden_threshold(log_oof$error, log_oof$.pred_1)

# -----------------------------------------------------------------------------
# 5. Random Forest tuning + training-only threshold
# -----------------------------------------------------------------------------
rf_spec <- rand_forest(
  trees = 500,
  mtry = tune(),
  min_n = tune()
) %>%
  set_engine("ranger", probability = TRUE, importance = "permutation") %>%
  set_mode("classification")

rf_grid <- tidyr::crossing(
  mtry = 1:4,
  min_n = c(2L, 5L, 10L, 20L, 40L)
)

rf_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(rf_spec)

set.seed(5003)
rf_tuned <- tune_grid(
  rf_wf,
  resamples = cv,
  grid = rf_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(save_pred = TRUE)
)

best_rf <- select_best(rf_tuned, metric = "roc_auc")
rf_oof <- collect_predictions(rf_tuned, parameters = best_rf)
rf_threshold <- youden_threshold(rf_oof$error, rf_oof$.pred_1)

# -----------------------------------------------------------------------------
# 6. XGBoost tuning + training-only threshold
# -----------------------------------------------------------------------------
xgb_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  mtry = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_params <- parameters(
  trees(range = c(100L, 500L)),
  tree_depth(range = c(2L, 8L)),
  learn_rate(range = c(-4, -1)),
  min_n(range = c(2L, 40L)),
  loss_reduction(),
  sample_prop(range = c(0.60, 1.00)),
  finalize(mtry(), train_ml %>% select(all_of(features)))
)

set.seed(5004)
xgb_grid <- grid_latin_hypercube(xgb_params, size = 30)

xgb_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(xgb_spec)

set.seed(5005)
xgb_tuned <- tune_grid(
  xgb_wf,
  resamples = cv,
  grid = xgb_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(save_pred = TRUE)
)

best_xgb <- select_best(xgb_tuned, metric = "roc_auc")
xgb_oof <- collect_predictions(xgb_tuned, parameters = best_xgb)
xgb_threshold <- youden_threshold(xgb_oof$error, xgb_oof$.pred_1)

threshold_table <- tibble(
  Model = c("Logistic Regression", "Random Forest", "XGBoost"),
  Threshold_source = "5-fold out-of-fold predictions from training data",
  Threshold = c(log_threshold, rf_threshold, xgb_threshold)
)
write_csv(threshold_table, "revision_results/ml_thresholds_training_only.csv")

write_csv(best_rf,  "revision_results/RF_hyperparameters.csv")
write_csv(best_xgb, "revision_results/XGB_hyperparameters.csv")

# -----------------------------------------------------------------------------
# 7. Final ML fits — test set remains untouched until this point
# -----------------------------------------------------------------------------
fit_log <- fit(log_wf, data = train_ml)

rf_final_wf <- finalize_workflow(rf_wf, best_rf)
fit_rf <- fit(rf_final_wf, data = train_ml)

xgb_final_wf <- finalize_workflow(xgb_wf, best_xgb)
fit_xgb <- fit(xgb_final_wf, data = train_ml)

predict_with_meta <- function(fit_obj, new_data, threshold) {
  predict(fit_obj, new_data, type = "prob") %>%
    bind_cols(
      new_data %>% select(error, scenario, patient_id, order_id)
    ) %>%
    mutate(
      class_num = if_else(.pred_1 >= threshold, 1L, 0L),
      class = factor(class_num, levels = c(0, 1))
    )
}

pred_log <- predict_with_meta(fit_log, test_ml, log_threshold)
pred_rf  <- predict_with_meta(fit_rf,  test_ml, rf_threshold)
pred_xgb <- predict_with_meta(fit_xgb, test_ml, xgb_threshold)

# -----------------------------------------------------------------------------
# 8. Tuned multianalyte PBRTQC
# -----------------------------------------------------------------------------
# Tune:
#   window size: 10, 20, 30, 50, 100
#   control limit: 1.5, 2.0, 2.5, 3.0 SD
#
# PBRTQC continuous score = maximum absolute standardized deviation of the
# analyte-specific moving averages. This gives a continuous ROC-AUC score.

moving_average <- function(x, k) {
  as.numeric(stats::filter(x, rep(1 / k, k), sides = 1))
}

fit_pbrtqc_reference <- function(df, window) {
  normal <- df %>% filter(error == "0")

  map_dfr(features, function(v) {
    ma_v <- moving_average(normal[[v]], window)
    tibble(
      analyte = v,
      ma_mean = mean(ma_v, na.rm = TRUE),
      ma_sd = sd(ma_v, na.rm = TRUE)
    )
  })
}

score_pbrtqc <- function(df, reference, window) {
  scored <- df
  z_names <- character(0)

  for (v in features) {
    ma_v <- moving_average(scored[[v]], window)
    ref_v <- reference %>% filter(analyte == v)
    z_v <- abs((ma_v - ref_v$ma_mean) / ref_v$ma_sd)
    z_name <- paste0(v, "_z")
    scored[[z_name]] <- z_v
    z_names <- c(z_names, z_name)
  }

  z_mat <- as.matrix(scored[, z_names])
  scored$pbrtqc_score <- apply(z_mat, 1, max, na.rm = FALSE)
  scored
}

calc_binary_metrics <- function(truth, pred) {
  truth <- as.integer(as.character(truth))
  pred <- as.integer(as.character(pred))

  tp <- sum(pred == 1 & truth == 1, na.rm = TRUE)
  tn <- sum(pred == 0 & truth == 0, na.rm = TRUE)
  fp <- sum(pred == 1 & truth == 0, na.rm = TRUE)
  fn <- sum(pred == 0 & truth == 1, na.rm = TRUE)

  tibble(
    TP = tp, FP = fp, TN = tn, FN = fn,
    Sensitivity = ifelse(tp + fn > 0, tp / (tp + fn), NA_real_),
    Specificity = ifelse(tn + fp > 0, tn / (tn + fp), NA_real_),
    Accuracy = (tp + tn) / (tp + tn + fp + fn),
    Balanced_Accuracy = mean(
      c(
        ifelse(tp + fn > 0, tp / (tp + fn), NA_real_),
        ifelse(tn + fp > 0, tn / (tn + fp), NA_real_)
      ),
      na.rm = TRUE
    )
  )
}

pbrtqc_grid <- tidyr::crossing(
  window = c(10L, 20L, 30L, 50L, 100L),
  z_limit = c(1.5, 2.0, 2.5, 3.0)
)

pbrtqc_tuning <- pmap_dfr(pbrtqc_grid, function(window, z_limit) {
  ref <- fit_pbrtqc_reference(train, window)
  scored <- score_pbrtqc(train, ref, window) %>%
    filter(!is.na(pbrtqc_score)) %>%
    mutate(flag = factor(if_else(pbrtqc_score >= z_limit, 1L, 0L),
                         levels = c(0, 1)))

  m <- calc_binary_metrics(scored$error, scored$flag)

  bind_cols(
    tibble(window = window, z_limit = z_limit),
    m
  )
})

best_pbrtqc <- pbrtqc_tuning %>%
  arrange(desc(Balanced_Accuracy), desc(Sensitivity), desc(Specificity)) %>%
  slice(1)

write_csv(pbrtqc_tuning, "revision_results/PBRTQC_tuning_grid.csv")
write_csv(best_pbrtqc, "revision_results/PBRTQC_best_parameters.csv")

pbrtqc_ref <- fit_pbrtqc_reference(train, best_pbrtqc$window)

pbrtqc_test <- score_pbrtqc(test, pbrtqc_ref, best_pbrtqc$window) %>%
  filter(!is.na(pbrtqc_score)) %>%
  mutate(
    class = factor(
      if_else(pbrtqc_score >= best_pbrtqc$z_limit, 1L, 0L),
      levels = c(0, 1)
    )
  )

# -----------------------------------------------------------------------------
# 9. Internal overall metrics + 95% CIs
# -----------------------------------------------------------------------------
binom_ci <- function(x, n) {
  if (n == 0) return(c(NA_real_, NA_real_))
  as.numeric(binom.test(x, n)$conf.int)
}

metric_table <- function(truth, pred, prob, model_name) {
  truth_num <- as.integer(as.character(truth))
  pred_num <- as.integer(as.character(pred))

  keep <- complete.cases(truth_num, pred_num, prob)
  truth_num <- truth_num[keep]
  pred_num <- pred_num[keep]
  prob <- prob[keep]

  tp <- sum(pred_num == 1 & truth_num == 1)
  tn <- sum(pred_num == 0 & truth_num == 0)
  fp <- sum(pred_num == 1 & truth_num == 0)
  fn <- sum(pred_num == 0 & truth_num == 1)

  sens <- tp / (tp + fn)
  spec <- tn / (tn + fp)
  acc <- (tp + tn) / length(truth_num)

  sens_ci <- binom_ci(tp, tp + fn)
  spec_ci <- binom_ci(tn, tn + fp)
  acc_ci <- binom_ci(tp + tn, length(truth_num))

  roc_obj <- pROC::roc(
    truth_num, prob,
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )
  auc_ci <- as.numeric(pROC::ci.auc(roc_obj))

  tibble(
    Model = model_name,
    N = length(truth_num),
    TP = tp, FP = fp, TN = tn, FN = fn,
    ROC_AUC = as.numeric(pROC::auc(roc_obj)),
    AUC_L = auc_ci[1], AUC_U = auc_ci[3],
    Sensitivity = sens, Sens_L = sens_ci[1], Sens_U = sens_ci[2],
    Specificity = spec, Spec_L = spec_ci[1], Spec_U = spec_ci[2],
    Accuracy = acc, Acc_L = acc_ci[1], Acc_U = acc_ci[2]
  )
}

internal_overall <- bind_rows(
  metric_table(pred_log$error, pred_log$class, pred_log$.pred_1,
               "Logistic Regression"),
  metric_table(pred_rf$error, pred_rf$class, pred_rf$.pred_1,
               "Random Forest"),
  metric_table(pred_xgb$error, pred_xgb$class, pred_xgb$.pred_1,
               "XGBoost"),
  metric_table(pbrtqc_test$error, pbrtqc_test$class,
               pbrtqc_test$pbrtqc_score, "Tuned multianalyte PBRTQC")
)

write_csv(internal_overall, "revision_results/internal_overall_metrics.csv")

roc_log <- roc(as.integer(as.character(pred_log$error)), pred_log$.pred_1,
               levels = c(0, 1), direction = "<", quiet = TRUE)
roc_rf <- roc(as.integer(as.character(pred_rf$error)), pred_rf$.pred_1,
              levels = c(0, 1), direction = "<", quiet = TRUE)
roc_xgb <- roc(as.integer(as.character(pred_xgb$error)), pred_xgb$.pred_1,
               levels = c(0, 1), direction = "<", quiet = TRUE)

delong_internal <- tibble(
  Comparison = c(
    "Random Forest vs Logistic Regression",
    "XGBoost vs Logistic Regression",
    "Random Forest vs XGBoost"
  ),
  P_value = c(
    roc.test(roc_rf, roc_log, method = "delong", paired = TRUE)$p.value,
    roc.test(roc_xgb, roc_log, method = "delong", paired = TRUE)$p.value,
    roc.test(roc_rf, roc_xgb, method = "delong", paired = TRUE)$p.value
  )
)
write_csv(delong_internal, "revision_results/internal_DeLong_tests.csv")

# -----------------------------------------------------------------------------
# 10. Performance by perturbation type
# -----------------------------------------------------------------------------
scenario_metric_ml <- function(pred_df, model_name, scenario_name) {
  d <- pred_df %>%
    filter(scenario == "normal" | scenario == scenario_name)

  metric_table(d$error, d$class, d$.pred_1, model_name) %>%
    mutate(Scenario = scenario_name, .before = 1)
}

scenario_metric_pbrtqc <- function(scenario_name) {
  d <- pbrtqc_test %>%
    filter(scenario == "normal" | scenario == scenario_name)

  metric_table(
    d$error, d$class, d$pbrtqc_score, "Tuned multianalyte PBRTQC"
  ) %>%
    mutate(Scenario = scenario_name, .before = 1)
}

scenarios_internal <- c("shift", "drift", "hemolysis", "delay")

internal_by_scenario <- map_dfr(scenarios_internal, function(sc) {
  bind_rows(
    scenario_metric_ml(pred_log, "Logistic Regression", sc),
    scenario_metric_ml(pred_rf, "Random Forest", sc),
    scenario_metric_ml(pred_xgb, "XGBoost", sc),
    scenario_metric_pbrtqc(sc)
  )
})

write_csv(
  internal_by_scenario,
  "revision_results/internal_performance_by_perturbation.csv"
)

# -----------------------------------------------------------------------------
# 11. Feature importance
# -----------------------------------------------------------------------------
rf_engine <- extract_fit_engine(fit_rf)
rf_importance <- tibble::enframe(
  rf_engine$variable.importance,
  name = "Feature",
  value = "Importance"
) %>%
  arrange(desc(Importance))

xgb_engine <- extract_fit_engine(fit_xgb)
xgb_importance <- xgb.importance(
  feature_names = features,
  model = xgb_engine
) %>%
  as_tibble()

write_csv(rf_importance, "revision_results/RF_permutation_importance.csv")
write_csv(xgb_importance, "revision_results/XGB_gain_importance.csv")

# -----------------------------------------------------------------------------
# 12. OPTIONAL NHANES transportability analysis
# -----------------------------------------------------------------------------
# IMPORTANT:
# This is NOT validation against documented real-world QC failures.
# It is a transportability/sensitivity analysis using independently sourced
# patient chemistry values with computationally imposed perturbations.
#
# Expected baseline file: NHANES_validation_400.csv
# Required columns: patient_id (or SEQN), glucose, sodium, potassium, creatinine
#
# Do NOT claim 2,000 independent patients. There are 400 unique patients, each
# represented under 5 scenarios (normal, shift, drift, hemolysis, mixed).

RUN_NHANES <- file.exists("NHANES_validation_400.csv")

make_nhanes_scenarios <- function(normal_data, seed = 7001) {
  set.seed(seed)

  if ("SEQN" %in% names(normal_data) && !"patient_id" %in% names(normal_data)) {
    normal_data <- normal_data %>% rename(patient_id = SEQN)
  }

  stopifnot(all(c("patient_id", features) %in% names(normal_data)))

  normal <- normal_data %>%
    select(patient_id, all_of(features)) %>%
    mutate(scenario = "normal", error = 0L)

  shift <- normal_data %>%
    select(patient_id, all_of(features)) %>%
    mutate(
      glucose = glucose * 1.10,
      sodium = sodium + 3,
      potassium = potassium + 0.5,
      creatinine = creatinine + 0.2,
      scenario = "shift",
      error = 1L
    )

  drift <- normal_data %>%
    select(patient_id, all_of(features)) %>%
    arrange(patient_id) %>%
    mutate(
      drift_factor = seq(1.00, 1.08, length.out = n()),
      glucose = glucose * drift_factor,
      sodium = sodium + seq(0, 3, length.out = n()),
      potassium = potassium + seq(0, 0.5, length.out = n()),
      creatinine = creatinine + seq(0, 0.2, length.out = n()),
      scenario = "drift",
      error = 1L
    ) %>%
    select(-drift_factor)

  hemolysis <- normal_data %>%
    select(patient_id, all_of(features)) %>%
    mutate(
      potassium = potassium + runif(n(), 0.5, 1.5),
      glucose = glucose * runif(n(), 0.90, 0.98),
      creatinine = creatinine + runif(n(), 0, 0.1),
      sodium = sodium + runif(n(), -1, 1),
      scenario = "hemolysis",
      error = 1L
    )

  mixed <- normal_data %>%
    select(patient_id, all_of(features)) %>%
    arrange(patient_id) %>%
    mutate(
      drift_factor = seq(1.00, 1.05, length.out = n()),
      glucose = glucose * 1.05 * drift_factor + rnorm(n(), 0, 5),
      sodium = sodium + seq(0, 3, length.out = n()) + rnorm(n(), 0, 1),
      potassium = potassium + runif(n(), 0.5, 1.5) +
        seq(0, 0.5, length.out = n()) + rnorm(n(), 0, 0.2),
      creatinine = creatinine + seq(0, 0.2, length.out = n()) +
        rnorm(n(), 0, 0.05),
      glucose = pmax(glucose, 0.01),
      potassium = pmax(potassium, 0.01),
      creatinine = pmax(creatinine, 0.01),
      scenario = "mixed",
      error = 1L
    ) %>%
    select(-drift_factor)

  bind_rows(normal, shift, drift, hemolysis, mixed) %>%
    mutate(
      error = factor(error, levels = c(0, 1)),
      scenario = factor(
        scenario,
        levels = c("normal", "shift", "drift", "hemolysis", "mixed")
      )
    )
}

cluster_boot_metrics_fast <- function(
    data,
    prob_col,
    class_col,
    model_name,
    B = 1000,
    seed = 123
) {
  
  set.seed(seed)
  
  # Keep only complete rows needed for analysis
  d <- data %>%
    filter(
      !is.na(patient_id),
      !is.na(error),
      !is.na(.data[[prob_col]]),
      !is.na(.data[[class_col]])
    )
  
  patient_ids <- unique(d$patient_id)
  n_patients <- length(patient_ids)
  
  # Pre-compute row indices for every patient
  patient_rows <- split(seq_len(nrow(d)), d$patient_id)
  
  boot_auc  <- numeric(B)
  boot_sens <- numeric(B)
  boot_spec <- numeric(B)
  boot_acc  <- numeric(B)
  
  for (b in seq_len(B)) {
    
    sampled_ids <- sample(
      patient_ids,
      size = n_patients,
      replace = TRUE
    )
    
    idx <- unlist(
      patient_rows[as.character(sampled_ids)],
      use.names = FALSE
    )
    
    truth <- as.integer(as.character(d$error[idx]))
    pred  <- as.integer(as.character(d[[class_col]][idx]))
    prob  <- d[[prob_col]][idx]
    
    TP <- sum(truth == 1 & pred == 1)
    TN <- sum(truth == 0 & pred == 0)
    FP <- sum(truth == 0 & pred == 1)
    FN <- sum(truth == 1 & pred == 0)
    
    boot_sens[b] <- ifelse(
      TP + FN > 0,
      TP / (TP + FN),
      NA_real_
    )
    
    boot_spec[b] <- ifelse(
      TN + FP > 0,
      TN / (TN + FP),
      NA_real_
    )
    
    boot_acc[b] <- (TP + TN) / length(truth)
    
    boot_auc[b] <- as.numeric(
      pROC::auc(
        response = truth,
        predictor = prob,
        levels = c(0, 1),
        direction = "<",
        quiet = TRUE
      )
    )
  }
  
  tibble(
    Model = model_name,
    
    AUC = as.numeric(
      pROC::auc(
        response = as.integer(as.character(d$error)),
        predictor = d[[prob_col]],
        levels = c(0, 1),
        direction = "<",
        quiet = TRUE
      )
    ),
    
    AUC_low = quantile(
      boot_auc,
      0.025,
      na.rm = TRUE,
      names = FALSE
    ),
    
    AUC_high = quantile(
      boot_auc,
      0.975,
      na.rm = TRUE,
      names = FALSE
    ),
    
    Sensitivity = mean(
      as.integer(as.character(d[[class_col]]))[d$error == 1] == 1
    ),
    
    Sensitivity_low = quantile(
      boot_sens,
      0.025,
      na.rm = TRUE,
      names = FALSE
    ),
    
    Sensitivity_high = quantile(
      boot_sens,
      0.975,
      na.rm = TRUE,
      names = FALSE
    ),
    
    Specificity = mean(
      as.integer(as.character(d[[class_col]]))[d$error == 0] == 0
    ),
    
    Specificity_low = quantile(
      boot_spec,
      0.025,
      na.rm = TRUE,
      names = FALSE
    ),
    
    Specificity_high = quantile(
      boot_spec,
      0.975,
      na.rm = TRUE,
      names = FALSE
    ),
    
    Accuracy = mean(
      as.integer(as.character(d[[class_col]])) ==
        as.integer(as.character(d$error))
    ),
    
    Accuracy_low = quantile(
      boot_acc,
      0.025,
      na.rm = TRUE,
      names = FALSE
    ),
    
    Accuracy_high = quantile(
      boot_acc,
      0.975,
      na.rm = TRUE,
      names = FALSE
    )
  )
}
# -----------------------------------------------------------------------------
# 13. NHANES TRANSPORTABILITY ANALYSIS
# -----------------------------------------------------------------------------

# Read the 400 baseline NHANES profiles
nhanes_normal <- readr::read_csv(
  "NHANES_validation_400.csv",
  show_col_types = FALSE
)

# Recreate all perturbation scenarios from the baseline profiles
nhanes_all <- make_nhanes_scenarios(nhanes_normal)

# Check structure
cat("\nNHANES rows:", nrow(nhanes_all), "\n")
cat("Unique patients:", dplyr::n_distinct(nhanes_all$patient_id), "\n")
print(table(nhanes_all$scenario, useNA = "ifany"))


# -----------------------------------------------------------------------------
# ML predictions using FINAL trained models and FROZEN training thresholds
# -----------------------------------------------------------------------------

nhanes_ml <- nhanes_all %>%
  select(patient_id, scenario, error, all_of(features)) %>%
  mutate(order_id = row_number())

pred_log_ext <- predict_with_meta(
  fit_log,
  nhanes_ml,
  log_threshold
)

pred_rf_ext <- predict_with_meta(
  fit_rf,
  nhanes_ml,
  rf_threshold
)

pred_xgb_ext <- predict_with_meta(
  fit_xgb,
  nhanes_ml,
  xgb_threshold
)


# -----------------------------------------------------------------------------
# PBRTQC
#
# IMPORTANT:
# Score each scenario independently so that the moving-average window
# never crosses from one NHANES scenario into another.
# -----------------------------------------------------------------------------

score_one_nhanes_scenario <- function(d) {
  
  d <- d %>%
    arrange(patient_id) %>%
    mutate(order_id = row_number())
  
  scored <- score_pbrtqc(
    d,
    pbrtqc_ref,
    best_pbrtqc$window[[1]]
  ) %>%
    filter(!is.na(pbrtqc_score)) %>%
    mutate(
      class = factor(
        if_else(
          pbrtqc_score >= best_pbrtqc$z_limit[[1]],
          1L,
          0L
        ),
        levels = c(0, 1)
      )
    )
  
  scored
}

pred_pbrtqc_ext <- nhanes_all %>%
  split(.$scenario) %>%
  purrr::map_dfr(score_one_nhanes_scenario)

cat(
  "\nPBRTQC scored rows:",
  nrow(pred_pbrtqc_ext),
  "\n"
)

print(table(pred_pbrtqc_ext$scenario, useNA = "ifany"))


# -----------------------------------------------------------------------------
# 14. NHANES PATIENT-LEVEL CLUSTER BOOTSTRAP - B = 1000
# -----------------------------------------------------------------------------

cat("\nStarting FINAL NHANES bootstrap (B = 1000)...\n")

ext_cluster_ci_final <- bind_rows(
  
  cluster_boot_metrics_fast(
    pred_log_ext,
    ".pred_1",
    "class",
    "Logistic Regression",
    B = 1000,
    seed = 8101
  ),
  
  cluster_boot_metrics_fast(
    pred_rf_ext,
    ".pred_1",
    "class",
    "Random Forest",
    B = 1000,
    seed = 8102
  ),
  
  cluster_boot_metrics_fast(
    pred_xgb_ext,
    ".pred_1",
    "class",
    "XGBoost",
    B = 1000,
    seed = 8103
  ),
  
  cluster_boot_metrics_fast(
    pred_pbrtqc_ext,
    "pbrtqc_score",
    "class",
    "Tuned multianalyte PBRTQC",
    B = 1000,
    seed = 8104
  )
)

print(ext_cluster_ci_final)

write_csv(
  ext_cluster_ci_final,
  "revision_results/NHANES_transportability_cluster_bootstrap_CI.csv"
)

# -----------------------------------------------------------------------------
# 15. NHANES PERFORMANCE BY PERTURBATION TYPE
# -----------------------------------------------------------------------------

ext_by_scenario <- purrr::map_dfr(
  c("shift", "drift", "hemolysis", "mixed"),
  
  function(sc) {
    
    pbrtqc_subset <- pred_pbrtqc_ext %>%
      filter(
        as.character(scenario) %in% c("normal", sc)
      )
    
    bind_rows(
      
      scenario_metric_ml(
        pred_log_ext,
        "Logistic Regression",
        sc
      ),
      
      scenario_metric_ml(
        pred_rf_ext,
        "Random Forest",
        sc
      ),
      
      scenario_metric_ml(
        pred_xgb_ext,
        "XGBoost",
        sc
      ),
      
      metric_table(
        pbrtqc_subset$error,
        pbrtqc_subset$class,
        pbrtqc_subset$pbrtqc_score,
        "Tuned multianalyte PBRTQC"
      ) %>%
        mutate(
          Scenario = sc,
          .before = 1
        )
    )
  }
)

write_csv(
  ext_by_scenario,
  "revision_results/NHANES_transportability_by_perturbation.csv"
)

cat("\nFINAL NHANES analysis completed.\n")

# =============================================================================
# 16. PUBLICATION-QUALITY FIGURES FOR REVISED MANUSCRIPT
# =============================================================================

# Create figure directory
dir.create(
  "revision_results/figures",
  showWarnings = FALSE,
  recursive = TRUE
)

# Publication color palette
model_colors <- c(
  "Logistic Regression" = "#D55E00",
  "Random Forest" = "#0072B2",
  "XGBoost" = "#009E73",
  "Tuned multianalyte PBRTQC" = "#CC79A7"
)

# Common publication theme
publication_theme <- theme_classic(base_size = 13) +
  theme(
    plot.title = element_text(
      face = "bold",
      size = 15,
      hjust = 0.5
    ),
    plot.subtitle = element_text(
      size = 11,
      hjust = 0.5
    ),
    axis.title = element_text(
      face = "bold",
      size = 12
    ),
    axis.text = element_text(
      size = 11,
      color = "black"
    ),
    legend.title = element_blank(),
    legend.text = element_text(size = 10),
    legend.position = "bottom",
    panel.grid.major.y = element_line(
      color = "grey90",
      linewidth = 0.3
    ),
    panel.grid.minor = element_blank(),
    plot.margin = margin(10, 15, 10, 10)
  )

# =============================================================================
# FIGURE 1. INTERNAL ROC CURVES
# =============================================================================
# Create internal PBRTQC predictions for Figure 1
pred_pbrtqc <- score_pbrtqc(
  test,
  pbrtqc_ref,
  best_pbrtqc$window[[1]]
) %>%
  filter(!is.na(pbrtqc_score)) %>%
  mutate(
    class = factor(
      if_else(
        pbrtqc_score >= best_pbrtqc$z_limit[[1]],
        1L,
        0L
      ),
      levels = c(0, 1)
    )
  )

roc_pbrtqc <- pROC::roc(
  response = as.integer(as.character(pred_pbrtqc$error)),
  predictor = pred_pbrtqc$pbrtqc_score,
  levels = c(0, 1),
  direction = "<",
  quiet = TRUE
) 

cat(
  "\nLogistic AUC:", as.numeric(pROC::auc(roc_log)),
  "\nRF AUC:", as.numeric(pROC::auc(roc_rf)),
  "\nXGBoost AUC:", as.numeric(pROC::auc(roc_xgb)),
  "\nPBRTQC AUC:", as.numeric(pROC::auc(roc_pbrtqc)),
  "\n"
)


roc_log <- pROC::roc(
  response = as.integer(as.character(pred_log$error)),
  predictor = pred_log$.pred_1,
  levels = c(0, 1),
  direction = "<",
  quiet = TRUE
)

roc_rf <- pROC::roc(
  response = as.integer(as.character(pred_rf$error)),
  predictor = pred_rf$.pred_1,
  levels = c(0, 1),
  direction = "<",
  quiet = TRUE
)

roc_xgb <- pROC::roc(
  response = as.integer(as.character(pred_xgb$error)),
  predictor = pred_xgb$.pred_1,
  levels = c(0, 1),
  direction = "<",
  quiet = TRUE
)

roc_pbrtqc <- pROC::roc(
  response = as.integer(as.character(pred_pbrtqc$error)),
  predictor = pred_pbrtqc$pbrtqc_score,
  levels = c(0, 1),
  direction = "<",
  quiet = TRUE
)

# Create labels containing the calculated AUC values
auc_labels <- c(
  "Logistic Regression" =
    sprintf("Logistic Regression (AUC = %.3f)", as.numeric(pROC::auc(roc_log))),
  
  "Random Forest" =
    sprintf("Random Forest (AUC = %.3f)", as.numeric(pROC::auc(roc_rf))),
  
  "XGBoost" =
    sprintf("XGBoost (AUC = %.3f)", as.numeric(pROC::auc(roc_xgb))),
  
  "Tuned multianalyte PBRTQC" =
    sprintf("Tuned multianalyte PBRTQC (AUC = %.3f)",
            as.numeric(pROC::auc(roc_pbrtqc)))
)

roc_df <- bind_rows(
  
  tibble(
    Specificity = roc_log$specificities,
    Sensitivity = roc_log$sensitivities,
    Model = "Logistic Regression"
  ),
  
  tibble(
    Specificity = roc_rf$specificities,
    Sensitivity = roc_rf$sensitivities,
    Model = "Random Forest"
  ),
  
  tibble(
    Specificity = roc_xgb$specificities,
    Sensitivity = roc_xgb$sensitivities,
    Model = "XGBoost"
  ),
  
  tibble(
    Specificity = roc_pbrtqc$specificities,
    Sensitivity = roc_pbrtqc$sensitivities,
    Model = "Tuned multianalyte PBRTQC"
  )
)

roc_df$Model <- factor(
  roc_df$Model,
  levels = c(
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "Tuned multianalyte PBRTQC"
  )
)

fig1 <- ggplot(
  roc_df,
  aes(
    x = 1 - Specificity,
    y = Sensitivity,
    color = Model
  )
) +
  geom_line(linewidth = 1.15) +
  
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed",
    color = "grey50",
    linewidth = 0.7
  ) +
  
  scale_color_manual(values = model_colors, labels = auc_labels) +
  
  coord_equal() +
  
  scale_x_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.2),
    expand = c(0, 0)
  ) +
  
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.2),
    expand = c(0, 0)
  ) +
  
  labs(
    title = "Figure 1. Internal test ROC curves",
    subtitle = "Independent synthetic test cohort",
    x = "False-positive rate (1 - specificity)",
    y = "Sensitivity",
    color = "Method"
  ) + 
  guides(
    color = guide_legend(
      nrow = 2,
      byrow = TRUE
    )
  ) +
  
  publication_theme

print(fig1)

ggsave(
  "revision_results/figures/Figure_1_Internal_ROC.png",
  fig1,
  width = 7.5,
  height = 6.5,
  dpi = 600,
  bg = "white"
)

ggsave(
  "revision_results/figures/Figure_1_Internal_ROC.tiff",
  fig1,
  width = 7.5,
  height = 6.5,
  dpi = 600,
  compression = "lzw",
  bg = "white"
)

# =============================================================================
# FIGURE 2. INTERNAL PERFORMANCE BY PERTURBATION TYPE
# =============================================================================

fig2_data <- internal_by_scenario %>%
  mutate(
    Scenario = factor(
      Scenario,
      levels = c(
        "shift",
        "drift",
        "hemolysis",
        "delay"
      ),
      labels = c(
        "Systematic shift",
        "Progressive drift",
        "Hemolysis-associated",
        "Delayed processing"
      )
    ),
    Model = factor(
      Model,
      levels = c(
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "Tuned multianalyte PBRTQC"
      )
    )
  )

fig2 <- ggplot(
  fig2_data,
  aes(
    x = Scenario,
    y = ROC_AUC,
    fill = Model
  )
) +
  
  geom_col(
    position = position_dodge(width = 0.82),
    width = 0.72
  ) +
  
  geom_text(
    aes(label = sprintf("%.3f", ROC_AUC)),
    position = position_dodge(width = 0.82),
    vjust = -0.35,
    size = 3.5,
    color = "black"
  ) +
  
  geom_hline(
    yintercept = 0.5,
    linetype = "dashed",
    color = "grey40",
    linewidth = 0.7
  ) +
  
  scale_fill_manual(values = model_colors) +
  
  scale_y_continuous(
    limits = c(0, 1.05),
    breaks = seq(0, 1, 0.1),
    expand = expansion(mult = c(0, 0.02))
  ) +
  
  labs(
    title = "Figure 2. Detection performance by perturbation type",
    subtitle = "Independent synthetic test cohort",
    x = "Simulated perturbation",
    y = "ROC-AUC",
    fill = "Method"
  ) +
  
  publication_theme +
  
  theme(
    axis.text.x = element_text(
      angle = 20,
      hjust = 1
    )
  )

print(fig2)

ggsave(
  "revision_results/figures/Figure_2_Internal_Perturbation_AUC.png",
  fig2,
  width = 9,
  height = 6.5,
  dpi = 600,
  bg = "white"
)

ggsave(
  "revision_results/figures/Figure_2_Internal_Perturbation_AUC.tiff",
  fig2,
  width = 9,
  height = 6.5,
  dpi = 600,
  compression = "lzw",
  bg = "white"
)


# =============================================================================
# FIGURE 3. FEATURE IMPORTANCE
# =============================================================================

# Read final feature-importance results
rf_importance <- readr::read_csv(
  "revision_results/RF_permutation_importance.csv",
  show_col_types = FALSE
)

xgb_importance <- readr::read_csv(
  "revision_results/XGB_gain_importance.csv",
  show_col_types = FALSE
)

# -------------------------------------------------------------------------
# Prepare Random Forest importance
# Normalize RF permutation importance so the four values sum to 1.
# This makes the scale visually comparable with XGBoost Gain.
# -------------------------------------------------------------------------

rf_imp_plot <- rf_importance %>%
  transmute(
    Feature = Feature,
    Importance = Importance / sum(Importance),
    Model = "Random Forest"
  )

# -------------------------------------------------------------------------
# Prepare XGBoost importance
# Gain is already expressed as relative importance and sums to ~1.
# -------------------------------------------------------------------------

xgb_imp_plot <- xgb_importance %>%
  transmute(
    Feature = Feature,
    Importance = Gain,
    Model = "XGBoost"
  )

# Combine
feature_plot_data <- bind_rows(
  rf_imp_plot,
  xgb_imp_plot
) %>%
  mutate(
    Feature = case_when(
      Feature == "glucose" ~ "Glucose",
      Feature == "sodium" ~ "Sodium",
      Feature == "potassium" ~ "Potassium",
      Feature == "creatinine" ~ "Creatinine",
      TRUE ~ Feature
    ),
    
    # Force analytes into a consistent order
    Feature = factor(
      Feature,
      levels = c(
        "Creatinine",
        "Sodium",
        "Glucose",
        "Potassium"
      )
    ),
    
    Model = factor(
      Model,
      levels = c(
        "Random Forest",
        "XGBoost"
      )
    )
  )

# Colors consistent with the other manuscript figures
feature_colors <- c(
  "Random Forest" = "#0072B2",
  "XGBoost" = "#009E73"
)

# -------------------------------------------------------------------------
# Generate Figure 3
# -------------------------------------------------------------------------

fig3 <- ggplot(
  feature_plot_data,
  aes(
    x = Feature,
    y = Importance,
    fill = Model
  )
) +
  
  geom_col(
    position = position_dodge(width = 0.75),
    width = 0.65
  ) +
  
  coord_flip() +
  
  scale_fill_manual(
    values = feature_colors
  ) +
  
  scale_y_continuous(
    limits = c(0, 0.70),
    breaks = seq(0, 0.70, 0.10),
    labels = scales::percent_format(accuracy = 1),
    expand = expansion(mult = c(0, 0.02))
  ) +
  
  labs(
    title = "Figure 3. Feature importance by machine-learning model",
    subtitle = "Random Forest permutation importance and XGBoost gain",
    x = "Clinical chemistry analyte",
    y = "Relative importance",
    fill = "Model"
  ) +
  
  publication_theme

print(fig3)

# -------------------------------------------------------------------------
# Save publication-quality versions
# -------------------------------------------------------------------------

ggsave(
  "revision_results/figures/Figure_3_Feature_Importance.png",
  plot = fig3,
  width = 8,
  height = 5.5,
  dpi = 600,
  bg = "white"
)

ggsave(
  "revision_results/figures/Figure_3_Feature_Importance.tiff",
  plot = fig3,
  width = 8,
  height = 5.5,
  dpi = 600,
  compression = "lzw",
  bg = "white"
)

cat(
  "\nFigure 3 saved successfully to:",
  "\nrevision_results/figures/Figure_3_Feature_Importance.png",
  "\nrevision_results/figures/Figure_3_Feature_Importance.tiff\n"
)


# =============================================================================
# FIGURE 4. NHANES TRANSPORTABILITY BY PERTURBATION
# =============================================================================

fig4_data <- ext_by_scenario %>%
  
  mutate(
    Scenario = factor(
      Scenario,
      levels = c(
        "shift",
        "drift",
        "hemolysis",
        "mixed"
      ),
      labels = c(
        "Systematic shift",
        "Progressive drift",
        "Hemolysis-associated",
        "Mixed perturbation"
      )
    ),
    
    Model = factor(
      Model,
      levels = c(
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "Tuned multianalyte PBRTQC"
      )
    )
  )

fig4 <- ggplot(
  fig4_data,
  aes(
    x = Scenario,
    y = ROC_AUC,
    fill = Model
  )
) +
  geom_text(
    aes(label = sprintf("%.3f", ROC_AUC)),
    position = position_dodge(width = 0.82),
    vjust = -0.35,
    size = 3.5,
    color = "black"
  ) +
  geom_col(
    position = position_dodge(width = 0.82),
    width = 0.72
  ) +
  
  geom_hline(
    yintercept = 0.5,
    linetype = "dashed",
    color = "grey40",
    linewidth = 0.7
  ) +
  
  scale_fill_manual(values = model_colors) +
  
  scale_y_continuous(
    limits = c(0, 1.05),
    breaks = seq(0, 1, 0.1),
    expand = expansion(mult = c(0, 0.02))
  ) +
  
  labs(
    title = "Figure 4. NHANES-based transportability by perturbation type",
    subtitle = "400 NHANES participants represented under simulated perturbation scenarios",
    x = "Simulated perturbation",
    y = "ROC-AUC",
    fill = "Method"
  ) +
  
  publication_theme +
  
  theme(
    axis.text.x = element_text(
      angle = 20,
      hjust = 1
    )
  )

print(fig4)

ggsave(
  "revision_results/figures/Figure_4_NHANES_Perturbation_AUC.png",
  fig4,
  width = 9,
  height = 6.5,
  dpi = 600,
  bg = "white"
)

ggsave(
  "revision_results/figures/Figure_4_NHANES_Perturbation_AUC.tiff",
  fig4,
  width = 9,
  height = 6.5,
  dpi = 600,
  compression = "lzw",
  bg = "white"
)

# -----------------------------------------------------------------------------
# 17. SESSION INFORMATION
# -----------------------------------------------------------------------------

capture.output(
  sessionInfo(),
  file = "revision_results/sessionInfo.txt"
)

cat("\nPBRTQC revised analysis completed successfully.\n")
cat("Results are available in: revision_results/\n")
