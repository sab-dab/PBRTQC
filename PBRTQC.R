setwd(setwd("C:/Users/sabir/OneDrive/Documents/Error detection by ML/PBRTQC"))
# ============================================================
# STEP 1: Simulate baseline clinical chemistry dataset
# ============================================================

set.seed(123)

n <- 10000

# Simulate normal lab values (realistic distributions)

data <- data.frame(
  patient_id = 1:n,
  
  glucose = rnorm(n, mean = 100, sd = 15),        # mg/dL
  sodium = rnorm(n, mean = 140, sd = 3),          # mmol/L
  potassium = rnorm(n, mean = 4.2, sd = 0.4),     # mmol/L
  creatinine = rnorm(n, mean = 1.0, sd = 0.2)     # mg/dL
)

# Ensure no negative values (clinical realism)
data$glucose[data$glucose < 40] <- 40
data$creatinine[data$creatinine < 0.3] <- 0.3

# Add label (no error initially)
data$error <- 0

# View summary
summary(data)

# ============================================================
# STEP 2: Inject analytical and pre-analytical errors
# ============================================================

inject_errors <- function(data, error_rate = 0.1) {
  
  data_err <- data
  n <- nrow(data_err) 
  data_err$error <- 0
  data_err$error_type <- "none"
  
  # Select indices for errors
  error_indices <- sample(1:n, size = error_rate * n)
  
  for (i in error_indices) {
    
    error_type <- sample(c("shift", "drift", "hemolysis", "delay"), 1)
    
    if (error_type == "shift") {
      data_err$glucose[i] <- data_err$glucose[i] + 30
    }
    
    if (error_type == "drift") {
      data_err$sodium[i] <- data_err$sodium[i] + rnorm(1, 5, 1)
    }
    
    if (error_type == "hemolysis") {
      data_err$potassium[i] <- data_err$potassium[i] + 1.5
    }
    
    if (error_type == "delay") {
      data_err$glucose[i] <- data_err$glucose[i] - 20
    }
    
    data_err$error[i] <- 1 
    data_err$error_type[i] <- error_type
  }
  
  return(data_err)
}

#Step 3 Create two datasets
# ============================================================
# Create Dataset 1 (Training - 10% errors)
# ============================================================

data_train <- inject_errors(data, error_rate = 0.10)

# ============================================================
# Create Dataset 2 (Testing - 5% errors)
# ============================================================

data_test <- inject_errors(data, error_rate = 0.05) 


# Ensure data is ordered (important for PBRTQC)
data_train <- data_train[order(data_train$patient_id), ]
data_test  <- data_test[order(data_test$patient_id), ]

#check counts of each error type
table(data_train$error_type)
table(data_test$error_type)

#Step 4 verify distribution

# Training dataset(error proportion)
prop.table(table(data_train$error))

# Testing dataset(error proportion)
prop.table(table(data_test$error)) 

#error subtype proportion 
prop.table(table(data_train$error_type))
prop.table(table(data_test$error_type))

str(data_train)
str(data_test) 



# Visualization
library(ggplot2)

ggplot(data_test, aes(x = glucose, fill = factor(error))) +
  geom_histogram(bins = 50, alpha = 0.6, position = "identity") +
  labs(title = "Test Dataset (5% Errors)", fill = "Error")

write.csv(data_train, "data_train_simulated.csv", row.names = FALSE)
write.csv(data_test, "data_test_simulated.csv", row.names = FALSE)

saveRDS(data_train, "data_train_simulated.rds")
saveRDS(data_test, "data_test_simulated.rds") 

# ============================================================
# Step 5: PBRTQC baseline using moving average on glucose
# ============================================================

library(dplyr)

# ---------- Function to calculate moving average ----------
calc_moving_average <- function(x, k = 20) {
  stats::filter(x, rep(1/k, k), sides = 1)
}

# Window size
k <- 20

# ---------- TRAINING DATA ----------
# Compute moving average for glucose in training set
data_train$glucose_ma <- as.numeric(calc_moving_average(data_train$glucose, k = k))

# Use only rows where MA is available
train_ma <- data_train %>%
  filter(!is.na(glucose_ma))

# Define PBRTQC control limits from training data
ma_mean <- mean(train_ma$glucose_ma)
ma_sd   <- sd(train_ma$glucose_ma)

upper_limit <- ma_mean + 2 * ma_sd
lower_limit <- ma_mean - 2 * ma_sd

cat("PBRTQC limits for glucose moving average:\n")
cat("Mean =", round(ma_mean, 2), "\n")
cat("SD =", round(ma_sd, 2), "\n")
cat("Lower limit =", round(lower_limit, 2), "\n")
cat("Upper limit =", round(upper_limit, 2), "\n")

# ---------- TEST DATA ----------
# Compute moving average for glucose in test set
data_test$glucose_ma <- as.numeric(calc_moving_average(data_test$glucose, k = k))

# Flag PBRTQC alerts
data_test$pbrtc_flag <- ifelse(
  !is.na(data_test$glucose_ma) &
    (data_test$glucose_ma < lower_limit | data_test$glucose_ma > upper_limit),
  1, 0
)

# Check how many alerts were generated
table(data_test$pbrtc_flag)
prop.table(table(data_test$pbrtc_flag))

# ============================================================
# Step 6: Compare PBRTQC alerts with true error labels
# ============================================================

# Confusion matrix
table(PBRTQC_Flag = data_test$pbrtc_flag, True_Error = data_test$error)

# Basic performance metrics
tp <- sum(data_test$pbrtc_flag == 1 & data_test$error == 1, na.rm = TRUE)
tn <- sum(data_test$pbrtc_flag == 0 & data_test$error == 0, na.rm = TRUE)
fp <- sum(data_test$pbrtc_flag == 1 & data_test$error == 0, na.rm = TRUE)
fn <- sum(data_test$pbrtc_flag == 0 & data_test$error == 1, na.rm = TRUE)

sensitivity <- tp / (tp + fn)
specificity <- tn / (tn + fp)
accuracy    <- (tp + tn) / (tp + tn + fp + fn)

cat("PBRTQC Performance on Test Data:\n")
cat("Sensitivity =", round(sensitivity, 3), "\n")
cat("Specificity =", round(specificity, 3), "\n")
cat("Accuracy    =", round(accuracy, 3), "\n")

# ============================================================
# Step 7: Plot moving average and PBRTQC limits
# ============================================================

library(ggplot2)

ggplot(data_test, aes(x = patient_id, y = glucose_ma)) +
  geom_line() +
  geom_hline(yintercept = upper_limit, linetype = "dashed") +
  geom_hline(yintercept = lower_limit, linetype = "dashed") +
  labs(
    title = "PBRTQC: Moving Average of Glucose in Test Dataset",
    x = "Patient Order",
    y = "Glucose Moving Average"
  )

# ============================================================
#  Multianalyte PBRTQC using moving averages
# Analytes: glucose, sodium, potassium
# ============================================================

library(dplyr)
library(ggplot2)

# ---------- Function to calculate moving average ----------
calc_moving_average <- function(x, k = 20) {
  stats::filter(x, rep(1 / k, k), sides = 1)
}

# Window size
k <- 20

# ============================================================
# 1. TRAINING DATA: compute analyte-specific moving averages
# ============================================================

data_train$glucose_ma   <- as.numeric(calc_moving_average(data_train$glucose,   k = k))
data_train$sodium_ma    <- as.numeric(calc_moving_average(data_train$sodium,    k = k))
data_train$potassium_ma <- as.numeric(calc_moving_average(data_train$potassium, k = k))

# Use non-missing moving averages to define control limits
train_glucose_ma   <- data_train %>% filter(!is.na(glucose_ma))
train_sodium_ma    <- data_train %>% filter(!is.na(sodium_ma))
train_potassium_ma <- data_train %>% filter(!is.na(potassium_ma))

# Glucose limits
glucose_ma_mean  <- mean(train_glucose_ma$glucose_ma)
glucose_ma_sd    <- sd(train_glucose_ma$glucose_ma)
glucose_lower    <- glucose_ma_mean - 2 * glucose_ma_sd
glucose_upper    <- glucose_ma_mean + 2 * glucose_ma_sd

# Sodium limits
sodium_ma_mean   <- mean(train_sodium_ma$sodium_ma)
sodium_ma_sd     <- sd(train_sodium_ma$sodium_ma)
sodium_lower     <- sodium_ma_mean - 2 * sodium_ma_sd
sodium_upper     <- sodium_ma_mean + 2 * sodium_ma_sd

# Potassium limits
potassium_ma_mean <- mean(train_potassium_ma$potassium_ma)
potassium_ma_sd   <- sd(train_potassium_ma$potassium_ma)
potassium_lower   <- potassium_ma_mean - 2 * potassium_ma_sd
potassium_upper   <- potassium_ma_mean + 2 * potassium_ma_sd

# Print limits
cat("Multianalyte PBRTQC limits:\n\n")

cat("Glucose MA:\n")
cat("Mean =", round(glucose_ma_mean, 2), "\n")
cat("SD =", round(glucose_ma_sd, 2), "\n")
cat("Lower =", round(glucose_lower, 2), "\n")
cat("Upper =", round(glucose_upper, 2), "\n\n")

cat("Sodium MA:\n")
cat("Mean =", round(sodium_ma_mean, 2), "\n")
cat("SD =", round(sodium_ma_sd, 2), "\n")
cat("Lower =", round(sodium_lower, 2), "\n")
cat("Upper =", round(sodium_upper, 2), "\n\n")

cat("Potassium MA:\n")
cat("Mean =", round(potassium_ma_mean, 3), "\n")
cat("SD =", round(potassium_ma_sd, 3), "\n")
cat("Lower =", round(potassium_lower, 3), "\n")
cat("Upper =", round(potassium_upper, 3), "\n\n") 

# ============================================================
# 2. TEST DATA: compute analyte-specific moving averages
# ============================================================

data_test$glucose_ma   <- as.numeric(calc_moving_average(data_test$glucose,   k = k))
data_test$sodium_ma    <- as.numeric(calc_moving_average(data_test$sodium,    k = k))
data_test$potassium_ma <- as.numeric(calc_moving_average(data_test$potassium, k = k))

# ============================================================
# 3. Create analyte-specific PBRTQC flags
# ============================================================

data_test$glucose_flag <- ifelse(
  !is.na(data_test$glucose_ma) &
    (data_test$glucose_ma < glucose_lower | data_test$glucose_ma > glucose_upper),
  1, 0
)

data_test$sodium_flag <- ifelse(
  !is.na(data_test$sodium_ma) &
    (data_test$sodium_ma < sodium_lower | data_test$sodium_ma > sodium_upper),
  1, 0
)

data_test$potassium_flag <- ifelse(
  !is.na(data_test$potassium_ma) &
    (data_test$potassium_ma < potassium_lower | data_test$potassium_ma > potassium_upper),
  1, 0
)

# ============================================================
# 4. Combined multianalyte PBRTQC flag
# Flag if ANY analyte is outside limits
# ============================================================

data_test$pbrtc_multi_flag <- ifelse(
  data_test$glucose_flag == 1 |
    data_test$sodium_flag == 1 |
    data_test$potassium_flag == 1,
  1, 0
)

# Check number of alerts
table(data_test$pbrtc_multi_flag)
prop.table(table(data_test$pbrtc_multi_flag)) 

# ============================================================
# 5. Confusion matrix and performance
# ============================================================

table(PBRTQC_Multi_Flag = data_test$pbrtc_multi_flag, True_Error = data_test$error)

tp <- sum(data_test$pbrtc_multi_flag == 1 & data_test$error == 1, na.rm = TRUE)
tn <- sum(data_test$pbrtc_multi_flag == 0 & data_test$error == 0, na.rm = TRUE)
fp <- sum(data_test$pbrtc_multi_flag == 1 & data_test$error == 0, na.rm = TRUE)
fn <- sum(data_test$pbrtc_multi_flag == 0 & data_test$error == 1, na.rm = TRUE)

sensitivity <- tp / (tp + fn)
specificity <- tn / (tn + fp)
accuracy    <- (tp + tn) / (tp + tn + fp + fn)

cat("Multianalyte PBRTQC Performance on Test Data:\n")
cat("Sensitivity =", round(sensitivity, 3), "\n")
cat("Specificity =", round(specificity, 3), "\n")
cat("Accuracy    =", round(accuracy, 3), "\n") 

# ============================================================
# 6. Inspect analyte-specific alert counts
# ============================================================

cat("Analyte-specific alert counts:\n")
cat("Glucose flags:", sum(data_test$glucose_flag, na.rm = TRUE), "\n")
cat("Sodium flags:", sum(data_test$sodium_flag, na.rm = TRUE), "\n")
cat("Potassium flags:", sum(data_test$potassium_flag, na.rm = TRUE), "\n") 

# Sodium moving average plot
ggplot(data_test, aes(x = patient_id, y = sodium_ma)) +
  geom_line() +
  geom_hline(yintercept = sodium_upper, linetype = "dashed") +
  geom_hline(yintercept = sodium_lower, linetype = "dashed") +
  labs(
    title = "PBRTQC: Moving Average of Sodium in Test Dataset",
    x = "Patient Order",
    y = "Sodium Moving Average"
  )

# Potassium moving average plot
ggplot(data_test, aes(x = patient_id, y = potassium_ma)) +
  geom_line() +
  geom_hline(yintercept = potassium_upper, linetype = "dashed") +
  geom_hline(yintercept = potassium_lower, linetype = "dashed") +
  labs(
    title = "PBRTQC: Moving Average of Potassium in Test Dataset",
    x = "Patient Order",
    y = "Potassium Moving Average"
  )

################################################################################
# ============================================================
# Load required packages
# ============================================================

library(tidymodels)
library(dplyr)

# Avoid conflicts
tidymodels::tidymodels_prefer() 

# ============================================================
# Prepare datasets
# ============================================================

# Select features (predictors)
features <- c("glucose", "sodium", "potassium", "creatinine")

train_ml <- data_train %>%
  select(all_of(features), error)

test_ml <- data_test %>%
  select(all_of(features), error)

# Convert outcome to factor
train_ml$error <- factor(train_ml$error, levels = c(0,1))
test_ml$error  <- factor(test_ml$error, levels = c(0,1)) 

# ============================================================
# Recipe (preprocessing)
# ============================================================

recipe_ml <- recipe(error ~ ., data = train_ml) %>%
  step_normalize(all_predictors()) 

log_model <- logistic_reg() %>%
  set_engine("glm") 

rf_model <- rand_forest(trees = 200) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification") 

xgb_model <- boost_tree(
  trees = 200,
  learn_rate = 0.05,
  tree_depth = 6
) %>%
  set_engine("xgboost") %>%
  set_mode("classification") 

wf_log <- workflow() %>%
  add_model(log_model) %>%
  add_recipe(recipe_ml)

wf_rf <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(recipe_ml)

wf_xgb <- workflow() %>%
  add_model(xgb_model) %>%
  add_recipe(recipe_ml) 

fit_log <- fit(wf_log, data = train_ml)
fit_rf  <- fit(wf_rf, data = train_ml)
fit_xgb <- fit(wf_xgb, data = train_ml) 

pred_log <- predict(fit_log, test_ml, type = "prob") %>%
  bind_cols(predict(fit_log, test_ml)) %>%
  bind_cols(test_ml)

pred_rf <- predict(fit_rf, test_ml, type = "prob") %>%
  bind_cols(predict(fit_rf, test_ml)) %>%
  bind_cols(test_ml)

pred_xgb <- predict(fit_xgb, test_ml, type = "prob") %>%
  bind_cols(predict(fit_xgb, test_ml)) %>%
  bind_cols(test_ml) 

# ============================================================
# Evaluation function
# ============================================================
evaluate_model <- function(pred) {
  
  roc  <- roc_auc(pred, truth = error, .pred_1, event_level = "second")
  acc  <- accuracy(pred, truth = error, .pred_class)
  sens <- sens(pred, truth = error, .pred_class, event_level = "second")
  spec <- spec(pred, truth = error, .pred_class, event_level = "second")
  
  data.frame(
    ROC_AUC = roc$.estimate,
    Accuracy = acc$.estimate,
    Sensitivity = sens$.estimate,
    Specificity = spec$.estimate
  )
}


results_log <- evaluate_model(pred_log)
results_rf  <- evaluate_model(pred_rf)
results_xgb <- evaluate_model(pred_xgb)

results_log
results_rf
results_xgb 

results <- bind_rows(
  Logistic = results_log,
  RandomForest = results_rf,
  XGBoost = results_xgb,
  .id = "Model"
)

results 

library(pROC)

roc_obj <- roc(pred_xgb$error, pred_xgb$.pred_1)

plot(roc_obj, main = "ROC Curve - XGBoost Model") 


# ============================================================
# XGBoost Feature Importance
# ============================================================

library(xgboost)

# Extract fitted model
xgb_fit <- extract_fit_parsnip(fit_xgb)$fit

# Get importance
importance_matrix <- xgb.importance(model = xgb_fit)

importance_matrix 

xgb.plot.importance(importance_matrix, main = "XGBoost Feature Importance") 

# ============================================================
# Random Forest Feature Importance
# ============================================================

rf_fit <- extract_fit_parsnip(fit_rf)$fit

rf_importance <- rf_fit$variable.importance

rf_importance 

# ============================================================
# Final Results Table
# ============================================================

final_results <- data.frame(
  Method = c(
    "PBRTQC (Single Analyte)",
    "PBRTQC (Multianalyte)",
    "Logistic Regression",
    "Random Forest",
    "XGBoost"
  ),
  
  ROC_AUC = c(
    NA,
    NA,
    round(results_log$ROC_AUC, 3),
    round(results_rf$ROC_AUC, 3),
    round(results_xgb$ROC_AUC, 3)
  ),
  
  Sensitivity = c(
    0.040,
    0.094,
    round(results_log$Sensitivity, 3),
    round(results_rf$Sensitivity, 3),
    round(results_xgb$Sensitivity, 3)
  ),
  
  Specificity = c(
    0.961,
    0.890,
    round(results_log$Specificity, 3),
    round(results_rf$Specificity, 3),
    round(results_xgb$Specificity, 3)
  )
)

final_results 

write.csv(final_results, "Table_Model_Comparison.csv", row.names = FALSE) 

