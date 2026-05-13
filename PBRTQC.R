set.seed(123)

required_packages <- c(
  "dplyr", "ggplot2", "tidymodels", "pROC",
  "xgboost", "ranger", "vip", "readr"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

library(dplyr)
library(ggplot2)
library(tidymodels)
library(pROC)
library(xgboost)
library(ranger)
library(vip)
library(readr)

tidymodels::tidymodels_prefer()
install.packages("Ckmeans.1d.dp")
library(haven)
n <- 10000

data <- data.frame(
  patient_id = 1:n,
  glucose = pmax(rnorm(n, 100, 15), 40),
  sodium = rnorm(n, 140, 3),
  potassium = rnorm(n, 4.2, 0.4),
  creatinine = pmax(rnorm(n, 1.0, 0.2), 0.3),
  error = 0
) 

inject_errors <- function(data, rate) {
  data$error <- 0
  idx <- sample(1:nrow(data), rate * nrow(data))
  
  for (i in idx) {
    type <- sample(c("shift","drift","hemolysis","delay"),1)
    
    if (type=="shift") data$glucose[i] <- data$glucose[i] + 30
    if (type=="drift") data$sodium[i] <- data$sodium[i] + rnorm(1,5,1)
    if (type=="hemolysis") data$potassium[i] <- data$potassium[i] + 1.5
    if (type=="delay") data$glucose[i] <- data$glucose[i] - 20
    
    data$error[i] <- 1
  }
  data
}

train <- inject_errors(data, 0.10)
test  <- inject_errors(data, 0.05) 

ma <- function(x,k=20) as.numeric(stats::filter(x, rep(1/k,k), sides=1))

train$ma <- ma(train$glucose)
limits <- c(mean(train$ma,na.rm=T), sd(train$ma,na.rm=T))

test$ma <- ma(test$glucose)

test$flag <- ifelse(test$ma < limits[1]-2*limits[2] |
                      test$ma > limits[1]+2*limits[2],1,0) 

features <- c("glucose","sodium","potassium","creatinine")

train_ml <- train[,c(features,"error")]
test_ml  <- test[,c(features,"error")]

train_ml$error <- factor(train_ml$error)
test_ml$error  <- factor(test_ml$error)

rec <- recipe(error ~ ., data=train_ml) %>%
  step_normalize(all_predictors()) 

log_mod <- logistic_reg() %>% set_engine("glm")

rf_mod <- rand_forest(trees=500,mtry=1,min_n=20) %>%
  set_engine("ranger") %>% set_mode("classification")

xgb_mod <- boost_tree(
  trees=200, tree_depth=6, learn_rate=0.003,
  loss_reduction=0.0002, sample_size=0.63
) %>% set_engine("xgboost") %>% set_mode("classification") 

wf <- function(model) workflow() %>% add_model(model) %>% add_recipe(rec)

fit_log <- fit(wf(log_mod), train_ml)
fit_rf  <- fit(wf(rf_mod), train_ml)
fit_xgb <- fit(wf(xgb_mod), train_ml) 

pred_log <- predict(fit_log, test_ml, type="prob") %>% bind_cols(test_ml)
pred_rf  <- predict(fit_rf, test_ml, type="prob") %>% bind_cols(test_ml)
pred_xgb <- predict(fit_xgb, test_ml, type="prob") %>% bind_cols(test_ml)

# Threshold optimization for XGBoost
roc_xgb <- roc(test_ml$error, pred_xgb$.pred_1)
coords_df <- pROC::coords(
  roc_xgb,
  "all",
  ret = c("threshold","sensitivity","specificity")
)

coords_df$youden <- coords_df$sensitivity + coords_df$specificity - 1

thresh <- coords_df$threshold[which.max(coords_df$youden)]

print(thresh)

pred_log$class <- ifelse(pred_log$.pred_1>0.5,1,0)
pred_rf$class  <- ifelse(pred_rf$.pred_1>0.5,1,0)
pred_xgb$class <- ifelse(pred_xgb$.pred_1>thresh,1,0)

pred_log$class <- factor(pred_log$class)
pred_rf$class  <- factor(pred_rf$class)
pred_xgb$class <- factor(pred_xgb$class, levels = c(0,1)) 
table(pred_xgb$class)

metrics <- function(df,name){
  tp <- sum(df$class==1 & df$error==1)
  tn <- sum(df$class==0 & df$error==0)
  fp <- sum(df$class==1 & df$error==0)
  fn <- sum(df$class==0 & df$error==1)
  
  data.frame(Model=name,TP=tp,FP=fp,TN=tn,FN=fn,
             Sensitivity=tp/(tp+fn),
             Specificity=tn/(tn+fp),
             Accuracy=(tp+tn)/(tp+tn+fp+fn))
}

res <- bind_rows(
  metrics(pred_log,"Logistic"),
  metrics(pred_rf,"Random Forest"),
  metrics(pred_xgb,"XGBoost")
)

print(res) 

roc_log <- roc(test_ml$error, pred_log$.pred_1)
roc_rf  <- roc(test_ml$error, pred_rf$.pred_1)

plot(roc_rf, col="blue")
plot(roc_log, col="red", add=TRUE)
plot(roc_xgb, col="green", add=TRUE)
legend("bottomright", legend=c("RF","Log","XGB"),
       col=c("blue","red","green"), lwd=2) 

imp <- xgb.importance(model = extract_fit_parsnip(fit_xgb)$fit)

ggplot(imp[1:10,], aes(x=reorder(Feature,Gain),y=Gain))+
  geom_col()+
  coord_flip()+
  ggtitle("XGBoost Feature Importance") 

dir.create("results", showWarnings=FALSE)

write.csv(res, "results/model_results.csv", row.names=FALSE)

ggsave("results/roc.png", width=6, height=4, dpi=300) 

print(res) 

# ============================================================
# Confidence Interval Function
# ============================================================

binom_ci <- function(x, n) {
  ci <- binom.test(x, n)$conf.int
  c(lower = ci[1], upper = ci[2])
}

metrics_with_ci <- function(df, name){
  
  tp <- sum(df$class==1 & df$error==1)
  tn <- sum(df$class==0 & df$error==0)
  fp <- sum(df$class==1 & df$error==0)
  fn <- sum(df$class==0 & df$error==1)
  
  sens <- tp/(tp+fn)
  spec <- tn/(tn+fp)
  acc  <- (tp+tn)/(tp+tn+fp+fn)
  
  sens_ci <- binom_ci(tp, tp+fn)
  spec_ci <- binom_ci(tn, tn+fp)
  acc_ci  <- binom_ci(tp+tn, tp+tn+fp+fn)
  
  data.frame(
    Model = name,
    TP=tp, FP=fp, TN=tn, FN=fn,
    
    Sensitivity = sens,
    Sens_L = sens_ci[1],
    Sens_U = sens_ci[2],
    
    Specificity = spec,
    Spec_L = spec_ci[1],
    Spec_U = spec_ci[2],
    
    Accuracy = acc,
    Acc_L = acc_ci[1],
    Acc_U = acc_ci[2]
  )
} 

res_ci <- bind_rows(
  metrics_with_ci(pred_log,"Logistic"),
  metrics_with_ci(pred_rf,"Random Forest"),
  metrics_with_ci(pred_xgb,"XGBoost")
)

print(res_ci) 

# ============================================================
# ROC + CI
# ============================================================

roc_log <- roc(test_ml$error, pred_log$.pred_1)
roc_rf  <- roc(test_ml$error, pred_rf$.pred_1)
roc_xgb <- roc(test_ml$error, pred_xgb$.pred_1)

auc_table <- data.frame(
  Model = c("Logistic","Random Forest","XGBoost"),
  
  AUC = c(auc(roc_log), auc(roc_rf), auc(roc_xgb)),
  
  CI_L = c(ci.auc(roc_log)[1],
           ci.auc(roc_rf)[1],
           ci.auc(roc_xgb)[1]),
  
  CI_U = c(ci.auc(roc_log)[3],
           ci.auc(roc_rf)[3],
           ci.auc(roc_xgb)[3])
)

print(auc_table) 

delong <- data.frame(
  Comparison = c(
    "RF vs Logistic",
    "XGB vs Logistic",
    "RF vs XGB"
  ),
  
  P_value = c(
    roc.test(roc_rf, roc_log)$p.value,
    roc.test(roc_xgb, roc_log)$p.value,
    roc.test(roc_rf, roc_xgb)$p.value
  )
)

print(delong) 

final_table <- res_ci %>%
  left_join(auc_table, by="Model") %>%
  mutate(
    ROC_AUC_95CI = paste0(round(AUC,3)," (",
                          round(CI_L,3),"-",
                          round(CI_U,3),")"),
    
    Sens_95CI = paste0(round(Sensitivity,3)," (",
                       round(Sens_L,3),"-",
                       round(Sens_U,3),")"),
    
    Spec_95CI = paste0(round(Specificity,3)," (",
                       round(Spec_L,3),"-",
                       round(Spec_U,3),")"),
    
    Acc_95CI = paste0(round(Accuracy,3)," (",
                      round(Acc_L,3),"-",
                      round(Acc_U,3),")")
  ) %>%
  select(Model, TP, FP, TN, FN,
         ROC_AUC_95CI,
         Sens_95CI,
         Spec_95CI,
         Acc_95CI)

print(final_table) 


library(tidymodels)
rf_tune <- rand_forest(
  trees = 500,   # FIXED
  mtry = tune(),
  min_n = tune()
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

grid <- grid_regular(
  mtry(range=c(1,4)),
  min_n(range=c(2,20)),
  levels=3
)

cv <- vfold_cv(train_ml, v=5)

rf_tuned <- tune_grid(
  workflow() %>% add_model(rf_tune) %>% add_recipe(rec),
  resamples=cv,
  grid=grid
)

best_rf <- select_best(rf_tuned, metric = "roc_auc")
print(best_rf)


library(tidymodels)

xgb_tune <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  mtry = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_grid <- grid_latin_hypercube(
  trees(range = c(100, 300)),
  tree_depth(range = c(3, 8)),
  learn_rate(range = c(-4, -1)),
  loss_reduction(),
  sample_prop(range = c(0.6, 1.0)),
  finalize(mtry(), train_ml),
  size = 20
)

xgb_tuned <- tune_grid(
  workflow() %>% add_model(xgb_tune) %>% add_recipe(rec),
  resamples = cv,
  grid = xgb_grid,
  metrics = metric_set(roc_auc)
)

best_xgb <- select_best(xgb_tuned, metric = "roc_auc")
print(best_xgb)








print(best_xgb) 








write.csv(best_rf, "RF_hyperparameters.csv", row.names=FALSE)
write.csv(best_xgb, "XGB_hyperparameters.csv", row.names=FALSE) 

# ============================================================
# DeLong Test (p-values for model comparison)
# ============================================================

library(pROC)

# ROC objects (you already have these)
roc_log <- roc(test_ml$error, pred_log$.pred_1)
roc_rf  <- roc(test_ml$error, pred_rf$.pred_1)
roc_xgb <- roc(test_ml$error, pred_xgb$.pred_1)

# Pairwise comparisons
p_rf_vs_log <- roc.test(roc_rf, roc_log, method = "delong")$p.value
p_xgb_vs_log <- roc.test(roc_xgb, roc_log, method = "delong")$p.value
p_rf_vs_xgb <- roc.test(roc_rf, roc_xgb, method = "delong")$p.value

# Create table
delong_table <- data.frame(
  Comparison = c(
    "Random Forest vs Logistic",
    "XGBoost vs Logistic",
    "Random Forest vs XGBoost"
  ),
  P_value = c(
    p_rf_vs_log,
    p_xgb_vs_log,
    p_rf_vs_xgb
  )
)

print(delong_table) 

delong_table$P_value <- formatC(delong_table$P_value, format = "e", digits = 2)
print(delong_table)



glu <- read_xpt("GLU_L.XPT")
bio <- read_xpt("BIOPRO_L.XPT")

merged <- merge(
  glu[, c("SEQN", "LBXGLU")],
  bio[, c("SEQN", "LBXSNASI", "LBXSKSI", "LBXSCR")],
  by = "SEQN"
)

names(merged) <- c(
  "SEQN",
  "glucose",
  "sodium",
  "potassium",
  "creatinine"
)
merged <- na.omit(merged)

set.seed(123)

merged_400 <- merged[sample(nrow(merged), 400), ]

write.csv(merged_400, "NHANES_validation_400.csv", row.names = FALSE)

####################################################################
# Create SHIFT error dataset from normal NHANES dataset

shift_dataset <- merged_400

# Apply systematic analytical shift
shift_dataset$glucose <- shift_dataset$glucose * 1.10
shift_dataset$sodium <- shift_dataset$sodium + 3
shift_dataset$potassium <- shift_dataset$potassium + 0.5
shift_dataset$creatinine <- shift_dataset$creatinine + 0.2

# Add labels
shift_dataset$error_type <- "shift"
shift_dataset$error_status <- 1

# Save dataset
write.csv(shift_dataset,
          "NHANES_shift_dataset.csv",
          row.names = FALSE)
#####################################################################
# Create DRIFT error dataset from normal NHANES dataset

drift_dataset <- merged_400

# Create gradual drift factors across 400 samples
drift_factor <- seq(1.00, 1.08, length.out = nrow(drift_dataset))

# Apply progressive analytical drift
drift_dataset$glucose <- drift_dataset$glucose * drift_factor

drift_dataset$sodium <- drift_dataset$sodium +
  seq(0, 3, length.out = nrow(drift_dataset))

drift_dataset$potassium <- drift_dataset$potassium +
  seq(0, 0.5, length.out = nrow(drift_dataset))

drift_dataset$creatinine <- drift_dataset$creatinine +
  seq(0, 0.2, length.out = nrow(drift_dataset))

# Add labels
drift_dataset$error_type <- "drift"
drift_dataset$error_status <- 1

# Save dataset
write.csv(drift_dataset,
          "NHANES_drift_dataset.csv",
          row.names = FALSE) 
###################################################################### 

# Create HEMOLYSIS error dataset from normal NHANES dataset

set.seed(123)

hemolysis_dataset <- merged_400

# Simulate hemolysis effects

# Potassium falsely elevated
hemolysis_dataset$potassium <- hemolysis_dataset$potassium +
  runif(nrow(hemolysis_dataset), 0.5, 1.5)

# Glucose mildly decreased (processing delay effect)
hemolysis_dataset$glucose <- hemolysis_dataset$glucose *
  runif(nrow(hemolysis_dataset), 0.90, 0.98)

# Creatinine slight increase
hemolysis_dataset$creatinine <- hemolysis_dataset$creatinine +
  runif(nrow(hemolysis_dataset), 0, 0.1)

# Sodium minimally affected
hemolysis_dataset$sodium <- hemolysis_dataset$sodium +
  runif(nrow(hemolysis_dataset), -1, 1)

# Add labels
hemolysis_dataset$error_type <- "hemolysis"
hemolysis_dataset$error_status <- 1

# Save dataset
write.csv(hemolysis_dataset,
          "NHANES_hemolysis_dataset.csv",
          row.names = FALSE)
##########################################################################

# Create MIXED ERROR dataset from normal NHANES dataset

set.seed(123)

mixed_dataset <- merged_400

# Create progressive drift component
drift_factor <- seq(1.00, 1.05, length.out = nrow(mixed_dataset))

# Apply mixed analytical and pre-analytical errors

# Glucose:
# shift + drift + random noise
mixed_dataset$glucose <- (
  mixed_dataset$glucose * 1.05 * drift_factor
) + rnorm(nrow(mixed_dataset), mean = 0, sd = 5)

# Sodium:
# mild shift + drift + small noise
mixed_dataset$sodium <- (
  mixed_dataset$sodium + seq(0, 3, length.out = nrow(mixed_dataset))
) + rnorm(nrow(mixed_dataset), mean = 0, sd = 1)

# Potassium:
# hemolysis effect + drift + noise
mixed_dataset$potassium <- (
  mixed_dataset$potassium +
    runif(nrow(mixed_dataset), 0.5, 1.5) +
    seq(0, 0.5, length.out = nrow(mixed_dataset))
) + rnorm(nrow(mixed_dataset), mean = 0, sd = 0.2)

# Creatinine:
# mild shift + drift + noise
mixed_dataset$creatinine <- (
  mixed_dataset$creatinine +
    seq(0, 0.2, length.out = nrow(mixed_dataset))
) + rnorm(nrow(mixed_dataset), mean = 0, sd = 0.05)

# Prevent negative values
mixed_dataset$glucose[mixed_dataset$glucose < 0] <- 0
mixed_dataset$potassium[mixed_dataset$potassium < 0] <- 0
mixed_dataset$creatinine[mixed_dataset$creatinine < 0] <- 0

# Add labels
mixed_dataset$error_type <- "mixed"
mixed_dataset$error_status <- 1

# Save dataset
write.csv(mixed_dataset,
          "NHANES_mixed_error_dataset.csv",
          row.names = FALSE)
##############################################################################

# ============================================================
# LOAD EXTERNAL VALIDATION DATASETS
# ============================================================

library(dplyr)
library(pROC)

normal_data <- read.csv("NHANES_validation_400.csv")
shift_data <- read.csv("NHANES_shift_dataset.csv")
drift_data <- read.csv("NHANES_drift_dataset.csv")
hemolysis_data <- read.csv("NHANES_hemolysis_dataset.csv")
mixed_data <- read.csv("NHANES_mixed_error_dataset.csv")

# ============================================================
# ADD ERROR LABELS
# ============================================================

normal_data$error <- 0
shift_data$error <- 1
drift_data$error <- 1
hemolysis_data$error <- 1
mixed_data$error <- 1

# ============================================================
# COMBINE DATASETS
# ============================================================

external_data <- bind_rows(
  normal_data,
  shift_data,
  drift_data,
  hemolysis_data,
  mixed_data
)

# Keep only required variables
external_ml <- external_data[, c(
  "glucose",
  "sodium",
  "potassium",
  "creatinine",
  "error"
)]

external_ml$error <- factor(external_ml$error)

# ============================================================
# EXTERNAL PREDICTIONS
# ============================================================

pred_log_ext <- predict(
  fit_log,
  external_ml,
  type = "prob"
) %>%
  bind_cols(external_ml)

pred_rf_ext <- predict(
  fit_rf,
  external_ml,
  type = "prob"
) %>%
  bind_cols(external_ml)

pred_xgb_ext <- predict(
  fit_xgb,
  external_ml,
  type = "prob"
) %>%
  bind_cols(external_ml)

# ============================================================
# CLASSIFICATION USING THRESHOLDS
# ============================================================

# Logistic regression optimized threshold
log_thresh <- 0.1110129

pred_log_ext$class <- ifelse(
  pred_log_ext$.pred_1 > log_thresh,
  1, 0
)

# Random Forest default threshold
pred_rf_ext$class <- ifelse(
  pred_rf_ext$.pred_1 > 0.5,
  1, 0
)

# XGBoost optimized threshold
pred_xgb_ext$class <- ifelse(
  pred_xgb_ext$.pred_1 > thresh,
  1, 0
)

# Convert predictions to factors
pred_log_ext$class <- factor(pred_log_ext$class)
pred_rf_ext$class <- factor(pred_rf_ext$class)
pred_xgb_ext$class <- factor(pred_xgb_ext$class)

# ============================================================
# ROC CURVES
# ===========================================================

roc_log_ext <- roc(
  external_ml$error,
  pred_log_ext$.pred_1
)

roc_rf_ext <- roc(
  external_ml$error,
  pred_rf_ext$.pred_1
)

roc_xgb_ext <- roc(
  external_ml$error,
  pred_xgb_ext$.pred_1
)

# ============================================================
# METRICS FUNCTION
# ============================================================

metrics_ext <- function(df, name){
  
  tp <- sum(df$class==1 & df$error==1)
  tn <- sum(df$class==0 & df$error==0)
  fp <- sum(df$class==1 & df$error==0)
  fn <- sum(df$class==0 & df$error==1)
  
  data.frame(
    Model=name,
    
    TP=tp,
    FP=fp,
    TN=tn,
    FN=fn,
    
    Sensitivity=tp/(tp+fn),
    Specificity=tn/(tn+fp),
    Accuracy=(tp+tn)/(tp+tn+fp+fn),
    
    ROC_AUC=as.numeric(
      auc(
        roc(df$error, df$.pred_1)
      )
    )
  )
}

# ============================================================
# FINAL EXTERNAL VALIDATION RESULTS
# ============================================================

external_results <- bind_rows(
  
  metrics_ext(pred_log_ext,"Logistic Regression"),
  
  metrics_ext(pred_rf_ext,"Random Forest"),
  
  metrics_ext(pred_xgb_ext,"XGBoost")
  
)

print(external_results)

# ============================================================
# SAVE RESULTS
# ============================================================

write.csv(
  external_results,
  "External_Validation_Results.csv",
  row.names=FALSE
)

# ============================================================
# PLOT ROC CURVES
# ============================================================

plot(
  roc_rf_ext,
  col="blue",
  main="External Validation ROC Curves"
)

plot(
  roc_log_ext,
  col="red",
  add=TRUE
)

plot(
  roc_xgb_ext,
  col="green",
  add=TRUE
)

legend(
  "bottomright",
  legend=c("RF","Logistic","XGBoost"),
  col=c("blue","red","green"),
  lwd=2
)

ggsave(
  "External_Validation_ROC.png",
  width=6,
  height=4,
  dpi=300
)

########################################################################
# ============================================================
# EXTERNAL VALIDATION
# CONFIDENCE INTERVALS + P VALUES
# ============================================================

library(pROC)
library(dplyr)

# ============================================================
# BINOMIAL CONFIDENCE INTERVAL FUNCTION
# ============================================================

binom_ci <- function(x, n){
  
  ci <- binom.test(x, n)$conf.int
  
  c(
    lower = ci[1],
    upper = ci[2]
  )
}

# ============================================================
# METRICS WITH CONFIDENCE INTERVALS
# ============================================================

metrics_with_ci_ext <- function(df, name){
  
  tp <- sum(df$class==1 & df$error==1)
  tn <- sum(df$class==0 & df$error==0)
  fp <- sum(df$class==1 & df$error==0)
  fn <- sum(df$class==0 & df$error==1)
  
  sens <- tp/(tp+fn)
  spec <- tn/(tn+fp)
  acc  <- (tp+tn)/(tp+tn+fp+fn)
  
  sens_ci <- binom_ci(tp, tp+fn)
  spec_ci <- binom_ci(tn, tn+fp)
  acc_ci  <- binom_ci(tp+tn, tp+tn+fp+fn)
  
  roc_obj <- roc(df$error, df$.pred_1)
  
  auc_val <- auc(roc_obj)
  
  auc_ci <- ci.auc(roc_obj)
  
  data.frame(
    
    Model = name,
    
    TP = tp,
    FP = fp,
    TN = tn,
    FN = fn,
    
    Sensitivity = sens,
    Sens_L = sens_ci[1],
    Sens_U = sens_ci[2],
    
    Specificity = spec,
    Spec_L = spec_ci[1],
    Spec_U = spec_ci[2],
    
    Accuracy = acc,
    Acc_L = acc_ci[1],
    Acc_U = acc_ci[2],
    
    AUC = as.numeric(auc_val),
    AUC_L = auc_ci[1],
    AUC_U = auc_ci[3]
  )
}

# ============================================================
# GENERATE RESULTS TABLE
# ============================================================

external_ci_results <- bind_rows(
  
  metrics_with_ci_ext(
    pred_log_ext,
    "Logistic Regression"
  ),
  
  metrics_with_ci_ext(
    pred_rf_ext,
    "Random Forest"
  ),
  
  metrics_with_ci_ext(
    pred_xgb_ext,
    "XGBoost"
  )
)

print(external_ci_results)

# ============================================================
# DELONG TEST P VALUES
# ============================================================

roc_log_ext <- roc(
  pred_log_ext$error,
  pred_log_ext$.pred_1
)

roc_rf_ext <- roc(
  pred_rf_ext$error,
  pred_rf_ext$.pred_1
)

roc_xgb_ext <- roc(
  pred_xgb_ext$error,
  pred_xgb_ext$.pred_1
)

delong_ext <- data.frame(
  
  Comparison = c(
    "RF vs Logistic",
    "XGBoost vs Logistic",
    "RF vs XGBoost"
  ),
  
  P_value = c(
    
    roc.test(
      roc_rf_ext,
      roc_log_ext,
      method="delong"
    )$p.value,
    
    roc.test(
      roc_xgb_ext,
      roc_log_ext,
      method="delong"
    )$p.value,
    
    roc.test(
      roc_rf_ext,
      roc_xgb_ext,
      method="delong"
    )$p.value
  )
)

print(delong_ext)

# ============================================================
# FORMAT FINAL TABLE
# ============================================================

final_external_table <- external_ci_results %>%
  
  mutate(
    
    ROC_AUC_95CI = paste0(
      round(AUC,3),
      " (",
      round(AUC_L,3),
      "-",
      round(AUC_U,3),
      ")"
    ),
    
    Sens_95CI = paste0(
      round(Sensitivity,3),
      " (",
      round(Sens_L,3),
      "-",
      round(Sens_U,3),
      ")"
    ),
    
    Spec_95CI = paste0(
      round(Specificity,3),
      " (",
      round(Spec_L,3),
      "-",
      round(Spec_U,3),
      ")"
    ),
    
    Acc_95CI = paste0(
      round(Accuracy,3),
      " (",
      round(Acc_L,3),
      "-",
      round(Acc_U,3),
      ")"
    )
  ) %>%
  
  select(
    Model,
    TP,
    FP,
    TN,
    FN,
    ROC_AUC_95CI,
    Sens_95CI,
    Spec_95CI,
    Acc_95CI
  )

print(final_external_table)

# ============================================================
# SAVE TABLES
# ============================================================

write.csv(
  final_external_table,
  "External_Validation_CI_Table.csv",
  row.names=FALSE
)

write.csv(
  delong_ext,
  "External_Validation_DeLong_Pvalues.csv",
  row.names=FALSE
)
#################################################################
library(pROC)

# Moving average function
ma <- function(x, k = 20) {
  as.numeric(stats::filter(x, rep(1/k, k), sides = 1))
}

features <- c("glucose", "sodium", "potassium", "creatinine")

# Create moving averages and limits from training data
for (var in features) {
  
  train[[paste0(var, "_ma")]] <- ma(train[[var]], k = 20)
  
  ma_mean <- mean(train[[paste0(var, "_ma")]], na.rm = TRUE)
  ma_sd   <- sd(train[[paste0(var, "_ma")]], na.rm = TRUE)
  
  test[[paste0(var, "_ma")]] <- ma(test[[var]], k = 20)
  
  test[[paste0(var, "_flag")]] <- ifelse(
    test[[paste0(var, "_ma")]] < ma_mean - 2 * ma_sd |
      test[[paste0(var, "_ma")]] > ma_mean + 2 * ma_sd,
    1, 0
  )
}

# Combined multianalyte PBRTQC flag
test$pbrtqc_multi_flag <- ifelse(
  test$glucose_flag == 1 |
    test$sodium_flag == 1 |
    test$potassium_flag == 1 |
    test$creatinine_flag == 1,
  1, 0
)

# Remove NA rows created by moving average
pbrtqc_multi_df <- test[complete.cases(test[, paste0(features, "_flag")]), ]

# Convert to factors
pbrtqc_multi_df$pbrtqc_multi_flag <- factor(
  pbrtqc_multi_df$pbrtqc_multi_flag,
  levels = c(0, 1)
)

pbrtqc_multi_df$error <- factor(
  pbrtqc_multi_df$error,
  levels = c(0, 1)
)

# Confusion matrix
tp <- sum(pbrtqc_multi_df$pbrtqc_multi_flag == 1 & pbrtqc_multi_df$error == 1)
tn <- sum(pbrtqc_multi_df$pbrtqc_multi_flag == 0 & pbrtqc_multi_df$error == 0)
fp <- sum(pbrtqc_multi_df$pbrtqc_multi_flag == 1 & pbrtqc_multi_df$error == 0)
fn <- sum(pbrtqc_multi_df$pbrtqc_multi_flag == 0 & pbrtqc_multi_df$error == 1)

# Metrics
sens <- tp / (tp + fn)
spec <- tn / (tn + fp)
acc  <- (tp + tn) / (tp + tn + fp + fn)

# Exact binomial CI
binom_ci <- function(x, n) {
  ci <- binom.test(x, n)$conf.int
  c(lower = ci[1], upper = ci[2])
}

sens_ci <- binom_ci(tp, tp + fn)
spec_ci <- binom_ci(tn, tn + fp)
acc_ci  <- binom_ci(tp + tn, tp + tn + fp + fn)

# ROC-AUC using binary PBRTQC flag
roc_pbrtqc_multi <- roc(
  pbrtqc_multi_df$error,
  as.numeric(as.character(pbrtqc_multi_df$pbrtqc_multi_flag))
)

auc_val <- auc(roc_pbrtqc_multi)
auc_ci  <- ci.auc(roc_pbrtqc_multi)

# Final table
pbrtqc_multi_table <- data.frame(
  Method = "Multianalyte PBRTQC (Moving Average)",
  TP = tp,
  FP = fp,
  TN = tn,
  FN = fn,
  
  ROC_AUC_95CI = paste0(
    round(auc_val, 3), " (",
    round(auc_ci[1], 3), "-",
    round(auc_ci[3], 3), ")"
  ),
  
  Sensitivity_95CI = paste0(
    round(sens, 3), " (",
    round(sens_ci[1], 3), "-",
    round(sens_ci[2], 3), ")"
  ),
  
  Specificity_95CI = paste0(
    round(spec, 3), " (",
    round(spec_ci[1], 3), "-",
    round(spec_ci[2], 3), ")"
  ),
  
  Accuracy_95CI = paste0(
    round(acc, 3), " (",
    round(acc_ci[1], 3), "-",
    round(acc_ci[2], 3), ")"
  )
)

print(pbrtqc_multi_table)


