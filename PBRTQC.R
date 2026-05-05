 -----------------------------
# 0. Setup
# -----------------------------

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

delong_table$P_value <- formatC(delong_table$P_value, format = "e", digits = 2)
print(delong_table)

