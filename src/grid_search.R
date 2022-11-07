# Predicting European Cities' Mitigation Performance with Machine Learning
# Angel Hsu, Xuewei Wang, Jonas Tan, Wayne Toh, and Nihit Goyal
# Grid Search Function
# November 2022

library(glue)
library(caret)
library(xgboost)


tune_parameters_tree_df_cv <- function(input_data, input_label, max_depth_range, eta_range, min_child_weight_range, gamma_range, score_parameter, objective_range )  {
  
  df <- data.frame(matrix(ncol = 8, nrow = 0))
  names_col <- c("min_train_RMSE","min_test_RMSE", "max_depth", "min_child_weight", "eta", "gamma", "objective", "nrounds")
  colnames(df) <- names_col
  
  for(a in 1:length(max_depth_range)){
    for(b in 1:length(min_child_weight_range)){
      for(c in 1:length(eta_range)){
        for(d in 1:length(gamma_range)){
          for (e in 1:length(objective_range)){
            
            test <-  xgb.cv(data = input_data, label = input_label, max.depth = max_depth_range[a], eta = eta_range[c], nrounds = 999, early_stopping_rounds = 5, min_child_weight = min_child_weight_range[b], gamma = gamma_range[d], objective = objective_range[e], verbose = 0, nfold = 5, metrics = "rmse")
            
            pred_score <- min((test$evaluation_log)$train_rmse_mean)
            
            rounds <- which((test$evaluation_log)$train_rmse_mean == min((test$evaluation_log)$train_rmse_mean))
            pred_score_test <- test$evaluation_log$test_rmse_mean[rounds]
            
            new_row <- data.frame(pred_score, pred_score_test, max_depth_range[a], min_child_weight_range[b], eta_range[c], gamma_range[d], objective_range[e], rounds)
            colnames(new_row) <- names_col
            
            df <- rbind(df, new_row)
            
          }
        }
      }
    }
  }
  
  return(df)
  
}
