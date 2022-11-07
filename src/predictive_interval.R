# Predicting European Cities' Mitigation Performance with Machine Learning
# Angel Hsu, Xuewei Wang, Jonas Tan, Wayne Toh, and Nihit Goyal
# Grid Search Function
# November 2022

library(glue)
library(caret)
library(xgboost)


# this function will create the predictive intervals for the X_pred data. By setting the subsample and colsample_bytree = 0.8, the model will be added randomness in each loop. 

# X is all available actors' independent variables.  Y is the all available actors' self reported emissions. These two are used to build the models. X_pred is the data matrix we will make predictions on
predictive_interval = function(iterations,X,Y,max_depth,eta,gamma,min_child_weight,round,X_pred){
  start = Sys.time()
  print("prediction interval starts")
  
  setClass(Class="predictive_interval",
           representation(
             quantiles="data.frame",
             result_mtx="data.frame"
           )
  )
  
  # create a matrix to save results
  predictions <- data.frame(matrix(0,nrow(X_pred),iterations))
  set.seed(123)
  seeds <- runif(iterations,1,100000)
  for (i in 1:ncol(predictions)){
    
    set.seed(seeds[i])
    
    model_temp <-  xgboost(data = as.matrix(X), label = Y,
                           max.depth = max_depth,
                           eta = eta,
                           gamma = gamma,
                           min_child_weight = min_child_weight,
                           objective = "reg:squarederror",
                           nrounds = round,
                           
                           verbose = 0,
                           subsample = .9
    )
    
    
    predictions[,i] <- predict(model_temp,newdata = as.matrix(X_pred))%>% as.data.frame()
    if (i%%200 == 0){
      print(paste("finished ", i," rows"))
    } 
    
  }
  result_qts = t(apply(predictions, 1, quantile, probs = c(0.05, 0.95),  na.rm = TRUE)) %>% as.data.frame()
  colnames(result_qts) = c("qt_5","qt_95")
  result_qts$pred_median = apply(predictions, 1, median, na.rm = T)
  result_qts$pred_mean = apply(predictions, 1, mean, na.rm = T)
  
  predictions = as.data.frame(predictions)
  ends = Sys.time()
  print('prediction interval function ends')
  print(ends-start)
  return(new("predictive_interval",
             quantiles=result_qts,
             result_mtx=predictions))
}