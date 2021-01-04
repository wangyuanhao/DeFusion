library(survival)
library(rms)
library(Hmisc)
library(pec)
library(survcomp)

computeCindex <- function(training_data, testing_data){
  
  # Input:
  #   training_data/testing_data: data matrix, sample_num x (status, follow_up, cluster)
  
  # transform matrix to data frame
  training_set <- as.data.frame(training_data)
  
  feat_num <- ncol(training_set) - 2
  colnames(training_set) <- c(c("status", "follow_up"), 
                              sapply(1:feat_num, function(x)paste0("V",x)))
  
  # training_set$cluster <- as.factor(training_set$cluster)
  
  testing_set <- as.data.frame(testing_data)
  colnames(testing_set) <- c(c("status", "follow_up"), 
                             sapply(1:feat_num, function(x)paste0("V",x)))
  
  # testing_set$cluster <- as.factor(testing_set$cluster)
  
  training_model <- coxph(Surv(follow_up, status)~., data=training_set)
  testing_model <- predict(training_model, newdata=testing_set)
  
  # browser()
  testing_Cindex_info <- concordance.index(testing_model, surv.time=testing_set$follow_up,
                                           surv.event=testing_set$status,
                                           method="noether")
  training_Cindex_info <- summary(training_model)
  
  training_Cindex <- unname(training_Cindex_info$concordance[1])
  training_Cindex_se <- unname(training_Cindex_info$concordance[2])
  
  testing_Cindex <- testing_Cindex_info$c.index
  testing_Cindex_se <- testing_Cindex_info$se
  
  return(c(training_Cindex, training_Cindex_se,
           testing_Cindex, testing_Cindex_se))
}