library(caret)
library(pROC)
library(MuMIn)
library(dplyr)

####################################################################
#LEARNING PROCESS
####################################################################
setwd("C:/Users/jihem/Google Drive/These/German credit scoring/interface")

# Read dataset german credit scoring
german_credit <- read.csv("german_credit.csv")

#Machine learning
set.seed(12420424)
in.train <- createDataPartition(as.factor(german_credit$default), p=0.8, list=FALSE)
german_credit.train <- german_credit[in.train,]
german_credit.test <- german_credit[-in.train,]
fit <- glm(default ~ ., family = binomial, german_credit.train,na.action = "na.fail")


proba <- predict(fit, newdata = german_credit.test, type = "response")
actual <- german_credit.test$default

##############################################################################
##PERFORMANCE METRICS
##############################################################################
roc_obj <- roc(actual, proba, direction="<")
threshold <- coords(roc_obj, "best", "threshold", transpose=TRUE)[1][[1]]
predicted_values <-ifelse(proba>threshold,1,0)
conf_matrix <- table(predicted_values,actual)
accuracy <- (conf_matrix[1] + conf_matrix[4]) /(conf_matrix[1] + conf_matrix[3] + conf_matrix[3] + conf_matrix[4])
AUC <- auc(actual, proba)

##############################################################################
#FAIRNESS METRICS
##############################################################################
conf_matrix_female <- table(factor(predicted_values[german_credit.test$sex=="female"], levels = 0:1), factor(actual[german_credit.test$sex=="female"], levels = 0:1))
true_positive_female <- conf_matrix_female[4] /(conf_matrix_female[2] + conf_matrix_female[4])
true_negative_female <- conf_matrix_female[1] /(conf_matrix_female[1] + conf_matrix_female[3])
refusal_rate_female <- (conf_matrix_female[2] + conf_matrix_female[4])/(conf_matrix_female[1] + conf_matrix_female[2] + conf_matrix_female[3] + conf_matrix_female[4])

conf_matrix_male <- table(factor(predicted_values[german_credit.test$sex=="male"], levels = 0:1), factor(actual[german_credit.test$sex=="male"], levels = 0:1))
true_positive_male <- conf_matrix_male[4] /(conf_matrix_male[2] + conf_matrix_male[4])
true_negative_male <- conf_matrix_male[1] /(conf_matrix_male[1] + conf_matrix_male[3])
refusal_rate_male <- (conf_matrix_male[2] + conf_matrix_male[4])/(conf_matrix_male[1] + conf_matrix_male[2] + conf_matrix_male[3] + conf_matrix_male[4])

fairness_reject <- true_positive_female - true_positive_male
fairness_acceptation <- true_negative_female - true_negative_male
demogrphic_parity_sex <- refusal_rate_female - refusal_rate_male