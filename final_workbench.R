# Libraries
################################################################################
library(ggplot2)
library(tree)
library(caret)
library(dplyr)
library(e1071)
library(readxl)
################################################################################



# Data Processing
################################################################################
# Load data
df <- read.csv2("data3.csv", sep=",")
df[,1]<-factor(df[,1])

heart_disease_data <- read.csv2("data3.csv", sep=",")
heart_disease_data[,1]<-factor(heart_disease_data[,1])

# Summary
summary(df)



# Factorize BMI data
temp <- df$BMI
for(i in 1:length(temp)){
  if(temp[i]>31) {
    temp[i] <-4
  } else if(temp[i]>27 && temp[i]<32) {
    temp[i] <- 3
  } else if(temp[i]>24 && temp[i]<28) {
    temp[i] <- 2
  } else {
    temp[i] <- 1
  }
}
df$BMI <- temp
summary(df$BMI)



# Removing NA's
df <- na.omit(df)



# Converting df2 columns into double
# for (i in 1:max(col(df))) {
#   df[,i] <- as.double(df[,i])
# }



# Data head
head(df)



# Null values check
colSums(is.na(df))



# Types of all the columns in dataframe
sapply(df, typeof)



# Number of patients with heart disease or attack
table(df['HeartDiseaseorAttack'])
################################################################################



# Partition data into training(40), validation(30) and test(30)
################################################################################
n=dim(df)[1]
set.seed(12345)



## Training Data
id=sample(1:n, floor(n*0.4))
train=df[id,]



id1=setdiff(1:n, id)
set.seed(12345)



## Validation data
id2=sample(id1, floor(n*0.3))
valid=df[id2,]



## Testing data
id3=setdiff(id1,id2)
test=df[id3,]
################################################################################



# Models
################################################################################
# Decision Tree with default settings
dt_default <- tree(formula = HeartDiseaseorAttack~.,
data = train,
method="class")



# SVM with default settings
svm_default = svm(formula = HeartDiseaseorAttack~.,
                  data = train,
                  type = 'C-classification',
                  kernel = 'linear')
################################################################################



# Predictions
################################################################################
# decision tree
pred_dt<- predict(dt_default, test, type = "class")



# SVM
pred_svm = predict(svm_default, newdata = test)
################################################################################



# Confusion Matrix
################################################################################
# decision tree
cm_dt<-table(pred_dt, test$HeartDiseaseorAttack)
cm_dt



# SVM
cm_svm = table(pred_svm, test[,1])
cm_svm
################################################################################



# Misclassification error
################################################################################
# decision tree
mmce_dt <- 1 - sum(diag(cm_dt)) / sum(cm_dt)
mmce_dt



# SVM
mmce_svm <- 1 - sum(diag(cm_svm)) / sum(cm_svm)
mmce_svm
################################################################################



# F1 Score
################################################################################
# SVM
TN_svm <- cm_svm[1,1]
TP_svm <- cm_svm[2,2]
FN_svm <- cm_svm[1,2]
FP_svm <- cm_svm[2,1]
precision_svm <- (TP_svm) / (TP_svm + FP_svm) # 0.6753752
recall_score_svm <- (FP_svm) / (FP_svm + TN_svm) # 0.05079365
f1_score_svm <- 2 * ((precision_svm * recall_score_svm) / (precision_svm + recall_score_svm))
f1_score_svm # 0.09448153
################################################################################



# Accuracy
################################################################################
# SVM
accuracy_svm <- (TP_svm + TN_svm) / (TP_svm + TN_svm + FP_svm + FN_svm)
accuracy_svm # 0.8034247
################################################################################



# Plot
################################################################################
roc_svm <- roc(response = test$HeartDiseaseorAttack, predictor =as.numeric(pred_svm))
plot(roc_svm, col = "green")
legend(0.3, 0.2, legend = c("SVM"), lty = c(1), col = c("green"))
################################################################################



set.seed(12345)
trainScore = rep(0, 100)
testScore = rep(0, 100)

for(i in 2:100) {
  prunedTree = prune.tree(dt_default, best=i)
  pd = predict(prunedTree, newdata = valid, type= "tree")
  trainScore[i] = deviance(prunedTree)
  testScore[i] = deviance(pd)
}

optimalLeaves <- which.min(testScore[2:100])

optimalTree = prune.tree(dt_default, best=optimalLeaves)
predOptimalTree = predict(optimalTree, newdata= valid, type="class")

cnfMatrixOptimalTree <- table(valid$HeartDiseaseorAttack, predOptimalTree)


pi_generator <- seq(0.05, 0.95, 0.05)
logiReg <- glm(formula = HeartDiseaseorAttack~., data = train, family = "binomial")
logiRegPred <- predict(logiReg, select(test, -c(HeartDiseaseorAttack)), type = "response")

confList <- list()
for(i in pi_generator) {
  a <- as.factor(ifelse(logiRegPred>i, 'yes', 'no'))
  b <- table(a, test$HeartDiseaseorAttack)
  confList <- c(confList, list(b))
}

tpr_logR <- c()
fpr_logR <- c()
total_loop <- length(pi_generator)-1
for (iter in 1:18) {
  tpr_value <- confList[[iter]][4]/(confList[[iter]][3]+confList[[iter]][4])
  tpr_logR <- c(tpr_logR, tpr_value)
  
  fpr_value <- confList[[iter]][2]/(confList[[iter]][1]+confList[[iter]][2])
  fpr_logR <- c(fpr_logR, fpr_value)
}


# Classify test data with Optimal Tree
optimalTreePred = predict(optimalTree, newdata= test, type="vector")
confListOptTree <- list()
for(i in pi_generator) {
  k <- as.factor(ifelse(optimalTreePred[,2]>i, 'yes', 'no'))
  l <- table(k, test$HeartDiseaseorAttack)
  confListOptTree <- c(confListOptTree, list(l))
}

tprOptTree <- c()
fprOptTree <- c()

for (iter1 in 1:19) {
  TP <- confListOptTree[[iter1]][4]
  P <- (confListOptTree[[iter1]][3]+confListOptTree[[iter1]][4])
  tpr_value <- TP/P
  tprOptTree <- c(tprOptTree, tpr_value)
  
  FP <- confListOptTree[[iter1]][2]
  N <- (confListOptTree[[iter1]][1]+confListOptTree[[iter1]][2])
  fpr_value <- FP/N
  fprOptTree <- c(fprOptTree, fpr_value)
}
tprOptTree[16:19] <- 0.0
fprOptTree[16:19] <- 0.0


plot(fpr_logR, tpr_logR, type="l", col="blue",
     xlab="FPR", ylab="TPR", xlim=c(0,0.8), ylim=c(0,1))
lines(fprOptTree, tprOptTree, col="red", type="l")
title("ROC Curve")
legend("bottomright", legend=c("Logistic Reg","Optimal Tree"),
       col=c("blue","red"), lty = 1)