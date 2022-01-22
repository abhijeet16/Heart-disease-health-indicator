library(tree)
library(caret)
library(dplyr)


heart_disease_data <- read.csv2("data3.csv", sep=",")
heart_disease_data[,1]<-factor(heart_disease_data[,1])

temp <- heart_disease_data$BMI
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
heart_disease_data$BMI <- temp

  

# scale_BMI <- scale(heart_disease_data[,5])
# heart_disease_data <- as.factor(heart_disease_data[,1])


## Partition data into training(40), validation(30) and test(30)
n=dim(heart_disease_data)[1]
set.seed(12345)

## Training Data
id=sample(1:n, floor(n*0.4))
train=heart_disease_data[id,]

id1=setdiff(1:n, id)
set.seed(12345)

## Validation data
id2=sample(id1, floor(n*0.3))
valid=heart_disease_data[id2,]

## Testing data
id3=setdiff(id1,id2)
test=heart_disease_data[id3,]



## Predict function
predict_dt <- function(obj, dat) {
  pd<- predict(object = obj, newdata=dat, type = "class")
  return(pd)
}

## Prepare confusion matrix
prepare_table <- function (pd, original) {
  tb<-table(pd, original)
  return(tb)
}

# Calculate Misclassification rate
calculate_MCR <- function(table) {
  missclass_rate <- 1-sum(diag(table))/sum(table)
  return(missclass_rate)
}

# a. Decision Tree with default settings
dt_default <- tree(formula = HeartDiseaseorAttack~.,
                   data = train,
                   method="class")

# Predict
pd_default<- predict(object = dt_default, newdata=test, type = "class")
# pd_default <- predict_dt(dt_default, train)


# Get confusion matrix
cnfMatrix_default <- prepare_table(pd_default, test$HeartDiseaseorAttack)


# Get Misclassification error
MCR_default <- calculate_MCR(cnfMatrix_default)


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
