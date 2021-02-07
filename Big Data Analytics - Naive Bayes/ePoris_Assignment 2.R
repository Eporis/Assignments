# The file Bank.csv contains data on 5000 customers. For this exercise, focus only 
# on two two predcitors: Online (whetehr or not a customer is an active user of 
# online banking services) and Credit Card (does the customer have a credit card 
# issued by the bank). Partition the data into training (60%) and validation (40%). 
# The goal is to predict customer response to a the personal loan campaign.
# 
# Use Naïve Bayes and compute the confusion matrix and ROC curve and comment. 
# The beginning of your report should contain a managerial conclusion.

## Initialize relevant packages and load data
install.packages("Rtools")
install.packages("yardstick")
library(yardstick)
library(e1071)
library(caret)
library(ROCR)
library(ggplot2)
library(dplyr)
library(tidyverse)
setwd('C:/Users/Ethan/Documents/Ivey/T2/Big Data Analytics')
bank <- read.csv('./Assignment 2/Bank-1.csv', header=TRUE)
head(bank)


# Select Personal Loans, Online and Credit card as variables and split into training and test sets
selected.var <- c(10, 13, 14)
train.index <- sample(c(1:dim(bank)[1]), dim(bank)[1]*0.6)
train.df <- bank[train.index, selected.var]
valid.df <- bank[-train.index, selected.var]

# Train NB model to predict Personal Loans using Online and Credit Cards
bank.nb <- naiveBayes(as.factor(Personal.Loan) ~ ., data = train.df)
bank.nb

# Use NB to predict P(accept loan) for people in test set; assign to 'pred.prob'
pred.prob <- predict(bank.nb, newdata = valid.df, type = "raw")
pred.prob

# Use NB to predict class of people in test set (1 = accept loan); assign to 'pred.class'
pred.class <- predict(bank.nb, newdata = valid.df)
pred.class

# assign 'df' four columns: actual class, pred class, prob(decline), prob(accept)
df <- data.frame(actual = valid.df$Personal.Loan, predicted = pred.class, pred.prob)
head(df)

#Classification uses .5 criterion, classification 2 is for playing with criterion
# Classification vector of 1s if pred.prob's P(accept) > criterion, 0 if < criterion
actual <- valid.df$Personal.Loan
classification <- ifelse(pred.prob[,2]>.5,1,0)
classification2 <- ifelse(pred.prob[,2]>mean(pred.prob[,2]),1,0)

summary(pred.prob[,2])

# Make confusion matrix
cm1 <- confusionMatrix(as.factor(classification), as.factor(actual))
cm2 <- confusionMatrix(as.factor(classification2), as.factor(actual))


# Draw confusion matrices
cm1$table %>%
  data.frame() %>% 
  group_by(Reference) %>% 
  mutate(total = sum(Freq)) %>% 
  ungroup() %>% 
  ggplot(aes(Reference, reorder(Prediction, desc(Prediction)), fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), size = 8) +
  scale_fill_gradient(low = "white", high = "#badb33") +
  scale_x_discrete(position = "top") +
  ggtitle("Confusion Matrix with Base Criterion") +
  theme(plot.title = element_text(hjust = 0.5)) +
  ylab("Prediction") + 
  geom_tile(color = "black", fill = "black", alpha = 0)

cm2$table %>%
  data.frame() %>% 
  group_by(Reference) %>% 
  mutate(total = sum(Freq)) %>% 
  ungroup() %>% 
  ggplot(aes(Reference, reorder(Prediction, desc(Prediction)), fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), size = 8) +
  scale_fill_gradient(low = "white", high = "#badb33") +
  scale_x_discrete(position = "top") +
  ggtitle("Model 1 - Confusion Matrix with Mean Criterion") +
  theme(plot.title = element_text(hjust = 0.5)) +
  ylab("Prediction") + 
  geom_tile(color = "black", fill = "black", alpha = 0)




# make predObj, rocObj, and aucObj to make ROC curve       
predObj <- prediction(pred.prob[,2], valid.df$Personal.Loan)
rocObj = performance(predObj, measure="tpr", x.measure="fpr")
aucObj = performance(predObj, measure="auc")

#plot ROC curve
plot(rocObj, main = paste("Model 1 - Area under the curve:", round(aucObj@y.values[[1]], 4)))
predObj


### 
# NB model 2 - include more variables to see if can make better model for predicting Personal.Loan
# Includes Age, Experience, Income, Family, CCAvg, Online and CreditCard

# Select Personal Loans, Online and Credit card as variables and split into training and test sets
selected.var <- c(10, 2, 3, 4, 6, 7, 13, 14)
train.df <- bank[train.index, selected.var]
valid.df <- bank[-train.index, selected.var]

# Train NB model to predict Personal Loans using Online and Credit Cards
bank.nb <- naiveBayes(as.factor(Personal.Loan) ~ ., data = train.df)

# Use NB to predict P(accept loan) for people in test set; assign to 'pred.prob'
pred.prob <- predict(bank.nb, newdata = valid.df, type = "raw")
pred.prob

# Use NB to predict class of people in test set (1 = accept loan); assign to 'pred.class'
pred.class <- predict(bank.nb, newdata = valid.df)
pred.class

# assign 'df' four columns: actual class, pred class, prob(decline), prob(accept)
df <- data.frame(actual = valid.df$Personal.Loan, predicted = pred.class, pred.prob)
head(df)

summary()
#Classification uses .5 criterion, classification 2 is for playing with criterion
# Classification vector of 1s if pred.prob's P(accept) > criterion, 0 if < criterion
actual <- valid.df$Personal.Loan
classification <- ifelse(pred.prob[,2]>.5,1,0)

# Make confusion matrix
cm1 <- confusionMatrix(as.factor(classification), as.factor(actual))

# Draw confusion matrices
cm1$table %>%
  data.frame() %>% 
  group_by(Reference) %>% 
  mutate(total = sum(Freq)) %>% 
  ungroup() %>% 
  ggplot(aes(Reference, reorder(Prediction, desc(Prediction)), fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), size = 8) +
  scale_fill_gradient(low = "white", high = "#badb33") +
  scale_x_discrete(position = "top") +
  ggtitle("Model 2 - Confusion Matrix") +
  theme(plot.title = element_text(hjust = 0.5)) +
  ylab("Prediction") + 
  geom_tile(color = "black", fill = "black", alpha = 0)


# make predObj, rocObj, and aucObj to make ROC curve       
predObj <- prediction(pred.prob[,2], valid.df$Personal.Loan)
rocObj = performance(predObj, measure="tpr", x.measure="fpr")
aucObj = performance(predObj, measure="auc")

#plot ROC curve
plot(rocObj, main = paste("Model 2 - Area under the curve:", round(aucObj@y.values[[1]], 4)))
predObj

#######

# NB model 3 - Use Age and Income to predict Personal.Loan

# Select Personal Loans, Online and Credit card as variables and split into training and test sets
selected.var <- c(10, 2, 4)
train.df <- bank[train.index, selected.var]
valid.df <- bank[-train.index, selected.var]

# Train NB model to predict Personal Loans using Online and Credit Cards
bank.nb <- naiveBayes(as.factor(Personal.Loan) ~ ., data = train.df)

# Use NB to predict P(accept loan) for people in test set; assign to 'pred.prob'
pred.prob <- predict(bank.nb, newdata = valid.df, type = "raw")
pred.prob

# Use NB to predict class of people in test set (1 = accept loan); assign to 'pred.class'
pred.class <- predict(bank.nb, newdata = valid.df)
pred.class

# assign 'df' four columns: actual class, pred class, prob(decline), prob(accept)
df <- data.frame(actual = valid.df$Personal.Loan, predicted = pred.class, pred.prob)
head(df)

summary()
#Classification uses .5 criterion, classification 2 is for playing with criterion
# Classification vector of 1s if pred.prob's P(accept) > criterion, 0 if < criterion
actual <- valid.df$Personal.Loan
#classification <- ifelse(pred.prob[,2]>.3,1,0)
classification <- ifelse(pred.prob[,2]>.5,1,0)

# Make confusion matrix
cm1 <- confusionMatrix(as.factor(classification), as.factor(actual))

# Draw confusion matrices
cm1$table %>%
  data.frame() %>% 
  group_by(Reference) %>% 
  mutate(total = sum(Freq)) %>% 
  ungroup() %>% 
  ggplot(aes(Reference, reorder(Prediction, desc(Prediction)), fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), size = 8) +
  scale_fill_gradient(low = "white", high = "#badb33") +
  scale_x_discrete(position = "top") +
  ggtitle("Model 3 - Confusion Matrix") +
  theme(plot.title = element_text(hjust = 0.5)) +
  ylab("Prediction") + 
  geom_tile(color = "black", fill = "black", alpha = 0)


# make predObj, rocObj, and aucObj to make ROC curve       
predObj <- prediction(pred.prob[,2], valid.df$Personal.Loan)
rocObj = performance(predObj, measure="tpr", x.measure="fpr")
aucObj = performance(predObj, measure="auc")

#plot ROC curve
plot(rocObj, main = paste("Model 3 - Area under the curve:", round(aucObj@y.values[[1]], 4)))
predObj