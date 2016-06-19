# This script trains a Random Forest model based on the data,
# saves a sample submission, and plots the relative importance
# of the variables in making predictions

# Download 1_random_forest_r_submission.csv from the output below
# and submit it through https://www.kaggle.com/c/titanic-gettingStarted/submissions/attach
# to enter this getting started competition!

library(ggplot2)
library(randomForest)

set.seed(1)
train <- read.csv("./datasets/train.csv", stringsAsFactors=FALSE)
test  <- read.csv("./datasets/test.csv",  stringsAsFactors=FALSE)

test$Survived <- NA

#Group all data
all_data <- rbind(train, test)

features <- all_data
features$Fare[is.na(features$Fare)] <- median(features$Fare, na.rm=TRUE)
features$Embarked[features$Embarked==""] = "S"
features$Sex      <- as.factor(features$Sex)
features$Embarked <- as.factor(features$Embarked)
features$Family <- features$Parch + features$SibSp + 1
features$Fare_pp <- features$Fare / features$Family

# How to fill in missing Age values?
# We make a prediction of a passengers Age using the other variables and a decision tree model. 
# This time you give method = "anova" since you are predicting a continuous variable.
library(rpart)
library("rattle")
library("rpart.plot")
library("RColorBrewer")
predicted_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch,
                       data = features[!is.na(features$Age),], method = "anova")
fancyRpartPlot(predicted_age)

features$Age[is.na(features$Age)] <- predict(predicted_age, features[is.na(features$Age),])

# Split the data back into a train set and a test set
train <- features[1:891,]
test <- features[892:1309,]


rf <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare_pp + Family, data = train, 
                   importance = TRUE, ntree = 1000)

submission <- data.frame(PassengerId = test$PassengerId)
submission$Survived <- predict(rf, test)
write.csv(submission, file = "submission.csv", row.names=FALSE)

imp <- importance(rf, type=1)
featureImportance <- data.frame(Feature=row.names(imp), Importance=imp[,1])

ggplot(featureImportance, aes(x=reorder(Feature, Importance), y=Importance)) +
     geom_bar(stat="identity", fill="green") +
     coord_flip() + 
     theme_light(base_size=20) +
     xlab("") +
     ylab("Importance") + 
     ggtitle("Random Forest Feature Importance\n") +
     theme(plot.title=element_text(size=18))

