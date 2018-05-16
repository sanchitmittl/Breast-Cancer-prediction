install.packages("ggplot2")
library(ggplot2)
install.packages("caret")
library(caret)
install.packages("stringr")
library(magrittr)
library(stringr)
install.packages("lattice")
library(lattice)


setwd("C:/Users/User 1/Desktop/NUS/CI/assignment")

rm(list=ls()) #clear workspace
set.seed(2) #to avoid random result

dat <- read.csv(file = "breast tissue data.csv", stringsAsFactors=FALSE)

#replace Class 1 with 0 and Class 2 with 1 for nnet's target range
z = dat$Class
z = replace(z, (dat$Class == "car"), 0)
z = replace(z, (dat$Class == "fad"), 1)
z = replace(z, (dat$Class == "mas"), 1)
z = replace(z, (dat$Class == "gla"), 1)
z = replace(z, (dat$Class == "con"), 2)
z = replace(z, (dat$Class == "adi"), 3)
z = as.numeric(z) #force z into numeric
dat$class = z
dat_full <- dat

dat_numeric <- dat_full[c("I0","PA500","HFS","DA","Area","A.DA","Max.IP","DR","P","class")]

#removingDA and Area
dat_numeric1 <- dat_full[c("I0","PA500","HFS","A.DA","Max.IP","DR","P","class")]

#exploring the data
str(dat_numeric)
cor(dat_numeric)
is.na(dat)
summary(dat)
counts <- table(dat$X1)
barplot(counts, main="Age Distribution", 
        xlab="Age")


dat$X1 = as.numeric(dat$X1)
dat$X14 = as.numeric(dat$X14)

#plot correlation in heatmap
cormat <- round(cor(dat_numeric),2)
head(cormat)
library(reshape2)
melted_cormat <- melt(cormat)
head(melted_cormat)

library(ggplot2)
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()


plt.title('Heatmap correlation')
plt.show()

#analysis of covariance
fit <- aov(class ~., data= dat_numeric)
summary(fit)

#convert target variables into factor
dat_numeric$class = as.factor(dat_numeric$class)

#Sample data with 0.7-0.3 ratio and split into "train_dat" and "test_dat"
train_set = sample(1:length(dat_numeric[,1]), 0.7*length(dat_numeric[,1]), replace = FALSE)
test_set = setdiff(1:length(dat_numeric[,1]), train_set)
train_dat <- dat_numeric[train_set, ]
test_dat <- dat_numeric[test_set, ]

#training

trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(3233)

svm_Linear <- train(class ~., data = train_dat, method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)




#tuning the soft margin (cost function)

grid <- expand.grid(C = c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75))
set.seed(3233)
svm_Linear_Grid <- train(class ~., data = train_dat, method = "svmLinear",
                           trControl=trctrl,
                           preProcess = c("center", "scale"),
                           tuneGrid = grid,
                           tuneLength = 10)

svm_Linear_Grid

plot(svm_Linear_Grid)


test_pred <- predict(svm_Linear, newdata = test_dat)
test_pred

confusionMatrix(test_pred, test_dat$class)

#using Kernel function

set.seed(3233)
svm_Radial <- train(class ~., data = train_dat, method = "svmPoly",
                      trControl=trctrl,
                      preProcess = c("center", "scale"),
                      tuneLength = 10)

plot(svm_Radial)

#on test data

test_pred_Radial <- predict(svm_Radial, newdata = test_dat)

#Confusion Matrix and Statistics
confusionMatrix(test_pred_Radial, test_dat$class )


#tuning our classifier with different values of C & sigma

grid_radial <- expand.grid(sigma = c(0,0.01, 0.02, 0.025, 0.03, 0.04,
                                       0.05, 0.06, 0.07,0.08, 0.09, 0.1, 0.25, 0.5, 0.75,0.9,1,1.2),
                             C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75,
                                   1, 1.5, 2,5))

set.seed(3233)
svm_Radial_Grid <- train(class ~., data = train_dat, method = "svmPoly",
                           trControl=trctrl,
                           preProcess = c("center", "scale"),
                           tuneGrid = grid_radial,
                           tuneLength = 10)



plot(svm_Radial_Grid)

plot(svm_Radial_Grid, train_dat, class ~., 
     slice = list(Sepal.Width = 1, Sepal.Length = 2))
 
##The final values used for the model were sigma = 0.75 and C = 5. ##
 
 #trying out test set 
test_pred_Radial_Grid <- predict(svm_Radial_Grid, newdata = test_dat)

confusionMatrix(test_pred_Radial_Grid, test_dat$class )

#using SVM poly
#for polynomial

set.seed(3233)
svm_Radial <- train(class ~., data = train_dat, method = "svmPoly",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)

svm_Radial_Grid <- train(class ~., data = train_dat, 
                         method = "svmPoly", 
                         trControl = trctrl,
                         preProc = c("center", "scale"))
plot(svm_Radial_Grid)
