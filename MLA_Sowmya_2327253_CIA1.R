## Load necessary libraries
library(tidyverse)
library(corrplot)
library(psych)
library(ggplot2)
library(DataExplorer)
library(car)
library(lmtest)
library(Metrics)
library(MASS)
library(glmnet)
library(dplyr)

################################################################################

## Load the dataset
setwd("C:\\Users\\Asus\\Desktop\\TRIM 4\\MLA")
getwd()
wine_data=read.csv("winequality-white.csv", header=T)

################################################################################

## DATA EXPLORATION
View(wine_data)
names(wine_data)

# Display the first few rows of the dataset
head(wine_data)
# Dimension of the dataset
dim(wine_data)
# Structure of the dataset
str(wine_data)

# Summary statistics of the dataset
summary(wine_data)

# Check for missing values
summary(wine_data)
is.na(wine_data)
sum(is.na(wine_data))
plot_missing(wine_data)

# Distribution of the target variable 'quality'
ggplot(wine_data, aes(x = quality)) + 
  geom_bar(fill = "skyblue", color = "black") + 
  ggtitle("Distribution of Wine Quality Scores") +
  xlab("Quality Score") + ylab("Count")

#Understand distributions and correlations
pairs.panels(wine_data)

# Correlation matrix
cor_matrix <- cor(wine_data)
corrplot::corrplot(cor_matrix, method = "circle", type = "upper", tl.cex = 0.8)

plot_histogram(wine_data)
plot_density(wine_data)
plot_correlation(wine_data)

################################################################################

# ASSESSING DATA QUALITY

# Check for missing values
sum(is.na(wine_data))

# Check for duplicates
duplicate_rows <- wine_data[duplicated(wine_data), ]
nrow(duplicate_rows) # will remove the duplicates in data cleaning step

################################################################################

#DATA CLEANING

# Check for missing values
sum(is.na(wine_data))

# Remove duplicates
wine_data_clean <- wine_data[!duplicated(wine_data), ]
duplicate_rows <- wine_data[duplicated(wine_data_clean), ]
nrow(duplicate_rows)

wine_data_clean
dim(wine_data_clean)


# Outlier detection using boxplots
boxplot(wine_data_clean[, -ncol(wine_data_clean)], main="Boxplot for Outlier Detection", col="orange", border="brown")

# Detecting outliers using IQR method
outliers <- apply(wine_data_clean, 2, function(x) {
  Q1 <- quantile(x, 0.25)
  Q3 <- quantile(x, 0.75)
  IQR <- Q3 - Q1
  sum(x < (Q1 - 1.5 * IQR) | x > (Q3 + 1.5 * IQR))
})
print(outliers)

# Visualize outliers using boxplots
boxplot(wine_data_cleaned[, -ncol(wine_data_clean)], main="Boxplot for Outlier Detection", col="orange", border="brown")

# Treating outliers (example: capping outliers)
cap_outliers <- function(x) {
  Q1 <- quantile(x, 0.25)
  Q3 <- quantile(x, 0.75)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  x[x < lower_bound] <- lower_bound
  x[x > upper_bound] <- upper_bound
  return(x)
}

wine_data_cleaned <- as.data.frame(lapply(wine_data_clean, cap_outliers))
dim(wine_data_cleaned)
wine_data_cleaned

################################################################################

# MODELING

# Shuffling / mixing the dataset
dim(wine_data_cleaned)
set.seed(1234)
winedata_mixed<-wine_data_cleaned[order(runif(3961)),]
training<-winedata_mixed[1:3150,]
testing<-winedata_mixed[3151:3961,]

#Building full model
fullmodel<-lm(quality~.,data=training)
fullmodel
summary(fullmodel)

#Building model with relevant features
lm_relevant<-lm(quality~fixed.acidity + volatile.acidity + residual.sugar + free.sulfur.dioxide
                + density + pH+ sulphates,data=training)
summary(lm_relevant)


# Predict and evaluate on test data
fullmodel_pred <- predict(fullmodel, newdata = testing)
fullmodel_pred
lm_relevant_pred <- predict(lm_relevant, newdata = testing)
lm_relevant_pred


# Calculate performance metrics
fullmodel_r2 <- summary(fullmodel)$r.squared
fullmodel_test_r2 <- cor(testing$quality, fullmodel_pred)^2


lm_relevant_r2 <- summary(lm_relevant)$r.squared
lm_relevant_test_r2 <- cor(testing$quality, lm_relevant_pred)^2

# Compare R-squared values
cat("Full Model - Train R2:", fullmodel_r2, "Test R2:", fullmodel_test_r2, "\n")

cat("Simplified Model - Train R2:", lm_relevant_r2, "Test R2:", lm_relevant_test_r2, "\n")

# MSE
fullmodel_mse <- mean((testing$quality - fullmodel_pred)^2)
cat("Full Model - MSE:", fullmodel_mse, "\n")

lm_relevant_mse <- mean((testing$quality - lm_relevant_pred)^2)
cat("Simplified Model  - MSE:", lm_relevant_mse, "\n")

################################################################################

# 

# Create model matrix 
X <- model.matrix(quality ~ ., wine_data_cleaned)[, -1]
X
Y <- wine_data_cleaned$quality
Y

# Define the lambda sequence
lambda <- 10^seq(10, -2, length = 100)
print(lambda)

# Split the data into training and validation sets
set.seed(567)
part <- sample(2, nrow(X), replace = TRUE, prob = c(0.7, 0.3))
X_train <- X[part == 1, ]
X_cv <- X[part == 2, ]
Y_train <- Y[part == 1]
Y_cv <- Y[part == 2]

################################################################################

# Perform Ridge regression
ridge_reg <- glmnet(X_train, Y_train, alpha = 0, lambda = lambda)
summary(ridge_reg)

# Find the best lambda via cross-validation
ridge_reg1 <- cv.glmnet(X_train, Y_train, alpha = 0)
bestlam <- ridge_reg1$lambda.min
print(bestlam)

# Predict on the validation set
ridge.pred <- predict(ridge_reg, s = bestlam, newx = X_cv)

# Calculate mean squared error
mse <- mean((Y_cv - ridge.pred)^2)
print(paste("Mean Squared Error:", mse))

################################################################################

# Perform Lasso regression
lasso_reg <- glmnet(X_train, Y_train, alpha = 1, lambda = lambda)

# Find the best lambda via cross-validation
lasso_reg1 <- cv.glmnet(X_train, Y_train, alpha = 1)
bestlam <- lasso_reg1$lambda.min
bestlam

# Predict on the validation set
lasso.pred <- predict(lasso_reg, s = bestlam, newx = X_cv)

# Calculate mean squared error
mse <- mean((Y_cv - lasso.pred)^2)
print(paste("Mean Squared Error:", mse))

# Calculate R2 value
sst <- sum((Y_cv - mean(Y_cv))^2)
sse <- sum((Y_cv - lasso.pred)^2)
r2 <- 1 - (sse / sst)
print(paste("R²:", r2))

# Get the Lasso regression coefficients
lasso.coef <- predict(lasso_reg, type = "coefficients", s = bestlam)
print("Lasso Coefficients:")
print(lasso.coef)

##################################################################################

# Compare MSE and R² for all models

# Multiple Linear Regression (Simplified Model)
simplified_mse <- mean((testing$quality - lm_relevant_pred)^2)
simplified_r2 <- cor(testing$quality, lm_relevant_pred)^2

# Ridge Regression
ridge_mse <- mean((Y_cv - ridge.pred)^2)
ridge_r2 <- 1 - (sum((Y_cv - ridge.pred)^2) / sum((Y_cv - mean(Y_cv))^2))

# Lasso Regression
lasso_mse <- mean((Y_cv - lasso.pred)^2)
lasso_r2 <- 1 - (sum((Y_cv - lasso.pred)^2) / sum((Y_cv - mean(Y_cv))^2))

# Print performance metrics
cat("Simplified Linear Regression - MSE:", simplified_mse, "R²:", simplified_r2, "\n")
cat("Ridge Regression - MSE:", ridge_mse, "R²:", ridge_r2, "\n")
cat("Lasso Regression - MSE:", lasso_mse, "R²:", lasso_r2, "\n")

# Choose the best model based on the lowest MSE and highest R²
model_performance <- data.frame(
  Model = c("Multiple Linear Regression", "Ridge Regression", "Lasso Regression"),
  MSE = c(simplified_mse, ridge_mse, lasso_mse),
  R_squared = c(simplified_r2, ridge_r2, lasso_r2)
)

print(model_performance)

# Identify the best model based on MSE
best_model <- model_performance[which.min(model_performance$MSE), ]
cat("Best Model Based on MSE:\n")
print(best_model)

# Identify the best model based on R-squared
best_model_r2 <- model_performance[which.max(model_performance$R_squared), ]
cat("Best Model Based on R-squared:\n")
print(best_model_r2)

