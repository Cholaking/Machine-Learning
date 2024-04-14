# Load required libraries
library(readr)
library(caret)
library(dplyr)
library(ggplot2)
library(glmnet) 
library(pROC)   

# Load and prepare the data
students_data <- read_csv("oulad-students.csv")
assessment_data <- read_csv("oulad-assessments.csv")

# Check for column existence and modify
common_id <- intersect(names(students_data), names(assessment_data))

# Assuming 'id_student' is a common identifier if it exists
if ('id_student' %in% common_id) {
  full_data <- left_join(students_data, assessment_data, by = "id_student")
} else {
  stop("No common identifier column found in both datasets.")
}

# Adjust this list based on actual column names in our dataset
factor_vars <- c("gender", "region", "highest_education", "imd_band", "age_band", "disability", "final_result")
numeric_vars <- c("num_of_prev_attempts", "score")

# Ensuring all factor variables exist before converting
factor_vars <- factor_vars[factor_vars %in% names(full_data)]
numeric_vars <- numeric_vars[numeric_vars %in% names(full_data)]

full_data <- full_data %>%
  mutate_at(vars(factor_vars), factor) %>%
  mutate_at(vars(numeric_vars), as.numeric) %>%
  na.omit()

# Splitting the data
set.seed(123)
train_index <- createDataPartition(full_data$final_result, p = 0.8, list = FALSE)
train_data <- full_data[train_index, ]
test_data <- full_data[-train_index, ]

# Prepare matrix for glmnet
x <- model.matrix(final_result ~ . - score - final_result, data = train_data)[,-1]  # remove intercept term
y <- train_data$final_result

# Fit model using glmnet
fit <- glmnet(x, as.numeric(y) - 1, family = "binomial", alpha = 0)  # alpha=0 for ridge regression

# Predict using fitted model
class_predictions <- predict(fit, s = 0.01, newx = model.matrix(~ . - score - final_result, data = test_data)[,-1], type = "response")
class_predictions_binary <- factor(ifelse(class_predictions > 0.5, "Pass", "Fail"), levels = levels(test_data$final_result))

# Confusion Matrix and other metrics
conf_matrix <- confusionMatrix(class_predictions_binary, test_data$final_result)
print(conf_matrix)

# Regression model predicting scores
regression_model <- lm(score ~ . - final_result, data = train_data)
reg_predictions <- predict(regression_model, newdata = test_data)

# Calculate and print RMSE and R-squared
rmse <- sqrt(mean((test_data$score - reg_predictions)^2))
rsq <- summary(regression_model)$r.squared
cat("RMSE: ", rmse, "\n")
cat("R-squared: ", rsq, "\n")

# Plotting actual vs predicted scores for regression model
ggplot(data = test_data, aes(x = score, y = reg_predictions)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = lm, col = "blue") +
  labs(x = "Actual Score", y = "Predicted Score", title = "Actual vs Predicted Scores")

# Adjusting the ROC analysis for multi-class to a binary case (e.g., Pass vs. Non-Pass)
binary_response <- ifelse(test_data$final_result == "Pass", "Pass", "Not Pass")

# Convert predictions to a binary format focusing on "Pass" class
binary_predictions <- ifelse(class_predictions_binary == "Pass", "Pass", "Not Pass")

# Compute ROC curve
binary_roc_curve <- roc(binary_response, as.numeric(binary_predictions == "Pass"))

# Plot ROC Curve
plot(binary_roc_curve, main = "ROC Curve for 'Pass' vs. All Other Classes")

# Box plots for key predictors
ggplot(full_data, aes(x = final_result, y = num_of_prev_attempts, color = final_result)) +
  geom_boxplot() +
  labs(title = "Box Plot of Previous Attempts by Final Result", y = "Number of Previous Attempts", x = "Final Result")

# Comparison plot of actual vs predicted categories
comparison_data <- data.frame(Actual = test_data$final_result, Predicted = class_predictions_binary)
ggplot(comparison_data, aes(x = Actual, fill = Predicted)) +
  geom_bar(position = "fill") +
  labs(y = "Proportion", x = "Actual Category", fill = "Predicted Category", title = "Actual vs Predicted Categories")

