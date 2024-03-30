library(readr)
data <- read_csv("oulad-students.csv")
data$code_module <- as.factor(data$code_module)
data$code_presentation <- as.factor(data$code_presentation)
data$gender <- as.factor(data$gender)
data$region <- as.factor(data$region)
data$highest_education <- as.factor(data$highest_education)
data$imd_band <- as.factor(data$imd_band)
data$age_band <- as.factor(data$age_band)
data$num_of_prev_attempts <- as.integer(as.character(data$num_of_prev_attempts))  # Convert to integer
data$disability <- as.factor(data$disability)
data$final_result <- as.factor(data$final_result)
data <- na.omit(data)
library(caret)
set.seed(123)
train_index <- createDataPartition(data$final_result, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]
model <- glm(final_result ~ ., data = train_data, family = binomial)
predictions <- predict(model, newdata = test_data, type = "response")
predictions <- ifelse(predictions > 0.5, "Pass", "Fail")
predictions <- factor(predictions, levels = levels(test_data$final_result))
confusionMatrix(predictions, test_data$final_result)

