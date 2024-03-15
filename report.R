# Importing packages
# install.packages("tidyverse")...
library(tidyverse)
library(data.table)
library(mlr3)
library(mlr3learners)
library(mlr3viz)
library(ggplot2)
library(skimr)
library(DataExplorer)
library(caret)
library(mlr3tuning)
library(paradox)

# Read data
bank <- read.csv("bank_personal_loan.csv",header = TRUE) 
head(bank)
dim(bank)
str(bank)
summary(bank)

# Previously, the feature Zip code will not be used.
bank <- bank[, -4]

# Check missing values
missing_values_per_column <- sapply(bank, function(x) sum(is.na(x)))
print(missing_values_per_column)

# skim data
skimr::skim(bank)
DataExplorer::plot_bar(bank, ncol = 3)
DataExplorer::plot_histogram(bank, ncol = 3)
DataExplorer::plot_boxplot(bank, by = "Personal.Loan", ncol = 3)

# Replace the negative experience values with the most commonly occurring value
bank[bank < 0] <- 32

# Set Task
set.seed(212) # set seed for reproducibility

bank$Personal.Loan <- as.factor(bank$Personal.Loan)

loan_task <- TaskClassif$new(id = "BankLoan",
                             backend = bank, 
                             target = "Personal.Loan")

train_set = sample(loan_task$row_ids, 0.8 * loan_task$nrow)
test_set = setdiff(loan_task$row_ids, train_set)

# Logistic Regression 
learner_logreg = lrn("classif.log_reg",predict_type = "response")
learner_logreg$train(loan_task, row_ids = train_set)

# Decision Tree
learner_tree = lrn("classif.rpart", predict_type = "response")
learner_tree$train(loan_task, row_ids = train_set)

# Random Forest
learner_rf = lrn("classif.ranger", predict_type = "response", importance = "permutation")
learner_rf$train(loan_task, row_ids = train_set)

# Plot the importance of feature
importance = as.data.table(learner_rf$importance(), keep.rownames = TRUE)
colnames(importance) = c("Feature", "Importance")
ggplot(data=importance,
       aes(x = reorder(Feature, Importance), y = Importance)) + 
  geom_col() + coord_flip() + xlab("")

# Predict
pred_logreg = learner_logreg$predict(loan_task, row_ids = test_set)
pred_tree = learner_tree$predict(loan_task, row_ids = test_set)
pred_rf = learner_rf$predict(loan_task, row_ids = test_set)
autoplot(pred_logreg)
autoplot(pred_tree)
autoplot(pred_rf)

# Confusion
pred_logreg$confusion
pred_tree$confusion
pred_rf$confusion

# Evaluation
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(loan_task)

res <- benchmark(data.table(
  task       = list(loan_task),
  learner    = list(learner_logreg,
                    learner_tree,
                    learner_rf),
  resampling = list(cv5)
), store_models = TRUE)
res$aggregate(measures = msr("classif.acc"))

# Desicion Tree Tuning
search_space <- ps(
  cp = p_dbl(lower = 0.001, upper = 0.1),
  minsplit = p_int(lower = 1, upper = 10)
)
evals20 <- trm("evals", n_evals = 40) 
instance <- tune(
  task = loan_task,
  learner = learner_tree,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.acc"),
  search_space = search_space,
  tuner = tnr("grid_search", resolution = 5),
  terminator = evals20
)

instance$result_learner_param_vals
instance$result_y

# Random Forest Tuning
search_space = ps(
  num.trees = p_int(lower=100,upper=1000),
  mtry = p_int(lower=1,upper=9),
  min.node.size = p_int(lower=1, upper=10),
  max.depth = p_int(lower=5,upper=30)
)
evals20 <- trm("evals", n_evals = 20) 
instance <- tune(
  task = loan_task,
  learner = learner_rf,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.acc"),
  search_space = search_space,
  tuner = tnr("random_search") ,
  terminator = evals20
)

instance$result_learner_param_vals
instance$result_y

learner_rf$param_set$values$k = instance$result_learner_param_vals$k
learner_rf$train(loan_task, row_ids = train_set)
learner_rf$predict(loan_task, row_ids = test_set)$confusion

res_rf_best <- resample(loan_task, learner_rf, cv5)
res_rf_best$aggregate(measures = msr("classif.acc"))