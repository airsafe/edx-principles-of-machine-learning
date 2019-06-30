# Processing start time
sink('boosting_output.txt')
timeStart = Sys.time()

# ADMINISTRATIVE NOTES
# Note: To describe database at any point, use str(*name*)
# Note: To clear R workspace, use rm(list = ls())
# Note: Searching R help files - RSiteSearch("character string")
# Note: To clear console, use CTRL + L 


# INSTALLING SPECIAL PACKAGES
# if("<package_name>" %in% rownames(installed.packages()) == FALSE) 
# {install.packages("<package_name>")}# Added due to error
# library(<package_name>)
# ====================

# Boosting and the AdaBoost Method
# Using ensemble methods can greatly improve the results achieved with weak machine learning algorithms, also called weak learners. 
# Ensemble methods achieve better performance by aggregating the results of many statistically independent models. 
# This process averages out the errors and produces a better, final, prediction.
# 
# In this lab you will work with widely used ensemble method known as boosting. 
# Boosting is a meta-algorithm since the method can be applied to many types of machine learning algorithms. 
# In summary, boosting iteratively improves the learning of the N models by giving greater weight to training cases with larger errors. 
# The basic boosting procedure is simple and follows these steps:
#         
#         1. N learners (machine learning models) are defined.
#         2. Each of i training data cases is given an initial equal weight of 1/i.
#         3. The N learners are trained on the weighted training data cases.
#         4. The prediction is computed based on a aggregation of the learners; averaging over the hypothesis of the N learners.
#         5. Weights for the training data cases are updated based on the aggregated errors made by the learners. 
#            Cases with larger errors are given larger weights.
#         6. Steps 3, 4, and 5 are repeated until a convergence criteria is met.
# 
# Classification and regression tree models are the weak learners most commonly used with boosting. 
# In this lab you will work with one of the most widely used and successful boosted methods, known as AdaBoost or adaptive boosting. 
# AdaBoost uses some large number, N, tree models. The rate at which weights are updated is adaptive with the errors.
# 
# It is important to keep in mind that boosted machine learning is not robust to significant noise or outliers in the training data. 
# The re-weighting process gives greater weight to the large errors, and therefore can give undue weight to outliers and errors. 
# In cases where data is noisy, the random forest algorithm may prove to be more robust.
# 
# Example: Iris dataset
# As a first example you will use AdaBoost to classify the species of iris flowers.
# 
# As a first step, execute the code in the cell below to load the required packages to run the rest of this notebook.
# 
# Note: If you are running in Azure Notebooks, make sure that you run the code in the setup.ipynb notebook at the start of you session to ensure your environment is correctly configured.

## Import packages

if("ggplot2" %in% rownames(installed.packages()) == FALSE) 
{install.packages("ggplot2")}
library(ggplot2)

if("gridExtra" %in% rownames(installed.packages()) == FALSE) 
{install.packages("gridExtra")}
library(gridExtra)

if("repr" %in% rownames(installed.packages()) == FALSE) 
{install.packages("repr")}
library(repr)

if("dplyr" %in% rownames(installed.packages()) == FALSE) 
{install.packages("dplyr")}
library(dplyr)

if("caret" %in% rownames(installed.packages()) == FALSE) 
{install.packages("caret")}
library(caret)

if("gbm" %in% rownames(installed.packages()) == FALSE) 
{install.packages("gbm")}
library(gbm)

if("e1071" %in% rownames(installed.packages()) == FALSE) 
{install.packages("e1071")}
library(e1071)

if("MLmetrics" %in% rownames(installed.packages()) == FALSE) 
{install.packages("MLmetrics")}
library(MLmetrics)

options(repr.plot.width=4, repr.plot.height=4) # Set the initial plot area dimensions

# The code in the cell below displays the head of the data frame and plots all pairwise combinations of the features with the species of the iris flower in colors. 
# Execute this code and examine the results.

single_plot = function(df, colx, coly){
        ggplot(df, aes_string(colx,coly)) +
                geom_point(aes(color = factor(df$Species)), alpha = 0.4)
}

plot_iris = function(df){
        options(repr.plot.width=8, repr.plot.height=5)
        grid.arrange(
                single_plot(df, 'Sepal.Length', 'Sepal.Width'),
                single_plot(df, 'Sepal.Length', 'Petal.Length'),
                single_plot(df, 'Petal.Length', 'Petal.Width'),
                single_plot(df, 'Sepal.Width', 'Petal.Length'),
                nrow = 2)
}

head(iris, 10)   
plot_iris(iris)

# You can see that Setosa (in blue) is well separated from the other two categories. 
# The Versicolor (in orange) and the Virginica (in green) show considerable overlap. 
# The question is how well our classifier will separate these categories.
# 
# Next, execute the code in the cell below to split the dataset into test and training set. 
# Notice that unusually, 67% of the cases are being used as the test dataset.

set.seed(1955)
## Randomly sample cases to create independent training and test data
partition = createDataPartition(iris[,'Species'], times = 1, p = 0.33, list = FALSE)
training = iris[partition,] # Create the training sample
dim(training)
test = iris[-partition,] # Create the test sample
dim(test)

# As is always the case with machine learning, numeric features must be scaled. 
# Execute the code in the cell below to scale the training and test datasets:

num_cols = c('Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width')
preProcValues <- preProcess(training[,num_cols], method = c("center", "scale"))

training[,num_cols] = predict(preProcValues, training[,num_cols])
test[,num_cols] = predict(preProcValues, test[,num_cols])
head(training[,num_cols])

# Now you will define and fit an AdaBoosted tree model. 
# The code in the cell below defines the model with 5 estimators (trees) using the gbm function from the R gbm package, and then fits the model. 
# Since this is a multi-class classification problem a multinomial distribution is specified for the model response. 
# Execute this code.

set.seed(3355)
gbm_mod = gbm(Species ~ ., data = training, distribution = "multinomial", n.trees = 5)

# Next, the code in the cell below uses the predict method is used to compute the multinomial class probabilities from the scaled features. 
# Execute this code.

test$probs = predict(gbm_mod, newdata = test, n.trees = 5)
test[1:10,]

# You can see the species and multinomial probabilities for the classes, along with the species of the the flower.
# 
# The code in the cell below uses the R which.max function to compute classes from the maximum of the probabilities. 
# Execute this code.

test$scores = apply(test$probs, 1, which.max) # Find the class with the maximum probability
test$scores = ifelse(test$scores == 1, 'sentosa', ifelse(test$scores == 2, 'versicolor', 'virginica'))
test[1:10,]

# It is time to evaluate the model results. 
# Keep in mind that the problem has been made deliberately difficult, by having more test cases than training cases. 
# The iris data has three species categories. 
# Therefore it is necessary to use evaluation code for a three category problem. 
# The function in the cell below extends code from previous labs to deal with a three category problem. 
# Execute this code and examine the results.

print_metrics = function(df, label){
        ## Compute and print the confusion matrix
        cm = as.matrix(table(Actual = df$Species, Predicted = df$scores))
        print(cm)
        
        ## Compute and print accuracy 
        accuracy = round(sum(sapply(1:nrow(cm), function(i) cm[i,i]))/sum(cm), 3)
        cat('\n')
        cat(paste('Accuracy = ', as.character(accuracy)), '\n \n')                           
        
        ## Compute and print precision, recall and F1
        precision = sapply(1:nrow(cm), function(i) cm[i,i]/sum(cm[i,]))
        recall = sapply(1:nrow(cm), function(i) cm[i,i]/sum(cm[,i]))    
        F1 = sapply(1:nrow(cm), function(i) 2*(recall[i] * precision[i])/(recall[i] + precision[i]))    
        metrics = sapply(c(precision, recall, F1), round, 3)        
        metrics = t(matrix(metrics, nrow = nrow(cm), ncol = ncol(cm)))       
        dimnames(metrics) = list(c('Precision', 'Recall', 'F1'), unique(test$Species))      
        print(metrics)
}  
print_metrics(test, 'Species')   

# Examine these results. 
# Notice the following:
#         
#         1. The confusion matrix has dimension 3X3. You can see that most cases are correctly classified, but with a few errors.
#         2. The overall accuracy is 0.83. Since the classes are roughly balanced, this metric indicates relatively good performance of the classifier, particularly since it was only trained on 50 cases.
#         3. The precision, recall and F1 for each of the classes are reasonable.
# 
# The above results were computed using an ensemble with only 5 trees. 
# Will more trees in the ensemble help? To find out, execute this code and examine the results.
# 
# Then, answer Question 1 on the course page.

set.seed(8899)
gbm_mod = gbm(Species ~ ., data = training, distribution = "multinomial", n.trees = 100)
test$probs = predict(gbm_mod, newdata = test, n.trees = 100)
test$scores = apply(test$probs, 1, which.max) # Find the class with the maximum probability
test$scores = ifelse(test$scores == 1, 'sentosa', ifelse(test$scores == 2, 'versicolor', 'virginica'))
print_metrics(test, 'Species')  

# These metrics show a small improvement. 
# Using more trees in the ensemble has resulted in a better model.
# 
# How important are each of the features for this model? 
# The R Caret package provides the capability to find out. 
# As a first step, gbm models must be trained using the the Caret train function. 
# The code in the cell below does this, using the default model arguments. 
# The default arguments for the model are specified with the tuneGrid argument of train. 
# Execute the code.

gbm_mod_train = train(Species ~ ., data = training, method = "gbm", verbose = FALSE,
                      distribution = "multinomial", trControl=trainControl(number=100),
                      tuneGrid=expand.grid(n.trees = 100,
                                           interaction.depth = 1, 
                                           n.minobsinnode = 10, 
                                           shrinkage = 0.001))

# With the Caret model object trained, the feature importance can be computed and displayed. 
# Execute this code and examine the results.

options(repr.plot.width=4, repr.plot.height=3)
imp = varImp(gbm_mod_train, scale = FALSE)$importance
imp[,'Feature'] = row.names(imp)
ggplot(imp, aes(x = Feature, y = Overall)) + geom_point(size = 4) +
        ggtitle('Variable importance for iris features')

# Observe the following:
#         
# 1. Petal length is the most importance feature.
# 2. Sepal width is the least important feature.
# 
# The question now is, if a model with a reduced feature set will be an improvement. 
# Execute the code in the cell below which creates a model using three of the four features.
# 
# Then, answer Question 2 on the course page.

set.seed(8899)
gbm_mod = gbm(Species ~ Petal.Length + Petal.Width + Sepal.Length, 
              data = training, distribution = "multinomial", n.trees = 100)
test$probs = predict(gbm_mod, newdata = test, n.trees = 100)
test$scores = apply(test$probs, 1, which.max) # Find the class with the maximum probability
test$scores = ifelse(test$scores == 1, 'sentosa', ifelse(test$scores == 2, 'versicolor', 'virginica'))
print_metrics(test, 'Species')  

# The model with reduced feature set has slightly improved metrics. 
# Overall, this model is preferred since it is more likely to generalize.
# 
# The code in the cell below plots the classes of the iris flower along with the classification errors shown by shape. 
# Execute this code and examine the results.

## Create column of correct-incorrect classification
test$correct = ifelse(test$Species == test$scores, 'correct', 'incorrect')

single_plot_classes = function(df, colx, coly){
        ggplot(df, aes_string(colx,coly)) +
                geom_point(aes(color = factor(df$Species), shape = correct), alpha = 0.4)
}

plot_iris_classes = function(df){
        options(repr.plot.width=8, repr.plot.height=5)
        grid.arrange(
                single_plot_classes(df, 'Sepal.Length', 'Sepal.Width'),
                single_plot_classes(df, 'Sepal.Length', 'Petal.Length'),
                single_plot_classes(df, 'Petal.Length', 'Petal.Width'),
                single_plot_classes(df, 'Sepal.Width', 'Petal.Length'),
                nrow = 2)
}

plot_iris_classes(test)

# Examine these plots. 
# You can see how the classifier has divided the feature space between the classes. 
# Notice that most of the errors occur in the overlap region between Virginica and Versicolor. 
# This behavior is to be expected.
# 
# Another example
# Now, you will try a more complex example using the credit scoring data. 
# You will use the prepared data which has been prepared by removing duplicate cases. 
# Some columns which are know not to be predictive are removed. 
# Execute the code in the cell below to load the dataset for the example.

credit = read.csv('German_Credit_Preped.csv', header = TRUE)
## Subset the data frame
credit = credit[,c('checking_account_status', 'loan_duration_mo', 'credit_history', 'loan_amount', 'savings_account_balance',
                   'time_employed_yrs', 'payment_pcnt_income', 'time_in_residence', 'property', 'age_yrs',
                   'other_credit_outstanding', 'number_loans', 'job_category', 'dependents', 'telephone', 'bad_credit' )]
print(dim(credit))
names(credit)

# Cross validation will be used to train the model. 
# Since folds will be selected from the entire dataset the numeric features are scaled in batch. 
# Execute the code in the cell below to accomplish this:
        
        num_cols = c('loan_duration_mo', 'loan_amount', 'payment_pcnt_income',
                     'time_in_residence', 'age_yrs', 'number_loans', 'dependents')

preProcValues <- preProcess(credit[,num_cols], method = c("center", "scale"))
credit[,num_cols] = predict(preProcValues, credit[,num_cols])
head(credit[,num_cols])

# The R Caret package computes most performance metrics using the positive cases. 
# For example, recall is a measure of correct classification of positive cases. 
# Therefore, it is important to have the coding of the label correct. 
# The code in the cell below creates a factor (categorical) variable and coerces the levels of the label column, bad_credit. 
# Execute this code.

credit$bad_credit <- ifelse(credit$bad_credit == 1, 'bad', 'good')
credit$bad_credit <- factor(credit$bad_credit, levels = c('bad', 'good'))
credit$bad_credit[1:5]

# In the results above you can see the new coding of the label column along with the levels, {'bad', 'good'}.
# 
# As the inner loop of a nested cross validation, the code in the cell below uses the capability of the R Caret package to estimate the best hyperparameters using 5 fold cross validation. 
# This first cross validation is performed using ROC as the metric. There are a few points to note here:
#         
#         1. A Caret trainControl object is used to define the 5 fold cross validation. 
#            The twoClassSummary function is specified, making ROC the metric for hyperparameter optimization. 
#            The gbm model does not accommodate case weights. 
#            To compensate for the class imbalance up-sampling or over-sampling is used. Up-sampling over-samples the minority case so that the number of classes in each training fold is balanced.
#         2. The model is trained using all features as can be seen from the model formula in the Caret train function.
#         3. ROC is specified as a metric in the call to train.
#         4. Weights are specified to help with the class imbalance and the cost imbalance of misclassification of bad credit customers.
#         5. The train function uses a tuneGrid argument to define the hyperparameters to search.
#            Execute this code, examine the result, and answer Question 3 on the course page.

fitControl <- trainControl(method = "cv",
                           number = 5,
                           sampling = 'up',
                           returnResamp="all",
                           savePredictions = TRUE,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

set.seed(1234)
bb_fit_inside_tw <- train(bad_credit ~ ., 
                          data = credit,  
                          method = "gbm", # Gradient boosted tree model
                          trControl = fitControl, 
                          verbose = FALSE,
                          metric="ROC")
print(bb_fit_inside_tw)

# The grid of hyperpameters searched by the Caret package are interaction.depth and n.trees. 
# The grid along with the ROC metric is shown in the printed table. 
# Notice that shrinkage and n.minobsinnode can be added to the tuning grid.
# 
# The hyperparameter optimization can also be performed using Recall as a metric. 
# The code in the cell below uses the prSummary function for the summaryFunction argument for trainControl and sets the metric as Recall. 
# Execute this call and examine the results.

fitControl <- trainControl(method = "cv",
                           number = 5,
                           sampling = 'up',
                           returnResamp="all",
                           savePredictions = TRUE,
                           classProbs = TRUE,
                           summaryFunction = prSummary)

set.seed(15534)
bb_fit_inside_pr <- train(bad_credit ~ ., 
                          data = credit,  
                          method = "gbm", # Gradient boosted tree model
                          trControl = fitControl, 
                          verbose = FALSE,
                          metric="Recall")
print(bb_fit_inside_pr)

# Let's stay with the model trained on ROC.

# Given the optimal hyperparameters for the model trained with ROC, which features are the most important? 
# The code in the cell below computes and displays feature importance using the Caret varImp function. 
# Execute this code and examine the results.

options(repr.plot.width=8, repr.plot.height=6)
var_imp = varImp(bb_fit_inside_tw)
print(var_imp)
plot(var_imp)

# In is clear that some of the features are not important to model performance. 
# Execute the code in the cell below to prune the feature set using the dplyr select function:

# Following did not work, replaced with following code
# credit_reduced = select(credit, -dependents, -time_employed_yrs, -job_category, -other_credit_outstanding)
credit_reduced = credit[,-which(colnames(credit) %in% c("dependents", "time_employed_yr", "job_category", "other_credit_outstanding"))]

# Execute the code in the cell below to perform the cross validation grid search using the reduced feature set:

fitControl <- trainControl(method = "cv",
number = 5,
sampling = 'up',
returnResamp="all",
savePredictions = TRUE,
classProbs = TRUE,
summaryFunction = twoClassSummary)

set.seed(4321)
bb_fit_inside_tw <- train(bad_credit ~ ., 
data = credit_reduced,  
method = "gbm", # Gradient boosted tree model
trControl = fitControl, 
verbose = FALSE,
metric="ROC")
print(bb_fit_inside_tw)

# The results of the cross validation grid search with the reduced feature set are nearly the same as the first result. 
# Evidentially, pruning these features was the correct step. 
# This process can be continued, but will not be in this lab in the interest of reducing length.
# 
# To better understand the parameter sweep, execute the code in the cell below to create a chart.

options(repr.plot.width=5, repr.plot.height=4)
trellis.par.set(caretTheme())
plot(bb_fit_inside_tw)  

# Examine these results. The ROC is in a narrow range, but there are clear differences for the various hyperpareter choices.
# 
# Finally, to verify that the model will generalize well it is time to perform the outside CV loop. 
# The code in the cell below defines a parameter grid with just the optimal hyperparameter values. 
# The CV then repeatedly fits the model with this single hyperparameter. Execute this code and examine the result.
# 
# Then, answer Question 4 on the course page.

## Set the hyperparameter grid to the optimal values from the inside loop
paramGrid <- expand.grid(n.trees = c(100), interaction.depth = c(3), shrinkage = c(0.1), n.minobsinnode = c(10))

set.seed(5678)
bb_fit_outside_tw <- train(bad_credit ~ ., 
data = credit_reduced,  
method = "gbm", # Gradient boosted tree model
trControl = fitControl, 
tuneGrid = paramGrid, 
verbose = FALSE,
metric="ROC")

print_metrics = function(mod){
means = c(apply(mod$resample[,1:3], 2, mean), n.trees = mod$resample[1,4], interaction.depth = mod$resample[1,5], 
shrinkage = mod$resample[1,6], n.minobsinnode = mod$resample[1,7], Resample = 'Mean')
stds = c(apply(mod$resample[,1:3], 2, sd), n.trees = mod$resample[1,4], interaction.depth = mod$resample[1,5], 
shrinkage = mod$resample[1,6], n.minobsinnode = mod$resample[1,7], Resample = 'STD')
out = rbind(mod$resample, means, stds)
out[,1:3] = lapply(out[,1:3], function(x) round(as.numeric(x), 3))
out
}
print_metrics(bb_fit_outside_tw)

# Examine these results. Notice that the standard deviation of the mean of the AUC are nearly an order of magnitude smaller than the mean. 
# This indicates that this model is likely to generalize well.
# 
# Note: The predict method can be used with this optimal model to classify unknown cases.
# 
# Summary
# In this lab you have accomplished the following:
# 
#         1. Used an AdaBoosted tree model to classify the cases of the iris data. 
#            This model produced quite good results.
#         2. Used 5 fold to find estimated optimal hyperparameters for an AdaBoosted tree model to classify credit risk cases. 
#            The model did not generalize well.
#         3. Applied up-sampling of the minority cases to create a balanced training dataset and retrained and evaluated the model.
#         4. Applied feature importance was used for feature selection. 
#            The model created and evaluated with the reduced feature set had essentially the same performance as the model with more features.
#         5.Used the outer loop of the nested cross validation to demonstrate that the model is likely to generalize.
# ====================
# Processing end time
timeEnd = Sys.time()

# Processing date and total processing time
cat(paste("","Processing end date and time",date(),"","",sep="\n"))
paste("Total processing time =",round(difftime(timeEnd,timeStart), digits=2),"seconds",sep=" ")


# Stop writing to an output file
sink()

################