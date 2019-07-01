# Processing start time
sink('support-vector-machines_output.txt')
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

# Support Vector Machine Models
# Support vector machines (SVMs) are a widely used and powerful category of machine learning algorithms. 
# There are many variations on the basic idea of an SVM. 
# An SVM attempts to maximally separate classes by finding the support vector with the lowest error rate or maximum separation. 
# SVMs can use many types of kernel functions. 
# The most common kernel functions are linear and the radial basis function or RBF. 
# The linear basis function attempts to separate classes by finding hyperplanes in the feature space that maximally separate classes. 
# The RBF uses set of local Gaussian shaped basis kernels to find a nonlinear separation of the classes.
# 
# Example: Iris dataset
# As a first example you will use SVMs to classify the species of iris flowers.
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

if("e1071" %in% rownames(installed.packages()) == FALSE) 
{install.packages("e1071")}
library(e1071)

if("MLmetrics" %in% rownames(installed.packages()) == FALSE) 
{install.packages("MLmetrics")}
library(MLmetrics)

# Had to add this package after initially getting an error
if("kernlab" %in% rownames(installed.packages()) == FALSE) 
{install.packages("kernlab")}
library(kernlabs)

options(repr.plot.width=4, repr.plot.height=4) # Set the initial plot area dimensions

# To get a feel for these data, you will now load and plot them. 
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

# You can see that Setosa (in red) is well separated from the other two categories. 
# The Versicolor (in green) and the Virginica (in blue) show considerable overlap. 
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

# Now you will define and fit an SVM model. The code in the cell uses the svm function from the e1071 package. 
# The model formula is specified along with the data, the requirement not to scale and the type of model required. 
# Execute this code.

svm_mod = svm(factor(Species) ~ ., data = training, scale = FALSE, type = 'C-classification')

# Next, the code in the cell below uses the predict method is used to compute the multinomial class probabilities from the scaled features. 
# Execute this code.

test[,'scores'] = predict(svm_mod, newdata = test)
test[1:10,]

# You can see actual species and predicted species.
# 
# It is time to evaluate the model results. 
# Keep in mind that the problem has been made deliberately difficult, by having more test cases than training cases. 
# The iris data has three species categories. 
# Therefore it is necessary to use evaluation code for a three category problem. 
# The function in the cell below extends code from previous labs to deal with a three category problem. 
# 
# Execute this code and examine the results.
# 
# Then, answer Question 1 on the course page.

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

# Examine these results. Notice the following:
#         
#         1. The confusion matrix has dimension 3X3. You can see that most cases are correctly classified.
#         2. The overall accuracy is 0.94. Since the classes are roughly balanced, this metric indicates relatively good performance of the classifier, particularly since it was only trained on 51 cases.
#         3. The precision, recall and F1 for each of the classes is relatively good. 
#            Versicolor has the worst metrics since it has the largest number of misclassified cases.
# 
# How important are each of the features for this model? 
# The R Caret package provides the capability to find out. 
# As a first step, gbm models must be trained using the the Caret train function. 
# The code in the cell below does this, using the default model arguments. 
# The default arguments for the model are specified with the tuneGrid argument of train. 
# 
# Execute the code.

trControl <- trainControl(method = "cv", number = 10)

svm_mod_train = train(factor(Species) ~ ., 
                      data = training, 
                      method = "svmLinear", 
                      verbose = FALSE,
                      trControl = trControl,
                      scale = FALSE, 
                      type = 'C-svc')
svm_mod_train

# With the Caret model object trained, the feature importance can be computed and displayed. 
# Execute this code and examine the results.

options(repr.plot.width=4, repr.plot.height=3)
imp = varImp(svm_mod_train, scale = FALSE)$importance
imp

# Examine the table above. Notice that Sepal.Width has the least importance for classifying each of the species.
# 
# Next, you will train and evaluate a model using the three most important features by executing the code in the cell below.
# 
# Execute this code, and answer Question 1 on the course page.

svm_mod_3 = svm(factor(Species) ~ Sepal.Length + Petal.Length + Petal.Width, data = training, 
                scale = FALSE, type = 'C-classification')
test[,'scores'] = predict(svm_mod, newdata = test)
print_metrics(test, 'Species')  

# These results are identical to those obtained with the model with all features. The simpler model is preferred since it is likely to generalize better.
# 
# The code in the cell below plots the classes of the iris flower along with the classification errors shown by shape. Execute this code and examine the results.

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
# 
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
# As the inner loop of a nested cross validation, the code in the cell below uses the capability of the R Caret package to estimate the best hyperparameters using 5 fold cross validation. This first cross validation is performed using ROC as the metric. 
# There are a few points to note here:
#         
#       1. A Caret trainControl object is used to define the 5 fold cross validation. 
#          The twoClassSummary function is specified, making ROC the metric for hyperparameter optimization. 
#          The svmRadial model does accommodate case weights. 
#          However, in the case down-sampling or under-sampling is used. 
#          Down-sampling randomly samples the majority case so that the number of classes in each training fold is balanced.
#       2. The model is trained using all features as can be seen from the model formula in the Caret train function.
#       3. ROC is specified as a metric in the call to train.
#       4. Weights are specified to help with the class imbalance and the cost imbalance of misclassification of bad credit customers.
#       5. The train function uses a tuneGrid argument to define the hyperparameters to search.

# Execute this code and examine the result.

fitControl <- trainControl(method = "cv",
                           number = 5,
                           sampling = 'down',
                           returnResamp="all",
                           savePredictions = TRUE,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)
paramGrid <- expand.grid(sigma = c(0.002, 0.02, 0.2), C = c(0.5, 1.0, 2.0))

set.seed(1234)
svm_fit_inside_tw <- train(bad_credit ~ ., 
                           data = credit,  
                           method = "svmRadial", # Radil SVM model 
                           trControl = fitControl, 
                           tuneGrid = paramGrid, 
                           metric="ROC")
print(svm_fit_inside_tw)

# The grid of hyperpameters searched by the Caret package is over kernel span parameter sigma and cost parameter C. 
# The grid along with the ROC and other metrics is shown in the printed table.
# 
# The hyperparameter optimization can also be performed using Recall as a metric. 
# The code in the cell below uses the prSummary function for the summaryFunction argument for trainControl and sets the metric as Recall. 
# Execute this call and examine the results.

fitControl <- trainControl(method = "cv",
                           number = 5,
                           sampling = 'down',
                           returnResamp="all",
                           savePredictions = TRUE,
                           classProbs = TRUE,
                           summaryFunction = prSummary)
paramGrid <- expand.grid(sigma = c(0.002, 0.02, 0.2), C = c(0.5, 1.0, 2.0))

set.seed(1234)
svm_fit_inside_pr <- train(bad_credit ~ ., 
                           data = credit,  
                           method = "svmRadial", # Radil SVM model 
                           trControl = fitControl, 
                           tuneGrid = paramGrid, 
                           metric="Recall")
print(svm_fit_inside_pr)

# These results are rather poor. The average AUC is barely above 0.5. 
# It is best to stay with the model trained on ROC.
# 
# Given the optimal hyperparameters for the model trained with ROC, which features are the most important? 
# The code in the cell below computes and displays feature importance using the Caret varImp function. 
# Execute this code and examine the results.

options(repr.plot.width=8, repr.plot.height=6)
var_imp = varImp(svm_fit_inside_tw)
print(var_imp)
plot(var_imp)

# It is clear that at least two of the features should be removed from the dataset. 
# Execute the code in the cell below to use the dplyr select function to remove two features.

credit_reduced = select(credit, -time_in_residence, -dependents)

# Execute the code in the cell below to perform the cross validation grid search using ROC as the metric with the reduced feature set:
        
        fitControl <- trainControl(method = "cv",
                                   number = 5,
                                   sampling = 'down',
                                   returnResamp="all",
                                   savePredictions = TRUE,
                                   classProbs = TRUE,
                                   summaryFunction = twoClassSummary)
paramGrid <- expand.grid(sigma = c(0.002, 0.02, 0.2), C = c(0.5, 1.0, 2.0))

set.seed(1234)
svm_fit_inside_tw <- train(bad_credit ~ ., 
                           data = credit_reduced,  
                           method = "svmRadial", # Radil SVM model 
                           trControl = fitControl, 
                           tuneGrid = paramGrid, 
                           metric="ROC")
print(svm_fit_inside_tw)

# The results of the cross validation grid search with the reduced feature set are nearly the same as the first result. 
# Evidentially, pruning these features was the correct step. 
# This process can be continued, but will not be in this lab in the interest of reducing length.
# 
# To better understand the parameter sweep, execute the code in the cell below to create a chart.

options(repr.plot.width=8, repr.plot.height=6)
var_imp = varImp(svm_fit_inside_tw)
print(var_imp)
plot(var_imp)

# In is clear that some of the features are not important to model performance. 
# Execute the code in the cell below to prune the feature set using the dplyr select function:
        
credit_reduced = select(credit, -telephone, -property, -job_category)

# Execute the code in the cell below to perform the cross validation grid search using the reduced feature set and the ROC metric:
        
        fitControl <- trainControl(method = "cv",
                                   number = 5,
                                   sampling = 'down',
                                   returnResamp="all",
                                   savePredictions = TRUE,
                                   classProbs = TRUE,
                                   summaryFunction = twoClassSummary)
paramGrid <- expand.grid(sigma = c(0.002, 0.02, 0.2), C = c(0.5, 1.0, 2.0))

set.seed(1234)
svm_fit_inside_tw <- train(bad_credit ~ ., 
                           data = credit_reduced,  
                           method = "svmRadial", # Radil SVM model 
                           trControl = fitControl, 
                           tuneGrid = paramGrid, 
                           metric="ROC")
print(svm_fit_inside_tw)

# The results of the cross validation grid search with the reduced feature set are nearly the same as the first result. Evidentially, pruning these features was the correct step. 
# This process can be continued, but will not be in this lab in the interest of reducing length.
# 
# To better understand the parameter sweep, execute the code in the cell below to create a chart.

options(repr.plot.width=5, repr.plot.height=4)
trellis.par.set(caretTheme())
plot(svm_fit_inside_tw)  

# Notice that at the low end of the range of sigma values none of the cost values really matter. It is often the case with model selection that only one or two hyperparameters have much of an effect.
# 
# Finally, to verify that the model will generalize well it is time to perform the outside CV loop. 
# The code in the cell below defines a parameter grid with just the optimal hyperparameter values. 
# The CV then repeatedly fits the model with these hyperparameter values.
# 
# Execute this code, examine the result, and answer Question 3 on the course page.

paramGrid <- expand.grid(sigma = c(0.02), C = c(0.5))

set.seed(1234)
svm_fit_outside_tw <- train(bad_credit ~ ., 
                            data = credit_reduced,  
                            method = "svmRadial", # Radial SVM model 
                            trControl = fitControl, 
                            tuneGrid = paramGrid, 
                            trace = FALSE,
                            metric="ROC")

print_metrics = function(mod){
        means = c(apply(mod$resample[,1:3], 2, mean), sigma = mod$resample[1,4], C = mod$resample[1,5], Resample = 'Mean')
        stds = c(apply(mod$resample[,1:3], 2, sd), sigma = mod$resample[1,4], C = mod$resample[1,5], Resample = 'STD')
        out = rbind(mod$resample, means, stds)
        out[,1:3] = lapply(out[,1:3], function(x) round(as.numeric(x), 3))
        out
}
print_metrics(svm_fit_outside_tw)

# Examine these results. Notice that the standard deviation of the mean of the AUC are nearly an order of magnitude smaller than the mean. 
# This indicates that this model is likely to generalize well.
# 
# Note: The predict method can be used with this optimal model to classify unknown cases.

# Summary
# In this lab you have accomplished the following:
#         
#       1. Used a radial support vector machine to classify the cases of the iris data. 
#          This model produced quite good results.
#       2. Used 5 fold to find estimated optimal hyperparameters for an svmRadial model to classify credit risk cases. 
#          The model trained using ROC as the metric performed reasonably well.
#       3. Applied down-sampling of the majority cases to create a balanced training dataset and retrained and evaluated the model.
#       4. Applied feature importance was used for feature selection. 
#          The model created and evaluated with the reduced feature set had essentially the same performance as the model with more features.
#       5. Used the outer loop of the nested cross validation to demonstrate that the model is likely to generalize.

# ====================
# Processing end time
timeEnd = Sys.time()

# Processing date and total processing time
cat(paste("","Processing end date and time",date(),"","",sep="\n"))
paste("Total processing time =",round(difftime(timeEnd,timeStart), digits=2),"seconds",sep=" ")


# Stop writing to an output file
sink()

################
