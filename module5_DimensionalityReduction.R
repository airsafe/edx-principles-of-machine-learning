# Processing start time
sink('dim_reduction_output.txt')
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

# Dimensionality reduction with principal components
# Principal component analysis, or PCA, is an alternative to regularization and straight-forward feature elimination. 
# PCA is particularly useful for problems with very large numbers of features compared to the number of training cases. 
# For example, when faced with a problem with many thousands of features and perhaps a few thousand cases, PCA can be a good choice to reduce the dimensionality of the feature space.
# 
# PCA is one of a family of transformation methods that reduce dimensionality. 
# PCA is the focus here, since it is the most widely used of these methods.
# 
# The basic idea of PCA is rather simple: Find a linear transformation of the feature space which projects the majority of the variance onto a few orthogonal dimensions in the transformed space. 
# The PCA transformation maps the data values to a new coordinate system defined by the principal components. 
# Assuming the highest variance directions, or components, are the most informative, low variance components can be eliminated from the space with little loss of information.
# 
# The projection along which the greatest variance occurs is called the first principal component. 
# The next projection, orthogonal to the first, with the greatest variance is called the second principal component. 
# Subsequent components are all mutually orthogonal with decreasing variance along the projected direction.
# 
# Widely used PCA algorithms compute the components sequentially, starting with the first principal component. 
# This means that it is computationally efficient to compute the first several components from a very large number of features. 
# Thus, PCA can make problems with very large numbers of features computationally tractable.
# 
# Note: It may help your understanding to realize that principal components are a scaled version of the eigenvectors of the feature matrix. 
# The scale for each dimensions is given by the eigenvalues. 
# The eigenvalues are the fraction of the variance explained by the components.
# 
# A simple example
# To cement the concepts of PCA you will now work through a simple example. 
# This example is restricted to 2-d data so that the results are easy to visualize.
# 
# As a first step, execute the code in cell below to load the packages required for the rest of this notebook.
# 
# Note: If you are running in Azure Notebooks, make sure that you run the code in the setup.ipynb notebook at the start of you session to ensure your environment is correctly configured.

# Dimensionality reduction with principal components
# Principal component analysis, or PCA, is an alternative to regularization and straight-forward feature elimination. 
# PCA is particularly useful for problems with very large numbers of features compared to the number of training cases. 
# For example, when faced with a problem with many thousands of features and perhaps a few thousand cases, PCA can be a good choice to reduce the dimensionality of the feature space.
# 
# PCA is one of a family of transformation methods that reduce dimensionality. 
# PCA is the focus here, since it is the most widely used of these methods.
# 
# The basic idea of PCA is rather simple: Find a linear transformation of the feature space which projects the majority of the variance onto a few orthogonal dimensions in the transformed space. 
# The PCA transformation maps the data values to a new coordinate system defined by the principal components. 
# Assuming the highest variance directions, or components, are the most informative, low variance components can be eliminated from the space with little loss of information.
# 
# The projection along which the greatest variance occurs is called the first principal component. 
# The next projection, orthogonal to the first, with the greatest variance is called the second principal component. 
# Subsequent components are all mutually orthogonal with decreasing variance along the projected direction.
# 
# Widely used PCA algorithms compute the components sequentially, starting with the first principal component. 
# This means that it is computationally efficient to compute the first several components from a very large number of features. 
# Thus, PCA can make problems with very large numbers of features computationally tractable.
# 
# Note: It may help your understanding to realize that principal components are a scaled version of the eigenvectors of the feature matrix. 
# The scale for each dimensions is given by the eigenvalues. 
# The eigenvalues are the fraction of the variance explained by the components.
# 
# A simple example
# To cement the concepts of PCA you will now work through a simple example. 
# This example is restricted to 2-d data so that the results are easy to visualize.
# 
# As a first step, execute the code in cell below to load the packages required for the rest of this notebook.
# 
# Note: If you are running in Azure Notebooks, make sure that you run the code in the setup.ipynb notebook at the start of you session to ensure your environment is correctly configured.

## Import packages

if("ggplot2" %in% rownames(installed.packages()) == FALSE) 
{install.packages("ggplot2")}
library(ggplot2)

if("repr" %in% rownames(installed.packages()) == FALSE) 
{install.packages("repr")}
library(repr)

if("dplyr" %in% rownames(installed.packages()) == FALSE) 
{install.packages("dplyr")}
library(dplyr)

if("caret" %in% rownames(installed.packages()) == FALSE) 
{install.packages("caret")}
library(caret)

if("MASS" %in% rownames(installed.packages()) == FALSE) 
{install.packages("MASS")}
library(MASS)

if("ROCR" %in% rownames(installed.packages()) == FALSE) 
{install.packages("ROCR")}
library(ROCR)

if("pROC" %in% rownames(installed.packages()) == FALSE) 
{install.packages("pROC")}
library(pROC)

# options(repr.plot.width=4, repr.plot.height=4) # Set the initial plot area dimensions
# The code in the cell below simulates data from a bivariate Normal distribution. 
# The distribution is deliberately centered on {0,0} and with unit variance on each dimension. There is considerable covariance between the two dimensions leading to a covariance matrix:
        
#       cov(X) =	[1.0	0.6]
#                       [0.6	1.0]
#
# Given the covariance matrix 100 draws from this distribution are computed using the mvrnorm function from the R MASS package. 
# Execute this code:
        
set.seed(124)
cov = matrix(c(1.0, 0.6, 0.6, 1.0), nrow =2, ncol = 2)
mean = c(0.0, 0.0)

sample = data.frame(mvrnorm(n = 100, mu = mean, Sigma = cov))
names(sample) = c('x','y')
print(dim(sample))
head(sample)

# To get a feel for this data, execute the code in the cell below to display a plot and examine the result.

ggplot(sample, aes(x,y)) + geom_point()

# You can see that the data have a roughly elliptical pattern.
# The correlation between the two dimensions is also visible.
# 
# With the simulated data set created, it is time to compute the PCA model. 
# The code in the cell below computes the principle component model using the R prcomp function. 
# This function contains a list with multiple elements including the eigenvalues. 
# The eigenvalues can be scaled to compute the variance explained:
#
#       VE(X) = Var_X-component(X)/Var_X-total(X)
# 
# Notice that by construction:
#         
#         𝑉𝐸(𝑋)=∑𝑖=1𝑁𝑉𝐸𝑖(𝑋)=1.0
# 
# In other words, the sum of the variance explained for each component must add to the total variance or 1.0 for standardized data.
# 
# Execute this code and examine the result.

pca_mod = prcomp(sample)
pca_mod

# Notice that the standard deviation of the first component is several times larger than for the second component. This is exactly the desired result indicating the first principal component explains the majority of the variance of the sample data. Mathematically the components are the eigenvectors and the standard deviations are the eigenvalues of the data covariance matrix.
# 
# The code in the cell below computes and prints the scaled magnitude of the components. These scaled components must add to 1.0. Execute this code:
        
sdev_scaled = pca_mod$sdev**2/sum(pca_mod$sdev**2)
sdev_scaled

# The components are scaled by element wise multiplication with percent variance explained. Execute this code and examine the results.

scaled_pca = data.frame(matrix(c(0,0,0,0), nrow = 2, ncol = 2))
for(i in 1:2){
        scaled_pca[i,] = pca_mod$rotation[i,] * sdev_scaled
}
names(scaled_pca) = c('PC1','PC2')
str(scaled_pca)

# The two component vectors have their origins at [0,0}, and are quite different magnitude, and are pointing in different directions. 
# To better understand how the projections of the components relate to the data, execute the code to plot the data along with the principal components. 
# Execute this code:
        
## Find the slopes
s1 = data.frame(x = c(0.0, scaled_pca$PC1[1]), y = c(0.0, scaled_pca$PC1[2]))
s2 = data.frame(x = c(0.0, scaled_pca$PC2[1]), y = c(0.0, scaled_pca$PC2[2]))

## Plot the data with the PCs

ggplot(sample, aes(x,y)) + geom_point() +
        geom_line(data = s1, aes(x,y), color = 'red', size = 1) +
        geom_line(data = s2, aes(x,y), color = 'red', size = 1)

# Notice the the first principal component (the long red line) is along the direction of greatest variance of the data. This is as expected. 
# The short red line is along the direction of the second principal component. 
# The lengths of these lines are the variance in the directions of the projection.
# 
# The ultimate goal of PCA is to transform data to a coordinate system with the highest variance directions along the axes. 
# The transform function in the cell below computes the projections of the data onto the new coordinate frames using matrix multiplication. 
# Execute this code to apply the transform and plot the result:
        
        pca_transform = function(df, pca, ncomps){
                data.frame(as.matrix(df) %*% as.matrix(pca)[,1:ncomps])
        }

trans_sample = pca_transform(sample, scaled_pca, 2)
names(trans_sample) = c('x', 'y')
ggplot(trans_sample, aes(x,y)) + geom_point()

# Notice that the scale along these two coordinates are quite different. 
# The first principal component is along the horizontal axis. The range of values on this direction is in the range of about {−2.5,2.5}. 
# The range of values on the vertical axis or second principal component are only about {−0.2,0.3}. 
# It is clear that most of the variance is along the direction of the fist principal component.
# 
# Load Features and Labels
# Keeping the foregoing simple example in mind, it is time to apply PCA to some real data.
# 
# The code in the cell below loads the dataset which has the following preprocessing:
#         
#         1. Cleaning missing values.
#         2. Aggregating categories of certain categorical variables.
# 
# Execute the code in the cell below to load the dataset:
        
credit = read.csv('German_Credit_Preped.csv', header = TRUE)
credit[,'Customer_ID'] = NULL
credit$bad_credit <- ifelse(credit$bad_credit == 1, 'bad', 'good')
credit$bad_credit <- factor(credit$bad_credit, levels = c('bad', 'good'))
dim(credit)
str(credit)
#
# There are 20 features in this data set.
# 
# The prcomp function can only work with numeric matrices. 
# Therefore, the categorical features are dummy variable encoded. 
# Executed the code in the cell below to compute the encoding for the dummy variables.

dummies = dummyVars(bad_credit ~ ., data = credit)

# The code in the cell below to split the data set into test and training subsets and dummy variable encode the categorical features. 
# The Caret createDataPartion function is used to randomly split the dataset. 
# The perdict method dummy variable encodes the categorical features. 
# Execute this code and examine the result.

set.seed(1955)
## Randomly sample cases to create independent training and test data
partition = createDataPartition(credit[,'bad_credit'], times = 1, p = 0.7, list = FALSE)
training = credit[partition,] # Create the training feature sample
training_label = credit[partition, 'bad_credit'] # Subset training labels
training = predict(dummies, newdata = training) # transform categorical to dummy vars
dim(training)
test = credit[-partition,] # Create the test sample
test_label = credit[-partition, 'bad_credit'] # Subset training labels
test = predict(dummies, newdata = test) # transform categorical to dummy vars
dim(test)
head(training)

# Before preforming PCA all features must be zero mean and unit variance. 
# Failure to do so will result in biased computation of the components and scales. 
# The preProcess function from Caret is used to compute scaling of the training data. 
# The same scaling is applied to the test data. Execute the code in the cell below to scale the features.

num_cols = c('loan_duration_mo', 'loan_amount', 'payment_pcnt_income', 'age_yrs')
preProcValues <- preProcess(training[,num_cols], method = c("center", "scale"))

training[,num_cols] = predict(preProcValues, training[,num_cols])
test[,num_cols] = predict(preProcValues, test[,num_cols])
head(training[,num_cols])

# Compute principal components
# The code in the cell below computes the principal components for the training feature subset. Execute this code:

pca_credit = prcomp(training)

# Execute the code in the cell below to print the variance explained for each component and the sum of the variance explained:

# var_exp = pca_credit$sdev**2/sum(pca_credit$sdev**2)
# var_exp
# sum(var_exp)

# These numbers are a bit abstract. 
# However, you can see that the variance ratios are in descending order and that the sum is 1.0.
# 
# Execute the code in the cell below to create a plot of the explained variance vs. the component:

plot_scree = function(pca_mod){
        ## Plot as variance explained
        df = data.frame(x = 1:length(var_exp), y = var_exp)
        ggplot(df, aes(x,y)) + geom_line(size = 1, color = 'blue') +
        xlab('Component number') + ylab('Variance explained') +
        ggtitle('Scree plot of variance explained vs. \n Principal Component')
        }

plot_scree(pca_credit)

# This curve is often referred to as a scree plot. 
# Notice that the explained variance decreases rapidly until the 10th component and then slowly, thereafter. 
# The first few components explain a large fraction of the variance and therefore contain much of the explanatory information in the data. 
# The components with small explained variance are unlikely to contain much explanatory information. 
# Often the inflection point or 'knee' in the scree curve is used to choose the number of components selected.
# 
# Now it is time to create a PCA model with a reduced number of components. 
# The code in the cell below trains and fits a PCA model with 10 components, and then transforms the features using that model. 
# Execute this code.

## Compute first 10 PCA components 
pca_credit_10 = prcomp(training, rank = 10)

## Scale the eigenvalues
var_exp_10 = pca_credit_10$sdev**2/sum(pca_credit_10$sdev**2)
Nrow = nrow(pca_credit_10$rotation)
Ncol = ncol(pca_credit_10$rotation)
scaled_pca_10 = data.frame(matrix(rep(0, Nrow * Ncol), nrow = Nrow, ncol = Ncol))

## Scale the rotations
for(i in 1:Nrow){
        scaled_pca_10[i,] = pca_credit_10$rotation[i,] * var_exp_10[1:Ncol]
}

## Print the dimensions of the scalled rotations and the first component
dim(scaled_pca_10)
pca_credit_10$rotation[1:10,1]

# The scaled rotation matrix has dimensions of 61 rows and 10 columns. 
# You can see the first 10 elements of the first rotation. 
# Multiplying these numbers by the features rotate each row (case) to the new coordinate system.
# 
# Compute and evaluate a logistic regression model
# Next, you will compute and evaluate a logistic regression model using the features transformed by the first 10 principal components. 
# The code in the cell below performs the matrix multiplication between the features and on the

training_10 = training %*% as.matrix(scaled_pca_10)
dim(training_10)

# There are now 10 transformed features.
# 
# Now you will now construct and evaluate a logistic regression model by executing the code below:
        
## Construct a data frame with the transformed features and label
training_10 = data.frame(training_10)
training_10[,'bad_credit'] = training_label

## Create a weight vector for the training cases.
weights = ifelse(training_10$bad_credit == 'bad', 0.66, 0.34)

## Define and fit the logistic regression model
set.seed(5566)
logistic_mod_10 = glm(bad_credit ~ ., data = training_10, 
                      weights = weights, family = quasibinomial)
logistic_mod_10$coefficients

# Notice that there are now 10 regression coefficients, one for each component plus an intercept. 
# This number is in contrast to the 61 features in the dummy variable array.
# 
# In order to test the model, the test feature array must also be transformed. 
# Execute the code in the cell below to apply the PCA transformation to the test features:
        
test_10 = test %*% as.matrix(scaled_pca_10)
test_10 = data.frame(test_10)
test_10[,'bad_credit'] = test_label
dim(test_10)

# Execute the code in the cell below to score the model using the test data:

score_model = function(df, threshold){
        df$score = ifelse(df$probs < threshold, 'bad', 'good')
        df
}

test_10$probs = predict(logistic_mod_10, newdata = test_10, type = 'response')
test_10 = score_model(test_10, 0.5)
test_10[1:10, c('bad_credit','probs', 'score')]# Execute the code in the cell below to evaluate the 10 PCA component logistic regression model.
# 
# Then, answer Question 1 on the course page.

logistic.eval <- function(df){ 
        # First step is to find the TP, FP, TN, FN cases
        df$conf = ifelse(df$bad_credit == 'bad' & df$score == 'bad', 'TP',
                         ifelse(df$bad_credit == 'bad' & df$score == 'good', 'FN',
                                ifelse(df$bad_credit == 'good' & df$score == 'good', 'TN', 'FP')))
        
        # Elements of the confusion matrix
        TP = length(df[df$conf == 'TP', 'conf'])
        FP = length(df[df$conf == 'FP', 'conf'])
        TN = length(df[df$conf == 'TN', 'conf'])
        FN = length(df[df$conf == 'FN', 'conf'])

        ## Confusion matrix as data frame
        out = data.frame(Negative = c(TN, FN), Positive = c(FP, TP))
        row.names(out) = c('Actual Negative', 'Actual Positive')
        print(out)  
        
        # Compute and print metrics
        P = TP/(TP + FP)
        R = TP/(TP + FN)  
        F1 = 2*P*R/(P+R)  
        cat('\n')
        cat(paste('accuracy  =', as.character(round((TP + TN)/(TP + TN + FP + FN), 3)), '\n'))      
        cat(paste('precision =', as.character(round(P, 3)), '\n'))     
        cat(paste('recall    =', as.character(round(R, 3)), '\n'))
        cat(paste('F1        =', as.character(round(F1,3)),'\n'))
        
        roc_obj <- roc(df$bad_credit, df$probs)
        cat(paste('AUC       =', as.character(round(auc(roc_obj),3)),'\n'))
}

ROC_AUC = function(df){
        options(repr.plot.width=5, repr.plot.height=5)
        pred_obj = prediction(df$probs, df$bad_credit)
        perf_obj <- performance(pred_obj, measure = "tpr", x.measure = "fpr")
        AUC = performance(pred_obj,"auc")@y.values[[1]] # Access the AUC from the slot of the S4 object
        plot(perf_obj)
        abline(a=0, b= 1, col = 'red')
        text(0.8, 0.2, paste('AUC = ', as.character(round(AUC, 3))))
}

logistic.eval(test_10)
ROC_AUC(test_10)

# These results are reasonably good. 
# Recall, accuracy and AUC have reasonable values, however precision and F1 are low. 
# Is it possible that more PCA components are required to achieve a good model?

# Add more components to the model
# Now you will compute and evaluate a logistic regression model using the first 20 principal components. 
# You will compare this model to the one created with 10 principal components.
# 
# Execute the code below to transform the training features using the first 20 principal components.

## Compute first 10 PCA components 
pca_credit_20 = prcomp(training, rank = 20)

## Scale the eigenvalues
var_exp_20 = pca_credit_20$sdev**2/sum(pca_credit_20$sdev**2)
Nrow = nrow(pca_credit_20$rotation)
Ncol = ncol(pca_credit_20$rotation)
scaled_pca_20 = data.frame(matrix(rep(0, Nrow * Ncol), nrow = Nrow, ncol = Ncol))

## Scale the rotations
for(i in 1:Nrow){
        scaled_pca_20[i,] = pca_credit_20$rotation[i,] * var_exp_20[1:Ncol]
}

## Print the dimensions of the scalled rotations and the first component
dim(scaled_pca_20)

# There are now 20 components in the PCA model.

# The code in the cell below computes the transformed feature set and creates a logistic regression model from this feature set. 
# Execute this code.

## Construct a data frame with the transformed features and label
training_20 = training %*% as.matrix(scaled_pca_20)
training_20 = data.frame(training_20)
training_20[,'bad_credit'] = training_label

weights = ifelse(training_20$bad_credit == 'bad', 0.66, 0.34)

## Define and fit the logistic regression model
set.seed(5566)
logistic_mod_20 = glm(bad_credit ~ ., data = training_20, 
        weights = weights, family = quasibinomial)
logistic_mod_20$coefficients

# The code in the cell below scores the logistic regression model and displays performance metrics, the ROC curve, and the AUC. 
# Execute this code and examine the result.

## Create the transformed test dataset
test_20 = test %*% as.matrix(scaled_pca_20)
test_20 = data.frame(test_20)
test_20[,'bad_credit'] = test_label

## Score the model
test_20$probs = predict(logistic_mod_20, newdata = test_20, type = 'response')
test_20 = score_model(test_20, 0.5)

## Evaluate the model
logistic.eval(test_20)
ROC_AUC(test_20)

# The metrics for the 20 component model are nearly the same as for the 10 component model. 
# It appears that 10 components is enough to represent the information in the feature set.
# 
# Summary
# In this lab you have applied principal component analysis to dimensionality reduction for supervised machine learning. 
# The first components computed contain most of the available information. 
# When faced with large number of features, PCA is an effective way to make supervised machine learning models tractable.
# 
# Specifically in this lab you have:
#         
#       1. Computed PCA models with different numbers of components.
#       2. Compared logistic regression models with different numbers of components. 
#          In this case, using 10 components produced a good model. 
#          Extending this to 20 components gained little if anything. 
#          In summary the dimensionality of the original 61 dummy variable array to just 10 components.


# ====================
# Processing end time
timeEnd = Sys.time()

# Processing date and total processing time
cat(paste("","Processing end date and time",date(),"","",sep="\n"))
paste("Total processing time =",round(difftime(timeEnd,timeStart), digits=2),"seconds",sep=" ")


# Stop writing to an output file
sink()

################