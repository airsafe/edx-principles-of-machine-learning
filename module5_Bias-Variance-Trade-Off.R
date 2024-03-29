# Processing start time
sink('bias-variance-trade-off_output.txt')
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

# Regularization and the bias-variance trade-off
# Over-fitting is a constant danger with machine learning models. 
# Over-fit models fit the training data well. 
# However, an over-fit model will not generalize. 
# A model that generalizes is a model which exhibits good performance on data cases beyond the ones used in training. 
# Models that generalize will be useful in production.
# 
# As a general rule, an over-fit model has learned the training data too well. 
# The over-fitting likely involved learning noise present in the training data. 
# The noise in the data is random and uninformative. 
# When a new data case is presented to such a model it may produce unexpected results since the random noise will be different.
# 
# So, what is one to do to prevent over-fitting of machine learning models? 
# The most widely used set of tools for preventing over-fitting are known as regularization methods. 
# Regularization methods take a number of forms, but all have the same goal, to prevent over-fitting of machine learning models.
# 
# Regularization is not free however. 
# While regularization reduces the variance in the model results, it introduces bias. 
# Whereas, an over-fit model exhibits low bias but the variance is high. 
# The high variance leads to unpredictable results when the model is exposed to new data cases. 
# On the other hand, the stronger the regularization of a model the lower the variance, but the greater the bias. 
# This all means that when applying regularization you will need to contend with the bias-variance trade-off.
# 
# To better understand the bias variance trade-off consider the following examples of extreme model cases:
#         
#       * If the prediction for all cases is just the mean (or median), variance is minimized. 
#         The estimate for all cases is the same, so the bias of the estimates is zero. 
#         However, there is likely considerable variance in these estimates.
#       * On the other hand, consider what happens when the data are fit with a kNN model with k=1. 
#         The training data will fit this model perfectly, since there is one model coefficient per training data point. 
#         The variance will be low. 
#         However, the model will have considerable bias when applied to test data.
#
# In either case, these extreme models will not generalize well and will exhibit large errors on any independent test data. 
# Any practical model must come to terms with the trade-off between bias and variance to make accurate predictions.
# 
# To better understand this trade-off you should consider the example of the mean square error, which can be decomposed into its components. 
# The mean square error can be written as:

#        delta_x = E[(Y - f_hat(X))^2 = (1/N)*Sum from 1 to N (y_i - f_hat(x_i))^2]

# Where,  Y =  the label vector.
# X = the feature matrix.
# f_hat(x) =  the trained model.

# Expanding the representation of the mean square error:
#        delta_x = (E[(f_hat(x)) - f_hat(X))^2 = E[(f_hat(x) - E[f_hat(x)])^2] + sigma^2
#	 delta_x = Bias^2 + Variance + Irreducible Error

# Where,  Y =  the label vector.
# X = the feature matrix.
# f_hat(x) =  the trained model.

# Expanding the representation of the mean square error:
#        delta_x = (E[(f_hat(x)) - f_hat(X))^2 = E[(f_hat(x) - E[f_hat(x)])^2] + sigma^2
#	 delta_x = Bias^2 + Variance + Irreducible Error

# Study this relationship. Notice that as regularization reduces variance, bias increases. 
# The irreducible error will remain unchanged. 
# Regularization parameters are chosen to minimize deta_x In many cases, this will prove challenging.
# 
# Load a data set
# With the above bit of theory in mind, it is time to try an example. 
# In this example you will compute and compare linear regression models using different levels and types of regularization.
# 
# Execute the code in the cell below to load the packages required for the rest of this notebook.

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

library(caret)
if("glmnet" %in% rownames(installed.packages()) == FALSE) 
{install.packages("glmnet")}
library(glmnet)

options(repr.plot.width=4, repr.plot.height=4) # Set the initial plot area dimensions
# The code in the cell below loads the dataset which was prepared in a previous lab. 
# Execute this code and ensure that the expected columns are present.

auto_prices = read.csv('Auto_Prices_Preped.csv')
print(dim(auto_prices))
names(auto_prices)

# Notice that there are 195 cases and two label columns.

# Split the dataset
# You must now create randomly sampled training and test data sets. 
# The createDataPartition function from the R caret package is used to create indices for the training data sample. 
# In this case 75% of the data will be used for training the model. 
# Since this data set is small, only 48 cases will be included in the test dataset. 
# Execute this code and note the dimensions of the resulting data frame.

set.seed(1955)
## Randomly sample cases to create independent training and test data
partition = createDataPartition(auto_prices[,'fuel.type'], times = 1, p = 0.75, list = FALSE)
training = auto_prices[partition,] # Create the training sample
dim(training)
test = auto_prices[-partition,] # Create the test sample
dim(test)

# Scale numeric features
# Numeric features must be rescaled so they have a similar range of values. 
# Rescaling prevents features from having an undue influence on model training simply because then have a larger range of numeric variables.
# 
# The code in the cell below uses the preProcess function from the caret function. 
# The processing is as follows:
#         
#       1. The preprocessing model object is computed. 
#          In this case the processing includes centering and scaling the numeric feature. 
#          Notice that this model is fit only to the training data.
#       2. The scaling is applied to both the test and training partitions.
# 
# Execute the code.

num_cols = c('wheel.base', 'length', 'width', 'height', 'curb.weight', 'engine.type', 
             'num.of.cylinders', 'engine.size', 'fuel.system', 'bore', 'stroke', 
             'compression.ratio', 'horsepower', 'peak.rpm', 'city.mpg')
preProcValues <- preProcess(training[,num_cols], method = c("center", "scale"))

training[,num_cols] = predict(preProcValues, training[,num_cols])
test[,num_cols] = predict(preProcValues, test[,num_cols])
head(training[,num_cols])

# A first linear regression model
# To create a baseline for comparison, you will first create a model using all 45 features and no regularization. 
# In the terminology used before this model has high variance and low bias. In other words, this model is over-fit.
# 
# The code in the cell below should be familiar. 
# In summary, it performs the following processing:
#         
#       1. Define and train the linear regression model using the training features and labels. 
#          The model is defined using the R modeling language.
#       2. Score the model using the test feature set.
#       3. Compute and display key performance metrics for the model using the test feature set.
#       4. Plot a histogram of the residuals of the model using the test partition.
#       5. Plot a Q-Q Normal plot of the residuals of the model using the test partition.
#       6. Plot the residuals of the model vs. the predicted values using the test partition.
# 
# Execute this code and examine the results for the linear regression model.

print_metrics = function(lin_mod, df, score, label){
        resids = df[,label] - score
        resids2 = resids**2
        N = length(score)
        r2 = as.character(round(summary(lin_mod)$r.squared, 4))
        adj_r2 = as.character(round(summary(lin_mod)$adj.r.squared, 4))
        cat(paste('Mean Square Error      = ', as.character(round(sum(resids2)/N, 4)), '\n'))
        cat(paste('Root Mean Square Error = ', as.character(round(sqrt(sum(resids2)/N), 4)), '\n'))
        cat(paste('Mean Absolute Error    = ', as.character(round(sum(abs(resids))/N, 4)), '\n'))
        cat(paste('Median Absolute Error  = ', as.character(round(median(abs(resids)), 4)), '\n'))
        cat(paste('R^2                    = ', r2, '\n'))
        cat(paste('Adjusted R^2           = ', adj_r2, '\n'))
}

hist_resids = function(df, score, label, bins = 10){
        options(repr.plot.width=4, repr.plot.height=3) # Set the initial plot area dimensions
        df$resids = df[,label] - score
        bw = (max(df$resids) - min(df$resids))/(bins + 1)
        ggplot(df, aes(resids)) + 
                geom_histogram(binwidth = bw, aes(y=..density..), alpha = 0.5) +
                geom_density(aes(y=..density..), color = 'blue') +
                xlab('Residual value') + ggtitle('Histogram of residuals')
}

resids_qq = function(df, score, label){
        options(repr.plot.width=4, repr.plot.height=3.5) # Set the initial plot area dimensions
        df$resids = df[,label] - score
        ggplot() + 
                geom_qq(data = df, aes(sample = resids)) + 
                ylab('Quantiles of residuals') + xlab('Quantiles of standard Normal') +
                ggtitle('QQ plot of residual values')
}

resid_plot = function(df, score, label){
        df$score = score
        df$resids = df[,label] - score
        ggplot(df, aes(score, resids)) + 
                geom_point() + 
                ggtitle('Residuals vs. Predicted Values') +
                xlab('Predicted values') + ylab('Residuals')
}

lin_mod = lm(log_price ~ make + fuel.type + aspiration + num.of.doors + body.style +
                     drive.wheels + engine.location + wheel.base + length +
                     width + height + curb.weight + num.of.cylinders +
                     engine.size + bore + stroke + compression.ratio +
                     horsepower + peak.rpm + city.mpg, data = training)

score = predict(lin_mod, newdata = test)
print_metrics(lin_mod, test, score, label = 'log_price')      
hist_resids(test, score, label = 'log_price')   
resids_qq(test, score, label = 'log_price')
resid_plot(test, score, label = 'log_price')

# Overall these results are reasonably good. 
# The error metrics are relatively small. 
# Further, the distribution of the residuals is a bit skewed, but otherwise well behaved. 
# There is one notable outlier.

# Create model matrix
# To explore the bias-variance trade-off for l1 and l2 regularization, you work with the R glmnet model. 
# The glmnet model will not work with data frames. 
# Rather, this model function requires a numeric matrix for the training features and a vector of labels.
# 
# To create model matrix the code in the cell below uses the dummyVars function from the caret package. 
# A predict method is applied to create numeric model matrices for training and test. 
# Execute the code and examine the resulting matrix.

cols = c('make', 'fuel.type', 'aspiration', 'num.of.doors', 'body.style',
         'drive.wheels', 'engine.location', 'wheel.base', 'length',
         'width', 'height', 'curb.weight', 'num.of.cylinders', 'engine.size', 
         'bore', 'stroke', 'compression.ratio', 'horsepower', 
         'peak.rpm', 'city.mpg', 'log_price')

dummies <- dummyVars(log_price ~ ., data = auto_prices[,cols])
training_dummies = predict(dummies, newdata = training[,cols])
print(dim(training_dummies))
head(training_dummies)

# Notice that there is now one dummy variable for each category of the categorical variables. 
# Only one dummy variable is coded with a one for each set of categories. 
# This is known as one hot encoding. 
# By using numeric dummy variable, the entire training feature array is now numeric.
# 
# Execute the code in the cell below to encode the test features as a dummy variable matrix.

test_dummies = predict(dummies, newdata = test[,cols])

# Apply l2 regularization
# Now, you will apply l2 regularization to constrain the model parameters. 
# Constraining the model parameters prevent over-fitting of the model. 
# This method is also known as Ridge Regression.
# 
# But, how does this work? l2 regularization applies a penalty proportional to the l2 or Euclidean norm of the model weights to the loss function. 
# For linear regression using squared error as the metric, the total loss function is the sum of the squared error and the regularization term. 
# The total loss function can then be written as follows:

# 𝐽(𝛽)=||𝐴𝛽+𝑏||2+𝜆||𝛽||2

# Where the penalty term on the model coefficients, 𝛽𝑖, is written:

# 𝜆||𝛽||2=𝜆(𝛽21+𝛽22+…+𝛽2𝑛)12=𝜆(∑𝑖=1𝑛𝛽2𝑖)12

# We call ||𝛽||2 the l2 norm of the coefficients, since we raise the weights of each 
# coefficient to the power of 2, sum the squares, and then raise the sum to the power of 12.

# Drawing
# **Geometric view of l2 regularization**
#         Notice that for a constant value of the l2 norm, the values of the model parameters 𝐵1 and 𝐵2 are related. 
# The Euclidean or l2 norm of the coefficients is shown as the dotted circle. 
# The constant value of the l2 norm is a constant value of the penalty. 
# Along this circle the coefficients change in relation to each other to maintain a constant l2 norm. 
# For example, if 𝐵1 is maximized then 𝐵2∼0, or vice versa. It is important to note that l2 regularization is a soft constraint. 
# Coefficients are driven close to, but likely not exactly to, zero.

# With this bit of theory in mind, it is time to try an example of l2 regularization. 
# The code in the cell below performs the following processing:
#         
# 1. Constructs a glmnet model using the feature matrix and vector of labels with the following arguments:
#         * The nlambda argument determines the number of regularization parameters to be tested.
#         * alpha determines the weighting between l1 (alpha = 1) and l2 regularization (alpha = 0).
#         * A gaussian distribution family is used for the response since this is a regression problem and the residuals are expected to have a Gaussian or Normal distribution.
# 2. A plot is created of the model parameters vs. the regularization constraint hyperparameter, lambda.
# 
# Execute this code and examine the results.
#         
options(repr.plot.width=6, repr.plot.height=6) # Set the initial plot area dimensions

glmnet_mod_l2 = glmnet(x = training_dummies, y = training[,'log_price'], 
                       nlambda = 20, alpha = 0, family = 'gaussian')
plot(glmnet_mod_l2, xlab = 'Inverse of regulariation')

# Next, execute the code in the cell below to compute a cross validation of the model with different regularization hyperparameters and display the result. 
# Cross validation is discussed in another lab.

cv_fit = cv.glmnet(x = training_dummies, y = training[,'log_price'], 
                   nlambda = 20, alpha = 0, family = 'gaussian')
plot(cv_fit)

# Examine these results.
# 
# The first plot shows the value of each model coefficient vs. the regularization hyperparameter. 
# The hyperparamter increases to the left. 
# On the right the hyperparameter is small and the model is under-constrained. 
# The model parameters have a wide range of values. 
# On the left the regularization hyperparameter is at a maximum and the model coefficients are constrained to near zero. 
# Notice how each coefficient value smoothly decrease toward zero. 
# There are a few coefficients which increase in value for some part of their trajectories. 
# This behavior results from the fact that l2 regularization is a soft constraint on the coefficient values.
# 
# The second plot shows the RMSE from the cross validation vs. the log of the hyperparameter. 
# Notice that in this chart the maximum of the hyperparameter is on the right, following the usual convention. 
# As the regularization is increased to the right, model variance decreases but with higher bias. 
# Less regularization produces a model with less bias but greater variance. 
# The dotted vertical lines indicate that the optimal hyperparameter is at the low end of the range.
# 
# Next, you will evaluate the model using the best l2 regularization hyperparameter discovered above. 
# The code in the cell below computes predicted values from the optimal l2 regression model using the test data. 
# The predict method for glmnet returns predictions for each value of the hyperparameter. 
# In this case the 18th value is chosen based on the analysis above. 
# Performance metrics and diagnostic plots are then displayed.
# 
# Execute the code and answer Question 1 on the course page.

print_metrics_glm = function(df, score, label){
        resids = df[,label] - score
        resids2 = resids**2
        N = length(score)
        SSR = sum(resids2)
        SST = sum((mean(df[,label]) - df[,label])**2)
        r2 = as.character(round(1 - SSR/SST, 4))
        cat(paste('Mean Square Error      = ', as.character(round(sum(resids2)/N, 4)), '\n'))
        cat(paste('Root Mean Square Error = ', as.character(round(sqrt(sum(resids2)/N), 4)), '\n'))
        cat(paste('Mean Absolute Error    = ', as.character(round(sum(abs(resids))/N, 4)), '\n'))
        cat(paste('Median Absolute Error  = ', as.character(round(median(abs(resids)), 4)), '\n'))
        cat(paste('R^2                    = ', r2, '\n'))
}

score = predict(glmnet_mod_l2, newx = test_dummies)[,18]

print_metrics_glm(test, score, 'log_price')
hist_resids(test, score, label = 'log_price')   
resids_qq(test, score, label = 'log_price')
resid_plot(test, score, label = 'log_price')

# Compare the error metrics achieved to those of the un-regularized model. 
# The error metrics for the regularized model are somewhat better. 
# This fact, indicates that the regularized model generalizes better than the un-regularized model. 
# Notice also that the residuals are a bit closer to Normally distributed than for the un-regularized model.
# 
# Apply l1 regularizaton
# Regularization can be performed using norms other than l2. 
# The l1 regularization or Lasso method limits the sum of the absolute values of the model coefficients. 
# The l1 norm is sometime know as the Manhattan norm, since distance are measured as if you were traveling on a rectangular grid of streets. 
# This is in contrast to the l2 norm that measures distance 'as the crow flies'.
# 
# We can compute the l1 norm of the model coefficients as follows:
        
# ||𝛽||1=(|𝛽1|+|𝛽2|+…+|𝛽𝑛|)=(∑𝑖=1𝑛|𝛽𝑖|)1
# where |𝛽𝑖| is the absolute value of 𝛽𝑖.

# The l1 norm is constrained by the sum of the absolute values of the coefficients. 
# This fact means that values of one parameter highly constrain another parameter. 
# The dotted line in the figure above looks as though someone has pulled a rope or lasso around pegs on the axes. 
# This behavior leads the name lasso for l1 regularization.
# 
# Notice that in the figure above that if  𝐵1=0  then  𝐵2  has a value at the limit, or vice versa
# . In other words, using a l1 norm constraint forces some weight values to zero to allow other coefficients to take non-zero values. 
# Thus, you can think of the l1 norm constraint knocking out some weights from the model altogether. 
# In contrast to l2 regularization, l1 regularization does drive some coefficients to exactly zero.
# 
# The code in the cell below computes l1 regularized or lasso regression over a grid of regularization values. 
# The alpha hyperparameter of the glmnet model is a pure l1 regularization model. 
# Execute the code and examine the results.

options(repr.plot.width=6, repr.plot.height=6) # Set the initial plot area dimensions

glmnet_mod_l1 = glmnet(x = training_dummies, y = training[,'log_price'], 
                       nlambda = 20, alpha = 1, family = 'gaussian')
plot(glmnet_mod_l1, xlab = 'Inverse of regulariation')

cv_fit = cv.glmnet(x = training_dummies, y = training[,'log_price'], 
                   nlambda = 20, alpha = 1, family = 'gaussian')
plot(cv_fit)

# The two plots created are the same types as used for the l2 regularization example.
# 
# The first plot shows the value of each model coefficient vs. the regularization hyperparameter. 
# The hyperparamter increases to the left. On the right the hyperparameter is small and the model is under-constrained. 
# The model parameters have a wide range of values. 
# On the left the regularization hyperparameter is at a maximum and the model coefficients are constrained to near zero. 
# Notice how the coefficient values are abruptly driven to zero as the hyperparameter increases. 
# There are a few coefficients which increase in value for some part of their trajectories. 
# These increases are abrupt as well, driven by another coefficient becoming zero. 
# This behavior results from the fact that l1 regularization is a hard constraint on the coefficient values.
# 
# The second plot shows the RMSE from the cross validation vs. the log of the hyperparameter. 
# Notice that in this chart the maximum of the hyperparameter is on the right, following the usual convention. 
# As the regularization is increased to the right, model variance decreases but with higher bias. 
# Less regularization produces a model with less bias but greater variance. 
# The dotted vertical lines indicate that the optimal hyperparameter is at the middle of the range.
# 
# Next, you will evaluate the model using the best l1 regularization parameter discovered above. 
# The code in the cell below computes predicted values from the optimal l1 regression model using the test data. 
# The predict method for glmnet returns predictions for each value of the hyperparameter. 
# In this case the 13th value is chosen based on the analysis above. 
# Performance metrics and diagnostic plots are then displayed.
# 
# Execute the code and answer Question 2 on the course page.

score = predict(glmnet_mod_l1, newx = test_dummies)[,13]
print_metrics_glm(test, score, 'log_price')
hist_resids(test, score, label = 'log_price')   
resids_qq(test, score, label = 'log_price')
resid_plot(test, score, label = 'log_price')

# Compare the error metrics achieved to those of the un-regularized model. 
# The error metrics for the regularized model are somewhat better. 
# This fact, indicates that the regularized model generalizes better than the un-regularized model. 
# Notice also that the residuals are a bit closer to Normally distributed than for the un-regularized model.
# 
# Summary
# In this lab you have explored the basics of regularization. 
# Regularization can prevent machine learning models from being over-fit. 
# Regularization is required to help machine learning models generalize when placed in production. 
# Selection of regularization strength involves consideration of the bias-variance trade-off. 
# As the regularization is increased, model variance decreases but with higher bias. 
# Less regularization produces a model with less bias but greater variance.
# 
# L2 and l1 regularization constrain model coefficients to prevent over-fitting. 
# L2 regularization constrains model coefficients using a Euclidean norm. 
# L2 regularization can drive some coefficients toward zero, usually not to zero. 
# On the other hand, l1 regularization can drive model coefficients to zero.
# 
# An optimal model can use a weighted mix of l1 and l2 regularization.


# ====================
# Processing end time
timeEnd = Sys.time()

# Processing date and total processing time
cat(paste("","Processing end date and time",date(),"","",sep="\n"))
paste("Total processing time =",round(difftime(timeEnd,timeStart), digits=2),"seconds",sep=" ")


# Stop writing to an output file
sink()

################
