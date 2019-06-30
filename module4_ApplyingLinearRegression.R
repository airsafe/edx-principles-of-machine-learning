# Processing start time
sink('applying_linear_regression_output.txt')
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

# Applications of Regression
# Overview
# In this lab you will apply regression to some realistic data. 
# In this lab you will work with the automotive price dataset. 
# Your goal is to construct a linear regression model to predict the price of automobiles from their characteristics.
# 
# In this lab will learn to:
#         
# 1. Use categorical data with R machine learning models.
# 2. Apply transformations to features and labels to improve model performance.
# 3. Compare regression models to improve model performance.
# 
# Load the dataset
# As a first, step you will load the dataset into the notebook environment.
# 
# First, execute the code in the cell below to load the packages you will need to run the rest of this notebook.
# 
# Note: If you are running in Azure Notebooks, make sure that you run the code in the setup.ipynb notebook at the start of you session to ensure your environment is correctly configured.

## Import packages
library(ggplot2)
library(repr)
library(dplyr)

if("caret" %in% rownames(installed.packages()) == FALSE)
{install.packages("caret")}
library(caret)

options(repr.plot.width=4, repr.plot.height=4) # Set the initial plot area dimensions

# The code in the cell below loads the dataset which was prepared from the Data Preparation lab. 
# Execute this code and ensure that the expected columns are present.

auto_prices = read.csv('Auto_Prices_Preped.csv')
names(auto_prices)

# Notice that both the label, price and the logrithmically transformed versions are included.
# 
# As a next step, execute the code in the cell below to display and examine the first few rows of the dataset.

head(auto_prices)

# Notice that there are both numeric and categorical features.

# Split the dataset
# You must now create randomly sampled training and test data sets. 
# The createDataPartition function from the R caret package is used to create indices for the training data sample. 
# In this case 75% of the data will be used for training the model. 
# Since this data set is small only 48 cases will be included in the test dataset. 
# Execute this code and note the dimensions of the resulting data frame.
# 
# Note The createDataPartition function allows you to balance categorical cases, creating stratified sub-samples. 
# For example, choosing fuel.type as the stratification creates sub-samples with equal numbers of cars with each fuel type.

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
# 1. The preprocessing model object is computed. 
# In this case the processing includes centering and scaling the numeric feature. Notice that this model is fit only to the training data.
# 2. The scaling is applied both the test and training partitions.
# 
# Execute the code.

num_cols = c('curb.weight', 'horsepower', 'city.mpg')
preProcValues <- preProcess(training[,num_cols], method = c("center", "scale"))

training[,num_cols] = predict(preProcValues, training[,num_cols])
test[,num_cols] = predict(preProcValues, test[,num_cols])
head(training[,num_cols])

# These three numeric features are now scaled.
# 
# Construct the linear regression model
# With data prepared and split into training and test subsets, you will now compute the linear regression model. 
# There are 28 features, so the model will require at least 28 coefficients. 
# The equation for such a multiple regression problem can be written as:
#         
#         𝑦̂ =𝑓(𝑥⃗ )=𝛽⃗ ⋅𝑥⃗ +𝑏=𝛽1𝑥1+𝛽2𝑥2+⋯+𝛽𝑛𝑥𝑛+𝑏
# 
# where;
# 𝑦̂  are the predicted values or scores,
# 𝑥⃗  is the vector of feature values with components {𝑥1,𝑥2,⋯,𝑥𝑛,
# 𝛽⃗  is vector of model coefficients with components {𝛽1,𝛽2,⋯,𝛽𝑛,
# 𝑏 is the intercept term, if there is one.
#                 
# You can think of the linear regression function 𝑓(𝑥⃗ ) as the dot product between the beta vector 𝛽⃗  and the feature vector 𝑥⃗ , plus the intercept term 𝑏.
#                 
# In R models are defined by an equation using the ∼ symbol to mean modeled by. In summary, the variable to be modeled is always on the left. The features are listed on the right. 
# This basic scheme can be written as shown here.
#         𝑙𝑎𝑏𝑒𝑙∼𝑓𝑒𝑎𝑡𝑢𝑟𝑒𝑠
# 
# Categorical and numerical features can be combined in the model formula. 
# Features not specified in the formula are ignored.
#                 
# The code in the cell below uses the R lm function to compute the model coefficients and create a model object. 
# Execute this code.
                
## define and fit the linear regression model
lin_mod = lm(log_price ~ curb.weight + horsepower + city.mpg + fuel.type + aspiration +
        body.style + drive.wheels + num.of.cylinders, data = training)

# The model has been fit to the training data. Execute the code in the cell below to examine the value of the intercept term and coefficients.
                
summary(lin_mod)$coefficients

# You can interpret these coefficients as follows:
#                         
# 1. The intercept is the mean of the (log) price with respect to the (scaled) features.
# 2. The coefficients numeric features are the change in the (log) price for a unit change in the (scaled) feature.
# 3. The coefficients of the categorical features are a bit harder to understand. R uses a technique known at the method of contrasts. 
#         The intercept is computed using one, arbitrarily chosen, category. 
#         The coefficients for other categories are the differences, or contrasts, with the intercept term for each category. 
#         This means that there are 𝑛−1 coefficients for each categorical feature with 𝑛 categories.
# 
# Now, answer Question 1 on the course page.
                
# Evaluate the model
# You will now use the test dataset to evaluate the performance of the regression model.
# As a first step, execute the code in the cell below to compute and display various performance metrics and examine the results. 
# Then, answer Question 2 on the course page.
                
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
                
score = predict(lin_mod, newdata = test)
print_metrics(lin_mod, test, score, label = 'log_price')      
                
# At first glance, these metrics look promising. 
# The RMSE, MAE and median absolute error are all small and in a similar range. 
# However, notice that the 𝑅2 and 𝑅2𝑎𝑑𝑗 are somewhat differ
# ent. This model has a large number of parameters compared to the number of cases available. 
# This result indicates that the model may be over-fit and might not generalize well.
#                 
# To continue the evaluation of the model performance, execute the code in the cell below to display a histogram of the residuals.
              
hist_resids = function(df, score, label, bins = 10){
        options(repr.plot.width=4, repr.plot.height=3) # Set the initial plot area dimensions
        df$resids = df[,label] - score
        bw = (max(df$resids) - min(df$resids))/(bins + 1)
        ggplot(df, aes(resids)) + 
                geom_histogram(binwidth = bw, aes(y=..density..), alpha = 0.5) +
                geom_density(aes(y=..density..), color = 'blue') +
                xlab('Residual value') + ggtitle('Histogram of residuals')
}
hist_resids(test, score, label = 'log_price')   
# ggsave("residual_histogram.png") # Save histogram as png
# This histogram shows that the residuals are in a small range. 
# However, there is some noticeable skew in the distribution, including an outlier.
#                 
# Next, execute the code in the cell below to display the Q-Q Normal plot.
             
resids_qq = function(df, score, label){
        options(repr.plot.width=4, repr.plot.height=3.5) # Set the initial plot area dimensions
        df$resids = df[,label] - score
        ggplot() + 
                geom_qq(data = df, aes(sample = resids)) + 
                ylab('Quantiles of residuals') + xlab('Quantiles of standard Normal') +
                ggtitle('QQ plot of residual values')
}

resids_qq(test, score, label = 'log_price') 
# ggsave("qq_plot.jpg") # Save Q-Q plot

# As with the histogram, the Q-Q Normal plot indicates the residuals are close to Normally distributed, show some skew (deviation from the straight line). 
# This is particularly for large positive residuals.
#                 
# There is one more diagnostic plot. Execute the code in the cell below to display the plot of residuals vs. predicted values.
                
resid_plot = function(df, score, label){
        df$score = score
        df$resids = df[,label] - score
        ggplot(df, aes(score, resids)) + 
                geom_point() + 
                ggtitle('Residuals vs. Predicted Values') +
                xlab('Predicted values') + ylab('Residuals')
}

 resid_plot(test, score, label = 'log_price')
# ggsave("log_price_residual_plog.png") # Save log price residual plot
 
# This plot looks reasonable. The residual values appear to have a fairly constant dispersion as the predicted value changes. 
# A few large residuals are noticeable, particularly on the positive side.
#                 
# But, wait! This residual plot is for the log of the auto price. 
# What does the plot look like when transformed to real prices? Execute the code in the cell below to find out.
                
score_untransform = exp(score)
resid_plot(test, score_untransform, label = 'price')

# Notice that the untransformed residuals show a definite trend. 
# The dispersion of the residuals has a cone-like pattern increasing to the right. 
# The regression model seems to do a good job of predicting the price of low cost cars, 
# but becomes progressively worse as the price of the car increases.
                
# Summary
# In this lesson you have done the following in the process of constructing and evaluating a multiple linear regression model:
#                         
# 1. Transformed the label value to make it more symmetric and closer to a Normal distribution.
# 2. Aggregated categories of a categorical variable to improve the statistical representation.
# 3. Scaled the numeric features.
# 4. Fit the linear regression model using the lm function. A categorical feature with  𝑛  categories produces  𝑛−1  model coefficients.
# 5. Evaluated the performance of the model using both numeric and graphical methods.
# 
# It is clear from the outcome of the performance evaluation that this model needs to be improved. As it is, the model shows poor generalization for high cost autos.

# ====================
# Processing end time
timeEnd = Sys.time()

# Processing date and total processing time
cat(paste("","Processing end date and time",date(),"","",sep="\n"))
paste("Total processing time =",round(difftime(timeEnd,timeStart), digits=2),"seconds",sep=" ")


# Stop writing to an output file
sink()

################
