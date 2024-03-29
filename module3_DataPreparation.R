# Processing start time
sink('data_preparation_output.txt')
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

# Data Preparation for Machine Learning
# Data preparation is a vital step in the machine learning pipeline. Just as visualization is necessary to understand the relationships in data, proper preparation or data munging is required to ensure machine learning models work optimally.
# 
# The process of data preparation is highly interactive and iterative. A typical process includes at least the following steps:
#         
#       1. Visualization of the dataset to understand the relationships and identify possible problems with the data.
#       2. Data cleaning and transformation to address the problems identified. 
#          It many cases, step 1 is then repeated to verify that the cleaning and transformation had the desired effect.
#       3. Construction and evaluation of a machine learning models. 
#          Visualization of the results will often lead to understanding of further data preparation that is required; going back to step 1.
# 
# In this lab you will learn the following:
#         
#       * Recode character strings to eliminate characters that will not be processed correctly.
#       * Find and treat missing values.
#       * Set correct data type of each column.
#       * Transform categorical features to create categories with more cases and coding likely to be useful in predicting the label.
#       * Apply transformations to numeric features and the label to improve the distribution properties.
#       * Locate and treat duplicate cases.
# 
# An example
# As a first example you will prepare the automotive dataset. 
# Careful preparation of this dataset, or any dataset, is required before attempting to train any machine learning model. 
# This dataset has a number of problems which must be addressed. 
# Further, some feature engineering will be applied.
# 
# Load the dataset
# As a first step you must load the dataset.
# 
# Execute the code in the cell below to load the packages required to run this notebook.
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

options(repr.plot.width=4, repr.plot.height=4) # Set the initial plot area dimensions

# Execute the code in the cell below to load the dataset and print the first few rows of the data frame.

auto_prices = read.csv('Automobile price data _Raw_.csv', stringsAsFactors = FALSE, header = TRUE)
head(auto_prices,20)

# Treat missing values
# Missing values are a common problem in data set. 
# Failure to deal with missing values before training a machine learning model will lead to biased training at best, and in many cases actual failure. 
# Many R models will not process arrays with missing values.
# 
# There are two problems that must be dealt with when treating missing values:
# 
#       1. First you must find the missing values. 
#          This can be difficult as there is no standard way missing values are coded. 
#          Some common possibilities for missing values are:
#               * Coded by some particular character string, or numeric value like -999.
#               * A NULL value or numeric missing value such as a NaN.
#       2. You must determine how to treat the missing values:
#               * Remove features with substantial numbers of missing values. 
#                 In many cases, such features are likely to have little information value.
#               * Remove rows with missing values. 
#                 If there are only a few rows with missing values it might be easier and more certain to simply remove them.
#               * Impute values. 
#                 Imputation can be done with simple algorithms such as replacing the missing values with the mean or median value. 
#                 There are also complex statistical methods such as the expectation maximization (EM) or SMOTE algorithms.
#               * Use nearest neighbor values. Alternatives for nearest neighbor values include, averaging, forward filling or backward filling.
# 
# Carefully observe the first few cases from the data frame and notice that missing values are coded with a '?' character. 
# Execute the code in the cell below to identify the columns with missing values.

#(auto_prices == '?').any
lapply(auto_prices, function(x){any(x == '?')})

# Execute the code in the cell below to display the data types of each column and a sample of the values.

str(auto_prices)

# Compare the columns with missing values to their data types. 
# In all cases, the columns with missing values have a character type as a result of using the '?' code. 
# As a result, some columns that should be numeric (bore, stroke, horsepower, peak.rpm, and price) are coded as character.
# 
# The next question is how many missing values are in each of these character type columns? Execute the code in the cell below to display the counts of missing values.

for(col in names(auto_prices)){
        if(is.character(auto_prices[,col])){
                count = sum(ifelse(auto_prices[,col] == '?', 1, 0))
                cat(paste(col, as.character(count), '\n'))
        }
}

# The normalize.losses column has a significant number of missing values and will be removed. 
# Columns that should be numeric, but contain missing values, are processed in the following manner:
#         
#         1. The '?' values are replaced with R NA values.
#         2. Rows containing NA values are removed with complete.cases.
#            Execute this code, noticing the resulting shape of the data frame.

## Drop column with too many missing values
auto_prices[,'normalized.losses'] = NULL
## Remove rows with missing values, accounting for mising values coded as '?'
cols = c('price', 'bore', 'stroke', 'horsepower', 'peak.rpm')
auto_prices[,cols] = lapply(auto_prices[,cols], function(x){ifelse(x == '?', NA, x)})
auto_prices = auto_prices[complete.cases(auto_prices[,cols]),]
dim(auto_prices)

# The data set now contains 195 cases and 25 columns. 
# 10 rows have been dropped by removing missing values.
# 
# Transform column data type
# As has been previously noted, there are five columns in this dataset which do not have the correct type as a result of missing values. 
# This is a common situation, as the methods used to automatically determine data type when loading files can fail when missing values are present.
# 
# The code in the cell below iterates over a list of columns setting them to numeric. Execute this code and observe the resulting types.

auto_prices[,cols] = lapply(auto_prices[,cols], as.numeric)
str(auto_prices[,cols])

# Feature engineering and transforming variables
# In most cases, machine learning is not performed using raw features. 
# Features are transformed, or combined to form new features in forms which are more predictive This process is known as feature engineering. In many cases, good feature engineering is more important than the details of the machine learning model used. 
# It is often the case that good features can make even poor machine learning models work well, whereas, given poor features even the best machine learning model will produce poor results. 
# Some common approaches to feature engineering include:
#         
#       * Aggregating categories of categorical variables to reduce the number. 
#         Categorical features or labels with too many unique categories will limit the predictive power of a machine learning model. 
#         Aggregating categories can improve this situation, sometime greatly. 
#         However, one must be careful. 
#         It only makes sense to aggregate categories that are similar in the domain of the problem. 
#         Thus, domain expertise must be applied.
#       * Transforming numeric variables to improve their distribution properties to make them more covariate with other variables.     
#         This process can be applied not only to features, but to labels for regression problems. 
#         Some common transformations include, logarithmic and power included squares and square roots.
#       * Compute new features from two or more existing features. 
#         These new features are often referred to as interaction terms. 
#         An interaction occurs when the behavior of say, the produce of the values of two features, is significantly more predictive than the two features by themselves. Consider the probability of purchase for a luxury mens' shoe. This probability depends on the interaction of the user being a man and the buyer being wealthy. As another example, consider the number of expected riders on a bus route. This value will depend on the interaction between the time of day and if it is a holiday.
# 
# Aggregating categorical variables
# When a dataset contains categorical variables these need to be investigated to ensure that each category has sufficient samples. 
# It is commonly the case that some categories may have very few samples, or have so many similar categories as to be meaningless.
# 
# As a specific case, you will examine the number of cylinders in the cars. 
# Execute the code in the cell below to print a frequency table for this variable and examine the result.

table(auto_prices[,'num.of.cylinders'])

# Notice that there is only one car with three and twelve cylinders. 
# There are only four cars with eight cylinders, and 10 cars with five cylinders. 
# It is likely that all of these categories will not have statistically significant difference in predicting auto price. 
# It is clear that these categories need to be aggregated.
# 
# The code in the cell below uses a list with named elements to recode the number of cylinder categories into a smaller number categories. 
# Notice that out vector is defined in advance. 
# Execute this code and examine the resulting frequency table.

cylinder_categories = c('three' = 'three_four', 'four' = 'three_four', 
                        'five' = 'five_six', 'six' = 'five_six',
                        'eight' = 'eight_twelve', 'twelve' = 'eight_twelve')

out = rep('i', length.out = nrow(auto_prices))
i = 1
for(x in auto_prices[,'num.of.cylinders']){
        out[i] = cylinder_categories[[x]]
        i = i + 1
}
auto_prices[,'num.of.cylinders'] = out

table(auto_prices[,'num.of.cylinders'])

# There are now three categories. 
# One of these categories only has five members. 
# However, it is likely that these autos will have different pricing from others.
# 
# Next, execute the code in the cell below to make box plots of the new cylinder categories.

ggplot(auto_prices, aes(num.of.cylinders,price)) + 
geom_boxplot()

# Indeed, the price range of these categories is distinctive. 
# It is likely that these new categories will be useful in predicting the price of autos.
# 
# Now, execute the code in the cell below and examine the frequency table for the body.style feature.

table(auto_prices[,'body.style'])

# Two of these categories have a limited number of cases. 
# These categories can be aggregated to increase the number of cases using a similar approach as used for the number of cylinders. 
# Execute the code in the cell below to aggregate these categories.

body_cats = c('sedan' = 'sedan', 'hatchback' = 'hatchback', 'wagon' = 'wagon', 
'hardtop' = 'hardtop_convert', 'convertible' = 'hardtop_convert')

out = rep('i', length.out = nrow(auto_prices))
i = 1
for(x in auto_prices[,'body.style']){
        out[i] = body_cats[[x]]
        i = i + 1
}
auto_prices[,'body.style'] = out

table(auto_prices[,'body.style'])

# To investigate if this aggregation of categories was a good idea, execute the code in the cell below to display a box plot.
# 
# Then, answer Question 1 on the course page.

ggplot(auto_prices, aes(body.style,price)) + 
geom_boxplot()  

# The hardtop_convert category does appear to have values distinct from the other body style.
# 
# Transforming numeric variables
# To improve performance of machine learning models transformations of the values are often applied. 
# Typically, transformations are used to make the relationships between variables more linear. 
# In other cases, transformations are performed to make distributions closer to Normal, or at least more symmetric. 
# These transformations can include taking logarithms, exponential transformations and power transformations.
# 
# In this case, you will transform the label, the price of the car. 
# Execute the code in the cell below to display and examine a histogram of the label.

plot_hist = function(df, col = 'price', bins = 10){
        options(repr.plot.width=4, repr.plot.height=3) # Set the initial plot area dimensions
        bw = (max(df[,col]) - min(df[,col]))/(bins + 1)
        p = ggplot(df, aes_string(col)) + 
                geom_histogram(binwidth = bw, aes(y=..density..), alpha = 0.5) +
                geom_density(aes(y=..density..), color = 'blue') + 
                geom_rug()
        print(p)
}    
plot_hist(auto_prices)  
# 
# The distribution of auto price is both quite skewed to the left and multi-modal. 
# Given the skew and the fact that there are no values less than or equal to zero, a log transformation might be appropriate.
# 
# The code in the cell below displays a histogram of the logarithm of prices. 
# Execute this code and examine the result.

auto_prices[, 'log_price'] = log(auto_prices[,'price'])
plot_hist(auto_prices, col = 'log_price')

# The distribution of the logarithm of price is more symmetric, but still shows some multi-modal tendency and skew. 
# Nonetheless, this is an improvement so we will use these values as our label.
# 
# The next question is, how does this transformation change the relationship between the label and some of the features? 
# To find out, execute the code in the cell below.

plot_scatter_sp = function(df, cols, col_y = 'log_price', alpha = 1.0){
        options(repr.plot.width=5, repr.plot.height=3.5) # Set the initial plot area dimensions
        for(col in cols){
                p = ggplot(df, aes_string(col, col_y)) + 
                        geom_point(aes(shape = factor(fuel.type)), alpha = alpha) +
                        ggtitle(paste('Scatter plot of', col_y, 'vs.', col, '\n with shape by fuel type'))
                print(p)
        }
}

num_cols = c('curb.weight', 'engine.size', 'horsepower', 'city.mpg')
plot_scatter_sp(auto_prices, num_cols, alpha = 0.2)

# Comparing the results to those obtained in the visualization lab, it does appear that the relationships between curb.weight and log_price and city.mpg and log_price are more linear.
# 
# The relationship with the log_price and categorical variables should likely also be investigated. 
# It is also possible that some type of power transformation should be applied to, say horsepower or engine size. 
# In the interest of brevity, these ideas are not pursued here.
# 
# Before proceeding, answer Question 2 on the course page.

# Let's save the dataframe to a csv file 
# We will use this in the next module so that we don't have to re-do the steps above
# You don't have to run this code as the csv file has been saved under the next module's folder
#write.csv(auto_prices, file = 'Auto_Prices_Preped.csv', row.names = FALSE)

# Another example
# Next, you will prepare the German credit data. 
# Execute the code in the cell below to load the dataset and print the head (first 5 rows) of the dataframe.

credit = read.csv('German_Credit.csv', stringsAsFactors = FALSE, header = FALSE)
head(credit, 5)

# This dataset is a bit hard to understand. 
# For a start, the column names are not human readable.
# 
# Recode character strings
# You have likely noticed that the the column names are not human readable. 
# This can be changed as was done for the previous dataset. 
# Execute the code in the cell below to add human-readable column names to the data frame.

names(credit) = c('Customer_ID', 'checking_account_status', 'loan_duration_mo', 'credit_history', 
                  'purpose', 'loan_amount', 'savings_account_balance', 
                  'time_employed_yrs', 'payment_pcnt_income','gender_status', 
                  'other_signators', 'time_in_residence', 'property', 'age_yrs',
                  'other_credit_outstanding', 'home_ownership', 'number_loans', 
                  'job_category', 'dependents', 'telephone', 'foreign_worker', 
                  'bad_credit')
head(credit, 5)

# Next, there is a trickier problem to deal with. 
# The current coding of the categorical variables is impossible to understand. 
# This makes interpreting these variables nearly impossible.
# 
# The code in the cell below uses a list of lists to recode the categorical features with human-readable text. 
# The last list recodes good and bad credit as a binary variable, {0,1}. This process is:
#         
#         1. The for loop iterates over the columns.
#         2. A lookup is performed on the list for each column to find the human-readable code which is then substituted.
# 
# Execute this code and examine the result:
        
checking_account_status = c('< 0 DM', '0 - 200 DM', '> 200 DM or salary assignment', 'none')
names(checking_account_status) = c('A11', 'A12', 'A13', 'A14')
credit_history = c('no credit - paid', 'all loans at bank paid', 'current loans paid', 
                   'past payment delays',  'critical account - other non-bank loans')
names(credit_history) = c('A30', 'A31', 'A32', 'A33', 'A34')
purpose = c( 'car (new)', 'car (used)', 'furniture/equipment', 'radio/television', 
             'domestic appliances', 'repairs', 'education', 'vacation', 'retraining',
             'business', 'other')
names(purpose) = c('A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A410')
savings_account_balance = c('< 100 DM', '100 - 500 DM', '500 - 1000 DM', '>= 1000 DM', 'unknown/none')
names(savings_account_balance) = c('A61', 'A62', 'A63', 'A64', 'A65')
time_employed_yrs = c('unemployed', '< 1 year', '1 - 4 years', '4 - 7 years', '>= 7 years')
names(time_employed_yrs) = c('A71', 'A72', 'A73', 'A74', 'A75')
gender_status = c('male-divorced/separated', 'female-divorced/separated/married',
                  'male-single', 'male-married/widowed', 'female-single')
names(gender_status) = c('A91', 'A92', 'A93', 'A94', 'A95')
other_signators = c('none', 'co-applicant', 'guarantor')
names(other_signators) = c('A101', 'A102', 'A103')
property =  c('real estate', 'building society savings/life insurance', 'car or other', 'unknown-none')
names(property) = c('A121', 'A122', 'A123', 'A124')
other_credit_outstanding = c('bank', 'stores', 'none')
names(other_credit_outstanding) = c('A141', 'A142', 'A143')
home_ownership = c('rent', 'own', 'for free')
names(home_ownership) = c('A151', 'A152', 'A153')
job_category = c('unemployed-unskilled-non-resident', 'unskilled-resident', 'skilled', 'highly skilled')
names(job_category) =c('A171', 'A172', 'A173', 'A174')
telephone = c('none', 'yes')
names(telephone) = c('A191', 'A192')
foreign_worker = c('yes', 'no')
names(foreign_worker) = c('A201', 'A202')
bad_credit = c(1, 0)
names(bad_credit) = c(2, 1)

codes = c('checking_account_status' = checking_account_status,
          'credit_history' = credit_history,
          'purpose' = purpose,
          'savings_account_balance' = savings_account_balance,
          'time_employed_yrs' = time_employed_yrs,
          'gender_status' = gender_status,
          'other_signators' = other_signators,
          'property' = property,
          'other_credit_outstanding' = other_credit_outstanding,
          'home_ownership' = home_ownership,
          'job_category' = job_category,
          'telephone' = telephone,
          'foreign_worker' = foreign_worker,
          'bad_credit' = bad_credit)         

cat_cols = c('checking_account_status', 'credit_history', 'purpose', 'savings_account_balance', 
             'time_employed_yrs','gender_status', 'other_signators', 'property',
             'other_credit_outstanding', 'home_ownership', 'job_category', 'telephone', 'foreign_worker', 
             'bad_credit')

for(col in cat_cols){
        credit[,col] = sapply(credit[,col], function(code){codes[[paste(col, '.', code, sep = '')]]})
}
#credit$bad_credit = as.numeric(credit$bad_credit)
head(credit, 5)

# The categorical values are now coded in a human readable manner.
# 
# Remove duplicate rows
# Duplicate cases can seriously bias the training of machine learning models. 
# In simple terms, cases which are duplicates add undue weight to that case when training a machine learning model. 
# Therefore, it is necessary to ensure there are no duplicates in the dataset before training a model.
# 
# One must be careful when determining if a case is a duplicate or not. 
# It is possible that some cases have identical values, particularly if most or all features are categorical. 
# On the other hand, if there are columns with values guaranteed to be unique these can be used to detect and remove duplicates.
# 
# Another consideration when removing duplicate cases is determining which case to remove. 
# If the duplicates have different dates of creation, the newest date is often selected. 
# In the absence of such a criteria, the choice is often arbitrary. 
# You may chose to keep the first case or the last case.
# 
# The German credit data has a Customer ID column which should be unique. 
# The presence of duplicates can be determined by comparing the number of rows to the number of unique values. 
# The code in the cell below prints the shape of the data frame and the number of unique rows determined by the dplyr distinct function.
# 
# Execute this code, examine the results, and answer Question 3 on the course page.

print(dim(credit))
dim(distinct(credit))

# There are 12 duplicate cases. 
# These need to be located and the duplicates removed. 
# In this case, the first instance will be kept.
# 
# The code in the cell below removes these duplicates from the data frame and resulting dimension is printed. Execute this code and examine the results.

credit = distinct(credit)
dim(credit)

# The duplicate rows have been successfully removed.

# Let's save the dataframe to a csv file 
# We will use this in the next module so that we don't have to re-do the steps above
# You don't have to run this code as the csv file has been saved under the next module's folder
#write.csv(credit, file = 'German_Credit_Preped.csv', row.names = FALSE)

# Feature engineering
# Some feature engineering needs to be investigated to determine if any improvement in predictive power can be expected. 
# From the previous data exploration, it is apparent that several of the numeric features had a strong left skew. 
# A log transformation may help in a case like this.
# 
# Execute the code in the cell below iterates over selected columns with lapply to apply the log function to each column.

credit[,c('log_loan_duration_mo', 'log_loan_amount', 'log_age_yrs')] = lapply(credit[,c('loan_duration_mo', 'loan_amount', 'age_yrs')], log)

# Next, execute the code in the cell below to visualize the differences in the distributions of the untransformed and transformed variables for the two label values.

plot_violin = function(df, cols, col_x = 'bad_credit'){
        options(repr.plot.width=4, repr.plot.height=3.5) # Set the initial plot area dimensions
        for(col in cols){
                p = ggplot(df, aes_string(col_x, col)) + 
                        geom_violin() +
                        ggtitle(paste('Box plot of', col, '\n vs.', col_x))
                print(p)
        }
}

num_cols = c('log_loan_duration_mo', 'log_loan_amount', 'log_age_yrs',
             'loan_duration_mo', 'loan_amount', 'age_yrs')
plot_violin(credit, num_cols) 

# The log transformed features have more symmetric distributions. 
# However, it does not appear that the separation of the label cases is improved.
# 
# Note: Recalling the visualization of the categorical features, there are quite a few categories with few cases. 
# However, it is not clear how these categories can be reasonably combined. 
# It may be the case that some of these categorical features are not terribly predictive.
# 
# Summary
# Good data preparation is the key to good machine learning performance. 
# Data preparation or data munging is a time interactive and iterative process. 
# Continue to visualize the results as you test ideas. 
# Expect to try many approaches, reject the ones that do not help, and keep the ones that do. 
# In summary, test a lot of ideas, fail fast, keep what works. 
# The reward is that well prepared data can improve the performance of almost any machine learning algorithm.

# ====================
# Processing end time
timeEnd = Sys.time()

# Processing date and total processing time
cat(paste("","Processing end date and time",date(),"","",sep="\n"))
paste("Total processing time =",round(difftime(timeEnd,timeStart), digits=2),"seconds",sep=" ")


# Stop writing to an output file
sink()

################