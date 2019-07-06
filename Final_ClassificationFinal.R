# Processing start time
sink('classification_final_output.txt')
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

# Classifying Data for Final Project

# Key objectives for the cleaning stage:

# Create a classification model that predicts whether or not a new customer will buy a bike.
# Do so under the following conditions:

        # 1. Use the Adventure Works Cycles customer data from the first part of the final to create a classification model that predicts whether or not a customer will purchase a bike. 
        #         - The model should predict bike purchasing for new customers for whom no information about average monthly spend or previous bike purchases is available.
        # 2. Download the test data which includes customer features but does not include bike purchasing or average monthly spend values.
        # 3. Use the model to predict the corresponding test dataset. 
        # 4. Check  prediction against the actual result.
# Note: Consider the prior data to be the training data

## Import packages
# 
if("ggplot2" %in% rownames(installed.packages()) == FALSE)
{install.packages("ggplot2")}
library(ggplot2)

if("gridExtra" %in% rownames(installed.packages()) == FALSE)
{install.packages("gridExtra")}
library(gridExtra)

if("MASS" %in% rownames(installed.packages()) == FALSE)
{install.packages("MASS")}
library(MASS)

if("cluster" %in% rownames(installed.packages()) == FALSE)
{install.packages("cluster")}
library(cluster)

if("caret" %in% rownames(installed.packages()) == FALSE)
{install.packages("caret")}
library(caret)

if("repr" %in% rownames(installed.packages()) == FALSE)
{install.packages("repr")}
library(repr)

if("dplyr" %in% rownames(installed.packages()) == FALSE) 
{install.packages("dplyr")}
library(dplyr)

if("stats" %in% rownames(installed.packages()) == FALSE) 
{install.packages("stats")}
library(stats)

if("ROCR" %in% rownames(installed.packages()) == FALSE)
{install.packages("ROCR")}
library(ROCR)

if("pROC" %in% rownames(installed.packages()) == FALSE)
{install.packages("pROC")}
library(pROC)

# Need for date calcluations
if("eeptools" %in% rownames(installed.packages()) == FALSE) 
{install.packages("eeptools")}
library(eeptools)

# GENERAL APPROACH

# Will approach this as a linear regression situation
# Where the label is a de facto binary case of '1' for 
# bike buyer and '0' for non bike buyer

# Will process the training data as in first part of final:
#       - Looking for missing values
#       - Identifying (and excluding from analysis) labels with too many missing values
#       - Using only the most recent of any duplicated record in the training data
#       - Deleting records that are duplicated in the test data 
# Will also test the training and test data for several things:
#       - Training cases duplicated in the test data


# Before moving to the logistic regression model, will
# do a linear regression based on AveMonthSpend (Which is not included in the test data)
# The labels deemed to be relevant in the linear regression for
# AveMonthSpend will be included in the logistic regression.

# Input training data
BikeBuyer = read.csv('AW_BikeBuyer.csv')
AveMonthSpend = read.csv('AW_AveMonthSpend.csv')
AdvWorksCusts = read.csv('AdvWorksCusts.csv')

# First column (CustomerID) is the same for all three data frames
# Goal is to have this be a unique identifier, so if it is duplicated,
#       will only consider the latest one in the data frame.

# AdvWorksCusts is the biggest data frame, the other two only have one additional column.

# Will cbind a record number variable (record_num),AdvWorksCusts, and the last columns of the other two 
Cust_data = cbind(record_num=seq(1,nrow(AdvWorksCusts)),AdvWorksCusts, BikeBuyer = BikeBuyer$BikeBuyer, AveMonthSpend = AveMonthSpend$AveMonthSpend)

# Need to find index of duplicated rows
last_dup_rows_ndx = which(duplicated(Cust_data$CustomerID))

# Need to check of there is more than one duplicate of a CustomerID
ifelse(
        length(last_dup_rows_ndx)==length(unique(Cust_data$CustomerID[last_dup_rows_ndx])),
        paste('No CustomerID is duplicated more than once'),
        paste('At least one CustomerID is duplicated more than once')) 

# If no more than one duplicate, should have one match for each 
# Identified duplicate CustomerID value

# Need to identify the index of the CustomerID values
Cust_data_redacted = Cust_data[-last_dup_rows_ndx,]
first_dup_rows_row = which(Cust_data_redacted$CustomerID %in% Cust_data$CustomerID[last_dup_rows_ndx] )

# The above comparison was on a redacted Cust_data data frame,
#       so must  get the index from the record_num column
first_dup_rows_ndx = Cust_data_redacted$record_num[first_dup_rows_row]

# Will now check to see if first duplicate rows were found 
#       and the first duplcates were to be removed

firstID=sort(Cust_data$CustomerID[first_dup_rows_ndx])
initial.row = first_dup_rows_ndx
lastID=sort(Cust_data$CustomerID[last_dup_rows_ndx])
duplicate.row = last_dup_rows_ndx

ifelse(
        (sum(firstID==lastID))==length(first_dup_rows_ndx),
        paste('Only true duplicates were removed'),
        paste('True duplicates were not removed')) 

Cust_data_no_dups = Cust_data[-first_dup_rows_ndx,]

# Checked for missing values, none if the number of rows equals the number of complete cases
ifelse(
        (sum(complete.cases(Cust_data_no_dups)) == nrow(Cust_data_no_dups)),
        paste('All cases complete, no missing values'),
        paste('There is at least one missing value')) 

# Will add an age variable
Cust_data_no_dups$Age = age_calc(dob = as.Date(Cust_data_no_dups$BirthDate), enddate = as.Date("1998-01-01"), units = "years")
Cust_data_no_dups$Age = round(Cust_data_no_dups$Age, digits = 2)

# Note: will not analyze features assumed to be irrelevant to 
# AveMonthSpend or otherwise unsuitable (eg. too many categories):
#       -"Title",
#       -"FirstName",
#       -"MiddleName",
#       -"LastName",
#       -"Suffix",
#       -"AddressLine1", 
#       -"AddressLine2",
#       -"City",
#       -"StateProvinceName",
#       -"PostalCode"
#       -"PhoneNumber"



#Rename to training data:
train = Cust_data_no_dups

head(train)
str(train)

# Will now read in the test data and do appropriate processing

test_raw =  read.csv('AW_test.csv')
dim(test_raw)
str(test_raw)

# Need to convert date from %m/%d%Y to %Y-%m-%d
test_raw$BirthDate = as.Date(test_raw$BirthDate, "%m/%d/%Y")

# Need to add an Age column to the test data
test_raw$Age = round(age_calc(dob = as.Date(test_raw$BirthDate), enddate = as.Date("1998-01-01"), units = "years"), digits = 2)
test = test_raw # This test data will be further processed before estimating the BikeBuyer value

# Tested for duplicated CustomerID values in raw test data, but none found
last_dup_rows_test_ndx = which(duplicated(test_raw$CustomerID))
last_dup_rows_test_ndx # No duplicates found

# Tested for training individuals in test data, specifically
# the same Customer ID in both training and testing
dup_in_train_ndx = train$CustomerID %in% test_raw$CustomerID
paste("The following",sum(dup_in_train_ndx), "training set Customer IDs were found in test set:")
paste(train$CustomerID[c(which(dup_in_train_ndx))])

# Duplictes removed from the training set
paste("These records will be removed from the training set")
train = train[!dup_in_train_ndx,]


# Now that the training set has had cases removed that were in the test set, will do a 
# split the training set into a pretrain and pretest set
# in order to do a linear regression for AveMonthSpend


set.seed(1955)
## Randomly sample cases to create independent training and test data
partition = createDataPartition(train[,"CustomerID"], times = 1, p = 0.75, list = FALSE)
pretrain = train[partition,] # Create the training sample

dim(train)
dim(pretrain)
pretest = train[-partition,] # Create the test sample
dim(pretest)
## Scale numeric features

# Scaling of numeric features is an important step when preparing data for training a machine learning model. The code in the cell below does the following:
# Can compute transformation using a caret package `preProcess` object for centering and scaling the data. Notice that these computations are done only with the training data. 
# 2. The transformations are applied to both the training and test dataset. 
# 
num_cols = c('YearlyIncome', 'Age', 'AveMonthSpend')
preProcValues = preProcess(pretrain[,num_cols], method = c("center", "scale"))

# pretrain[,num_cols] = predict(preProcValues, pretrain[,num_cols])
# pretest[,num_cols] = predict(preProcValues, pretest[,num_cols])
# head(pretrain[,num_cols])

# The other option (used here) is to apply two basic transformations,
# Normalizing 

# ----
# pretrain$norm_YearlyIncome = (pretrain$YearlyIncome - mean(pretrain$YearlyIncome))/sd(pretrain$YearlyIncome)
# pretrain$norm_Age = (pretrain$Age - mean(pretrain$Age))/sd(pretrain$Age)
# 
# pretest$norm_YearlyIncome = (pretest$YearlyIncome - mean(pretest$YearlyIncome))/sd(pretest$YearlyIncome)
# pretest$norm_Age = (pretest$Age - mean(pretest$Age))/sd(pretest$Age)
# ==========
# LINEAR REGRESSSION
paste("Base model with pretrain data")
lin_mod_pretrain = lm(AveMonthSpend ~ YearlyIncome + MaritalStatus +  
                         Gender + NumberChildrenAtHome +  Age, data = pretrain)


summary(lin_mod_pretrain)

# Evaluate the model
# You will now use the test dataset to evaluate the performance of the regression model.
# As a first step, execute the code in below to compute and display various performance metrics and examine the results.

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

score = predict(lin_mod_pretrain, newdata = pretest)
print_metrics(lin_mod_pretrain, pretest, score, label = 'AveMonthSpend')

# NOTE: After several runs, the following core variables remained:
#       YearlyIncome, MaritalStatus, Gender, NumberChildrenAt,Home, Age
# Two new binary features were added (both modified from existing catagorical variables):
#       GraduateDegree, ManagementJob

pretrain$ManagementJob = pretrain$Occupation == "Management"
pretrain$GraduateDegree = pretrain$Education == "Graduate Degree"

pretest$ManagementJob = pretest$Occupation == "Management"
pretest$GraduateDegree = pretest$Education == "Graduate Degree"

# Running this revised linear regression gives: 
lin_mod_pretrain_revised = lm(AveMonthSpend ~ YearlyIncome + MaritalStatus + 
                              ManagementJob + GraduateDegree +
                              Gender + NumberChildrenAtHome +  Age, data = pretrain)


summary(lin_mod_pretrain_revised)
score = predict(lin_mod_pretrain_revised, newdata = pretest)
print_metrics(lin_mod_pretrain_revised, pretest, score, label = 'AveMonthSpend')

# Running the linear regression with these additional variables gives a similar result
# as when the full Occupation and Education variables included. 
# The result is a more streamlined regression equation, so will proceed with this as the 
# set of logistic variables:
#       - GraduateDegree (binary, based on Education)
#       - ManagementJob (binary, based on Occupation)
#       - YearlyIncome (normalized)
#       - MaritalStatus
#       - Gender
#       - NumberChildrenAtHome
#       - Age (normalized)





# ==========
# LOGISTIC REGRESSION
# If instead of getting AveMonthSpend, the goal were to get 
# the binary variable BikeBuyer, would have done the first part differently
# Would have used  a normlized value rather log(AveMonthSpend) 
# Used the log above because the reverse transformation was a conceptually simpler process.

# Assuming use of the same prediction variables, the outcoume would have been as follows
# Will come up with a prediction model based on using the training data
# That was split earlier for the linear model for AveMonthSpend

set.seed(5566)
logistic_mod = glm(BikeBuyer ~ YearlyIncome + MaritalStatus + GraduateDegree + NumberCarsOwned + 
                           Gender + NumberChildrenAtHome + TotalChildren + Age + ManagementJob,
                   family = binomial, data = pretrain)


# Now, print and examine the model coefficients by executing the code in the cell below. 

paste("Logistic model coefficients")
logistic_mod$coefficients

# First of all, notice that model coefficients are similar to what you might expect for an linear regression model. 
# As previously explained the logistic regression is indeed a linear model.  
# 
# Recall that the logistic regression model outputs log likelihoods. 
# The class with the highest probability is taken as the score (prediction). 

pretest$probs = predict(logistic_mod, newdata = pretest, type = 'response')
pretest[1:20, c('BikeBuyer','probs')]

# The first column is the label, the second is the log likelihood of a positive score 
# The code below will add a third column, which is the prediction based on a threshold value of 0.5.
paste("The first column is the label, the second is the log likelihood of a positive score.")

score_model = function(df, threshold){
        df$score = ifelse(df$probs < threshold, 0, 1)
        df
}

paste("The third column is the prediction which is based on a threshold value of 0.5.")
pretest = score_model(pretest, 0.5)
pretest[1:30, c('BikeBuyer','probs','score')]

# Evaluate the classification model

# The code in the cell below implements a function that computes a confusion matrix. 
# The confusion matrix is then used to compute the performance metrics. 
# Execute this code and examine the results for the logistic regression model. 

logistic.eval <- function(df){ 
        # First step is to find the TP, FP, TN, FN cases
        df$conf = ifelse(df$BikeBuyer == 1 & df$score == 1, 'TP',
                         ifelse(df$BikeBuyer == 1 & df$score == 0, 'FN',
                                ifelse(df$BikeBuyer == 0 & df$score == 0, 'TN', 'FP')))
        
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
        
        roc_obj <- roc(df$BikeBuyer, df$probs)
        cat(paste('AUC       =', as.character(round(auc(roc_obj),3)),'\n'))
}
logistic.eval(pretest)      

# You can also use the function confusionMatrix if you have converted the score into the same categorical (factor) variable as the label.

confusionMatrix(factor(pretest$score), factor(pretest$BikeBuyer))
# ====================
# CLASSIFICATION OF THE TEST DATA

# If one had a set of data excluding BikeBuyer vlaues, the original training data and the
# raw test data will be processed similarly to what was done for the pretrain data

# First will process the relevant numerical variables 
num_cols = c('YearlyIncome', 'Age')
preProcValues = preProcess(train[,num_cols], method = c("center", "scale"))

# The processing of the test data is based on what was done 
# with the training data as per example from the Classification Module

train[,num_cols] = predict(preProcValues, train[,num_cols])
test[,num_cols] = predict(preProcValues, test[,num_cols])
head(train[,num_cols])
head(test[,num_cols])

# Will now add the binary variables for ManagementJob and GraduateDegree
train$ManagementJob = train$Occupation == "Management"
train$GraduateDegree = train$Education == "Graduate Degree"

test$ManagementJob = test$Occupation == "Management"
test$GraduateDegree = test$Education == "Graduate Degree"

# Now will perform the logistic regression model
set.seed(5566)
logistic_mod_final = glm(BikeBuyer ~ YearlyIncome + MaritalStatus + GraduateDegree + NumberCarsOwned + 
                           Gender + NumberChildrenAtHome + TotalChildren + Age + ManagementJob,
                   family = binomial, data = train)

# Now, print and examine the model coefficients by executing the code in the cell below. 

paste("Logistic model coefficients")
logistic_mod_final$coefficients

# The test data does not have a BikeBuyer label, so will only 
# create the estimates for each CustomerID

test$probs = predict(logistic_mod, newdata = test, type = 'response')
test[1:30, c('CustomerID','probs')]

# Now will add a column with the scores based on a threshold value of 0.5

test = score_model(test, 0.5)
test[1:30, c('CustomerID','probs','score')]

# Downloading the result
write.csv(test[, c('CustomerID','probs','score')], file = "BikeBuyer_predicted.csv")
# ====================
# # Processing end time
timeEnd = Sys.time()
# 
# # Processing date and total processing time
cat(paste("","Processing end date and time",date(),"","",sep="\n"))
paste("Total processing time =",round(difftime(timeEnd,timeStart), digits=2),"seconds",sep=" ")


# Stop writing to an output file
sink()

################
