# Processing start time
sink('visualizing_data_for_classification_output.txt')
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

# Visualizing Data for Classification
# In a previous lab you explored the automotive price dataset to understand the relationships for a regression problem. 
# In this lab you will explore the German bank credit dataset to understand the relationships for a classification problem. 
# The difference being, that in classification problems the label is a categorical variable.
# 
# Visualization for classification problems shares much in common with visualization for regression problems. 
# Colinear features should be identified so they can be eliminated or otherwise dealt with. 
# However, for classification problems you are looking for features that help separate the label categories. 
# Separation is achieved when there are distinctive feature values for each label category. 
# Good separation results in low classification error rate.
# 
# Load and prepare the data set
# 
# 
# Note: If you are running in Azure Notebooks, make sure that you run the code in the setup.ipynb notebook at the start of you session to ensure your environment is correctly configured.

## Import packages
if("ggplot2" %in% rownames(installed.packages()) == FALSE) 
{install.packages("ggplot2")}
library(ggplot2)

if("repr" %in% rownames(installed.packages()) == FALSE) 
{install.packages("repr")}
library(repr)

options(repr.plot.width=4, repr.plot.height=4) # Set the initial plot area dimensions

# The code in the cell below loads the dataset and assigns human-readable names to the columns. 
# The dimension and head of the data frame are then printed. Execute this code:
        
        credit = read.csv('German_Credit.csv', header = FALSE)
names(credit) = c('Customer_ID','checking_account_status', 'loan_duration_mo', 'credit_history', 
                  'purpose', 'loan_amount', 'savings_account_balance', 
                  'time_employed_yrs', 'payment_pcnt_income','gender_status', 
                  'other_signators', 'time_in_residence', 'property', 'age_yrs',
                  'other_credit_outstanding', 'home_ownership', 'number_loans', 
                  'job_category', 'dependents', 'telephone', 'foreign_worker', 
                  'bad_credit')
print(dim(credit))
head(credit)

# The first column is Customer_ID, which is an identifier. 
# And then there are 20 features plus a label column. 
# These features represent information a bank might have on its customers. 
# There are both numeric and categorical features. 
# However, the categorical features are coded in a way that makes them hard to understand. 
# Further, the label is coded as {1,2} which is a bit awkward.
# 
# The code in the cell below using a list of lists to recode the categorical features with human-readable text. 
# The processing is performed with these steps:
#         
#       1. Lists for each of the human readable codes are created for each column. 
#          The names of these lists are the codes in the raw data.
#       2. A list of lists is created with the column names used as the list names.
#       3. A list of categorical columns is created.
#       4. A for loop iterates over the column names. sapply is used to iterate over the codes in each column. 
#          The codes are used to generate names for the list lookup.

# Execute this code and examine the result:
#         
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
head(credit)

# The categorical features now have meaningful coding. Additionally, the label is now coded as a binary variable.
# 
# Examine classes and class imbalance
# In this case, the label has significant class imbalance. 
# Class imbalance means that there are unequal numbers of cases for the categories of the label. 
# Class imbalance can seriously bias the training of classifier algorithms. 
# It many cases, the imbalance leads to a higher error rate for the minority class. 
# Most real-world classification problems have class imbalance, sometimes severe class imbalance, so it is important to test for this before training any model.
# 
# Fortunately, it is easy to test for class imbalance using a frequency table. 
# Execute the code in the cell below to display a frequency table of the classes:
        
        table(credit$bad_credit)

# Notice that only 30% of the cases have bad credit. This is not suprising, since a bank would typically retain customers with good credit. 
# While this is not a cases of sereve imbalance, it is enough to bias the traing of any model.
# 
# Visualize class separation by numeric features
# As stated previously, the primary goal of visualization for classification problems is to understand which features are useful for class separation. 
# In this section, you will start by visualizing the separation quality of numeric features.
# 
# Execute the code, examine the results, and answer Question 1 on the course page.

        plot_box = function(df, cols, col_x = 'bad_credit'){
                options(repr.plot.width=4, repr.plot.height=3.5) # Set the initial plot area dimensions
                for(col in cols){
                        p = ggplot(df, aes_string(col_x, col)) + 
                                geom_boxplot() +
                                ggtitle(paste('Box plot of', col, '\n vs.', col_x))
                        print(p)
                }
        }
        
        num_cols = c('loan_duration_mo', 'loan_amount', 'payment_pcnt_income',
                     'age_yrs', 'number_loans', 'dependents')
        plot_box(credit, num_cols)  
        
# How can you interpret these results? 
# Box plots are useful, since by their very construction you are forced to focus on the overlap (or not) of the quartiles of the distribution. 
# In this case, the question is there sufficient differences in the quartiles for the feature to be useful in separation the label classes? 
# There are there are three cases displayed above:
#         
#       1. For loan_duration_mo, loan_amount, and payment as a percent of income (payment_pcnt_income), there is useful separation between good and bad credit customers. 
#          As one might expect, bad credit customers have longer loan duration on larger loans and with payments being a greater percentage of their income.
#       2. On the other hand, age in years, number_loans and dependents does not seem to matter. 
#          In latter two cases, this situation seems to result from the median value being zero. There are just not enough non-zero cases to make these useful features.

        # As an alternative to box plots, you can use violin plots to examine the separation of label cases by numeric features. Execute the code in the cell below and examine the results:
        
        plot_violin = function(df, cols, col_x = 'bad_credit'){
                options(repr.plot.width=4, repr.plot.height=3.5) # Set the initial plot area dimensions
                for(col in cols){
                        p = ggplot(df, aes_string(col_x, col)) + 
                                geom_violin() +
                                ggtitle(paste('Box plot of', col, '\n vs.', col_x))
                        print(p)
                }
        }

plot_violin(credit, num_cols)  

# The interpretation of these plots is largely the same as the box plots. However, there is one detail worth noting. The differences between loan_duration_mo and loan_amount for good and bad credit customers are only for the more extreme values. It may be that these features are less useful and the box plot indicates.
# 
# Visualizing class separation by categorical features
# Now you will turn to the problem of visualizing the ability of categorical features to separate classes of the label. Ideally, a categorical feature will have very different counts of the categories for each of the label values. A good way to visualize these relationships is with bar plots.
# 
# The code in the cell below creates side by side plots of the categorical variables for each of the labels categories. The grid.arrange function from the gridExtra package is used to arrange the two plots side by side.
# 
# Execute this code, examine the results, and answer Question 2 on the course page.

if("gridExtra" %in% rownames(installed.packages()) == FALSE) 
{install.packages("gridExtra")}# Added due to error
library(gridExtra)

plot_bars = function(df, catcols){
        options(repr.plot.width=6, repr.plot.height=5) # Set the initial plot area dimensions
        temp0 = df[df$bad_credit == 0,]
        temp1 = df[df$bad_credit == 1,]
        for(col in cat_cols){
                p1 = ggplot(temp0, aes_string(col)) + 
                        geom_bar() +
                        ggtitle(paste('Bar plot of \n', col, '\n for good credit')) +  
                        theme(axis.text.x = element_text(angle = 90, hjust = 1))
                p2 = ggplot(temp1, aes_string(col)) + 
                        geom_bar() +
                        ggtitle(paste('Bar plot of \n', col, '\n for bad credit')) +  
                        theme(axis.text.x = element_text(angle = 90, hjust = 1))
                grid.arrange(p1,p2, nrow = 1)
        }
}

plot_bars(credit, cat_cols)    
# There is a lot of information in these plots. The key to interpretation of these plots is comparing the proportion of the categories for each of the label values. If these proportions are distinctly different for each label category, the feature is likely to be useful in separating the label.
# 
# There are several cases evident in these plots:
#         
#       1. Some features such as checking_account_status and credit_history have significantly different distribution of categories between the label categories.
#       2. Others features such as gender_status and telephone show small differences, but these differences are unlikely to be significant.
#       3. Other features like other_signators, foreign_worker, home_ownership, and job_category have a dominant category with very few cases of other categories. 
#          These features will likely have very little power to separate the cases.

# Notice that only a few of these categorical features will be useful in separating the cases.
# 
# Summary
# In this lab you have performed exploration and visualization to understand the relationships in a classification dataset. Specifically:
#         
#       1. Looked for imbalance in the label cases using a frequency table.
#       2. The goal of visualization is to find numeric or categorical features that separate the cases.
# 

# ====================
# Processing end time
timeEnd = Sys.time()

# Processing date and total processing time
cat(paste("","Processing end date and time",date(),"","",sep="\n"))
paste("Total processing time =",round(difftime(timeEnd,timeStart), digits=2),"seconds",sep=" ")


# Stop writing to an output file
sink()

################
