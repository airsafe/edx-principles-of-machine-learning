# Processing start time
sink('application_of_clustering_output.txt')
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

# Cleaning Data for Final Project

# Key objectives for the cleaning stage:
#       * Replace any missing values and remove duplicate rows. 
#         In this dataset, each customer is identified by a unique customer ID. 
#         The most recent version of a duplicated record should be retained.
# 
#       * Explore the data by calculating summary and descriptive statistics for 
#         the features in the dataset, calculating correlations between features, 
#         and creating data visualizations to determine apparent relationships in the data.
# 
#       * Based on your analysis of the customer data after removing all duplicate customer records, answer the questions below:

#       1. Minimum AveMonthSpend
#       2. Maximum AveMonthSpend
#       3. Mean AveMonthSpend
#       4. Median AveMonthSpend
#       5. Standard Deviation AveMonthSpend
#       6. The distribution of the values in the BikeBuyer column indicates which:
#               - Fewer customers have bought bikes than have not bought bikes.
#               - More customers have bought bikes than have not bought bikes.
#               - The same number of customers have bought bikes as have not bought bikes.

## Import packages
# 
# if("ggplot2" %in% rownames(installed.packages()) == FALSE) 
# {install.packages("ggplot2")}
# library(ggplot2)
# 
# if("gridExtra" %in% rownames(installed.packages()) == FALSE) 
# {install.packages("gridExtra")}
# library(gridExtra)
# 
# if("MASS" %in% rownames(installed.packages()) == FALSE) 
# {install.packages("MASS")}
# library(MASS)
# 
# if("cluster" %in% rownames(installed.packages()) == FALSE) 
# {install.packages("cluster")}
# library(cluster)
# 
# if("caret" %in% rownames(installed.packages()) == FALSE) 
# {install.packages("caret")}
# library(caret)
# 
# if("repr" %in% rownames(installed.packages()) == FALSE) 
# {install.packages("repr")}
# library(repr)

if("dplyr" %in% rownames(installed.packages()) == FALSE) 
{install.packages("dplyr")}
library(dplyr)

if("stats" %in% rownames(installed.packages()) == FALSE) 
{install.packages("stats")}
library(stats)

# Need for date calcluations
if("eeptools" %in% rownames(installed.packages()) == FALSE) 
{install.packages("eeptools")}
library(eeptools)
# Import data

# All data frames have the same CustomerID values and order
# Will identify the duplicated CustomerID values and choosing the 
#       earlier versions to delete.

BikeBuyer = read.csv('AW_BikeBuyer.csv')
head(BikeBuyer,n=2)
str(BikeBuyer)

AveMonthSpend = read.csv('AW_AveMonthSpend.csv')
head(AveMonthSpend,n=2)
str(AveMonthSpend)

AdvWorksCusts = read.csv('AdvWorksCusts.csv')
head(AdvWorksCusts,n=2)
str(AdvWorksCusts)


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


# Will now answer the test questions with this data frame
# of unique records

#       1. Minimum AveMonthSpend
paste('1. Minimum AveMonthSpend = ', min(Cust_data_no_dups$AveMonthSpend))


#       2. Maximum AveMonthSpend
paste('2. Maximum AveMonthSpend = ', max(Cust_data_no_dups$AveMonthSpend))

#       3. Mean AveMonthSpend
paste('3. Mean AveMonthSpend = ', round(mean(Cust_data_no_dups$AveMonthSpend), digits=3))

#       4. Median AveMonthSpend
paste('4. Median AveMonthSpend = ', round(median(Cust_data_no_dups$AveMonthSpend), digits=3))

#       5. Standard Deviation AveMonthSpend
paste('5. Standard Deviation AveMonthSpend = ', round(sd(Cust_data_no_dups$AveMonthSpend), digits=3))

#       6. The distribution of the values in the BikeBuyer column indicates:
paste("6. There were", sum(Cust_data_no_dups$BikeBuyer==0), "who did not purchase bikes, and ",
      sum(Cust_data_no_dups$BikeBuyer==1), "who did.")


#       7. Select the correct order (from lowest to highest) that ranks the median YearlyIncome by Occupation:
paste("7. Median yearly income ranked by occupation from lowest to highest:")
#       Will use aggregate function from {stats} library

Rank_Median_Income = aggregate(Cust_data_no_dups[,"YearlyIncome"],
                by = list(Cust_data_no_dups$Occupation),
                FUN = median)
colnames(Rank_Median_Income) = c('Occupation','Median Ann Inc')
Rank_Median_Income = Rank_Median_Income[order(Rank_Median_Income[,2]),] 
Rank_Median_Income

#       8.Based on their age at the time when the data was collected (1st January 1998),
#         which group of customers accounts for the highest AveMonthSpend values?

# Need to create a new colume caculate age (from 1 Jan 1998) and add a new column
Cust_data_no_dups$Age = age_calc(dob = as.Date(Cust_data_no_dups$BirthDate), enddate = as.Date("1998-01-01"), units = "years")

# Need indices for males and females
female_ndx = Cust_data_no_dups$Gender=="F"
male_ndx = Cust_data_no_dups$Gender=="M"

# Need indices for the four age ranges of interest
over_55 = Cust_data_no_dups$Age >55
from_25_to_45 = Cust_data_no_dups$Age >=25 && Cust_data_no_dups$Age <= 45
under_25 = Cust_data_no_dups$Age < 25

# Will create and sort data frame SpendRank
SpendRank = cbind(
        "Group" = c("Females under 25", 
        "Males under 25", 
        "Females between 25 and 45",
        "Males between 25 and 45", 
        "Females over 55 ", 
        "Males over 55 "), 

        "Average_Monthly_Spend" = 
        c(round(mean(Cust_data_no_dups$AveMonthSpend[female_ndx & under_25]), digits = 2),
        round(mean(Cust_data_no_dups$AveMonthSpend[male_ndx & under_25]), digits = 2),
        round(mean(Cust_data_no_dups$AveMonthSpend[female_ndx & from_25_to_45]), digits = 2),
        round(mean(Cust_data_no_dups$AveMonthSpend[male_ndx & from_25_to_45]), digits = 2),
        round(mean(Cust_data_no_dups$AveMonthSpend[female_ndx & over_55]), digits = 2),
        round(mean(Cust_data_no_dups$AveMonthSpend[male_ndx & over_55]), digits = 2) )
        ) # End of cbind

# Ensure SpendRank is data frame
SpendRank = as.data.frame(SpendRank) 

# Ensure SpendRank$Average_Monthly_Spend is numeric 
SpendRank$Average_Monthly_Spend = as.numeric(as.character(SpendRank$Average_Monthly_Spend))

SpendRank = SpendRank[order(SpendRank$Average_Monthly_Spend, decreasing = TRUE),]

paste("8. The group with the highest average monthly spend is", SpendRank[1,1], "with and average spend of", SpendRank[1,2])

#       9. Which of the following statements about AveMonthSpend are true?

#       - Higher median AveMonthSpend: Married over single - TRUE
married_ndx  = Cust_data_no_dups$MaritalStatus == "M"
paste("Median monthly spend married = ", round(median(Cust_data_no_dups$AveMonthSpend[married_ndx]), digits = 2) )
paste("Median monthly spend not married = ", round(median(Cust_data_no_dups$AveMonthSpend[!married_ndx]), digits = 2) )


#       - Higher median AveMonthSpend: No cars over three or more cars - FALSE
three_plus_cars_ndx = Cust_data_no_dups$NumberCarsOwned >= 3
no_cars_ndx = Cust_data_no_dups$NumberCarsOwned == 0
paste("Median monthly spend three or more cars = ", round(median(Cust_data_no_dups$AveMonthSpend[three_plus_cars_ndx]), digits = 2) )
paste("Median monthly spend no cars = ", round(median(Cust_data_no_dups$AveMonthSpend[no_cars_ndx]), digits = 2) )


#       - Higher median AveMonthSpend: Males over females - TRUE
paste("Median monthly spend  males = ", round(median(Cust_data_no_dups$AveMonthSpend[male_ndx]), digits = 2) )
paste("Median monthly spend females = ", round(median(Cust_data_no_dups$AveMonthSpend[female_ndx]), digits = 2) )

#       - Wider range AveMonthSpend: Females over males - FALSE
paste("Spend range males = ", round(max(Cust_data_no_dups$AveMonthSpend[male_ndx]) - min(Cust_data_no_dups$AveMonthSpend[male_ndx]), digits = 2) )
paste("Spend range females = ", round(max(Cust_data_no_dups$AveMonthSpend[female_ndx]) - min(Cust_data_no_dups$AveMonthSpend[female_ndx]), digits = 2) )

#       - Lower Median AveMonthSpend: No children over one or more children - TRUE
child_ndx = Cust_data_no_dups$NumberChildrenAtHome > 0
paste("Median monthly spend no children = ", round(median(Cust_data_no_dups$AveMonthSpend[!child_ndx]), digits = 2) )
paste("Median monthly spend with children = ", round(median(Cust_data_no_dups$AveMonthSpend[child_ndx]), digits = 2) )

#       10. Which of the following statements about BikeBuyer are true?
bike_buyer_ndx = Cust_data_no_dups$BikeBuyer == 1

#       - The median YearlyIncome is higher for customers who bought a bike than for customers who didn't. - TRUE
paste(
        "Is the median yearly income for customers who bought a bike higher than for customers who didn't? -",
        median(Cust_data_no_dups$YearlyIncome[bike_buyer_ndx]) > median(Cust_data_no_dups$YearlyIncome[!bike_buyer_ndx])
)

#       - The median number of cars owned by customers who bought a bike lower than for customers who didn't. - FALSE
paste(
        "Is the median number of cars owned by customers who bought a bike is lower than for customers who didn't? -",
        median(Cust_data_no_dups$NumberCarsOwned[bike_buyer_ndx]) < median(Cust_data_no_dups$NumberCarsOwned[!bike_buyer_ndx])
        )
#       - The most common occupation type for customers who bought a bike is skilled manual. - FALSE
top.occupation = sort(table(Cust_data_no_dups$Occupation[bike_buyer_ndx]), decreasing = TRUE)[1]

paste("The most common occupation type for customers who bought a bike is '",
        names(top.occupation), "' with a total of ", as.vector(top.occupation), " people")

#       - Male customers are more likely to buy bikes than female customers. - TRUE
paste(
        "Are male customers more likely to buy bikes than female customers? -",
        sum(male_ndx & bike_buyer_ndx)/sum(male_ndx) > sum(female_ndx & bike_buyer_ndx)/sum(female_ndx)
)

#       - A maried customer is more likely to buy a bike. - FALSE
paste(
        "Are married customers more likely to buy bikes than unmarried customers? -",
        sum(married_ndx & bike_buyer_ndx)/sum(married_ndx) > sum(!married_ndx & bike_buyer_ndx)/sum(!married_ndx)
)

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
