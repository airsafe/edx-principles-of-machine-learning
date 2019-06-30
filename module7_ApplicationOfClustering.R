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

# Applying Cluster Models
# In this lab you will apply K-means and agglomerative to clustering to finding structure in the automotive data set. 
# Finding meaningful clusters in such a complex data set will prove challenging. The challenge is two-fold. 
# First, the optimal number of clusters must be determined. Then the clusters must be interpreted in some useful manner. 
# These challenges are typical of unsupervised learning.
# 
# Prepare the dataset
# Before you start building and evaluating cluster models, the dataset must be prepared. 
# First, execute the code in the cell below to load the packages required to run the rest of this notebook.
# 
# Note: If you are running in Azure Notebooks, make sure that you run the code in the setup.ipynb notebook at the start of you session to ensure your environment is correctly configured.

## Import packages

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

options(repr.plot.width=4, repr.plot.height=4) # Set the initial plot area dimensions

# The code in the cell below loads a prepared version of the autos dataset which has the the following preprocessing:
#         
#       1. Clean missing values.
#       2. Aggregate categories of certain categorical variables.
# 
# However, for this case, some additional processing is required:
#         
#       1. Select columns of interest to eliminate columns known to not be useful as features.
#       2. Encode the categorical features as dummy variables.
#       3. Log transform certain numeric columns.
#       4. Z-score normalize the numeric features.
#       5. Near zero variance features are removed.
# 
# Execute the code in the cell below to import and select columns of the dataset.

auto_prices = read.csv('Auto_Prices_Preped.csv')
auto_prices = auto_prices[,c('make', 'fuel.type', 'aspiration', 'num.of.doors', 'body.style', 
                             'drive.wheels', 'wheel.base', 'length', 'width', 'height',
                             'curb.weight', 'num.of.cylinders', 'engine.size', 'bore', 
                             'stroke', 'compression.ratio', 'horsepower', 'peak.rpm', 
                             'city.mpg', 'highway.mpg', 'log_price')]
names(auto_prices)
dim(auto_prices)

# Notice that the dataset has 21 columns (dimensions) for a small number of cases, 195. 
# The small number of rows compared to the number of features adds to the challenge of this problem.
# 
# Next, execute the code below to encode the categorical variables as dummy variables.

dummies = dummyVars(highway.mpg ~ ., data = auto_prices)
auto_dummies = data.frame(predict(dummies, newdata = auto_prices))
names(auto_dummies)

# Execute the code in the cell below to logarithmically transform certain numeric columns.

num_cols = c('wheel.base', 'curb.weight', 'engine.size', 'bore', 'stroke', 'horsepower', 
             'city.mpg', 'log_price')
auto_dummies[,num_cols] = lapply(auto_dummies[,num_cols], log)

# Execute the code in the cell below to Z Score normalize the numeric variables using the preProcess function from the Caret package.

num_cols = c('wheel.base', 'length', 'width', 'height', 'curb.weight', 'engine.size',
             'bore', 'stroke', 'compression.ratio', 'horsepower', 'peak.rpm', 'city.mpg',
             'log_price')
preProcValues <- preProcess(auto_dummies[,num_cols], method = c("center", "scale"))

auto_dummies[,num_cols] = predict(preProcValues, auto_dummies[,num_cols])

# Now, execute the code in the cell below to identify near zero variance variables using the nearZeroVar function from the Caret package.

near_zero = nearZeroVar(auto_dummies, freqCut = 95/5, uniqueCut = 10, saveMetrics = TRUE)
near_zero[(near_zero$zeroVar == TRUE) | (near_zero$nzv == TRUE), ]

# The code in the cell below removes columns with near zero variance. The select verb from the R dplyr package along with the starts_with function is used. Execute this code.

auto_dummies = select(auto_dummies, -starts_with("make"), -starts_with("num.of.doors"), 
                      -drive.wheels.4wd, -num.of.cylinders.eight_twelve)

names(auto_dummies)

# Apply K-means clustering
# With the data prepared, you will now create and evaluate a series of K-means clustering models applied to the automotive data set. 
# The code in the cell below computes a k=2 k-means cluster model. 
# The cluster assignments are appended to the data frame. 
# Execute this code.

set.seed(4455)
kmeans_2 = kmeans(auto_dummies, centers = 2)
auto_prices[,'assignment'] = kmeans_2$cluster

# Next, the code in the cell below plots four views of the cluster assignments. 
# With high dimensional data many views are possible. 
# However, given limits of perception it is often best to select a few meaningful views. 
# In this case 4 numeric columns and 1 categorical variable are displayed, for a total of 5 of 25 possible dimensions. 
# The function in the cell below displays 4 projections of the cluster assignments. 
# Fuel type is shown by shape. 
# Legend (scales) are displayed only for cluster assignment to reduce clutter. 
# Execute this code to display the cluster assignments for the K=2 model.

plot_auto_cluster = function(auto_dummies){
        options(repr.plot.width=8, repr.plot.height=4.5)
        grid.arrange(ggplot(auto_dummies, aes_string('city.mpg','log_price')) +
                             geom_point(aes(color = factor(assignment), shape = factor(fuel.type), alpha = 0.2)) +
                             scale_shape(guide = FALSE) + scale_alpha(guide = FALSE),
                     ggplot(auto_dummies, aes_string('curb.weight','log_price')) +
                             geom_point(aes(color = factor(assignment), shape = factor(fuel.type), alpha = 0.2)) +
                             scale_shape(guide = FALSE) + scale_alpha(guide = FALSE),
                     ggplot(auto_dummies, aes_string('curb.weight','city.mpg')) +
                             geom_point(aes(color = factor(assignment), shape = factor(fuel.type), alpha = 0.2)) +
                             scale_shape(guide = FALSE) + scale_alpha(guide = FALSE),
                     ggplot(auto_dummies, aes_string('horsepower','log_price')) +
                             geom_point(aes(color = factor(assignment), shape = factor(fuel.type), alpha = 0.2)) +
                             scale_shape(guide = FALSE) + scale_alpha(guide = FALSE),
                     ncol = 2)
}

plot_auto_cluster(auto_prices)

# The K=2 clustering model has divided the data between high price, low fuel efficiency, high weight and high horsepower autos and ones that have the opposite characteristics. 
# While this clustering is interesting, it can hardly be considered surprising.
# 
# Next, execute the code in the cell below to compute and display the cluster assignments for the K=3 model.

set.seed(4665)
kmeans_3 = kmeans(auto_dummies, centers =3)
auto_prices[,'assignment'] = kmeans_3$cluster
plot_auto_cluster(auto_prices)

# The basic divisions of the dataset between the clusters is similar to the K=2 model case. 
# Diesel autos are shown with circular markers and are largely separated into a cluster.
# 
# Execute the code in the cell below to compute and display the cluster assignments for the K=4 model.

set.seed(475)
kmeans_4 = kmeans(auto_dummies, centers =4)
auto_prices[,'assignment'] = kmeans_4$cluster
plot_auto_cluster(auto_prices)

# There appears to be a bit more overlap between the clusters for this model. 
# Some additional interesting structure is starting to emerge. 
# Primary divisions of these clusters are by price, weight, fuel efficiency, horsepower and fuel type. 
# All of the diesel autos are in two clusters, one with high cost, weight and horse power, and one for lower cost, weight and horse power.
# 
# Execute the code in the cell below to compute and display the cluster assignments for a K=5 model.

set.seed(475)
kmeans_5 = kmeans(auto_dummies, centers = 5)
auto_prices[,'assignment'] = kmeans_5$cluster
plot_auto_cluster(auto_prices)

# The structure of these clusters is rather complex. 
# The general pattern is similar to the K=4 model, but with finer grained division of the cases and more overlap between the clusters.
# 
# Finally, execute the code in the cell below to compute and display the class assignments for the K=6 model.

set.seed(475)
kmeans_6 = kmeans(auto_dummies, centers =6)
auto_prices[,'assignment'] = kmeans_6$cluster
plot_auto_cluster(auto_prices)

# The structure of these clusters follows the general pattern of the K=4 and K=5 models. 
# The difference being that there is a finer grained divisions between the clusters and yet more overlap.
# 
# While these visualizations are interesting, it is hard to select a best model based on just this evidence. 
# To establish a quantitative basis for model selection, you will now compute and compare the within cluster sum of squares (WCSS), between cluster sum of squares (BCSS) and silhouette coefficient (SC) metrics. 
# Execute the code in the cell below and examine the results.
# 
# Then, answer Question 1 on the course page.

dist_mat = dist(auto_dummies)
plot_clust_metrics = function(kmeans_2, kmeans_3, kmeans_4, kmeans_5, kmeans_6){
        options(repr.plot.width=7, repr.plot.height=6) # Set the plot area dimensions
        
        ## Create a data frame with the sum of the WCSS and BCSS and approximate ave SC as columns
        
        kmeans_metrics = data.frame(model = c('k=2', 'k=3', 'k=4', 'k=5', 'k=6'), 
                                    WCSS = c(sum(kmeans_2$withinss), sum(kmeans_3$withinss), sum(kmeans_4$withinss),
                                             sum(kmeans_5$withinss), sum(kmeans_6$withinss)),
                                    BCSS = c(sum(kmeans_2$betweenss), sum(kmeans_3$betweenss), sum(kmeans_4$betweenss),
                                             sum(kmeans_5$betweenss), sum(kmeans_6$betweenss)),
                                    SC = c(mean(silhouette(kmeans_2$cluster, dist_mat)[,3]),
                                           mean(silhouette(kmeans_3$cluster, dist_mat)[,3]),
                                           mean(silhouette(kmeans_4$cluster, dist_mat)[,3]),
                                           mean(silhouette(kmeans_5$cluster, dist_mat)[,3]),
                                           mean(silhouette(kmeans_6$cluster, dist_mat)[,3])))
        ## Create side by side plots of WCSS and BCSS vs. the model
        p_wcss = ggplot(kmeans_metrics, aes(model, WCSS)) + geom_point(size = 3) +
                ggtitle('Within cluster sum of squares \n vs. model')
        p_bcss = ggplot(kmeans_metrics, aes(model, BCSS)) + geom_point(size = 3) +
                ggtitle('Between cluster sum of squares \n vs. model')
        p_sc = ggplot(kmeans_metrics, aes(model, SC)) + geom_point(size = 3) +
                ggtitle('Average silhouette coefficient \n vs. model')
        grid.arrange(p_wcss, p_bcss, p_sc, ncol = 2)
}

plot_clust_metrics(kmeans_2, kmeans_3, kmeans_4, kmeans_5, kmeans_6)

# WCSS decreases and BCSS increases with increasing numbers of clusters. 
# The range of WCSS and BCSS values is relatively narrow. 
# However, the SC is highest for the k=2 and k=3 models. 
# The other models have noticeably lower SC. 
# However, all these SC values are fairly low. 
# Overall, it appears that the k=3 model might be the best overall compromise between these metrics. 
# The greater level of detail with greater numbers of clusters will be important for some applications.
# 
# Apply agglomerative clustering
# Having tried the K-means clustering mode with various numbers of clusters, you will now try agglomerative clustering models. 
# You will compare these models using both visualization and the SC metric.
# 
# The code in the cell below computes a 2 cluster agglomerative model and displays the cluster assignments. Execute this code.

set.seed(7799)
a_clusts = hclust(dist_mat, method = 'average')
agglomerative_2 = cutree(a_clusts, k = 2)
auto_prices[,'assignment'] = agglomerative_2
plot_auto_cluster(auto_prices)

# Examine the above plots and compare them to the cluster assignments for the K=2 K-means model. Whereas the K-means model created an approximately even split of the dataset, the agglomerative clustering model has placed the majority of points in one cluster.
# 
# Next, execute the code in the cell below to compute and display the assignments for the 3 cluster agglomerative model.

agglomerative_3 = cutree(a_clusts, k = 3)
auto_prices[,'assignment'] = agglomerative_3
plot_auto_cluster(auto_prices)

# Examine these plots and compare them to the 2 cluster model. 
# It appears the 3 cluster model has split the larger cluster, but with considerable overlap in these views.
# 
# Execute the code in the cell below to compute and display the cluster assignments for the 4 cluster agglomerative model.

agglomerative_4 = cutree(a_clusts, k = 4)
auto_prices[,'assignment'] = agglomerative_4
plot_auto_cluster(auto_prices)

# Compare these cluster assignments to the 3 cluster model. 
# Notice that low weight, low horsepower and low cost autos have been split into two clusters.
# 
# Execute the code in the cell below to compute and display the cluster assignments for a 5 cluster model.

agglomerative_5 = cutree(a_clusts, k = 5)
auto_prices[,'assignment'] = agglomerative_5
plot_auto_cluster(auto_prices)

# The cases are now split into 5 fairly distinct groups with minimal overlap. 
# Compare each of the four views above to see how the clusters divide these cases.
# 
# Finally, execute the code in the cell below to compute and display the assignments for the 6 cluster agglomerative model.

agglomerative_6 = cutree(a_clusts, k = 6)
auto_prices[,'assignment'] = agglomerative_6
plot_auto_cluster(auto_prices)

# These results appear similar to the 5 cluster model. 
# As should be expected, there is a slightly finer division of some of the cases.
# 
# Finally, execute the code in the cell below to compute and display the SC for the agglomerative clustering models.
# 
# Then, answer Question 2 on the course page.

options(repr.plot.width=4, repr.plot.height=4) # Set the plot area dimensions

SC_metrics = data.frame(model = c('2 cluster', '3 cluster', '4 cluster',
                                  '5 cluster', '6 cluster'),
                        SC = c(mean(silhouette(agglomerative_2, dist_mat)[,3]),
                               mean(silhouette(agglomerative_3, dist_mat)[,3]),
                               mean(silhouette(agglomerative_4, dist_mat)[,3]),
                               mean(silhouette(agglomerative_5, dist_mat)[,3]),
                               mean(silhouette(agglomerative_6, dist_mat)[,3])))

ggplot(SC_metrics, aes(model, SC)) + geom_point(size = 3) +
        ggtitle('Average silhouette coefficient \n vs. model') +
        theme(axis.text.x = element_text(angle = 90, hjust = 1))

# The SC values are in a narrow range. 
# The 2 cluster model has the highest SC. 
# However, the 5 and 6 cluster models exhibit reasonable divisions of the cases and have reasonable SC values. 
# Therefore these models are preferred.
# 
# Summary
# In this lab you have computed, evaluated and compared K-means and agglomerative clustering models with 2, 3, 4, 5 and 6 clusters applied to the automotive dataset. 
# As is often the case with unsupervised learning, it has proven difficult to compare models. 
# It is also challenging to determine the most interesting aspects of data structure discovered by the clustering process.
# 
# Specifically, your analysis discovered:
#         
#         1. The k=3 model appears to be the best compromise between the metrics for the of the k-means models.
#         2. The 5 or 6 cluster agglomerative models appear the be the best of those tried. 
#            As with the K-means model, some interesting structure was revealed, but the SC values were relatively low.
# 
# Cluster analysis of the automotive data can be extended in a number of ways, including:
#         
#         1. Use larger numbers of clusters to determine if finer groupings reveal structure.
#         2. For agglomerative clustering model try other linkage functions and distance metrics.


# ====================
# Processing end time
timeEnd = Sys.time()

# Processing date and total processing time
cat(paste("","Processing end date and time",date(),"","",sep="\n"))
paste("Total processing time =",round(difftime(timeEnd,timeStart), digits=2),"seconds",sep=" ")


# Stop writing to an output file
sink()

################

