#DATA SOURCE:
#[https://developers.themoviedb.org/3/getting-started/popularity#movies]

#LIBRARIES:

library(dplyr)
library(tree)
library(randomForest)

#==================================================================================================================================================================================================================================

#DATA IMPORTATION:
#setwd("C:/Users/39392/Documents/Luca/Università/Laurea Magistrale Stochastics and Data Science/II° Year/I° Semester/Statistical Machine Learning/Project")
movies = read.csv("movies_cleaned.csv")
movies = movies %>% dplyr :: select(-"X")
#View(movies)

#Merge with the genres information:
genres_df = read.csv("genres_df.csv")
to_merge = merge(movies, genres_df, by = "id")
to_merge = to_merge %>% dplyr :: select(-c("X", "original_title.y", "Mystery", "Thriller", "Comedy", "War", "Music", "TV.Movie"))

#Encode the categorical variables as factors:
to_factorize = colnames(to_merge)[c(3, 5, 6, seq(10, length(colnames(to_merge))))]
for(variable in to_factorize){
  to_merge[, variable] = as.factor(to_merge[, variable])
}

#Remove useless variables:
to_merge = dplyr::select(to_merge, -c("id", "original_title.x"))

#==================================================================================================================================================================================================================================

#Outliers analysis:

par(mfrow = c(2, 2))

out_rev = which(movies$revenue %in% boxplot(movies$revenue, main = "Revenue")$out)

out_bud = which(movies$budget %in% boxplot(movies$budget, main = "Budget")$out)

out_run = which(movies$runtime %in% boxplot(movies$runtime, main = "Runtime")$out)

out_fam = which(movies$famous_count %in% boxplot(movies$famous_count, main = "Famous Count")$out)

par(mfrow = c(1, 1))

#Multivariate Outliers (we briefly look at the squared Mahalanobis distances to see if there're some multivariate strange observations): 
numerical = movies %>% dplyr :: select(c(revenue, budget, runtime, famous_count)) #Numerical predictors
numerical = as.matrix(numerical)
numerical = scale(numerical) #We scale data in order to avoid issues related to the different dimensionalities

distances = mahalanobis(numerical, colMeans(numerical), cor(numerical)) #Mahalanobis

#Extract sure Multivariate outliers:
mult_out = which(distances > 100)

#Remove outliers:
remove = c(which(movies$revenue > 1500000000), which(movies$budget > 250000000), which(movies$runtime > 300),
           mult_out) #We consider the multivariate outliers and some univariate's
no_out = to_merge[-remove, ]

#==================================================================================================================================================================================================================================

#CLASSIFICATION TREES:

#Scale data in order to make comparisons with Linear Models:
no_out_s = no_out
no_out_s$revenue = scale(no_out_s$revenue)
no_out_s$budget = scale(no_out_s$budget)
no_out_s$runtime = scale(no_out_s$runtime)
no_out_s$famous_count = scale(no_out_s$famous_count)

#Base model:
#We consider the entire dataset and perform cross validation for different train/test split considering
#each time the deepest possible tree and then pruning it via cross-validation by setting the tree size (number of terminal nodes)
#Once the tree is pruned we get a measure of fit to the train data (Residual Mean Deviance) and also a 
#measure of prediction performance (Root Mean Squared Error)

#Additionally, we keep track of the variables actually used in partitioning the sample space in order to
#get an idea of their importance

K = 10
models = list() #List of pairs containing the couple Train and Test set for each fitted model

#Cross Validation:
#We track the measures before and after pruning the Tree to see the changes induced by pruning procedure:
rmd_bp = c()
rmses_bp = c()
rmses_ns_bp = c()
r_squared_bp = c()

rmd = c()
rmses = c()
rmses_ns = c()
r_squared = c()

used = list()
for(k in seq(1, K)){ #K-fold CV
  
  #Train-Test split (we'll keep track of them to select the best model):
  n = nrow(no_out_s)
  n_train = floor(.80*n)
  training = sample(1:n, size = n_train, replace = F)
  
  movies_train = no_out_s[training, ] #Train set
  movies_test = no_out_s[-training, ] #Test set
  models[[k]] = list(movies_train, movies_test)
  
  #Fit the deepest tree:
  current_model = tree(revenue ~ ., data = movies_train, mindev = 0.005) #We fit the tree in depth by setting a very small value for "mindev"
  
  #Store its performances:
  
  #On Train Set:
  SSR = summary(current_model)$dev #Sum of Squared Residuals
  n_tr = dim(movies_train)[1]
  rmd_bp[k] = SSR / (n_tr - summary(current_model)$size) #Residual Mean Deviance
  
  y_bar = mean(movies_train$revenue)
  SST = sum((movies_train$revenue - y_bar)^2)
  r_squared_bp[k] = 1 - SSR/SST
  
  #On Test Set:
  predictions = predict(current_model, movies_test) #Test Predictions
  rmse = sqrt(mean((predictions - movies_test$revenue)**2)) #Test RMSE
  rmses_bp[k] = rmse
  
  prediction_ns = predictions * sd(no_out$revenue) + mean(no_out$revenue)
  real_ns = movies_test$revenue * sd(no_out$revenue) + mean(no_out$revenue)
  rmses_ns_bp[k] = sqrt(mean((prediction_ns - real_ns)**2)) #Test RMSE not scaled
  
  #Prune the tree selecting the best size via Cross Validation:
  pruning_path = cv.tree(current_model) #Perform Cross Validation
  index = which(pruning_path$dev == min(pruning_path$dev)) #Select the minimum Deviance
  best_size = min(pruning_path$size[index]) #Select the corresponding size
  pruned = prune.tree(current_model, best = best_size) #Pruned Tree
  
  #Track performances of the pruned tree:
  
  #Important Features:
  used[[k]] = as.character(summary(pruned)$used)
  
  #On Train Set:
  SSR = summary(pruned)$dev #Sum of Squared Residuals
  n_tr = dim(movies_train)[1]
  rmd[k] = SSR / (n_tr - best_size) #Residual Mean Deviance
  
  y_bar = mean(movies_train$revenue)
  SST = sum((movies_train$revenue - y_bar)^2)
  r_squared[k] = 1 - SSR/SST
  
  #On Test Set:
  predictions = predict(pruned, movies_test) #Test Predictions
  rmse = sqrt(mean((predictions - movies_test$revenue)**2)) #Test RMSE
  rmses[k] = rmse
  
  prediction_ns = predictions * sd(no_out$revenue) + mean(no_out$revenue)
  real_ns = movies_test$revenue * sd(no_out$revenue) + mean(no_out$revenue)
  rmses_ns[k] = sqrt(mean((prediction_ns - real_ns)**2)) #Test RMSE not scaled
}

#Effects of pruning:
mean(r_squared_bp - r_squared) #Fitting
mean(rmses_bp - rmses) #Predicting

#Average performances:
mean(r_squared) #On Train
mean(rmses) #On Test
mean(rmses_ns)

#Relevant Features:
relevant = c()
for(features in used){
  for (feature in features){
    if(!(feature %in% relevant)){
      relevant = c(relevant, feature)
    }
  }
}
relevant

#Select the best model obtained via Cross Validation:
index = which(r_squared == max(r_squared))
movies_train = models[[index]][[1]]
movies_test = models[[index]][[2]]

#Fit the model:
model = tree(revenue ~ ., data = movies_train, mindev = 0.005)

#Tree structure before pruning:
summary(model)
plot(model)
title(main = "Tree before pruning")
text(model)

#Prune the model:
pruning = cv.tree(model)

plot(pruning, type = "b")
title("Tree's pruning path", line = 3)
title(main = expression(alpha), line = 2)

index_p = which(pruning$dev == min(pruning$dev))
best = min(pruning$size[index_p])
tree.movies = prune.tree(model, best = best)

abline(v = best, col = "red", lwd = 3, lty = "dashed")

#Summary of the Tree:
summary(tree.movies)
plot(tree.movies)
title(main = "Tree after pruning")
text(tree.movies)
summary(tree.movies)$used

#Prediction performances:
predictions = predict(tree.movies, movies_test)
plot(predictions, movies_test$revenue, main = "Real Values vs Predictions", xlab = "Test predictions", ylab = "Real values")

#==================================================================================================================================================================================================================================

#RANDOM FOREST:
#We perform cross validation to find the best Rand Forest Model

#Remarks on the model fit:
#We initially train the model using all the possible predictors and then we try to use just the
#relevant ones to see

#We train the model on scaled data always to be able to compare them with the linear regression results
#Then the RMSE on the Test set will be computed also on the original scale

#We subset the predictors used at each split randomly using m = p/3 variables each time

#We always select the best model in terms of R-squared index

#For each Forest we track:
#R-squared
#RMSE on test (scaled and not scaled)
#Importance of features
#Optimal number of Trees
#Train-Test partition

#Entire predictor set:

#Cross Validation:
models_rf = list()
r_squared_rf = c()
rmses_rf = c()
rmses_ns_rf = c()
imp_features = list() #list of matrices containing the relevance of features for both the available metrics
n_trees = c()
for(k in seq(1, K)){ #K-fold CV
  
  #Train-Test split (we'll keep track of them to select the best model):
  n = nrow(no_out_s)
  n_train = floor(.80*n)
  training = sample(1:n, size = n_train, replace = F)
  
  movies_train = no_out_s[training, ] #Train set
  movies_test = no_out_s[-training, ] #Test set
  models_rf[[k]] = list(movies_train, movies_test)
  
  #Fit the Forest:
  current_model = randomForest(revenue ~ ., data = movies_train, importance = TRUE)
  
  #Check progresses:
  cat("Iteration Number ")
  cat(k)
  cat("\n============\n")
  
  #Store its performances:
  
  #On Train Set:
  r_squared_rf[k] = current_model$rsq[length(current_model$rsq)] #R-squared obtained using all the trees fitted
  n_trees[k] = which(current_model$rsq == max(current_model$rsq))
  imp_features[[k]] = current_model$importance
  
  #On Test Set:
  predictions = predict(current_model, newdata = movies_test) #Test Predictions
  rmse = sqrt(mean((predictions - movies_test$revenue)**2)) #Test RMSE
  rmses_rf[k] = rmse
  
  prediction_ns = predictions * sd(no_out$revenue) + mean(no_out$revenue)
  real_ns = movies_test$revenue * sd(no_out$revenue) + mean(no_out$revenue)
  rmses_ns_rf[k] = sqrt(mean((prediction_ns - real_ns)**2, na.rm = T)) #Test RMSE not scaled
}

#Average Performances:
mean(r_squared_rf) #On Train

mean(rmses_rf) #On Test
mean(rmses_ns_rf)

mean(n_trees) #Tuning parameter

#Summary of Features Importance over over different Cross-Validation sets:
features = colnames(no_out_s)[-5] #Extract features
summary_imp = c()
for(feature in features){ #For any feature
  importance = c(0, 0)
  for(k in seq(1, K)){ #Sum the two measures of importance
    importance[1] = importance[1] + imp_features[[k]][feature, 1]
    importance[2] = importance[2] + imp_features[[k]][feature, 2]
  }
  
  #Average them
  importance[1] = 100 * (importance[1] / K) #Express in percentage
  importance[2] = importance[2] / K
  
  summary_imp = rbind(summary_imp, importance) #Build the summary
}

rownames(summary_imp) = features #Correct names
colnames(summary_imp) = colnames(imp_features[[1]]) #Correct metrics
round(summary_imp, digits = 3)

#Select the best Forest:
index_rf = which(r_squared_rf == max(r_squared_rf))
movies_train_rf = models_rf[[index_rf]][[1]]
movies_test_rf = models_rf[[index_rf]][[2]]

forest = randomForest(revenue ~ ., data = movies_train_rf, importance = TRUE)

#Train fit:
forest$rsq[length(forest$rsq)] 

par(mfrow = c(2, 1))

plot(forest$rsq, main = "R-squared evolution", ylab = "R-squared", xlab = "Number of Trees", 
     type = "l", lwd = 2) #R-squared evolution
abline(v = which(forest$rsq == max(forest$rsq)), lwd = 3, lty = "dashed", col = "blue")
text(x = which(forest$rsq == max(forest$rsq)), y = 0.5, pos = 4, labels = "max", col = "blue")
abline(v = 100, lwd = 3, lty = "dashed", col = "red")
abline(v = 200, lwd = 3, lty = "dashed", col = "red")
text(x = 150, y = 0.55, pos = 1, labels = "Acceptable\nrange", col = "red")

plot(forest$mse, main = "MSE evolution", ylab = "MSE", xlab = "Number of Trees",
     type = "l", lwd = 2) #MSE Evolution
abline(v = which(forest$mse == min(forest$mse)), lwd = 3, lty = "dashed", col = "blue")
text(which(forest$rsq == max(forest$rsq)), y = 0.45, pos = 4, labels = "min", col = "blue")
abline(v = 100, lwd = 3, lty = "dashed", col = "red")
abline(v = 200, lwd = 3, lty = "dashed", col = "red")
text(x = 150, y = 0.5, pos = 1, labels = "Acceptable\nrange", col = "red")

par(mfrow = c(1, 1))

#Feature Importance Plot:
varImpPlot(forest, main = "Features Importance of the best Random Forest")

#More beautiful plot:
par(mar = c(5,6.2,3,1)+.1)
par(mfrow = c(1, 2))

feature_1 = sort(forest$importance[, 1] / forest$importanceSD)
names(feature_1)[6] = "Sci.Fi"
names(feature_1)[9] = "language"
names(feature_1)[15] = "companies"
names(feature_1)[18] = "famous"
names(feature_1)[19] = "collections"

feature_2 = sort(forest$importance[, 2])
names(feature_2)[6] = "language"
names(feature_2)[8] = "Sci.Fi"
names(feature_2)[16] = "companies"
names(feature_2)[17] = "collections"
names(feature_2)[19] = "famous"

barplot(feature_1, horiz = T, las = 2)
title("Percentage increase in MSE")

barplot(feature_2, horiz = T, las = 2, col = "red")
title("Node purity")

#Test Predictions:
predictions = predict(forest, newdata = movies_test_rf)
plot(predictions, movies_test_rf$revenue, main = "Real Values vs Predictions", xlab = "Test predictions", ylab = "Real values")
abline(b = 1, a = 0, lwd = 3, col = "red")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Model with lower predictors:
#We select only the predictors that have been found as relevant both by the Decision Tree and the
#Random Forest Models

useful = c(relevant, c("budget", "belongs_to_collection", "famous_count", "runtime", "production_companies",
           "Drama", "Animation"))
lower = dplyr::select(no_out_s, c(useful, "revenue"))

#Cross Validation:
models_rf_l = list()
r_squared_rf_l = c()
rmses_rf_l = c()
rmses_ns_rf_l = c()
imp_features_l = list() #list of matrices containing the relevance of features for both the available metrics
n_trees_l = c()
for(k in seq(1, K)){ #K-fold CV
  
  #Train-Test split (we'll keep track of them to select the best model):
  n = nrow(lower)
  n_train = floor(.80*n)
  training = sample(1:n, size = n_train, replace = F)
  
  movies_train = lower[training, ] #Train set
  movies_test = lower[-training, ] #Test set
  models_rf_l[[k]] = list(movies_train, movies_test)
  
  #Fit the Forest:
  current_model = randomForest(revenue ~ ., data = movies_train, importance = TRUE)
  
  #Check progresses:
  cat("Iteration Number ")
  cat(k)
  cat("\n============\n")
  
  #Store its performances:
  
  #On Train Set:
  r_squared_rf_l[k] = current_model$rsq[length(current_model$rsq)] #R-squared obtained using all the trees fitted
  n_trees_l[k] = which(current_model$rsq == max(current_model$rsq))
  imp_features_l[[k]] = current_model$importance
  
  #On Test Set:
  predictions = predict(current_model, newdata = movies_test) #Test Predictions
  rmse = sqrt(mean((predictions - movies_test$revenue)**2)) #Test RMSE
  rmses_rf_l[k] = rmse
  
  prediction_ns = predictions * sd(no_out$revenue) + mean(no_out$revenue)
  real_ns = movies_test$revenue * sd(no_out$revenue) + mean(no_out$revenue)
  rmses_ns_rf_l[k] = sqrt(mean((prediction_ns - real_ns)**2, na.rm = T)) #Test RMSE not scaled
}

#Average Performances:
mean(r_squared_rf_l) #On Train

mean(rmses_rf_l) #On Test
mean(rmses_ns_rf_l)

mean(n_trees_l) #Tuning parameter

#Summary of Features Importance over over different Cross-Validation sets:
features = colnames(lower)[-8] #Extract features
summary_imp_l = c()
for(feature in features){ #For any feature
  importance = c(0, 0)
  for(k in seq(1, K)){ #Sum the two measures of importance
    importance[1] = importance[1] + imp_features_l[[k]][feature, 1]
    importance[2] = importance[2] + imp_features_l[[k]][feature, 2]
  }
  
  #Average them
  importance[1] = 100 * (importance[1] / K) #Express in percentage
  importance[2] = importance[2] / K
  
  summary_imp_l = rbind(summary_imp_l, importance) #Build the summary
}

rownames(summary_imp_l) = features #Correct names
colnames(summary_imp_l) = colnames(imp_features[[1]]) #Correct metrics
round(summary_imp_l, digits = 3)

#Select the best Forest:
index_rf_l = which(r_squared_rf_l == max(r_squared_rf_l))
movies_train_rf_l = models_rf_l[[index_rf_l]][[1]]
movies_test_rf_l = models_rf_l[[index_rf_l]][[2]]

forest_l = randomForest(revenue ~ ., data = movies_train_rf_l, importance = TRUE)
forest_l

#Train fit:
forest_l$rsq[length(forest_l$rsq)] 

par(mfrow = c(2, 1))

plot(forest_l$rsq, main = "R-squared evolution", ylab = "R-squared", xlab = "Number of Trees") #R-squared evolution
abline(v = which(forest_l$rsq == max(forest_l$rsq)), lwd = 3, lty = "dashed", col = "red")
plot(forest_l$mse, main = "MSE evolution", ylab = "MSE", xlab = "Number of Trees") #MSE Evolution
abline(v = which(forest_l$mse == min(forest_l$mse)), lwd = 3, lty = "dashed", col = "blue")

par(mfrow = c(1, 1))

#Feature Importance Plot:
varImpPlot(forest_l, main = "Features Importance of the best Random Forest")

#Test Predictions:
predictions = predict(forest_l, newdata = movies_test_rf_l)
plot(predictions, movies_test_rf_l$revenue, main = "Real Values vs Predictions", xlab = "Test predictions", ylab = "Real values")
abline(b = 1, a = 0, lwd = 3, col = "red")

#==================================================================================================================================================================================================================================

#BAGGING:

#We reproduce the same steps seen for Random Forests but considering all the predictors ar each
#cut (bagging with more correlated trees):

#Entire predictor set:

#Cross Validation:
models_bag = list()
r_squared_bag = c()
rmses_bag = c() 
rmses_ns_bag = c()
for(k in seq(1, K)){ #K-fold CV
  
  #Train-Test split (we'll keep track of them to select the best model):
  n = nrow(no_out_s)
  n_train = floor(.80*n)
  training = sample(1:n, size = n_train, replace = F)
  
  movies_train = no_out_s[training, ] #Train set
  movies_test = no_out_s[-training, ] #Test set
  models_bag[[k]] = list(movies_train, movies_test)
  
  #Fit the Bagging Model (m = p):
  current_model = randomForest(revenue ~ ., data = movies_train, importance = TRUE, mtry = dim(no_out_s)[2] - 1)
  
  #Check progresses:
  cat("Iteration Number ")
  cat(k)
  cat("\n============\n")
  
  #Store its performances:
  
  #On Train Set:
  r_squared_bag[k] = current_model$rsq[length(current_model$rsq)] #R-squared obtained using all the trees fitted

  #On Test Set:
  predictions = predict(current_model, newdata = movies_test) #Test Predictions
  rmse = sqrt(mean((predictions - movies_test$revenue)**2)) #Test RMSE
  rmses_bag[k] = rmse
  
  prediction_ns = predictions * sd(no_out$revenue) + mean(no_out$revenue)
  real_ns = movies_test$revenue * sd(no_out$revenue) + mean(no_out$revenue)
  rmses_ns_bag[k] = sqrt(mean((prediction_ns - real_ns)**2, na.rm = T)) #Test RMSE not scaled
}

#Average Performances:
mean(r_squared_bag) #On Train

mean(rmses_bag) #On Test
mean(rmses_ns_bag)

#Select the best Bagging Model:
index_bag = which(r_squared_bag == max(r_squared_bag))
movies_train_bag = models_bag[[index_bag]][[1]]
movies_test_bag = models_bag[[index_bag]][[2]]

bagging = randomForest(revenue ~ ., data = movies_train_bag, importance = TRUE, mtry = dim(no_out_s)[2] - 1)
bagging

#Train fit:
bagging$rsq[length(bagging$rsq)] 

par(mfrow = c(2, 1))

plot(bagging$rsq, main = "R-squared evolution", ylab = "R-squared", xlab = "Number of Trees") #R-squared evolution
abline(v = which(bagging$rsq == max(bagging$rsq)), lwd = 3, lty = "dashed", col = "red")
plot(bagging$mse, main = "MSE evolution", ylab = "MSE", xlab = "Number of Trees") #MSE Evolution
abline(v = which(bagging$mse == min(bagging$mse)), lwd = 3, lty = "dashed", col = "blue")

par(mfrow = c(1, 1))

#Feature Importance Plot:
varImpPlot(bagging, main = "Features Importance of the best Random Forest")

#Test Predictions:
predictions = predict(bagging, newdata = movies_test_bag)
plot(predictions, movies_test_bag$revenue, main = "Real Values vs Predictions", xlab = "Test predictions", ylab = "Real values")
abline(b = 1, a = 0, lwd = 3, col = "red")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Model with lower predictors:

#Cross Validation:
models_bag_l = list()
r_squared_bag_l = c()
rmses_bag_l = c() 
rmses_ns_bag_l = c()
for(k in seq(1, K)){ #K-fold CV
  
  #Train-Test split (we'll keep track of them to select the best model):
  n = nrow(lower)
  n_train = floor(.80*n)
  training = sample(1:n, size = n_train, replace = F)
  
  movies_train = lower[training, ] #Train set
  movies_test = lower[-training, ] #Test set
  models_bag_l[[k]] = list(movies_train, movies_test)
  
  #Fit the Bagging Model (m = p):
  current_model = randomForest(revenue ~ ., data = movies_train, importance = TRUE, mtry = dim(lower)[2] - 1)
  
  #Check progresses:
  cat("Iteration Number ")
  cat(k)
  cat("\n============\n")
  
  #Store its performances:
  
  #On Train Set:
  r_squared_bag_l[k] = current_model$rsq[length(current_model$rsq)] #R-squared obtained using all the trees fitted
  
  #On Test Set:
  predictions = predict(current_model, newdata = movies_test) #Test Predictions
  rmse = sqrt(mean((predictions - movies_test$revenue)**2)) #Test RMSE
  rmses_bag_l[k] = rmse
  
  prediction_ns = predictions * sd(no_out$revenue) + mean(no_out$revenue)
  real_ns = movies_test$revenue * sd(no_out$revenue) + mean(no_out$revenue)
  rmses_ns_bag_l[k] = sqrt(mean((prediction_ns - real_ns)**2, na.rm = T)) #Test RMSE not scaled
}

#Average Performances:
mean(r_squared_bag_l) #On Train

mean(rmses_bag_l) #On Test
mean(rmses_ns_bag_l)

#Select the best Bagging Model:
index_bag_l = which(r_squared_bag_l == max(r_squared_bag_l))
movies_train_bag_l = models_bag_l[[index_bag_l]][[1]]
movies_test_bag_l = models_bag_l[[index_bag_l]][[2]]

bagging_l = randomForest(revenue ~ ., data = movies_train_bag_l, importance = TRUE, mtry = dim(lower)[2] - 1)
bagging_l

#Train fit:
bagging_l$rsq[length(bagging_l$rsq)] 

par(mfrow = c(2, 1))

plot(bagging_l$rsq, main = "R-squared evolution", ylab = "R-squared", xlab = "Number of Trees") #R-squared evolution
abline(v = which(bagging_l$rsq == max(bagging_l$rsq)), lwd = 3, lty = "dashed", col = "red")
plot(bagging_l$mse, main = "MSE evolution", ylab = "MSE", xlab = "Number of Trees") #MSE Evolution
abline(v = which(bagging_l$mse == min(bagging_l$mse)), lwd = 3, lty = "dashed", col = "blue")

par(mfrow = c(1, 1))

#Feature Importance Plot:
varImpPlot(bagging_l, main = "Features Importance of the best Random Forest")

#Test Predictions:
predictions = predict(bagging_l, newdata = movies_test_bag_l)
plot(predictions, movies_test_bag_l$revenue, main = "Real Values vs Predictions", xlab = "Test predictions", ylab = "Real values")
abline(b = 1, a = 0, lwd = 3, col = "red")

#==================================================================================================================================================================================================================================

#SUMMARY OF THE DIFFERENT MODELS:
train = c(mean(r_squared), mean(r_squared_bag), mean(r_squared_bag_l), mean(r_squared_rf), 
          mean(r_squared_rf_l))
test1 = c(mean(rmses), mean(rmses_bag), mean(rmses_bag_l), mean(rmses_rf), mean(rmses_rf_l))
test2 = c(mean(rmses_ns), mean(rmses_ns_bag), mean(rmses_ns_bag_l), mean(rmses_ns_rf), mean(rmses_ns_rf_l))

comparison_r = cbind(train, test1, test2)
rownames(comparison_r) = c("Decision Tree", "Bagging", "Bagging (Relevant Predictors)", 
                         "Random Forest", "Random Forest (Relevant Predictors)")
colnames(comparison_r) = c("R-squared", "RMSE", "RMSE (not scaled)")
comparison_r

#==================================================================================================================================================================================================================================
