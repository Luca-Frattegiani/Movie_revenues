#DATA SOURCE:
#[https://developers.themoviedb.org/3/getting-started/popularity#movies]

#LIBRARIES:

library(dplyr)
library(ggplot2)
library(stringr)
library(cvms)
library(ggpubr)
library(glmnet)
library(car)
library(MASS)

#==================================================================================================================================================================================================================================

#DATA IMPORTATION:
#setwd("C:/Users/39392/Documents/Luca/Università/Laurea Magistrale Stochastics and Data Science/II° Year/I° Semester/Statistical Machine Learning/Project")
movies = read.csv("movies_cleaned.csv")
movies = movies %>% dplyr :: select(-"X")
#View(movies)

#Set factor type for categorical predictors:
categoricals = colnames(movies)[c(3, 5, 6)]
for(variable in categoricals){
  movies[, variable] = as.factor(movies[, variable])
}

#==================================================================================================================================================================================================================================

#Outliers analysis:

par(mfrow = c(2, 2))

out_rev = which(movies$revenue %in% boxplot(movies$revenue, main = "Revenue")$out)

out_bud = which(movies$budget %in% boxplot(movies$budget, main = "Budget")$out)

out_run = which(movies$runtime %in% boxplot(movies$runtime, main = "Runtime")$out)

out_fam = which(movies$famous_count %in% boxplot(movies$famous_count, main = "Famous count")$out)

par(mfrow = c(1, 1))

#Multivariate Outliers (we briefly look at the squared Mahalanobis distances to see if there're some multivariate strange observations): 
numerical = movies %>% dplyr :: select(c(revenue, budget, runtime, famous_count)) #Numerical predictors
numerical = as.matrix(numerical)
numerical = scale(numerical) #We scale data in order to avoid issues related to the different dimensionalities

distances = mahalanobis(numerical, colMeans(numerical), cor(numerical)) #Mahalanobis

#Extract sure Multivariate outliers:
mult_out = which(distances > 100)

#==================================================================================================================================================================================================================================

#USUAL LINEAR REGRESSION:
#Each model is performed using only scaled data (to improve the interpretation of coefficients) and allow
#better comparisons with the subsequent models that require necessarily scaled data,
#Anyway to get a better idea of the model performances especially on the test set, we'll compute
#the RMSE also on the original scale
#We produce a K-fold cross-validation for each model, computing the mean for the metrics of interest and 
#then choosing the best model
#Metrics considered: R^2 (on trained data), RMSE and RMSE normalized (test), MAPE (scale invariant)
#We analyze the diagnostic plot (of the best model, according to its R^2)
#We show graphically the matching with the predictions (of the best model)

#Scaling:
movies_s = movies
movies_s$revenue = scale(movies_s$revenue)
movies_s$budget = scale(movies_s$budget)
movies_s$runtime = scale(movies_s$runtime)
movies_s$famous_count = scale(movies_s$famous_count)

#NO GENRES, WITH OUTLIERS:

#Cross Validation:
K = 10
models = list() #List of pairs containing the couple Train and Test set for each fitted model
r_squared = c()
rmses = c() #RMSE on scaled data
rmses_ns = c() #RMSE on not scaled data
for(k in seq(1, K)){ #K-fold CV
  
  #Train-Test split (we'll keep track of them to select the best model):
  n = nrow(movies)
  n_train = floor(.80*n)
  training = sample(1:n, size = n_train, replace = F)
  
  movies_train = movies_s[training, ] #Train set
  movies_test = movies_s[-training, ] #Test set
  models[[k]] = list(movies_train, movies_test)
  
  #Fit the model:
  current_model = lm(revenue ~ belongs_to_collection + budget + original_language + 
                   production_companies + runtime + famous_count, data = movies_train)
  
  #Track the R^2:
  r_squared[k] = summary(current_model)$r.squared
  
  #Track the normalized Root Mean Squared Errors:
  
  #Test set:
  predictions = predict(current_model, movies_test) #Test Predictions
  rmse = sqrt(mean((predictions - movies_test$revenue)**2)) #Test RMSE scaled
  rmses[k] = rmse

  prediction_ns = predictions * sd(movies$revenue) + mean(movies$revenue)
  real_ns = movies_test$revenue * sd(movies$revenue) + mean(movies$revenue)
  rmses_ns[k] = sqrt(mean((prediction_ns - real_ns)**2)) #Test RMSE not scaled  
}

#Mean Values scored:
mean(r_squared)
mean(rmses)
mean(rmses_ns)

#Select the best model obtained via Cross Validation and look at the coefficients:
index = which(r_squared == max(r_squared))
movies_train = models[[index]][[1]]
movies_test = models[[index]][[2]]
movies.lm = lm(revenue ~ belongs_to_collection + budget + original_language + 
                 production_companies + runtime + famous_count, data = movies_train)

summary(movies.lm)

#Diagnostic Plot:
par(mfrow = c(2, 2))
plot(movies.lm, ask = F)
par(mfrow = c(1, 1))
#plot(rstudent(movies.lm) ~ hatvalues(movies.lm)) #Cercare di interpretare questo plot

#Plot the Prediction vs Real on Test set:
predictions = predict(movies.lm, movies_test) #Predictions
plot(predictions, movies_test$revenue, main = "Real Values vs Predictions", xlab = "Test predictions", ylab = "Real values")
abline(b = 1, a = 0, lwd = 3, col = "red")

#Brief diagnostic considerations:

#Hypothesis:
#Multivariate Normality: Not respected both for the predictors and the target (as we can detect from the
#univariate EDA's plot and from the QQ-plot above, especially for large quantiles)

#Zero-Mean Residuals: From the residuals vs fitted diagnostic plot, we can detect that this assumption is not
#perfectly respected especially for the highest fitted values where the residuals seems to be higher

#Homoschedasticity: This assumption is clearly not respected looking at the Scale-Location Plot, indeed variance
#of the error component show an increase value as the predicted output grows

#Multicollinearity among predictors: Not particularly relevant in our case as we can notice from the Correlation
#Matrix

#Linearity with the target: Fairly respected especially for budget and a little bit for famous_count

#Independent Normal Errors: from the Residuals vs Fitted we see a not strong but still evident pattern between
#the values of the residuals and the magnitude of the predictions.

#Outliers:
#Also from the Real vs Fitted plot we can see some few particularly strange (high) observations

#Influencial points:
#The model doesn't seem to be affected by influencial points since leverages and Cook's distances hire normal/low
#values

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#NO GENRES, NO OUTLIERS:

#Remove the outliers:
remove = c(which(movies$revenue > 1500000000), which(movies$budget > 250000000), 
           which(movies$runtime > 300), mult_out) #We consider the multivariate outliers and some univariate
no_out = movies[-remove, ]

#Scale the new dataset:
no_out_s = no_out
no_out_s$revenue = scale(no_out_s$revenue)
no_out_s$budget = scale(no_out_s$budget) 
no_out_s$runtime = scale(no_out_s$runtime)
no_out_s$famous_count = scale(no_out_s$famous_count)

#Cross Validation:
models_no = list() #List of pairs containing the couple Train and Test set for each fitted model
r_squared_no = c()
rmses_no = c()
rmses_ns_no = c() #RMSE on not scaled data
for(k in seq(1, K)){ #K-fold CV
  
  #Train-Test split (we'll keep track of them to select the best model):
  n = nrow(no_out)
  n_train = floor(.80*n)
  training = sample(1:n, size = n_train, replace = F)
  
  movies_train = no_out_s[training, ] #Train set
  movies_test = no_out_s[-training, ] #Test set
  models_no[[k]] = list(movies_train, movies_test)
  
  #Fit the model:
  current_model = lm(revenue ~ belongs_to_collection + budget + original_language + 
                       production_companies + runtime + famous_count, data = movies_train)
  
  #Track the R^2:
  r_squared_no[k] = summary(current_model)$r.squared
  
  #Track the normalized Root Mean Squared Errors:
  
  #Test set:
  predictions = predict(current_model, movies_test) #Test Predictions
  rmse = sqrt(mean((predictions - movies_test$revenue)**2)) #Test RMSE
  rmses_no[k] = rmse
  
  prediction_ns = predictions * sd(no_out$revenue) + mean(no_out$revenue)
  real_ns = movies_test$revenue * sd(no_out$revenue) + mean(no_out$revenue)
  rmses_ns_no[k] = sqrt(mean((prediction_ns - real_ns)**2)) #Test RMSE not scaled
}

#Mean Values scored:
mean(r_squared_no)
mean(rmses_no) 
mean(rmses_ns_no)

#Select the best model obtained via Cross Validation:
index_no = which(r_squared_no == max(r_squared_no))
movies_train_no = models_no[[index_no]][[1]]
movies_test_no = models_no[[index_no]][[2]]
movies.lm_no = lm(revenue ~ belongs_to_collection + budget + original_language + 
                 production_companies + runtime + famous_count, data = movies_train_no)
summary(movies.lm_no)

#Diagnostic Plot:
par(mfrow = c(2, 2))
plot(movies.lm_no, ask = F)
par(mfrow = c(1, 1))
#plot(rstudent(movies.lm) ~ hatvalues(movies.lm)) #Cercare di interpretare questo plot

#Plot the Prediction vs Real on Test set:
predictions_no = predict(movies.lm_no, movies_test_no) #Predictions
plot(predictions_no, movies_test_no$revenue, main = "Real Values vs Predictions", xlab = "Test predictions", ylab = "Real values")
abline(b = 1, a = 0, lwd = 3, col = "red")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#NO OUTLIERS WITH ALL GENRES:

#Merge genres information (and scaling):
genres_df = read.csv("genres_df.csv")
to_merge = merge(no_out, genres_df, by = "id")

to_merge = to_merge %>% dplyr :: select(-c("X", "original_title.y"))
#View(to_merge)

#Encode genres as factors:
genres = colnames(to_merge)[seq(10, length(colnames(to_merge)))]
for(genre in genres){
  to_merge[, genre] = as.factor(to_merge[, genre])
}

#Scale:
to_merge_s = to_merge
to_merge_s$revenue = scale(to_merge_s$revenue)
to_merge_s$budget = scale(to_merge_s$budget)
to_merge_s$runtime = scale(to_merge_s$runtime)
to_merge_s$famous_count = scale(to_merge_s$famous_count)

#Cross Validation:
models_no_g = list() #List of pairs containing the couple Train and Test set for each fitted model
r_squared_no_g = c()
rmses_no_g = c()
rmses_ns_no_g = c()
for(k in seq(1, K)){ #K-fold CV
  
  #Train-Test split (we'll keep track of them to select the best model):
  n = nrow(no_out)
  n_train = floor(.80*n)
  training = sample(1:n, size = n_train, replace = F)
  
  movies_train = dplyr :: select(to_merge_s[training, ], -c(id, original_title.x, TV.Movie)) #Train set 
  movies_test = dplyr :: select(to_merge_s[-training, ], -c(id, original_title.x, TV.Movie)) #Test set
  models_no_g[[k]] = list(movies_train, movies_test)
  
  #Fit the model:
  current_model = lm(revenue ~ ., data = movies_train)
  
  #Track the R^2:
  r_squared_no_g[k] = summary(current_model)$r.squared
  
  #Track the normalized Root Mean Squared Errors:
  
  #Test set:
  predictions = predict(current_model, movies_test) #Test Predictions
  rmse = sqrt(mean((predictions - movies_test$revenue)**2)) #Test RMSE
  rmses_no_g[k] = rmse
  prediction_ns = predictions * sd(to_merge$revenue) + mean(to_merge$revenue)
  real_ns = movies_test$revenue * sd(to_merge$revenue) + mean(to_merge$revenue)
  rmses_ns_no_g[k] = sqrt(mean((prediction_ns - real_ns)**2)) #Test RMSE not scaled    
}

#Mean Values scored:
mean(r_squared_no_g)
mean(rmses_no_g)
mean(rmses_ns_no_g)

#Select the best model obtained via Cross Validation:
index_no_g = which(r_squared_no_g == max(r_squared_no_g))
movies_train_no_g = models_no_g[[index_no_g]][[1]]
movies_test_no_g = models_no_g[[index_no_g]][[2]]
movies.lm_no_g = lm(revenue ~ ., data = movies_train_no_g)

summary(movies.lm_no_g)

#Diagnostic Plot:
par(mfrow = c(2, 2))
plot(movies.lm_no_g, ask = F)
par(mfrow = c(1, 1))
#plot(rstudent(movies.lm) ~ hatvalues(movies.lm)) #Cercare di interpretare questo plot

#Plot the Prediction vs Real on Test set:
predictions_no_g = predict(movies.lm_no_g, movies_test_no_g) #Predictions
plot(predictions_no_g, movies_test_no_g$revenue, main = "Real Values vs Predictions", xlab = "Test predictions", ylab = "Real values")
abline(b = 1, a = 0, lwd = 3, col = "red")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#NO OUTLIERS, ONLY RELEVANT GENRES:

#We remove the genres with both a low impact on the mean explored in the EDA and non significant coefficients:
#-(mystery, thriller, comedy, war, music) and also TV Movie which show just one occurence 

#Cross Validation:
models_no_pg = list() #List of pairs containing the couple Train and Test set for each fitted model
r_squared_no_pg = c()
rmses_no_pg = c()
rmses_ns_no_pg = c()
for(k in seq(1, K)){ #K-fold CV
  
  #Train-Test split (we'll keep track of them to select the best model):
  n = nrow(no_out)
  n_train = floor(.80*n)
  training = sample(1:n, size = n_train, replace = F)
  
  movies_train = dplyr :: select(to_merge_s[training, ], -c(id, original_title.x, Mystery, Thriller, Comedy, War, Music, TV.Movie)) #Train set 
  movies_test = dplyr :: select(to_merge_s[-training, ], -c(id, original_title.x, Mystery, Thriller, Comedy, War, Music, TV.Movie)) #Test set
  models_no_pg[[k]] = list(movies_train, movies_test)
  
  #Fit the model:
  current_model = lm(revenue ~ ., data = movies_train)
  
  #Track the R^2:
  r_squared_no_pg[k] = summary(current_model)$r.squared
  
  #Track the normalized Root Mean Squared Errors:
  
  #Test set:
  predictions = predict(current_model, movies_test) #Test Predictions
  rmse = sqrt(mean((predictions - movies_test$revenue)**2)) #Test RMSE
  rmses_no_pg[k] = rmse
  prediction_ns = predictions * sd(to_merge$revenue) + mean(to_merge$revenue)
  real_ns = movies_test$revenue * sd(to_merge$revenue) + mean(to_merge$revenue)
  rmses_ns_no_pg[k] = sqrt(mean((prediction_ns - real_ns)**2)) #Test RMSE not scaled  
}

#Mean Values scored:
mean(r_squared_no_pg)
mean(rmses_no_pg)
mean(rmses_ns_no_pg)

#Select the best model obtained via Cross Validation:
index_no_pg = which(r_squared_no_pg == max(r_squared_no_pg))
movies_train_no_pg = models_no_pg[[index_no_pg]][[1]]
movies_test_no_pg = models_no_pg[[index_no_pg]][[2]]
movies.lm_no_pg = lm(revenue ~ ., data = movies_train_no_pg)

summary(movies.lm_no_pg)

#Diagnostic Plot:
par(mfrow = c(2, 2))
plot(movies.lm_no_pg, ask = F)
par(mfrow = c(1, 1))
#plot(rstudent(movies.lm) ~ hatvalues(movies.lm)) #Cercare di interpretare questo plot

#Plot the Prediction vs Real on Test set:
predictions_no_pg = predict(movies.lm_no_pg, movies_test_no_pg) #Predictions
plot(predictions_no_pg, movies_test_no_pg$revenue, main = "Real Values vs Predictions", xlab = "Test predictions", ylab = "Real values")
abline(b = 1, a = 0, lwd = 3, col = "red")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ANOVA TEST:
#We aim to compare the reduction in RSS between the different models trained 
#(without including genres, with all genres, with partial genres)

#Result:
#It seems that the best model is the one that includes all the genres, but since differences are 
#considerably low, we can use the model with a lower number of genres to have an easier interpretation of the coefficients

anova(movies.lm_no, movies.lm_no_g, movies.lm_no_pg)

#==================================================================================================================================================================================================================================

#INTERACTIONS (REMOVE ADDITIVE ASSUMPTION):
#We try to include some interactions of the first order:
#Here we always scale data in order to reduce possible multicollinearity that comes from interactions

#Interaction between Production Companies and Famous Count:

#Cross Validation:
models_no_pg_pf = list() #List of pairs containing the couple Train and Test set for each fitted model
r_squared_no_pg_pf = c()
rmses_no_pg_pf = c()
rmses_ns_no_pg_pf = c()
for(k in seq(1, K)){ #K-fold CV
  
  #Train-Test split (we'll keep track of them to select the best model):
  n = nrow(no_out)
  n_train = floor(.80*n)
  training = sample(1:n, size = n_train, replace = F)
  
  movies_train = dplyr :: select(to_merge_s[training, ], -c(id, original_title.x, Mystery, Thriller, Comedy, War, Music, TV.Movie)) #Train set 
  movies_test = dplyr :: select(to_merge_s[-training, ], -c(id, original_title.x, Mystery, Thriller, Comedy, War, Music, TV.Movie)) #Test set
  
  models_no_pg_pf[[k]] = list(movies_train, movies_test)
  
  #Fit the model:
  current_model = lm(revenue ~ . + production_companies*famous_count, data = movies_train)
  
  #Track the R^2:
  r_squared_no_pg_pf[k] = summary(current_model)$r.squared
  
  #Track the normalized Root Mean Squared Errors:
  
  #Test set:
  predictions = predict(current_model, movies_test) #Test Predictions
  rss_test = sum((predictions - movies_test$revenue)^2) #Test RSS
  rmse = sqrt(mean((predictions - movies_test$revenue)**2)) #Test RMSE
  rmses_no_pg_pf[k] = rmse

  prediction_ns = predictions * sd(to_merge$revenue) + mean(to_merge$revenue)
  real_ns = movies_test$revenue * sd(to_merge$revenue) + mean(to_merge$revenue)
  rmses_ns_no_pg_pf[k] = sqrt(mean((prediction_ns - real_ns)**2)) #Test RMSE not scaled  
}

#Mean Values scored:
mean(r_squared_no_pg_pf)
mean(rmses_no_pg_pf)
mean(rmses_ns_no_pg_pf)

#Select the best model obtained via Cross Validation:
index_no_pg_pf = which(r_squared_no_pg_pf == max(r_squared_no_pg_pf))
movies_train_no_pg_pf = models_no_pg_pf[[index_no_pg_pf]][[1]]
movies_test_no_pg_pf = models_no_pg_pf[[index_no_pg_pf]][[2]]
movies.lm_no_pg_pf = lm(revenue ~ . + production_companies*famous_count, data = movies_train_no_pg_pf)
summary(movies.lm_no_pg_pf)

#Anova result:
anova(movies.lm_no_g, movies.lm_no_pg, movies.lm_no_pg_pf) #Accepted

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Interaction between Production Companies and Budget:

#Cross Validation:
models_no_pg_pb = list() #List of pairs containing the couple Train and Test set for each fitted model
r_squared_no_pg_pb = c()
rmses_no_pg_pb = c()
rmses_ns_no_pg_pb = c()
for(k in seq(1, K)){ #K-fold CV
  
  #Train-Test split (we'll keep track of them to select the best model):
  n = nrow(no_out)
  n_train = floor(.80*n)
  training = sample(1:n, size = n_train, replace = F)
  
  movies_train = dplyr :: select(to_merge_s[training, ], -c(id, original_title.x, Mystery, Thriller, Comedy, War, Music, TV.Movie)) #Train set 
  movies_test = dplyr :: select(to_merge_s[-training, ], -c(id, original_title.x, Mystery, Thriller, Comedy, War, Music, TV.Movie)) #Test set
  
  models_no_pg_pb[[k]] = list(movies_train, movies_test)
  
  #Fit the model:
  current_model = lm(revenue ~ . + budget*production_companies, data = movies_train)
  
  #Track the R^2:
  r_squared_no_pg_pb[k] = summary(current_model)$r.squared
  
  #Track the normalized Root Mean Squared Errors:
  
  #Test set:
  predictions = predict(current_model, movies_test) #Test Predictions
  rss_test = sum((predictions - movies_test$revenue)^2) #Test RSS
  rmse = sqrt(mean((predictions - movies_test$revenue)**2)) #Test RMSE
  rmses_no_pg_pb[k] = rmse

  prediction_ns = predictions * sd(to_merge$revenue) + mean(to_merge$revenue)
  real_ns = movies_test$revenue * sd(to_merge$revenue) + mean(to_merge$revenue)
  rmses_ns_no_pg_pb[k] = sqrt(mean((prediction_ns - real_ns)**2)) #Test RMSE not scaled  
}

#Mean Values scored:
mean(r_squared_no_pg_pb)
mean(rmses_no_pg_pb)
mean(rmses_ns_no_pg_pb)

#Select the best model obtained via Cross Validation:
index_no_pg_pb = which(r_squared_no_pg_pb == max(r_squared_no_pg_pb))
movies_train_no_pg_pb = models_no_pg_pb[[index_no_pg_pb]][[1]]
movies_test_no_pg_pb = models_no_pg_pb[[index_no_pg_pb]][[2]]
movies.lm_no_pg_pb = lm(revenue ~ . + budget*production_companies, data = movies_train_no_pg_pb)
summary(movies.lm_no_pg_pb)

#Anova result:
anova(movies.lm_no_g, movies.lm_no_pg, movies.lm_no_pg_pb) #Accepted

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Interaction between Budget and Famous count:

#Cross Validation:
models_no_pg_bf = list() #List of pairs containing the couple Train and Test set for each fitted model
r_squared_no_pg_bf = c()
rmses_no_pg_bf = c()
rmses_ns_no_pg_bf = c()
for(k in seq(1, K)){ #K-fold CV
  
  #Train-Test split (we'll keep track of them to select the best model):
  n = nrow(no_out)
  n_train = floor(.80*n)
  training = sample(1:n, size = n_train, replace = F)
  
  movies_train = dplyr :: select(to_merge_s[training, ], -c(id, original_title.x, Mystery, Thriller, Comedy, War, Music, TV.Movie)) #Train set 
  movies_test = dplyr :: select(to_merge_s[-training, ], -c(id, original_title.x, Mystery, Thriller, Comedy, War, Music, TV.Movie)) #Test set
  
  models_no_pg_bf[[k]] = list(movies_train, movies_test)
  
  #Fit the model:
  current_model = lm(revenue ~ . + budget*production_companies, data = movies_train)
  
  #Track the R^2:
  r_squared_no_pg_bf[k] = summary(current_model)$r.squared
  
  #Track the normalized Root Mean Squared Errors:
  
  #Test set:
  predictions = predict(current_model, movies_test) #Test Predictions
  rss_test = sum((predictions - movies_test$revenue)^2) #Test RSS
  rmse = sqrt(mean((predictions - movies_test$revenue)**2)) #Test RMSE
  rmses_no_pg_bf[k] = rmse

  prediction_ns = predictions * sd(to_merge$revenue) + mean(to_merge$revenue)
  real_ns = movies_test$revenue * sd(to_merge$revenue) + mean(to_merge$revenue)
  rmses_ns_no_pg_bf[k] = sqrt(mean((prediction_ns - real_ns)**2)) #Test RMSE not scaled  
}

#Mean Values scored:
mean(r_squared_no_pg_bf)
mean(rmses_no_pg_bf)
mean(rmses_ns_no_pg_bf)

#Select the best model obtained via Cross Validation:
index_no_pg_bf = which(r_squared_no_pg_bf == max(r_squared_no_pg_bf))
movies_train_no_pg_bf = models_no_pg_bf[[index_no_pg_bf]][[1]]
movies_test_no_pg_bf = models_no_pg_bf[[index_no_pg_bf]][[2]]
movies.lm_no_pg_bf = lm(revenue ~ . + budget*production_companies, data = movies_train_no_pg_bf)
summary(movies.lm_no_pg_bf)

#Anova result:
anova(movies.lm_no_g, movies.lm_no_pg, movies.lm_no_pg_bf) #Accepted

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Include all:

#Cross Validation:
models_no_pg_in = list() #List of pairs containing the couple Train and Test set for each fitted model
r_squared_no_pg_in = c()
rmses_no_pg_in = c()
rmses_ns_no_pg_in = c()
for(k in seq(1, K)){ #K-fold CV
  
  #Train-Test split (we'll keep track of them to select the best model):
  n = nrow(no_out)
  n_train = floor(.80*n)
  training = sample(1:n, size = n_train, replace = F)
  
  movies_train = dplyr :: select(to_merge_s[training, ], -c(id, original_title.x, Mystery, Thriller, Comedy, War, Music, TV.Movie)) #Train set 
  movies_test = dplyr :: select(to_merge_s[-training, ], -c(id, original_title.x, Mystery, Thriller, Comedy, War, Music, TV.Movie)) #Test set
  
  models_no_pg_in[[k]] = list(movies_train, movies_test)
  
  #Fit the model:
  current_model = lm(revenue ~ . + budget*production_companies + production_companies*famous_count
                     + budget*famous_count, data = movies_train)
  
  #Track the R^2:
  r_squared_no_pg_in[k] = summary(current_model)$r.squared
  
  #Track the normalized Root Mean Squared Errors:
  
  #Test set:
  predictions = predict(current_model, movies_test) #Test Predictions
  rss_test = sum((predictions - movies_test$revenue)^2) #Test RSS
  rmse = sqrt(mean((predictions - movies_test$revenue)**2)) #Test RMSE
  rmses_no_pg_in[k] = rmse
  
  prediction_ns = predictions * sd(to_merge$revenue) + mean(to_merge$revenue)
  real_ns = movies_test$revenue * sd(to_merge$revenue) + mean(to_merge$revenue)
  rmses_ns_no_pg_in[k] = sqrt(mean((prediction_ns - real_ns)**2)) #Test RMSE not scaled  
}

#Mean Values scored:
mean(r_squared_no_pg_in)
mean(rmses_no_pg_in) 
mean(rmses_ns_no_pg_in)

#Select the best model obtained via Cross Validation:
index_no_pg_in = which(r_squared_no_pg_in == max(r_squared_no_pg_in))
movies_train_no_pg_in = models_no_pg_in[[index_no_pg_in]][[1]]
movies_test_no_pg_in = models_no_pg_in[[index_no_pg_in]][[2]]
movies.lm_no_pg_in = lm(revenue ~ . + budget*production_companies + production_companies*famous_count
                        + budget*famous_count, data = movies_train_no_pg_in)
summary(movies.lm_no_pg_in)

#Anova result:
anova(movies.lm_no_pg, movies.lm_no_pg_pf, movies.lm_no_pg_in) #Accepted
anova(movies.lm_no_pg, movies.lm_no_pg_pb, movies.lm_no_pg_in) #Accepted
anova(movies.lm_no_pg, movies.lm_no_pg_bf, movies.lm_no_pg_in) #Accepted

#Diagnostic Plot:
par(mfrow = c(2, 2)) 
plot(movies.lm_no_pg_in, ask = F)
par(mfrow = c(1, 1))

#RESULT:
#The result of the anova procedure point out that the best model is the one that includes all 
#the interactions related to budget, famous count and production companies:

best_model = movies.lm_no_pg_in

#==================================================================================================================================================================================================================================

#SHRINKAGE METHODS (INTERACTION VERSION):
#The goal is to obtain a model with a lower number of coefficients, shrinking toward zero the non significant 
#ones

#All the models estimated will have standardized predictors since different scales can lead to penalize more
#the coefficients associated to low-scale variables

#The shrinkage will be applied on the best model found at the end of the previous section (without outliers and
#partial genres). So the Train Set and the Test set will never be changed along this section.

#Ridge Regression:

#Cross Validation:
models_ridge = list() #List of pairs containing the couple Train and Test set for each fitted model
r_squared_ridge = c()
rmses_ridge = c()
rmses_ns_ridge = c()
for(k in seq(1, K)){ #K-fold CV
  
  #Train-Test split (we'll keep track of them to select the best model):
  n = nrow(to_merge)
  n_train = floor(.80*n)
  training = sample(1:n, size = n_train, replace = F)
  
  movies_train = dplyr :: select(to_merge_s[training, ], -c(id, original_title.x, Mystery, Thriller, Comedy, War, Music, TV.Movie)) #Train set 
  movies_test = dplyr :: select(to_merge_s[-training, ], -c(id, original_title.x, Mystery, Thriller, Comedy, War, Music, TV.Movie)) #Test set
  
  #Store the partitions:
  models_ridge[[k]] = list(movies_train, movies_test)
  
  #Fit the model with interactions:
  current_model = lm(revenue ~ . + budget*production_companies + production_companies*famous_count
                     + budget*famous_count, data = movies_train)
  
  #Extract the Train Model Matrix for Ridge Regression:
  X_tr_in = model.matrix(current_model)[, -1] #Exclude intercept because it will be added by glmnet
  Y_tr_in = movies_train$revenue
  
  #Design Matrix for testing:
  test.lm = lm(revenue ~ . + budget*production_companies + production_companies*famous_count
               + budget*famous_count, data = movies_test)
  X_test_in = model.matrix(test.lm)[, -1]
  
  #Encode as Factors:
  to_factorize = colnames(X_tr_in)[c(1, seq(3, 11), seq(14, 27))]
  for(variable in to_factorize){
    X_tr_in[, variable] = as.factor(X_tr_in[, variable])
    if(variable %in% colnames(X_test_in)){
      X_test_in[, variable] = as.factor(X_test_in[, variable]) 
    }
    else{
      X_test_in = cbind(X_test_in, rep(0, dim(X_test_in)[1]))
      colnames(X_test_in)[length(colnames(X_test_in))] = variable
      X_test_in[, variable] = as.factor(X_test_in[, variable])
    }
  }
  
  #Fit Regression (the following object is propedeutic for the Cross Validation Procedure): 
  fit = glmnet(X_tr_in, Y_tr_in, alpha = 0) #We have different attempts for different lambdas to produce a Ridge Regression
  
  #Perform Cross Validation to select the best value of lambda (importance of the penalty) among the fits produced at the step before:
  cv = cv.glmnet(X_tr_in, Y_tr_in, alpha = 0) #test 100 values of lambda and evaluates
  lambda = cv$lambda.min #Select the best value obtained via Cross Validation
  
  #Evaluate predictions of the best Cross-Validation lambda:
  predictions = predict(fit, newx = X_test_in, s = lambda)
  rmses_ridge[k] = sqrt(mean((predictions - movies_test$revenue)**2)) #Root Mean Squared Error
  
  prediction_ns = predictions * sd(to_merge$revenue) + mean(to_merge$revenue)
  real_ns = movies_test$revenue * sd(to_merge$revenue) + mean(to_merge$revenue)
  rmses_ns_ridge[k] = sqrt(mean((prediction_ns - real_ns)**2)) #Test RMSE not scaled  
  
  #We compute the R-squared:
  tr_pred = predict(fit, newx = X_tr_in, s = lambda)
  sst = sum((Y_tr_in - mean(Y_tr_in))^2)
  sse = sum((tr_pred - Y_tr_in)^2)
  r_squared_ridge[k] = 1 - (sse/sst)
}

#Mean Values scored:
mean(r_squared_ridge)
mean(rmses_ridge) 
mean(rmses_ns_ridge)

#Select the best model obtained via Cross Validation:
index_ridge = which(r_squared_ridge == max(r_squared_ridge))
movies_train_ridge = models_ridge[[index_ridge]][[1]]
movies_test_ridge = models_ridge[[index_ridge]][[2]]

#Design Matrices:

#Train:
model = lm(revenue ~ . + budget*production_companies + production_companies*famous_count
                        + budget*famous_count, data = movies_train_ridge)
X_tr_in = model.matrix(model)[, -1] #Exclude intercept because it will be added by glmnet
Y_tr_in = movies_train_ridge$revenue

#Test:
test.lm = lm(revenue ~ . + budget*production_companies + production_companies*famous_count
             + budget*famous_count, data = movies_test_ridge)
X_test_in = model.matrix(test.lm)[, -1]

#Encode as Factors:
to_factorize = colnames(X_tr_in)[c(1, seq(3, 11), seq(14, 27))]
for(variable in to_factorize){
  X_tr_in[, variable] = as.factor(X_tr_in[, variable])
  if(variable %in% colnames(X_test_in)){
    X_test_in[, variable] = as.factor(X_test_in[, variable]) 
  }
  else{
    X_test_in = cbind(X_test_in, rep(0, dim(X_test_in)[1]))
    colnames(X_test_in)[length(colnames(X_test_in))] = variable
    X_test_in[, variable] = as.factor(X_test_in[, variable])
  }
}

#Fit Regression (the following object is propedeutic for the Cross Validation Procedure): 
ridgefit_in = glmnet(X_tr_in, Y_tr_in, alpha = 0) #We have different attempts for different lambdas to produce a Ridge Regression

#Perform Cross Validation to select the best value of lambda (importance of the penalty) among the fits produced at the step before:
ridgecv_in = cv.glmnet(X_tr_in, Y_tr_in, alpha = 0) #test 100 values of lambda and evaluates
bestlam_r_in = ridgecv_in$lambda.min #Select the best value obtained via Cross Validation

#Evaluate predictions of the best Cross-Validation lambda:
ridge.pred_in = predict(ridgefit_in, newx = X_test_in, s = bestlam_r_in)
plot(ridge.pred_in, movies_test_ridge$revenue, main = "Real Values vs Predictions", xlab = "Test predictions", ylab = "Real values")
abline(b = 1, a = 0, lwd = 3, col = "red")

#We compute the R-squared:
r_squared_ridge[index_ridge]

#We graphically show the values of our coefficients:
coeff_sum <- apply(coef(ridgefit_in)[-1,]^2, 2, sum) #Sum of the coefficients over the different ridge attempts (numerator of the x-ax)
ratio <- coeff_sum/max(coeff_sum) #Maximum sum obtained (denominator of the x-ax)
matplot(ratio, t(coef(ridgefit_in)[-1,]), type = "l", lwd = 2, lty = 1, col = "slateblue", xlab = "", ylab = "Coefficients", 
        main = "Ridge regression coefficients path", xlim = c(0,1.2)) #Plot the evolution of the coefficients
mtext(expression(sum(beta[j]^2)/max(sum(beta[j]^2))), 1, line = 3) #X-ax

#Extract coefficients of the lambda found via Cross Validation and plot their position:
intercept = coef(ridgefit_in, bestlam_r_in)[1]
coeff_values = coef(ridgefit_in, bestlam_r_in)[-1]
abline(v = sum(coeff_values^2)/max(coeff_sum), col = "red", lwd = 2)

ridge.coeff_in = data.frame(c("Intercept", colnames(X_tr_in)), c(intercept, coeff_values))
colnames(ridge.coeff_in) = c("Predictors", "Coefficients")
ridge.coeff_in

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Lasso Regression:

#Cross Validation:
models_lasso = list() #List of pairs containing the couple Train and Test set for each fitted model
r_squared_lasso = c()
rmses_lasso = c()
rmses_ns_lasso = c()
for(k in seq(1, K)){ #K-fold CV
  
  #Train-Test split (we'll keep track of them to select the best model):
  n = nrow(to_merge)
  n_train = floor(.80*n)
  training = sample(1:n, size = n_train, replace = F)
  
  movies_train = dplyr :: select(to_merge_s[training, ], -c(id, original_title.x, Mystery, Thriller, Comedy, War, Music, TV.Movie)) #Train set 
  movies_test = dplyr :: select(to_merge_s[-training, ], -c(id, original_title.x, Mystery, Thriller, Comedy, War, Music, TV.Movie)) #Test set
  
  #Store the partitions:
  models_lasso[[k]] = list(movies_train, movies_test)
  
  #Fit the model with interactions:
  current_model = lm(revenue ~ . + budget*production_companies + production_companies*famous_count
                     + budget*famous_count, data = movies_train)
  
  #Extract the Train Model Matrix for Ridge Regression:
  X_tr_in = model.matrix(current_model)[, -1] #Exclude intercept because it will be added by glmnet
  Y_tr_in = movies_train$revenue
  
  #Design Matrix for testing:
  test.lm = lm(revenue ~ . + budget*production_companies + production_companies*famous_count
               + budget*famous_count, data = movies_test)
  X_test_in = model.matrix(test.lm)[, -1]
  
  #Encode as Factors:
  to_factorize = colnames(X_tr_in)[c(1, seq(3, 11), seq(14, 27))]
  for(variable in to_factorize){
    X_tr_in[, variable] = as.factor(X_tr_in[, variable])
    if(variable %in% colnames(X_test_in)){
      X_test_in[, variable] = as.factor(X_test_in[, variable]) 
    }
    else{
      X_test_in = cbind(X_test_in, rep(0, dim(X_test_in)[1]))
      colnames(X_test_in)[length(colnames(X_test_in))] = variable
      X_test_in[, variable] = as.factor(X_test_in[, variable])
    }
  }
  
  #Fit Regression (the following object is propedeutic for the Cross Validation Procedure): 
  fit = glmnet(X_tr_in, Y_tr_in, alpha = 1) #We have different attempts for different lambdas to produce a Ridge Regression
  
  #Perform Cross Validation to select the best value of lambda (importance of the penalty) among the fits produced at the step before:
  cv = cv.glmnet(X_tr_in, Y_tr_in, alpha = 1) #test 100 values of lambda and evaluates
  lambda = cv$lambda.min #Select the best value obtained via Cross Validation
  
  #Evaluate predictions of the best Cross-Validation lambda:
  predictions = predict(fit, newx = X_test_in, s = lambda)
  rmses_lasso[k] = sqrt(mean((predictions - movies_test$revenue)**2)) #Root Mean Squared Error
  
  prediction_ns = predictions * sd(to_merge$revenue) + mean(to_merge$revenue)
  real_ns = movies_test$revenue * sd(to_merge$revenue) + mean(to_merge$revenue)
  rmses_ns_lasso[k] = sqrt(mean((prediction_ns - real_ns)**2)) #Test RMSE not scaled  
  
  #We compute the R-squared:
  tr_pred = predict(fit, newx = X_tr_in, s = lambda)
  sst = sum((Y_tr_in - mean(Y_tr_in))^2)
  sse = sum((tr_pred - Y_tr_in)^2)
  r_squared_lasso[k] = 1 - (sse/sst)
}

#Mean Values scored:
mean(r_squared_lasso)
mean(rmses_lasso) 
mean(rmses_ns_lasso)

#Select the best model obtained via Cross Validation:
index_lasso = which(r_squared_lasso == max(r_squared_lasso))
movies_train_lasso = models_lasso[[index_lasso]][[1]]
movies_test_lasso = models_lasso[[index_lasso]][[2]]

#Design Matrices:

#Train:
model = lm(revenue ~ . + budget*production_companies + production_companies*famous_count
           + budget*famous_count, data = movies_train_lasso)
X_tr_in = model.matrix(model)[, -1] #Exclude intercept because it will be added by glmnet
Y_tr_in = movies_train_lasso$revenue

#Test:
test.lm = lm(revenue ~ . + budget*production_companies + production_companies*famous_count
             + budget*famous_count, data = movies_test_lasso)
X_test_in = model.matrix(test.lm)[, -1]

#Encode as Factors:
to_factorize = colnames(X_tr_in)[c(1, seq(3, 11), seq(14, 27))]
for(variable in to_factorize){
  X_tr_in[, variable] = as.factor(X_tr_in[, variable])
  if(variable %in% colnames(X_test_in)){
    X_test_in[, variable] = as.factor(X_test_in[, variable]) 
  }
  else{
    X_test_in = cbind(X_test_in, rep(0, dim(X_test_in)[1]))
    colnames(X_test_in)[length(colnames(X_test_in))] = variable
    X_test_in[, variable] = as.factor(X_test_in[, variable])
  }
}

#Fit Regression (the following object is propedeutic for the Cross Validation Procedure): 
lassofit_in = glmnet(X_tr_in, Y_tr_in, alpha = 1) #We have different attempts for different lambdas to produce a Ridge Regression

#Perform Cross Validation to select the best value of lambda (importance of the penalty) among the fits produced at the step before:
lassocv_in = cv.glmnet(X_tr_in, Y_tr_in, alpha = 1) #test 100 values of lambda and evaluates
bestlam_l_in = lassocv_in$lambda.min #Select the best value obtained via Cross Validation

#Evaluate predictions of the best Cross-Validation lambda:
lasso.pred_in = predict(lassofit_in, newx = X_test_in, s = bestlam_l_in)
plot(lasso.pred_in, movies_test_lasso$revenue, main = "Real Values vs Predictions", xlab = "Test predictions", ylab = "Real values")
abline(b = 1, a = 0, lwd = 3, col = "red")

#We compute the R-squared:
r_squared_lasso[index_lasso]

#We graphically show the values of our coefficients:
coeff_sum <- apply(coef(lassofit_in)[-1,]^2, 2, sum) #Sum of the coefficients over the different ridge attempts (numerator of the x-ax)
ratio <- coeff_sum/max(coeff_sum) #Maximum sum obtained (denominator of the x-ax)
matplot(ratio, t(coef(lassofit_in)[-1,]), type = "l", lwd = 2, lty = 1, col = "slateblue", xlab = "", ylab = "Coefficients", 
        main = "Lasso regression coefficients path", xlim = c(0,1.2)) #Plot the evolution of the coefficients
mtext(expression(sum(beta[j]^2)/max(sum(beta[j]^2))), 1, line = 3) #X-ax

#Extract coefficients of the lambda found via Cross Validation and plot their position:
intercept = coef(lassofit_in, bestlam_l_in)[1]
coeff_values = coef(lassofit_in, bestlam_l_in)[-1]
abline(v = sum(coeff_values^2)/max(coeff_sum), col = "red", lwd = 2)

lasso.coeff_in = data.frame(c("Intercept", colnames(X_tr_in)), c(intercept, coeff_values))
colnames(lasso.coeff_in) = c("Predictors", "Coefficients")
lasso.coeff_in

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Elastic Net Regression:
#Since Elastic Net has a dependence not only on the value of "lambda" but also on the value of the 
#parameter "alpha", we'll produce cross validation for different values of alpha (and for each of them
#there'll be a cross validation for lambda optimal) in order to find the best model in terms of RMSE

#Cross Validation:
models_en = list() #List of pairs containing the couple Train and Test set for each fitted model
r_squared_en = c()
rmses_en = c()
rmses_ns_en = c()
optimal_parameters = list() #List of pairs containing the couples (alpha, lambda) optimized parameters for Elastic Net
for(k in seq(1, K)){ #K-fold CV
  
  #Train-Test split (we'll keep track of them to select the best model):
  n = nrow(to_merge)
  n_train = floor(.80*n)
  training = sample(1:n, size = n_train, replace = F)
  
  movies_train = dplyr :: select(to_merge_s[training, ], -c(id, original_title.x, Mystery, Thriller, Comedy, War, Music, TV.Movie)) #Train set 
  movies_test = dplyr :: select(to_merge_s[-training, ], -c(id, original_title.x, Mystery, Thriller, Comedy, War, Music, TV.Movie)) #Test set
  
  #Store the partitions:
  models_en[[k]] = list(movies_train, movies_test)
  
  #Fit the model with interactions:
  current_model = lm(revenue ~ . + budget*production_companies + production_companies*famous_count
                     + budget*famous_count, data = movies_train)
  
  #Extract the Train Model Matrix for Ridge Regression:
  X_tr_in = model.matrix(current_model)[, -1] #Exclude intercept because it will be added by glmnet
  Y_tr_in = movies_train$revenue
  
  #Design Matrix for testing:
  test.lm = lm(revenue ~ . + budget*production_companies + production_companies*famous_count
               + budget*famous_count, data = movies_test)
  X_test_in = model.matrix(test.lm)[, -1]
  
  #Encode as Factors:
  to_factorize = colnames(X_tr_in)[c(1, seq(3, 11), seq(14, 27))]
  for(variable in to_factorize){
    X_tr_in[, variable] = as.factor(X_tr_in[, variable])
    if(variable %in% colnames(X_test_in)){
      X_test_in[, variable] = as.factor(X_test_in[, variable]) 
    }
    else{
      X_test_in = cbind(X_test_in, rep(0, dim(X_test_in)[1]))
      colnames(X_test_in)[length(colnames(X_test_in))] = variable
      X_test_in[, variable] = as.factor(X_test_in[, variable])
    }
  }
  
  #Optimize alpha via Cross Validation:
  alphas = seq(0, 1, by = 0.1) #Different values tested for alpha
  lambdas = c() #Best lambda obtained for each alpha
  rsq = c()
  for(iteration in seq(1, length(alphas))){ #Different mixtures of Lasso and Ridge regressions
    
    #Fit several Elastic Net regressions for different values of the tuning parameter lambda:
    fit_intermediate = glmnet(X_tr_in, Y_tr_in, alpha = alphas[iteration]) 
    
    #Perform Cross Validation to select the best value of lambda:
    cv_intermediate = cv.glmnet(X_tr_in, Y_tr_in, alpha = alphas[iteration])
    lambdas[iteration] = cv_intermediate$lambda.min
    
    #We compute the R-squared:
    train_pred = predict(fit_intermediate, newx = X_tr_in, s = lambdas[iteration])
    sst = sum((Y_tr_in - mean(Y_tr_in))^2)
    sse = sum((train_pred - Y_tr_in)^2)
    rsq[iteration] = 1 - (sse/sst)
  }
  
  #Track performances for each optimized model:
  index = which(rsq == max(rsq)) #Criterion: Maximize R-squared
  alpha = alphas[index] #Best Mixture
  lambda = lambdas[index] #Best Penalty relevance
  optimal_parameters[[k]] = c(alpha, lambda)
  
  #Train Performances:
  r_squared_en[k] = rsq[index]
  
  #Test Performances:
  fit = glmnet(X_tr_in, Y_tr_in, alpha = alpha)
  
  predictions = predict(fit, newx = X_test_in, s = lambda)
  rmses_en[k] = sqrt(mean((predictions - movies_test$revenue)**2)) #Root Mean Squared Error
  
  prediction_ns = predictions * sd(to_merge$revenue) + mean(to_merge$revenue)
  real_ns = movies_test$revenue * sd(to_merge$revenue) + mean(to_merge$revenue)
  rmses_ns_en[k] = sqrt(mean((prediction_ns - real_ns)**2)) #Test RMSE not scaled
}

#Mean Values scored:
mean(r_squared_en)
mean(rmses_en) 
mean(rmses_ns_en)

#Select the best model obtained via Cross Validation:
index_en = which(r_squared_en == max(r_squared_en))
movies_train_en = models_en[[index_en]][[1]]
movies_test_en = models_en[[index_en]][[2]]

alpha_in = optimal_parameters[[index_en]][1]
lambda_in = optimal_parameters[[index_en]][2]

#Design Matrices:

#Train:
model = lm(revenue ~ . + budget*production_companies + production_companies*famous_count
           + budget*famous_count, data = movies_train_en)
X_tr_in = model.matrix(model)[, -1] #Exclude intercept because it will be added by glmnet
Y_tr_in = movies_train_en$revenue

#Test:
test.lm = lm(revenue ~ . + budget*production_companies + production_companies*famous_count
             + budget*famous_count, data = movies_test_en)
X_test_in = model.matrix(test.lm)[, -1]

#Encode as Factors:
to_factorize = colnames(X_tr_in)[c(1, seq(3, 11), seq(14, 27))]
for(variable in to_factorize){
  X_tr_in[, variable] = as.factor(X_tr_in[, variable])
  if(variable %in% colnames(X_test_in)){
    X_test_in[, variable] = as.factor(X_test_in[, variable]) 
  }
  else{
    X_test_in = cbind(X_test_in, rep(0, dim(X_test_in)[1]))
    colnames(X_test_in)[length(colnames(X_test_in))] = variable
    X_test_in[, variable] = as.factor(X_test_in[, variable])
  }
}

#Fit Regression (the following object is propedeutic for the Cross Validation Procedure): 
elasticfit_in = glmnet(X_tr_in, Y_tr_in, alpha = alpha_in) #We have different attempts for different lambdas to produce a Ridge Regression

#Evaluate predictions of the best Cross-Validation lambda:
elastic.pred_in = predict(elasticfit_in, newx = X_test_in, s = lambda_in)
plot(elastic.pred_in, movies_test_en$revenue, main = "Real Values vs Predictions", xlab = "Test predictions", ylab = "Real values")
abline(b = 1, a = 0, lwd = 3, col = "red")

#We compute the R-squared:
r_squared_en[index_en]

#We graphically show the values of our coefficients:
coeff_sum <- apply(coef(elasticfit_in)[-1,]^2, 2, sum) #Sum of the coefficients over the different ridge attempts (numerator of the x-ax)
ratio <- coeff_sum/max(coeff_sum) #Maximum sum obtained (denominator of the x-ax)
matplot(ratio, t(coef(elasticfit_in)[-1,]), type = "l", lwd = 2, lty = 1, col = "slateblue", xlab = "", ylab = "Coefficients", 
        main = "Elastic Net regression coefficients path", xlim = c(0,1.2)) #Plot the evolution of the coefficients
mtext(expression(sum(beta[j]^2)/max(sum(beta[j]^2))), 1, line = 3) #X-ax

#Extract coefficients of the lambda found via Cross Validation and plot their position:
intercept = coef(elasticfit_in, lambda_in)[1]
coeff_values = coef(elasticfit_in, lambda_in)[-1]
abline(v = sum(coeff_values^2)/max(coeff_sum), col = "red", lwd = 2)

elastic.coeff_in = data.frame(c("Intercept", colnames(X_tr_in)), c(intercept, coeff_values))
colnames(elastic.coeff_in) = c("Predictors", "Coefficients")
elastic.coeff_in

#==================================================================================================================================================================================================================================

#SUMMARY OF DIFFERENT ATTEMPTS:
#We'll briefly compare the different linear regressions estimated in terms of performances on the train
#set (through the r-squared indexes) and on the test set (through the Normalized Root Mean Squared Errors)

#Performances:
train = c(mean(r_squared), mean(r_squared_no), mean(r_squared_no_g), mean(r_squared_no_pg), mean(r_squared_no_pg_in), 
          mean(r_squared_ridge), mean(r_squared_lasso), mean(r_squared_en))
test_1 = c(mean(rmses), mean(rmses_no), mean(rmses_no_g), mean(rmses_no_pg),
         mean(rmses_no_pg_in), mean(rmses_ridge), mean(rmses_lasso), mean(rmses_en))
test_2 = c(mean(rmses_ns), mean(rmses_ns_no), mean(rmses_ns_no_g), mean(rmses_ns_no_pg),
           mean(rmses_ns_no_pg_in), mean(rmses_ns_ridge), mean(rmses_ns_lasso),
           mean(rmses_ns_en))

#Comparison Matrix:
comparison_l = data.frame(train, test_1, test_2)
rownames(comparison_l) = c("Standard", "No Outliers", "All the Genres", "Relevant Genres", "Interaction effects",
                         "Ridge", "Lasso", "Elastic Net")
colnames(comparison_l) = c("R-squared", "RMSE (scaled data)", "RMSE (original scale)")
comparison_l

#==================================================================================================================================================================================================================================

#VARIABLE LOG TRANSFORMATION:
#Our aim is to apply some transformation to our variables (target and predictors) in order to see if we can improve
#the model assumptions/diagnostic and the predictive performances

#We apply the transformation at the end of the section since we detected common problems in the diagnostic 
#of our models

#Take non scaled data to manage only positive quantities:
target = movies$revenue
predictors = dplyr :: select(movies, budget, runtime, famous_count)

#Analyze how transformation can impact:
#It turns out that the only useful log transformation can be the one related to "runtime" since it makes
#the variable more "normal" and don't worse the correlation

#Impact on Correlation Patterns:
pairs(cbind(log(target), predictors), main = "Log Transform only the target")
pairs(cbind(target, log(predictors)), main = "Log Transform only the predictors")
pairs(cbind(log(target), log(predictors)), main = "Log Transform both target and predictors")

#Impact on distributions:
color1 = rgb(red = 0, green = 0, blue = 0.5, alpha = 0.5)
color2 = rgb(red = 0.5, green = 0, blue = 0, alpha = 0.5)
hist(scale(log(target)), main = "Log Revenues", xlab = "revenues", probability = T, ylim = c(0, 0.8), col = color1)
hist(scale(target), add = T, probability = T, col = color2)

hist(scale(log(predictors[, 1])), main = "Log Budget", xlab = "budget", probability = T, ylim = c(0, 0.8), col = color1)
hist(scale(predictors[, 1]), add = T, probability = T, col = color2)

hist(log(predictors[, 2]), main = "Log Runtime", xlab = "runtime", col = color1)

hist(log(predictors[, 3]), main = "Log Famous Count", xlab = "famous count", col = color1)

#Fit model:
#Taking the logarithm of "runtime" doesn't allow to estimate the model since NaNs are produced.

#==================================================================================================================================================================================================================================

#ALTERNATIVE TRANSFORMATIONS:
#We try to apply both the Box-Cox and the Box-Tidwell transformations to see if we can improove
#the diagnostics of the linear regression

#Box-Tidwell Transformation:
#Looking at the hist-density plots
#we decide to keep just the Box-Tidwell transformations for Runtime and Famous Count

#Find the best lambda for the transformation:
numericals = dplyr :: select(movies, revenue, budget, runtime, famous_count) #Extract numerical predictors

#Predictors (we add a little positive quantity in order to have all positive values):
numericals_sp = numericals
numericals_sp[, "runtime"] = numericals[, "runtime"] + 0.01
numericals_sp[, "famous_count"] = numericals[, "famous_count"] + 0.01

#Find best transformation of predictors:
bt = boxTidwell(revenue ~ budget + runtime + famous_count, data = numericals_sp)

transformed = numericals_sp
for(variable in colnames(numericals_sp)[-1]){
  lambda = bt$result[,1][variable]
  transformed[, variable] = (numericals_sp[, variable]^lambda - 1)/lambda
}

#Check Normality Approximation:
par(mfrow = c(2, 1))

#Budget:
#Before Transformation:
probabilities = ppoints(length(movies[, "budget"]))
quantiles = qnorm(probabilities, mean = mean(movies[, "budget"]), 
                  sd = sd(movies[, "budget"]))
plot(density(quantiles), lwd = 3, col = "red", 
     main = "Budget before transformation BT", xlab = "", ylim = c(0, 0.00000003))
hist(movies[, "budget"], probability = T, add = T, col = color2)

#After Transformation:
probabilities = ppoints(length(transformed[, "budget"]))
quantiles = qnorm(probabilities, mean = mean(transformed[, "budget"]), 
                  sd = sd(transformed[, "budget"]))
plot(density(quantiles), lwd = 3, col = "black", 
     main = "Budget after transformation BT", xlab = "", ylim = c(0, 0.00000000004))
hist(transformed[, "budget"], probability = T, add = T, col = color1)

#Famous count:
#Before Transformation:
probabilities = ppoints(length(movies[, "famous_count"]))
quantiles = qnorm(probabilities, mean = mean(movies[, "famous_count"]), 
                  sd = sd(movies[, "famous_count"]))
plot(density(quantiles), lwd = 3, col = "red", 
     main = "Famous count before transformation BT", xlab = "", ylim = c(0, 0.18))
hist(movies[, "famous_count"], probability = T, add = T, col = color2)

#After Transformation:
probabilities = ppoints(length(transformed[, "famous_count"]))
quantiles = qnorm(probabilities, mean = mean(transformed[, "famous_count"]), 
                  sd = sd(transformed[, "famous_count"]))
plot(density(quantiles), lwd = 3, col = "black", ylim = c(0, 0.03), 
     main = "Famous count after transformation BT", xlab = "")
hist(transformed[, "famous_count"], probability = T, add = T, col = color1)

#Runtime:
#Before Transformation:
probabilities = ppoints(length(movies[, "runtime"]))
quantiles = qnorm(probabilities, mean = mean(movies[, "runtime"]), 
                  sd = sd(movies[, "runtime"]))
plot(density(quantiles), lwd = 3, col = "black", 
     main = "Runtime before transformation BT", xlab = "")
hist(movies[, "runtime"], probability = T, add = T, col = color1)

#After Transformation:
probabilities = ppoints(length(transformed[, "runtime"]))
quantiles = qnorm(probabilities, mean = mean(transformed[, "runtime"]), 
                  sd = sd(transformed[, "runtime"]))
plot(density(quantiles), lwd = 3, col = "black", ylim = c(0, 0.006), 
     main = "Runtime after transformation BT", xlab = "")
hist(transformed[, "runtime"], probability = T, add = T, col = color1)

par(mfrow = c(1, 1))

#Anti-transform budget:
transformed[, "budget"] = numericals_sp[, "budget"]

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Box-Cox Transformation:
#Since the hist-density plot suggests that budget beahves better with a BC transformation, 
#wedecide to keep the BT Transformation for Runtime and Famous Count and the BC Transformation
#for Revenue and Budget

model = lm(revenue ~ budget + runtime + famous_count, data = transformed) # Train the model
boxcox(model) #Show result
bc = boxcox(model) 

#Extract the best value of lambda:
index_bc = which(bc$y == max(bc$y)) 
lambda_bc = boxcox(model)$x[index_bc]

#Transform data:
trans_numerical = transformed
for(variable in colnames(transformed)[c(1, 2)]){
  trans_numerical[, variable] = (transformed[, variable]^lambda_bc - 1)/lambda_bc
}

#Check Normality Approximation:
par(mfrow = c(2, 1))

#Budget:
#Before Transformation:
probabilities = ppoints(length(movies[, "budget"]))
quantiles = qnorm(probabilities, mean = mean(movies[, "budget"]), 
                  sd = sd(movies[, "budget"]))
plot(density(quantiles), lwd = 3, col = "red", 
     main = "Budget before transformation BC", xlab = "", ylim = c(0, 0.00000003))
hist(movies[, "budget"], probability = T, add = T, col = color2)

#After Transformation:
probabilities = ppoints(length(trans_numerical[, "budget"]))
quantiles = qnorm(probabilities, mean = mean(trans_numerical[, "budget"]), 
                  sd = sd(trans_numerical[, "budget"]))
plot(density(quantiles), lwd = 3, col = "black", 
     main = "Budget after transformation BC", xlab = "", ylim = c(0, 0.004))
hist(trans_numerical[, "budget"], probability = T, add = T, col = color1)

#Revenues:
#Before Transformation:
probabilities = ppoints(length(movies[, "revenue"]))
quantiles = qnorm(probabilities, mean = mean(movies[, "revenue"]), 
                  sd = sd(movies[, "revenue"]))
plot(density(quantiles), lwd = 3, col = "red", 
     main = "Revenue before transformation BC", xlab = "", ylim = c(0, 0.0000000045))
hist(movies[, "revenue"], probability = T, add = T, col = color2)

#After Transformation:
probabilities = ppoints(length(trans_numerical[, "revenue"]))
quantiles = qnorm(probabilities, mean = mean(trans_numerical[, "revenue"]), 
                  sd = sd(trans_numerical[, "revenue"]))
plot(density(quantiles), lwd = 3, col = "black", 
     main = "Revenue after transformation BC", xlab = "")
hist(trans_numerical[, "revenue"], probability = T, add = T, col = color1)

par(mfrow = c(1, 1))

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Model on Transformed data:

#Scale data and add categorical predictors:
trans_s = data.frame(cbind(scale(trans_numerical), rep(0, dim(movies)[1]), rep(0, dim(movies)[1]), rep(0, dim(movies)[1])))
colnames(trans_s) = c("revenue", "budget", "runtime", "famous_count", "belongs_to_collection", 
                      "original_language", "production_companies")
categorical = colnames(movies)[c(3, 5, 6)]

#Add categoricals:
for(variable in categorical){
  trans_s[, variable] = movies[, variable]
}

#Impact on correlations:
cor(trans_numerical) #Not disturbed

#Fit the model with transformed predictors:

#Cross Validation:
models_t = list() #List of pairs containing the couple Train and Test set for each fitted model
r_squared_t = c()
rmses_t = c()
rmses_ns_t = c()
for(k in seq(1, K)){ #K-fold CV
  
  #Train-Test split (we'll keep track of them to select the best model):
  n = nrow(trans_s)
  n_train = floor(.80*n)
  training = sample(1:n, size = n_train, replace = F)
  
  movies_train = trans_s[training, ] #Train set
  movies_test = trans_s[-training, ] #Test set
  models_t[[k]] = list(movies_train, movies_test)
  
  #Fit the model:
  current_model = lm(revenue ~ ., data = movies_train)
  
  #Track the R^2:
  r_squared_t[k] = summary(current_model)$r.squared
  
  #Track the normalized Root Mean Squared Errors:
  
  #Test set:
  predictions = predict(current_model, movies_test) #Test Predictions
  rmse = sqrt(mean((predictions - movies_test$revenue)**2)) #Test RMSE
  rmses_t[k] = rmse
  
  #Invert scaling:
  prediction_ns = predictions * sd(trans_numerical$revenue) + mean(trans_numerical$revenue)
  real_ns = movies_test$revenue * sd(trans_numerical$revenue) + mean(trans_numerical$revenue)
  
  #Invert Transformation:
  prediction_ns_nt = ((prediction_ns * lambda_bc) + 1)^(1/lambda_bc)
  real_ns_nt = ((real_ns * lambda_bc) + 1)^(1/lambda_bc)
  
  #Compute original RMSE:
  rmses_ns_t[k] = sqrt(mean((prediction_ns_nt - real_ns_nt)**2, na.rm = T)) #Test RMSE not scaled  
}

#Mean Values scored:
mean(r_squared_t)
mean(rmses_t)
mean(rmses_ns_t)

#Select the best model obtained via Cross Validation:
index_t = which(r_squared_t == max(r_squared_t))
movies_train_t = models_t[[index_t]][[1]]
movies_test_t = models_t[[index_t]][[2]]
movies.lm_t = lm(revenue ~ ., data = movies_train_t)

summary(movies.lm_t)

#Diagnostic Plot:
par(mfrow = c(2, 2))
plot(movies.lm_t, ask = F)
par(mfrow = c(1, 1))
#plot(rstudent(movies.lm) ~ hatvalues(movies.lm)) #Cercare di interpretare questo plot

#Plot the Prediction vs Real on Test set:
predictions_t = predict(movies.lm_t, movies_test_t) #Predictions
plot(predictions_t, movies_test_t$revenue, main = "Real Values vs Predictions", xlab = "Test predictions", ylab = "Real values")
abline(b = 1, a = 0, lwd = 3, col = "red")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Try to build the best predictive model (including relevant genres and interactions on transformed data):

no_out_ns = movies[-remove, ]
to_merge_ns = merge(no_out_ns, genres_df, by = "id")
to_merge_ns = to_merge_ns %>% dplyr :: select(-c("X", "original_title.y"))

#Encode genres as factors:
genres = colnames(to_merge_ns)[seq(10, length(colnames(to_merge_ns)))]
for(genre in genres){
  to_merge_ns[, genre] = as.factor(to_merge_ns[, genre])
}

final_numericals = dplyr :: select(to_merge_ns, revenue, budget, runtime, famous_count)

#Predictors (we add a little positive quantity in order to have all positive values):
final_numericals_sp = final_numericals
final_numericals_sp[, "runtime"] = final_numericals[, "runtime"] + 0.01
final_numericals_sp[, "famous_count"] = final_numericals[, "famous_count"] + 0.01

#BT transforms:
bt_f = boxTidwell(revenue ~ budget + runtime + famous_count, data = final_numericals_sp)

transformed_f = final_numericals_sp
for(variable in colnames(final_numericals_sp)[-c(1, 2)]){
  lambda = bt_f$result[,1][variable]
  transformed_f[, variable] = (final_numericals_sp[, variable]^lambda - 1)/lambda
}

#BC Transforms:
model_final = lm(revenue ~ budget + runtime + famous_count, data = transformed_f) # Train the model
boxcox(model_final) #Show result
bc_final = boxcox(model_final) 

#Extract the best value of lambda:
index_bc_final = which(bc_final$y == max(bc_final$y)) 
lambda_bc_final = boxcox(model_final)$x[index_bc_final]

#Transform data:
trans_numerical_final = transformed_f

for(variable in colnames(trans_numerical_final)[c(1, 2)]){
  trans_numerical_final[, variable] = (transformed_f[, variable]^lambda_bc_final - 1)/lambda_bc_final
}

#Scale data and add categorical predictors:
columns = c()
for(iteration in seq(1, 23)){
  columns = cbind(columns, rep(0, dim(to_merge)[1]))
}

trans_numerical_final_s = data.frame(cbind(scale(trans_numerical_final), columns))
colnames(trans_numerical_final_s) = c("revenue", "budget", "runtime", "famous_count", "belongs_to_collection", 
                      "original_language", "production_companies", 
                      colnames(to_merge_ns)[seq(10, length(colnames(to_merge_ns)))])
categorical = colnames(to_merge_ns)[c(3, 5, 6, seq(10, length(colnames(to_merge_ns))))]

#Add categoricals:
for(variable in categorical){
  trans_numerical_final_s[, variable] = to_merge_ns[, variable]
}

#Fit the model with transformed predictors:

#Cross Validation:
models_t_final = list() #List of pairs containing the couple Train and Test set for each fitted model
r_squared_t_final = c()
rmses_t_final = c()
rmses_ns_t_final = c()
for(k in seq(1, K)){ #K-fold CV
  
  #Train-Test split (we'll keep track of them to select the best model):
  n = nrow(trans_numerical_final)
  n_train = floor(.80*n)
  training = sample(1:n, size = n_train, replace = F)
  
  movies_train = dplyr :: select(trans_numerical_final_s[training, ], -c(Mystery, Thriller, Comedy, War, Music, TV.Movie)) #Train set
  movies_test = dplyr :: select(trans_numerical_final_s[-training, ], -c(Mystery, Thriller, Comedy, War, Music, TV.Movie)) #Test set
  models_t_final[[k]] = list(movies_train, movies_test)
  
  #Fit the model:
  current_model = lm(revenue ~ . + budget*production_companies + production_companies*famous_count
                     + budget*famous_count, data = movies_train)
  
  #Track the R^2:
  r_squared_t_final[k] = summary(current_model)$r.squared
  
  #Track the normalized Root Mean Squared Errors:
  
  #Test set:
  predictions = predict(current_model, movies_test) #Test Predictions
  rmse = sqrt(mean((predictions - movies_test$revenue)**2)) #Test RMSE
  rmses_t_final[k] = rmse
  
  #Invert scaling:
  prediction_ns = predictions * sd(trans_numerical_final$revenue) + mean(trans_numerical_final$revenue)
  real_ns = movies_test$revenue * sd(trans_numerical_final$revenue) + mean(trans_numerical_final$revenue)
  
  #Invert Transformation:
  prediction_ns_nt = ((prediction_ns * lambda_bc_final) + 1)^(1/lambda_bc_final)
  real_ns_nt = ((real_ns * lambda_bc_final) + 1)^(1/lambda_bc_final)
  
  #Compute original RMSE:
  rmses_ns_t_final[k] = sqrt(mean((prediction_ns_nt - real_ns_nt)**2, na.rm = T)) #Test RMSE not scaled  
}

#Mean Values scored:
mean(r_squared_t_final)
mean(rmses_t_final)
mean(rmses_ns_t_final)

#Select the best model obtained via Cross Validation:
index_t_final = which(r_squared_t_final == max(r_squared_t_final))
movies_train_t_final = models_t_final[[index_t_final]][[1]]
movies_test_t_final = models_t_final[[index_t_final]][[2]]
movies.lm_t_final = lm(revenue ~ . + budget*production_companies + production_companies*famous_count
                       + budget*famous_count, data = movies_train_t_final)

summary(movies.lm_t_final)

#Diagnostic Plot:
par(mfrow = c(2, 2))
plot(movies.lm_t_final, ask = F)
par(mfrow = c(1, 1))
#plot(rstudent(movies.lm) ~ hatvalues(movies.lm)) #Cercare di interpretare questo plot

#Plot the Prediction vs Real on Test set:
predictions_t_final = predict(movies.lm_t_final, movies_test_t_final) #Predictions
plot(predictions_t_final, movies_test_t_final$revenue, main = "Real Values vs Predictions", xlab = "Test predictions", ylab = "Real values")
abline(b = 1, a = 0, lwd = 3, col = "red")