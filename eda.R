#LIBRARIES:

library(dplyr)
library(ggplot2)
library(stringr)
library(cvms)
library(ggpubr)

#==================================================================================================================================================================================================================================

#IMPORT DATA:

#setwd("C:/Users/39392/Documents/Luca/Università/Laurea Magistrale Stochastics and Data Science/II° Year/I° Semester/Statistical Machine Learning/Project")
movies=read.csv("movies_cleaned.csv") #Read csv after pre-processing
movies=movies %>% select(-"X")

#==================================================================================================================================================================================================================================

#IMPLEMENTED FUNCTIONS:

order_corrs = function(X, p){
  R = cor(X)
  
  m = matrix(sort(R[p, -p], decreasing = T), ncol = 1)
  rownames(m) = names(sort(R[p, -p], decreasing = T))
  colnames(m) = paste("Ordered Correlations: ", colnames(X)[p])
  
  return(m)
}

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

corrs_report = function(X){
  
  corrs = list()
  for(p in seq(1, dim(X)[2])){
    corrs[[p]] = order_corrs(X, p)
  }
  
  names(corrs) = colnames(X)
  return(corrs)
}

#=================================================================================================================================================================================================================================

#EXPLORATORY DATA ANALYSIS:

#We look at the pie plots of the categorical predictors:

#Language:
mycols <- c("red", "green", "blue", "yellow", "pink")
language <- movies %>% 
  group_by(original_language) %>% 
  dplyr::summarise(n = n()) %>% 
  ungroup() %>% 
  mutate(
    prop = n / sum(n),
    prop = round(prop*100, 1)
  ) %>% 
  arrange(desc(prop)) %>% 
  mutate(original_language = factor(original_language, original_language)) %>% 
  arrange(desc(original_language)) %>% 
  mutate(lab.ypos = cumsum(prop) - 0.5*prop) 

pie1 <- ggplot(language, aes(x = "", y = prop, fill = original_language)) +
  geom_bar(width = 1, stat = "identity", color = "white") +
  coord_polar("y", start = 0)+
  scale_fill_manual(name = "Language",
                    values = mycols) +
  theme_void() +
  theme(legend.justification = "left")
pie1

#Belonging to a collection:
mycols <- c("red", "green")
collection <- movies %>% 
  group_by(belongs_to_collection) %>% 
  dplyr::summarise(n = n()) %>% 
  ungroup() %>% 
  mutate(
    prop = n / sum(n),
    prop = round(prop*100, 1)
  ) %>% 
  arrange(desc(prop)) %>% 
  mutate(belongs_to_collection = factor(belongs_to_collection, belongs_to_collection)) %>% 
  arrange(desc(belongs_to_collection)) %>% 
  mutate(lab.ypos = cumsum(prop) - 0.5*prop) 

pie2 <- ggplot(collection, aes(x = "", y = prop, fill = belongs_to_collection)) +
  geom_bar(width = 1, stat = "identity", color = "white") +
  coord_polar("y", start = 0)+
  scale_fill_manual(name = "Belongs to collection",
                    values = mycols) +
  theme_void() +
  theme(legend.justification = "left")
pie2

#Production Companies:
mycols <- c("red", "green", "blue", "yellow", "pink", "lightblue")
company <- movies %>% 
  group_by(production_companies) %>% 
  dplyr::summarise(n = n()) %>% 
  ungroup() %>% 
  mutate(
    prop = n / sum(n),
    prop = round(prop*100, 1)
  ) %>% 
  arrange(desc(prop)) %>% 
  mutate(production_companies = factor(production_companies, production_companies)) %>% 
  arrange(desc(production_companies)) %>% 
  mutate(lab.ypos = cumsum(prop) - 0.5*prop) 

pie3 <- ggplot(company, aes(x = "", y = prop, fill = production_companies)) +
  geom_bar(width = 1, stat = "identity", color = "white") +
  coord_polar("y", start = 0)+
  scale_fill_manual(name = "Companies",
                    values = mycols) +
  theme_void() +
  theme(legend.justification = "left")
pie3

pie = ggarrange(pie1, pie3, align="h")
pie

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Genres, we aim to merge the dataset containing the movie's genre information with the original data:
genres_df = read.csv("genres_df.csv") 
genres_df=genres_df %>% dplyr::select(-"X")
to_merge=merge(movies, genres_df, by="id", all=FALSE)

#We create a vector containing the variation in mean for each different movie's genre with respect to the 
#overall dataset mean of the target (revenue) (scaled by the standard deviation of the variable):
different_means = c() #Vector of variations
genres = colnames(dplyr::select(genres_df, -c("original_title", "id"))) #All the genres

#We populate the vector
for(genre in genres){
  different_means[genre] = (colMeans(dplyr::select(to_merge[which(to_merge[, genre] == 1), ], revenue)) - colMeans(dplyr::select(movies, revenue))) / sd(as.matrix(dplyr::select(movies, revenue)))
}

#We consider the "non significant means" to be the one with a difference lower than 0.10 on absolute value:
sort(different_means)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#We analyze the numerical predictors starting by their histogram and density plots in order to possibly detect 
#some particular distributions:

#Revenues:
plot(density(movies$revenue), lwd = 3, col = "black", main =  "Revenues: Histogram vs Density", xlab = "Movies revenues")
color = rgb(red = 0, green = 0, blue = 0.5, alpha = 0.1)
hist(movies$revenue, add = T, probability = T, col = color)

#Budget:
plot(density(movies$budget), lwd = 3, col = "black", main =  "Budget: Histogram vs Density", xlab = "Movie's budget", ylim = c(0, 0.000000027))
color = rgb(red = 0.2, green = 0.4, blue = 0.3, alpha = 0.1)
hist(movies$budget, add = T, probability = T, col = color)

#Runtime:
plot(density(movies[which(!is.na(movies$runtime)), "runtime"]), lwd = 3, col = "black", main =  "Runtime: Histogram vs Density", xlab = "Movie's runtime")
color = rgb(red = 1, green = 0.4, blue = 0.4, alpha = 0.2)
hist(movies$runtime, add = T, probability = T, col = color)

#Famous count:
plot(density(movies$famous_count), lwd = 3, col = "black", main =  "Famous count: Histogram vs Density", xlab = "Famous actors in the movie", ylim = c(0, 0.17))
color = rgb(red = 0, green = 0.5, blue = 0.5, alpha = 0.2)
hist(movies$famous_count, add = T, probability = T, col = color)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Now we look at the correlations through the common Pearson correlation coefficients and the scatterplots:
corrs_report(movies %>% dplyr :: select (famous_count, revenue, budget, runtime))$revenue
corrs_report(movies %>% select (famous_count, budget, runtime))

pairs(movies %>% select (famous_count, revenue, budget, runtime), main = "Scatterplots between numerical predictors") #We can detect some nice relationships, but the presence of outliers makes this evaluation difficult

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Outliers analysis:

#Multivariate Outliers: we briefly look at the squared Mahalanobis distances to see 
#if there're some multivariate strange observations: 
numerical = movies %>% select(c(famous_count, revenue, budget, runtime)) #Numerical predictors
numerical=as.matrix(numerical)
numerical = scale(numerical) #We scale data in order to avoid issues related to the different dimensionalities

distances=mahalanobis(numerical, colMeans(numerical), cor(numerical)) #Mahalanobis

plot(seq(1, length(distances)),distances, main = "Multivariate Outliers Analysis", xlab = "Observations", ylab = "Squared Mahalanobi's distances") #We detect a very strange observation and we remove it
abline(h = 300, col = "blue", lty = "dashed")
abline(h = 100, col = "red", lty = "dashed")
text(x = 500, y = 300, label = "Extremely High values", col = "blue", pos = 3)
text(x = 200, y = 100, label = "High values", col = "red", pos = 3)

#Extract sure Multivariate outliers:
mult_out = which(distances>100)

for(movie in mult_out){
  text(x = movie, y = distances[movie], label = movies[movie, "original_title"], col = "black", pos = 2, cex=0.5)
}

plot(seq(1, length(distances)-5), mahalanobis(numerical, colMeans(numerical), cor(numerical))[-which(distances>100)]) #We repeat the plot removing the strange observation

#We look at the scatterplots for quantitative variables after removing the multivariate outliers
pairs(movies[-mult_out, ] %>% select (revenue, budget, runtime, famous_count))

#Univariate outliers (BoxPlots): we plot and extract the indexes of the observations

#Looking at the boxplots we decide to store in a vector the multivariate outliers found before 
#and the sure univariate outliers for budget and runtime
remove = c(which(movies$budget > 250000000), which(movies$runtime > 300), mult_out)

#Now we are going to see the observations in remove in the boxplots of all the 4
#quantitative variables
par(mfrow=c(2,2))

out_rev = which(movies$revenue %in% boxplot(movies$revenue, main="Revenue")$out)
i = 1
for(out in remove){
  if(out %in% out_rev){
    if(out == 3143){
      text(x = 1, y = movies[out, "revenue"], labels = movies[out, "original_title"], col = "red", pos = 4, cex = 0.6)
      points(x = 1, y = movies[out, "revenue"], pch = 16, col = "red")
      next
    }
    if(out == 578){
      text(x = 1, y = movies[out, "revenue"], labels = movies[out, "original_title"], col = "red", pos = 3, cex = 0.6)
      points(x = 1, y = movies[out, "revenue"], pch = 16, col = "red")
      next
    }
    if(i%%2 == 0){
      text(x = 1, y = movies[out, "revenue"], labels = movies[out, "original_title"], col = "red", pos = 4, cex = 0.6)
      points(x = 1, y = movies[out, "revenue"], pch = 16, col = "red")
    }
    else{
      text(x = 1, y = movies[out, "revenue"], labels = movies[out, "original_title"], col = "red", pos = 2, cex = 0.6)
      points(x = 1, y = movies[out, "revenue"], pch = 16, col = "red")
    }
  }
  i = i + 1
}

out_bud = which(movies$budget %in% boxplot(movies$budget, main="Budget")$out)
i = 1
for(out in remove){
  if(out %in% out_bud){
    if(out == 133){
      text(x = 1, y = movies[out, "budget"], labels = movies[out, "original_title"], col = "red", pos = 3, cex = 0.6)
      points(x = 1, y = movies[out, "budget"], pch = 16, col = "red")
      next
    }
    if(out == 578){
      text(x = 1, y = movies[out, "budget"], labels = movies[out, "original_title"], col = "red", pos = 1, cex = 0.6)
      points(x = 1, y = movies[out, "budget"], pch = 16, col = "red")
      next
    }
    if(out == 4696){
      text(x = 1, y = movies[out, "budget"] + 20000000, labels = movies[out, "original_title"], col = "red", pos = 4, cex = 0.6)
      points(x = 1, y = movies[out, "budget"], pch = 16, col = "red")
      next
    }
    if(out == 249){
      text(x = 1, y = movies[out, "budget"], labels = movies[out, "original_title"], col = "red", pos = 2, cex = 0.6)
      points(x = 1, y = movies[out, "budget"], pch = 16, col = "red")
      next
    }
    if(out == 2666){
      text(x = 1, y = movies[out, "budget"], labels = movies[out, "original_title"], col = "red", pos = 2, cex = 0.6)
      points(x = 1, y = movies[out, "budget"], pch = 16, col = "red")
      next
    }
    if(out == 4069){
      text(x = 1, y = movies[out, "budget"], labels = movies[out, "original_title"], col = "red", pos = 2, cex = 0.6)
      points(x = 1, y = movies[out, "budget"], pch = 16, col = "red")
      next
    }
    if(out == 3502){
      text(x = 1, y = movies[out, "budget"] - 10000000, labels = movies[out, "original_title"], col = "red", pos = 4, cex = 0.6)
      points(x = 1, y = movies[out, "budget"], pch = 16, col = "red")
      next
    }
    if(out == 230){
      text(x = 1.2, y = movies[out, "budget"], labels = movies[out, "original_title"], col = "red", pos = 4, cex = 0.6)
      points(x = 1, y = movies[out, "budget"], pch = 16, col = "red")
      next
    }
    if(i%%2 == 0){
      text(x = 1, y = movies[out, "budget"], labels = movies[out, "original_title"], col = "red", pos = 4, cex = 0.6)
      points(x = 1, y = movies[out, "budget"], pch = 16, col = "red")
    }
    else{
      text(x = 1, y = movies[out, "budget"], labels = movies[out, "original_title"], col = "red", pos = 2, cex = 0.6)
      points(x = 1, y = movies[out, "budget"], pch = 16, col = "red")
    }
  }
  i = i + 1
}

out_run = which(movies$runtime %in% boxplot(movies$runtime, main="Runtime")$out)
for(out in remove){
  if(out %in% out_run){
    if(out == 578){
      text(x = 1, y = movies[out, "runtime"], labels = movies[out, "original_title"], col = "red", pos = 3, cex = 0.6)
      points(x = 1, y = movies[out, "runtime"], pch = 16, col = "red")
      next
    }
    if(out == 133){
      text(x = 1, y = movies[out, "runtime"], labels = movies[out, "original_title"], col = "red", pos = 3, cex = 0.6)
      points(x = 1, y = movies[out, "runtime"], pch = 16, col = "red")
      next
    }
    
    if(i%%2 == 0){
      text(x = 1, y = movies[out, "runtime"], labels = movies[out, "original_title"], col = "red", pos = 4, cex = 0.6)
      points(x = 1, y = movies[out, "runtime"], pch = 16, col = "red")
    }
    else{
      text(x = 1, y = movies[out, "runtime"], labels = movies[out, "original_title"], col = "red", pos = 2, cex = 0.6)
      points(x = 1, y = movies[out, "runtime"], pch = 16, col = "red")
    }
  }
  i = i + 1
}

out_fam = which(movies$famous_count %in% boxplot(movies$famous_count, main="Famous Count")$out)
for(out in remove){
  if(out %in% out_fam){
    if(out == 578){
      text(x = 1, y = movies[out, "famous_count"], labels = movies[out, "original_title"], col = "red", pos = 3, cex = 0.6)
      points(x = 1, y = movies[out, "famous_count"], pch = 16, col = "red")
      next
    }
    if(i%%2 == 0){
      text(x = 1, y = movies[out, "famous_count"], labels = movies[out, "original_title"], col = "red", pos = 4, cex = 0.6)
      points(x = 1, y = movies[out, "famous_count"], pch = 16, col = "red")
    }
    else{
      text(x = 1, y = movies[out, "famous_count"], labels = movies[out, "original_title"], col = "red", pos = 2, cex = 0.6)
      points(x = 1, y = movies[out, "famous_count"], pch = 16, col = "red")
    }
  }
  i = i + 1
}

par(mfrow=c(1,1))