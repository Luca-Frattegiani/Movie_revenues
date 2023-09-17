#DATA SOURCE:
#[https://developers.themoviedb.org/3/getting-started/popularity#movies]

#LIBRARIES:

library(dplyr)
library(ggplot2)
library(stringr)
library(cvms)
library(ggpubr)

#==================================================================================================================================================================================================================================

#DATA IMPORTATION:
#setwd("C:/Users/39392/Documents/Luca/Università/Laurea Magistrale Stochastics and Data Science/II° Year/I° Semester/Statistical Machine Learning/Project")
movies=read.csv2("movies_data.csv")
View(movies)

#==================================================================================================================================================================================================================================

#DATA PRE-PROCESSING:

#Convert all the quantitative variables into numeric type:
movies$vote_average=as.numeric(movies$vote_average)
movies$vote_count=as.numeric(movies$vote_count)
movies$budget=as.numeric(movies$budget)
movies$revenue=as.numeric(movies$revenue)
movies$popularity=as.numeric(movies$popularity)
movies$runtime=as.numeric(movies$runtime)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Now we are going to handle the categorical variables 

#Adult predictor: we simply delete it from our dataframe since actually it shows only one modality
dim(movies %>% filter(adult != "False"))
dim(movies %>% filter(adult == "False"))
movies = movies %>% select(-"adult")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Release Date: since movies from remote past suffers of sparsity for some important 
#variables and since they're strongly different from more recent films, 
#we decide to focus the analysis only on films between 1980 and 2017

#Firstly we extract the year from the "release_date" column:
dates = c() #Vector with dates expressed in years
i = 1
for(row in seq(1, length(movies$release_date))){
  dates[i] = strsplit(movies$release_date[row], split = "-")[[1]][1]
  i = i + 1
}

movies$release_date = as.numeric(dates) #We turn the column in a variable containing only the year information

#Now we filter the dataset considering only the period of time 1980-2017:
movies = movies %>% filter(release_date >= 1980)

#Discussion about maintaining or not the Release Date information:
#Our idea was to aggregate the variable in classes that represent the decades (80's, 90's, 00's and 10's) but we think
#that under a prediction task, maintaining the time information is useless since new film for which we can think of predict
#the revenue are all of the newest era for sure, so we prefer to remove the temporal information

movies = movies %>% select(-"release_date")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Budget and Revenues: since revenue is our target and budget will be a very important predictor 
#we consider only data which show an acceptable value for the budget and the revenue amount:

movies=movies %>% filter(movies$budget!=0)
movies=movies %>% filter(movies$revenue!=0)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Collections: we aim to turn this column into a boolean variable
#with True if the film is part of a collection and False otherwise

collection=movies$belongs_to_collection #vector with the new encoding for the variable
#We populate the vector assigning TRUE if there's an object in the row or FALSE otherwise
for (i in 1:length(collection)){
  if(collection[i]==""){
    collection[i]=F
  }
  else {
    collection[i]=T
  }
  collection[i] = as.logical(collection[i])
}

movies$belongs_to_collection=as.logical(collection) #Modify the column that now is a boolean binary variable

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Original language: we aim to merge together some of the languages to have 
#less categories 
movies$original_language=as.factor(movies$original_language) #First of all transform the variable into a factor 
languages=levels(movies$original_language)

freq=table(movies$original_language) #Table with the frequencies of the languages

for(i in 2:length(languages)){ #loop over all the levels 
  if(freq[languages[i]]<1){
    languages[i]="Others"
  }
  if(languages[i] %in% c("af", "bm", "cn", "xx")){ #African languages or artificial languages
    languages[i]="Others"
  }
  if(languages[i] %in% c("ca", "da", "de", "es", "fi", "fr", "hu", "is", "it", "nb",
                         "nl", "no", "pl", "pt", "ro", "ru", "sr", "sv")){ #European languages different from english
    languages[i]="Europe"
  }
  if(languages[i] %in% c("fa", "he", "id", "ja", "ko", "th", "tr", "ur", "vi", "zh")){ #Asian languages excluding India
    languages[i]="Asia"
  }
  if(languages[i] %in% c("hi", "kn", "ml", "mr", "ta", "te")){ #Languages spoken in India
    languages[i]="India"
  }
  if(languages[i]=="en"){
    languages[i]="English"
  }
}
languages[1]="Others" #Put the na'a into the Others category
levels(movies$original_language)=languages
#Now we have just 5 categories: English, Europe, Asia, India and Others
summary(movies$original_language)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Production company: we decide to keep just the first company, which is the most
#important one in the production process
companies = list() #List of the firts company for every observation
company_string=list() #Temporary list that we need to easy the process

for(row in seq(1, length(movies$production_companies))){ #Loop over all the rows
  company_string[row]=strsplit(toString(movies$production_companies[row]), 
                               split = ",")[[1]][1]
  companies[row]=strsplit(toString(company_string[row]), split="'")[[1]][4]
}

#Now companies is a list of lists. We need to transform it into a vector 
company_vector=c()
for(i in 1:length(companies)){
  company_vector[i]=companies[[i]][1]
}

movies$production_companies=company_vector
movies$production_companies=as.factor(movies$production_companies)
#Now movies$production_companies is a factor variablee with the most important production company
#for each observation

#Now we want to put together the companies that belong to the 5 mayor of Hollywood
#and label all the other companies as smaller company
summary(movies$production_companies)
new_companies=rep("Smaller company", length(levels(movies$production_companies)))
for(i in 1:length(new_companies)){
  if(levels(movies$production_companies)[i] %in% c("Universal Pictures","Dreamworks", "DreamWorks Animation",
                                                   "Nu Image Films", "Universal Studios", "Focus Features",
                                                   "PolyGram Filmed Entertainment", "DreamWorks", "Universal",
                                                   "Universal TV", "DreamWorks Pictures", "Universal Pictures Corporation" )){
    new_companies[i]="NBCUniversal"
  }
  if(levels(movies$production_companies)[i] %in% c("Paramount Pictures", "Miramax Films", "Paramount Vantage",
                                                   "Miramax", "Paramount Classics", "Paramount")){
    new_companies[i]="Paramount Global"
  }
  if(levels(movies$production_companies)[i] %in% c("Warner Bros.", "New Line Cinema", "Castle Rock Entertainment",
                                                   "Fine Line Features", "Dune Entertainment", 
                                                   "Spyglass Entertainment", "Alcon Entertainment", "DC Comics",
                                                   "DreamWorks SKG", "New Line Productions", "Warner Bros. Animation",
                                                   "Warner Bros. Family Entertainment", "Warner Bros. Pictures")){
    new_companies[i]="Warner Bros. Entertainment"
  }
  if(levels(movies$production_companies)[i] %in% c("Walt Disney Pictures", "Twentieth Century Fox Film Corporation",
                                                   "Fox Searchlight Pictures", "Touchstone Pictures", "Regency Enterprises",
                                                   "Hollywood Pictures", "Fox 2000 Pictures", "UTV Motion Pictures",
                                                   "Blue Sky Studios", "Lucasfilm", "Marvel Studios", "Caravan Pictures",
                                                   "20th Century Fox", "Twentieth Century-Fox", "Walt Disney",
                                                   "Walt Disney Animation Studios", "Walt Disney Animation Studios",
                                                   "Walt Disney Television Animation", "Pixar Animation Studios",
                                                   "Twentieth Century Fox", "Walt Disney Productions", "Walt Disney Studios Motion Pictures" )){
    new_companies[i]="The Walt Disney Studios"
  }
  if(levels(movies$production_companies)[i] %in% c("Columbia Pictures", "TriStar Pictures", "Columbia Pictures Corporation",
                                                   "Screen Gems", "Sony Pictures Classics", "Sony Pictures",
                                                   "Sony Pictures Animation", "Sony Music Entertainment Japan",
                                                   "Columbia Pictures Industries", "Columbia TriStar")){
    new_companies[i]="Sony Pictures Entertainment"
  }
}
levels(movies$production_companies)=new_companies
summary(movies$production_companies)
#Now production companies has just 6 categories. We are going to handle the missing 
#values later on

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Production country: we decide to keep just the first country, which is the most
#important one in the production process
#The process is very similar to the one used for production_companies
countries = list()
country_string=list()

for(row in seq(1, length(movies$production_countries))){ #Loop over all the rows
  country_string[row]=strsplit(toString(movies$production_countries[row]), 
                               split = ",")[[1]][1]
  countries[row]=strsplit(toString(country_string[row]), split=":")[[1]][2]
  countries[row]=str_replace_all(countries[row], "[[:punct:]]", "")
  countries[row]=str_replace_all(countries[row], " ", "")
}

country_vector=c()
for(i in 1:length(countries)){
  country_vector[i]=countries[[i]][1]
}

movies$production_countries=country_vector
movies$production_countries=as.factor(movies$production_countries)
summary(movies$production_countries)

#Now we want to understand if this information is useful or if this can be discarded
#given that we already have the original_language variable. To do that, we are going 
#to divide the countries into the same categories of original_language and then we are 
#going to produce the confusion matrix
Country=levels(movies$production_countries)
for(i in 1:length(Country)){
  if(Country[i] %in% c("AE", "CN", "HK", "ID", "IL", "IR", "KH", "KR", "MY", "PH", 
                       "PK", "QA", "SG", "TH", "TR", "TW", "JP")){
    Country[i]="Asia"
  }
  if(Country[i] %in% c("AT", "BE", "BG", "CH", "CZ", "DE", "DK", "ES", "FI", "FR",
                       "GR", "HU", "IS", "IE", "IT", "LU", "NL", "NO", "PL", "RO", 
                       "RS", "SE", "UA", "RU")){
    Country[i]="Europe"
  }
  if(Country[i] %in% c("AU", "CA", "GB", "MT", "NZ", "US", "ZA")){
    Country[i]="English"
  }
  if(Country[i]=="IN"){
    Country[i]="India"
  }
  if(Country[i] %in% c("BR", "CL", "DZ", "PE", "UY", "BF", "AR", "ML", "MX", "EC", "NA")) {
    Country[i]="Others"
  }
}

levels(movies$production_countries)=Country
summary(movies$production_countries)
movies2=movies %>% filter(!is.na(production_countries))

matrix=confusion_matrix(movies2$production_countries, movies2$original_language)
cvms::plot_confusion_matrix(matrix$`Confusion Matrix`[[1]],
                            add_normalized = FALSE, add_col_percentages = FALSE,
                            add_row_percentages = FALSE)

#Since the confusion matrix has almost all the observations on the diagonal we decide
#to remove the variable production_countries
movies=movies %>% select(-"production_countries")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#REMOVING DOUBLE ID'S

#Before moving to the genre variable we check if there are some problems
#with the id column, since we are going to create a new dataset using this a key variable 
#and then we are going to merge datasets using this variable as a join. 
#We found out that some movies had two observations and the only difference was in the value 
#of popularity (a variables that we are not going to use). So we remove one of the observation
#so that when we are going to merge movies with other datasets there will be no mistakes.

movies_id=movies
movies_id$id=as.factor(movies_id$id)
summary(movies_id$id)
which(movies_id$id==4912)
movies=movies[-1638,]

movies_id=movies
movies_id$id=as.factor(movies_id$id)
summary(movies_id$id)
which(movies_id$id==10991)
movies=movies[-1226,]

movies_id=movies
movies_id$id=as.factor(movies_id$id)
summary(movies_id$id)
which(movies_id$id==15028)
movies=movies[-1466,]

movies_id=movies
movies_id$id=as.factor(movies_id$id)
summary(movies_id$id)
which(movies_id$id==77221)
movies=movies[-2368,]

movies_id=movies
movies_id$id=as.factor(movies_id$id)
summary(movies_id$id)
which(movies_id$id==110428)
movies=movies[-1278,]

movies_id=movies
movies_id$id=as.factor(movies_id$id)
summary(movies_id$id)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Genre: we aim to understand which are the most important categories and produce 
#a number of boolean columns for each relevant category
categories = list() #List of vectors containing all the genres for each film
stranges = c() #vector of the film without any genre
k = 1
j = 1

for(row in seq(1, length(movies$genres))){ #Loop over all the rows
  num_gen = length(strsplit(toString(movies$genre[row]), split = ":")[[1]]) #Extract the number of genres*2
  if(num_gen <= 1){ #Manage the cases where we don't have any genre assigned
    stranges[k] = row #Store the rows without any genre
    k = k + 1
    categories[[j]] = c()
    j = j + 1
    next
  }
  indexes = seq(from = 3, to = num_gen, by = 2) 
  genres = c() #Vector to store the genres assigned to the film
  i = 1
  for(index in indexes){
    genres[i] = strsplit(strsplit(toString(movies$genre[row]), split = ":")[[1]][index], split = "'")[[1]][2] #Storage of the genres
    i = i + 1 
  }
  categories[[j]] = genres
  j = j + 1
}

#Now we create a list with all the possible genres ordered by number of occurences (to evaluate how we can manage this variable):
labels = list()
for(film in categories){
  for(category in film){
    check = F
    for(name in names(labels)){
      if(category == name){
        check = T
        break
      }
    }
    if(check){
      labels[[category]] = labels[[category]] + 1
    }
    else{
      labels[[category]] = 1
    }
  }
}

new = list()
temporary = labels
for(countdown in seq(length(temporary), 1)){
  max = names(temporary)[1]
  for(i in seq(1, length(temporary))){
    if(temporary[[i]] > temporary[[max]]){
      max = names(temporary)[i]
    }
  }
  new[[max]] = temporary[[max]]
  temporary[[max]] = 0
}
new

#Now we create a separate dataframe containing the information of the genres:
genres_df = select(movies, original_title, genres, id)  #Separate dataframe

for(genre in names(new)){ #Loop through all the categories
  genres_df[, genre] = rep(0, dim(genres_df)[1]) #Define a new column for each of them
  row = 1 #Initialize row index
  for(movie in categories){ #Loop over all movies
    for(type in movie){ #Loop over all the genres of any film
      if(type == genre){ #Check the correspondences
        genres_df[row, genre] = 1 #If match, we set the value equal 1
      }
    }
    row = row + 1 #Increase the row
  }
}

genres_df = genres_df[, -2]
movies = select(movies, -genres) #Remove the predictor

write.csv(genres_df, file = "genres_df.csv")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Cast information:
#We extract the cast information from another dataset in order to create a column that expresses the number of famous actors in a film and create a new predictor

cast = read.csv("credits.csv")
cast = cast %>% filter(id %in% movies$id)
cast=cast %>% filter(cast!="[]")

#Again there are some movies that are repeated twice and the difference is in the crew variable,
#which we are not going to use. So we remove one of the two observation so that we do not 
#have problems with the merge

cast_id=cast
cast_id$id=as.factor(cast_id$id)
summary(cast_id$id)
which(cast_id$id==4912)
cast=cast[-1636,]

cast_id=cast
cast_id$id=as.factor(cast_id$id)
summary(cast_id$id)
which(cast_id$id==10991)
cast=cast[-1226,]

cast_id=cast
cast_id$id=as.factor(cast_id$id)
summary(cast_id$id)
which(cast_id$id==15028)
cast=cast[-1466,]

cast_id=cast
cast_id$id=as.factor(cast_id$id)
summary(cast_id$id)
which(cast_id$id==77221)
cast=cast[-2365,]

cast_id=cast
cast_id$id=as.factor(cast_id$id)
summary(cast_id$id)
which(cast_id$id==110428)
cast=cast[-1278,]

cast_id=cast
cast_id$id=as.factor(cast_id$id)
summary(cast_id$id)

movies=movies %>% filter(id %in% cast$id)

names=c() #Vector that will contain all the actors stored in cast
all_strings=list() #List of lists of the actors in every movie
for(row in 1:length(cast$cast)){
  all_strings[row]=strsplit(toString(cast$cast[row]), split=",")
  for(i in 1:length(all_strings[[row]])){
    if(startsWith(all_strings[[row]][i], " 'name':")){
      names=append(names, toString(all_strings[[row]][i]))
    }
  }
}

#Choose just the name of the actor
for(row in 1:length(all_strings)){
  for(i in seq(length(all_strings[[row]]), 1, -1)){
    if(!(startsWith(all_strings[[row]][i], " 'name':"))){
      all_strings[[row]]=all_strings[[row]][-i]
    }
  }
}

for(row in 1:length(all_strings)){
  for(i in 1:length(all_strings[[row]])){
    all_strings[[row]][i]=strsplit(all_strings[[row]][i], split=":")[[1]][2]
    all_strings[[row]][i]=str_replace_all(all_strings[[row]][i], "[[:punct:]]", "")
  }
}

for(j in 1:length(names)){
  names[j]=strsplit(names[j], split=":")[[1]][2]
  names[j]=str_replace_all(names[j], "[[:punct:]]", "")
}
names=as.factor(names)
summary(names)

#Now we want to differentiate between famous actors and not famous actors. We decide 
#that famous actors will be only the ones that are present in more than 10 movies
famous=c()
freq_names=table(names)
factors=levels(names)
for(i in 1:length(factors)){
  if(freq_names[i]>10){
    famous=append(famous, factors[i])
  }
}
length(famous)

#Count for every movie the number of famous actors
famous_count=rep(0, length(all_strings))
for(row in 1:length(all_strings)){
  for(i in 1:length(all_strings[[row]])){
    if(all_strings[[row]][i] %in% famous){
      famous_count[row]=famous_count[row]+1
    }
  }
}

#Variables famous_count is added to cast
cast=data.frame(cast, famous_count)
#Merge cast and movies using id as a join 
movies3=merge(movies, cast, by="id")
#Remove useless information
movies=movies3 %>% select(-c("cast", "crew"))

#==================================================================================================================================================================================================================================

#MISSING DATA MANAGEMENT: check all the variables that we are going to use and 
#handle with the missing values

#Production company predictor:

#We need to find a way to manage the NA's for this variable since they don't show any strange behavior for the other predictors: 
filter(movies, is.na(production_companies)) #Other values are present

#These observations have strange values (detected by high negative differences in their means) only as regards the economic variables, 
#this probably can be interpreted by saying that this movies have been produced by small companies
colMeans(select(filter(movies, is.na(production_companies)), 
                revenue, budget, vote_count, vote_average, runtime, popularity)) -
  colMeans(select(movies, revenue, budget, vote_count, vote_average, runtime, popularity))

for(i in 1:length(movies$production_companies)){
  if (is.na(movies$production_companies[i])){
    movies$production_companies[i]="Smaller company"
  }
}
summary(movies$production_companies)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Original Title (index variable): no missing values

dim(filter(movies, !is.character(original_title)))[1]
dim(filter(movies, is.na(original_title)))[1]

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Belongs to collection: no missing values

dim(filter(movies, is.na(belongs_to_collection)))[1]
dim(filter(movies, !is.logical(belongs_to_collection)))[1]

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Original Language: no missing values

dim(filter(movies, is.na(original_language)))[1]

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Runtime: just one missing values, we are going to remove this observation

dim(filter(movies, is.na(runtime)))[1]
dim(filter(movies, is.nan(runtime)))[1]
dim(filter(movies, !is.numeric(runtime)))[1]
movies=movies %>% filter(!is.na(runtime))

#==================================================================================================================================================================================================================================

#To have the final dataset that we are going to study we remove the variables that we are
#not going to use since they are information that are found together with our target
#and not in advance like the others
#se vuoi il dataset come prima basta commentare questo comando 
movies=movies %>% select(-c("popularity", "vote_average", "vote_count"))

write.csv(movies, "movies_cleaned.csv")