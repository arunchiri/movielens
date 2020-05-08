
# Introduction
#Here is the code which is used in the report for making the movielens recommendation system.
# Data Wrangling
## Getting and Cleaning Data

#First step in any data analysis project is to get the data. 
#We can download the data and clean the code using the below 

#return memory to operating system
gc(full = TRUE)
# Note: this process could take a couple of minutes

if(!require(tidyverse))
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) 
  install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) 
  install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) 
  install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(Metrics)) 
  install.packages("Metrics", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

## Create edx(Train) and Validation datasets

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in train set

validation <- temp %>% 
     semi_join(edx, by = "movieId") %>%
     semi_join(edx, by = "userId")

# Add rows removed from validation set back into train set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


# Data Exploration

#We can see the validation & Train(edx) sets are in Tidy format using the head() function.

head(edx)
head(validation)

#Each row represents a rating given by one user *u* to one movie *i*.
#We can see the number of unique users that provide ratings and for how many 
#unique movies they provided them using this code.

edx %>% 
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

#seeing how sparse the data using the below code
users <- sample(unique(edx$userId), 100)
rafalib::mypar()
edx %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")
rm(users)


#Each outcome *y* has a different set of predictors. 
#To see this, note that if we are predicting the rating for movie *i* by user *u*, 
#in principle, all other ratings related to movie *i* and by user *u* may be used as predictors.
#But different users rate a different number of movies and different movies.Furthermore,
#we may be able to use information from other movies that we have determined are similar to movie *i*
#or from users determined to be similar to user *u*.
#So in essence, the entire matrix can be used as predictors for each cell.

## Biases in Data

#Lets' see some biases in data

### Movie Bias
#In this we notice that some movies are rated more than other movies. 
#Given below is the distribution for that:

edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

### User Bias
#In this we notice that the ratings are baised on the users. 
#Some users rate lot of movies, but others rate only very few movies. 


edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")


## Create Test set and Training Set from Edx dataset

#since we cannot use the validation set as the test set, 
#we need to split the edx dataset into test set and train set again. The code is as follows.

# Create train set and test set
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, p=0.2, list = FALSE)
trainSet <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set
testSet <- temp %>% 
  semi_join(trainSet, by = "movieId") %>%
  semi_join(trainSet, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, testSet)
trainSet <- rbind(trainSet, removed)
#rm can be used to remove objects and free space
rm(test_index, temp, removed)


# Data Analysis

## Selecting loss Function 
#To compare different models or to see how well we're doing compared to some baseline, 
#we need to quantify what it means to do well. We need a loss function. 
#The Netflix challenge used the typical error and thus decided on a winner based on the 
#residual mean squared error on a test set.

#Function for RMSE calculation
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#And now we're ready to build models and compare them to each other using RMSE.

## Different Models

### Naive approach

#A model that assumes the same rating for all movies and users with all the differences explained by random variation would look like this:

mu_hat <- mean(trainSet$rating)
mu_hat


#So that is the average rating of all movies across all users. 
#If we compare all the know ratings in test dataset with $\hat{\mu}$ we obtain the following RMSE:

naive_rmse <- RMSE(testSet$rating, mu_hat)
naive_rmse


#This RMSE is quite big. As per the requite of this project, we need to get an RMSE  < 0.86490. 
#Now because as we go along we will be comparing different approaches,
#we're going to create a table that's going to store the results that we obtain as we go along. 
#Let’s start by creating a results table with this naive approach:

rmse_results <- data.frame(method = "Naive approach", RMSE = naive_rmse)
rmse_results %>% knitr::kable()


### Modeling the movie effects

#checking the movie bais

mu <- mean(trainSet$rating) 
movie_avgs <- trainSet %>% 
     group_by(movieId) %>% 
     summarize(b_i = mean(rating - mu))


#We can see that these estimates vary substantially, not surprisingly. Some movies are good, but others are bad. 


movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))



#The overall average which we got is about 3.5.

 
# calculate predictions considering movie effect
predicted_ratings <- mu + testSet %>% 
     left_join(movie_avgs, by='movieId') %>%
     .$b_i
# calculate rmse after modelling movie effect
model_1_rmse <- RMSE(predicted_ratings, testSet$rating)
 
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie effect model",  
                                     RMSE = model_1_rmse))
rmse_results %>% knitr::kable()


#Movie effects model shows a improvement over naive method. But RMSE didn't reached it's goal of less than  0.86490.
#So we need to go after other approaches which can bring down the RMSE further below the goal. 

### Modeling the user effects

#ploting user effect
trainSet %>% group_by(userId) %>% 
            summarize(b_u = mean(rating)) %>% 
            filter(n() >= 100) %>% 
            ggplot(aes(b_u)) + 
            geom_histogram(bins = 30, color = "black")

#adding the userid effect to the movie effects
user_avgs <- trainSet %>% 
                left_join(movie_avgs, by='movieId') %>%
                group_by(userId) %>%
                summarize(b_u = mean(rating - mu - b_i))


#We can now construct predictors and see how much the RMSE improves:

# calculate predictions considering user effects in previous model
predicted_ratings <- testSet %>% 
                        left_join(movie_avgs, by='movieId') %>%
                        left_join(user_avgs, by='userId') %>%
                        mutate(pred = mu + b_i + b_u) %>%
                        .$pred
# calculate rmse after modelling user specific effect in previous model
model_2_rmse <- RMSE(predicted_ratings, testSet$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User effects model",  
                                     RMSE = model_2_rmse))
#adding the RMSE results to the table
rmse_results %>% knitr::kable()


#Movie + User effects model shows a improvement over naive method. But RMSE didn't reached it's goal of less than  0.86490. So we need to go after other approaches which can bring down the RMSE further below the goal. 

### Regularizing movie and user effects

#While investigating the cause of less improvement in RMSE, we could see that the top 10 movies and top worst movies are not correct using the movie effect. 

#TOP 10 BEST MOVIES*

movie_titles <- edx %>% 
     select(movieId, title) %>%
     distinct()
movie_avgs %>% left_join(movie_titles, by="movieId") %>%
     arrange(desc(b_i)) %>% 
     select(title, b_i) %>% 
     slice(1:10) %>%  
     knitr::kable()

#TOP 10 WORST MOVIES*

movie_avgs %>% left_join(movie_titles, by="movieId") %>%
     arrange(b_i) %>% 
     select(title, b_i) %>% 
     slice(1:10) %>%  
     knitr::kable()

#So this doesn't make any sense. So let's look at how often they are rated. 

#TOP BEST MOVIES WITH COUNT OF RATINGS*

trainSet %>% dplyr::count(movieId) %>% 
     left_join(movie_avgs) %>%
     left_join(movie_titles, by="movieId") %>%
     arrange(desc(b_i)) %>% 
     select(title, b_i, n) %>% 
     slice(1:10) %>% 
     knitr::kable()


#TOP WORST MOVIES WITH COUNT OF RATINGS*

trainSet %>% dplyr::count(movieId) %>% 
     left_join(movie_avgs) %>%
     left_join(movie_titles, by="movieId") %>%
     arrange(b_i) %>% 
     select(title, b_i, n) %>% 
     slice(1:10) %>% 
     knitr::kable()


#So the supposed best and worst movies are rated by very few users in most of the cases. 

#This happened because with just a few users, we have more uncertainity. Therefore larger estimates of $b_{i}$, negative or positive, are more likely when fewer users rate the movies. 

#These are basically noisy estimates that we should not trust, especially when it comes to prediction. Large errors can increase our residual mean squared error, so we would rather be conservative when we're not sure.

#When making predictions, we need one number, one prediction, not an interval. 

#### Regularization
#For this, we introduce the concept of $Regularization$. Regularization permits us to penalize large estimates that comes from small sample sizes. The general idea is to add a penalty for large values of $b$ to the sum of squares equations that we minimize.


lambda <- 3
mu <- mean(trainSet$rating)
movie_reg_avgs <- trainSet %>% 
     group_by(movieId) %>% 
     summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n())


#To see how the estimates shrink, let's make a plot of the regularized estimate versus the least 
#square estimates with the size of the circle telling us how large ni was.You can see that when n is small,\
#the values are shrinking more towards zero.


data_frame(original = movie_avgs$b_i, 
           regularlized = movie_reg_avgs$b_i, 
           n = movie_reg_avgs$n_i) %>%
     ggplot(aes(original, regularlized, size=sqrt(n))) + 
     geom_point(shape=1, alpha=0.5)



#TOP BEST MOVIES WITH COUNT OF RATINGS AFTER REGULARIZATION*

trainSet %>%
     dplyr::count(movieId) %>% 
     left_join(movie_reg_avgs) %>%
     left_join(movie_titles, by="movieId") %>%
     arrange(desc(b_i)) %>% 
     select(title, b_i, n) %>% 
     slice(1:10) %>% 
     knitr::kable()

#TOP WORST MOVIES WITH COUNT OF RATINGS AFTER REGULARIZATION*

trainSet %>%
     dplyr::count(movieId) %>% 
     left_join(movie_reg_avgs) %>%
     left_join(movie_titles, by="movieId") %>%
     arrange(b_i) %>% 
     select(title, b_i, n) %>% 
     slice(1:10) %>% 
     knitr::kable()


#We can see the predictions has been improved a lot for TOP movies and WORST movies. 

#Note that $\lambda$ is a tuning parameter in this equation. We can use cross validation to choose it.
#Also going to add the user effects to improve our predictions. 

lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
     mu <- mean(trainSet$rating)
     b_i <- trainSet %>%
          group_by(movieId) %>%
          summarize(b_i = sum(rating - mu)/(n()+l))
     b_u <- trainSet %>% 
          left_join(b_i, by="movieId") %>%
          group_by(userId) %>%
          summarize(b_u = sum(rating - b_i - mu)/(n()+l))
     predicted_ratings <- 
          testSet %>% 
          left_join(b_i, by = "movieId") %>%
          left_join(b_u, by = "userId") %>%
          mutate(pred = mu + b_i + b_u) %>%
          .$pred
     return(RMSE(predicted_ratings, testSet$rating))
})


#When we plot the lambdas aganist RMSEs geenrated, we will get the value of lambda when RMSE is small. 

qplot(lambdas, rmses) 


#So from the plot we understood that, for which value of lambda we are getting the lowest RMSE. 


lambda <- lambdas[which.min(rmses)]
lambda

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User Effect Model",  
                                     RMSE = min(rmses)))

#adding the RMSE to existin table
rmse_results %>% knitr::kable()


#Regularized Movie + User effects model shows a improvement over Movie + User effects model method.
#But RMSE didn't reached it's goal of less than  0.86490. 
#So we need to go after other approaches which can bring down the RMSE further below the goal. 

### Other Biases

#One of the other division we saw in the edx dataset is by *genere*. 
#On plotting error bar plots for average ratings of movies grouped by genres with more than 30,000 ratings, 
#evidence of genre effect $b_{g}$ is found.

#Group ratings by genres and plot the error bar plots for genres with over 30,000 ratings
edx %>% group_by(genres) %>%
  summarize(n=n(),avg=mean(rating),se=sd(rating)/sqrt(n())) %>% #mean and standard errors
  filter(n >= 30000) %>% #Keeping genres with ratings over 30,000
  mutate(genres = reorder(genres, avg)) %>% #order genres by mean ratings
  ggplot(aes(x=genres,y=avg,ymin=avg-2*se,ymax=avg+2*se)) + #lower and upper confidence intervals
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  ggtitle("Average Ratings of Genres with >=30,000 Ratings")


#if we remove the axis text we can see the average error in the rating for different generes. 
#So the genere effect $b_{u}$ also we need to take into account. 

#Group ratings by genres and plot the error bar plots for genres with over 30,000 ratings
edx %>% group_by(genres) %>%
  summarize(n=n(),avg=mean(rating),se=sd(rating)/sqrt(n())) %>% #mean and standard errors
  filter(n>1) %>%
  mutate(genres = reorder(genres, avg)) %>% #order genres by mean ratings
  ggplot(aes(x=genres,y=avg,ymin=avg-2*se,ymax=avg+2*se)) + #lower and upper confidence intervals
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_blank()) + #Removing X-Axis Genre labels
  ggtitle("Average Ratings of Different Genres")


#Now the equation for the model without considering the regularization is as below. 

#$$Y_{u,i} = \mu + b_{i} + b_{u} + b_{g} +\epsilon_{u,i}$$

#On further examination, we could see that there is a time effect coming in the rating.
#The following graph represents this. 

#Group ratings by date and plot the smoothed conditional mean over the days
edx %>% mutate(time=as_datetime(timestamp)) %>% #Converting epoch time to human readable time
  group_by(day=floor_date(time,"day")) %>% #rounding date-time down to nearest day
  summarise(mean_rating=mean(rating)) %>%
  ggplot(aes(day,mean_rating)) + 
  geom_smooth(method='loess', formula = y~x) +
  ggtitle("Variation of Average Rating over time")

#$$Y_{u,i} = \mu + b_{i} + b_{u} + b_{g} + b_{t} + \epsilon_{u,i}$$

#In the plot "Average ratings of Different Genres", some genres have high standard errors.
#We can filter out genres with high standard errors on their mean ratings to further decrease the RMSE. 
#For genres having high standard errors, it is better to be conservative and not assign a $b_g$ value. 
#The reason is as mentioned earlier we need to have a single value for b_g not a set of values. 
#So we are going to penalize the $b_g$ with high standard error. Since we need to  penalize for all the ratings,
#we need cross validation to find a optimum cutoff which reduces the RMSE. 

#For determining the cutoff value of the standard error, we use cross-validation.

# Using cross validation to determine the optimal standard Error cut off value
ses <- seq(0,1,0.1) #Range of Standard Error values
rmses <- sapply(ses, function(s){
     mu <- mean(trainSet$rating)
     b_i <- trainSet %>%
          group_by(movieId) %>%
          summarize(b_i = sum(rating - mu)/(n()))
     b_u <- trainSet %>% 
          left_join(b_i, by="movieId") %>%
          group_by(userId) %>%
          summarize(b_u = sum(rating - b_i - mu)/(n()))
     b_t <- trainSet %>% 
          left_join(b_u, by="userId") %>% 
          left_join(b_i, by = "movieId") %>% 
          mutate(time=as_datetime(timestamp)) %>% 
          group_by(day=floor_date(time,"day")) %>% 
          summarise(b_t = sum(rating - b_i - b_u - mu)/(n()))
    b_g <- trainSet %>% 
        left_join(b_u, by="userId") %>% 
        left_join(b_i, by = "movieId") %>% 
        mutate(day = floor_date(as_datetime(timestamp),"day")) %>%
        left_join(b_t, by="day") %>%
        mutate(b_t=replace_na(b_t,0)) %>%
        group_by(genres) %>%
        summarise(b_g=(sum(rating - b_i - b_u - b_t -mu ))/(n()),se = sd(rating)/sqrt(n())) %>%
        filter(se<=s) # Retaining b_g values that correspond to Standard Error less than or equal to S 

# Predicting movie ratings on test set   
predicted_ratings <- testSet %>% 
        left_join(b_i, by = "movieId") %>%
        left_join(b_u, by = "userId") %>%
        mutate(day = floor_date(as_datetime(timestamp),"day")) %>% 
        left_join(b_t, by="day") %>%
        mutate(b_t=replace_na(b_t,0)) %>%
        left_join(b_g, by="genres") %>%
        mutate(b_g=replace_na(b_g,0)) %>%
        mutate(pred = mu + b_i + b_u + b_t + b_g) %>%
        .$pred
    
    return(RMSE(predicted_ratings, testSet$rating))
})


#The optimal value of the standard error after cross validation is as below. 
# Storing the optimal standard error error value i.e the one which gives lowest RMSE
s_e <- ses[which.min(rmses)]
s_e


#So including regularization to each effects, the model equation changes to the following

#Since lambda is a tuning parameter in this equation. We can use cross validation to choose it as mentioned earlier. 

lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
     mu <- mean(trainSet$rating)
     b_i <- trainSet %>%
          group_by(movieId) %>%
          summarize(b_i = sum(rating - mu)/(n()+l))
     b_u <- trainSet %>% 
          left_join(b_i, by="movieId") %>%
          group_by(userId) %>%
          summarize(b_u = sum(rating - b_i - mu)/(n()+l))
     b_t <- trainSet %>% 
          left_join(b_u, by="userId") %>% 
          left_join(b_i, by = "movieId") %>% 
          mutate(time=as_datetime(timestamp)) %>% 
          group_by(day=floor_date(time,"day")) %>% 
          summarise(b_t = sum(rating - b_i - b_u - mu)/(n()+l))
     b_g <- trainSet %>% 
          left_join(b_u, by="userId") %>% 
          left_join(b_i, by = "movieId") %>% 
          mutate(day=floor_date(as_datetime(timestamp),"day")) %>%
          left_join(b_t, by="day") %>%
          mutate(b_t=replace_na(b_t,0)) %>%
          group_by(genres) %>%
          summarise(b_g=(sum(rating-b_i-b_u-b_t-mu))/(n()+l),se = sd(rating)/sqrt(n())) %>%
          filter(se<=s_e)
    predicted_ratings <- testSet %>% 
        left_join(b_i, by = "movieId") %>%
        left_join(b_u, by = "userId") %>%
        mutate(day = floor_date(as_datetime(timestamp),"day")) %>% 
        left_join(b_t, by="day") %>%
        mutate(b_t=replace_na(b_t,0)) %>% #filling the 'NA' values due to the SE cutoff
        left_join(b_g, by="genres") %>%
        mutate(b_g=replace_na(b_g,0)) %>% #filling the 'NA' values due to the SE cutoff
        mutate(pred = mu + b_i + b_u + b_t + b_g) %>%
        .$pred
    
    return(RMSE(predicted_ratings, testSet$rating))
})



qplot(lambdas, rmses) 


#So from the plot we understood that when $\lambda$ we are getting the lowest RMSE. 


lambda <- lambdas[which.min(rmses)]
lambda



rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Reglarised Movie + User + Time + Genre Effects",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()


# Results 

#So we have reached our final goal of the RMSE below 0.86490 using the model 
#$Reglarised Movie + User + Time + Genre Effects$. This model performed better 
#than other models. So now lets use the validation dataset and precit the RMSE. 


# Calculating mean rating
     mu <- mean(trainSet$rating)

# Computing b_i for each movie as mean of residuals
     b_i <- trainSet %>%
          group_by(movieId) %>%
          summarize(b_i = sum(rating - mu)/(n()+lambda))
     
# Computing b_u for each user as mean of residuals     
     b_u <- edx %>% 
          left_join(b_i, by="movieId") %>%
          group_by(userId) %>%
          summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
     
# Computing b_t for each day as mean of residuals
     b_t <- trainSet %>% 
          left_join(b_u, by="userId") %>% 
          left_join(b_i, by = "movieId") %>% 
          mutate(time=as_datetime(timestamp)) %>% 
          group_by(day=floor_date(time,"day")) %>% 
          summarise(b_t = sum(rating - b_i - b_u - mu)/(n()+lambda))
     
# Computing b_g for each genre as mean of residuals
     b_g <- trainSet %>% 
          left_join(b_u, by="userId") %>% 
          left_join(b_i, by = "movieId") %>% 
          mutate(day=floor_date(as_datetime(timestamp),"day")) %>%
          left_join(b_t, by="day") %>%
          mutate(b_t=replace_na(b_t,0)) %>% #filling the 'NA' values due to the SE cutoff
          group_by(genres) %>%
          summarise(b_g=(sum(rating-b_i-b_u-b_t-mu))/(n()+lambda),se = sd(rating)/sqrt(n())) %>%
          filter(se<=s_e)
     
# Predicitng the ratings on the validation set
    predicted_ratings <- validation %>% 
        left_join(b_i, by = "movieId") %>%
        left_join(b_u, by = "userId") %>%
        mutate(day = floor_date(as_datetime(timestamp),"day")) %>% 
        left_join(b_t, by="day") %>%
        mutate(b_t=replace_na(b_t,0)) %>% #filling the 'NA' values due to the SE cutoff
        left_join(b_g, by="genres") %>%
        mutate(b_g=replace_na(b_g,0)) %>% #filling the 'NA' values due to the SE cutoff
        mutate(pred = mu + b_i + b_u + b_t + b_g) %>%
        .$pred
    

#The  final RMSE is as given below: 

# RMSE for predictions made on validation set 
    RMSE_Valid <- RMSE(predicted_ratings, validation$rating)
    RMSE_Valid


# Conclusion

#In our modeling we have used different models which is having movie effects, time effect and user effects 
#to get a RMSE below 0.86490. We have succeed using the $Reglarised Movie + User + Time + Genre Effects$ in our goal 
#and attained RMSE < 0.86490. There is even enough scope of improvement using more highend analysis techniques
#like  gradient desecent approach, Principal Component Analysis and Singular Value Decomposition. 

#These approaches requrie more computer intensive setups and cannot be done on normal PCs.
#They required huge RAM and processing power. 

