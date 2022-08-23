library(tidyverse)

## First, we'll load the data and get caught up to where we left off in Session 1
raw_listings <- read.csv('data/listings_clean.csv', stringsAsFactors=TRUE)

listings <- raw_listings %>%
  filter(!is.na(bedrooms), !is.na(bathrooms)) %>%
  mutate(price = as.numeric(gsub('[$,]', '', price)), # we are now replacing the price column, not creating a new one
         cleaning_fee = as.numeric(gsub('[$,]', '', cleaning_fee)))

################## Bonus: Imputing Missing Data ######################
## --------------------------------------------------------------------
#' Now we are going to talk about approaches to handling missing data. Often the data you have at hand 
#' will contain missing values, or you create them when you merging datasets.

# Some algorithms can automatically work with missing data (CART, random forest).
# Some will need to exclude observations with missing data (logistic regression).

listings %>% select(name, review_scores_rating, reviews_per_month) %>% head()

# OPTION 1: Only include complete cases (i.e. no NAs in the row) # Warning: you are going to lose data!
listings_cc <- listings %>%
  filter(complete.cases(.))

listings_cc %>% select(name, review_scores_rating, reviews_per_month) %>% head()

# OPTION 2a: You can impute the values using median/mode value for each column
# Median/Mode Imputation is easy using a function in randomForest. 
# Imputes median for numerical columns, and mode for categorical columns
library(randomForest)
listings_imputeMedian <- na.roughfix(listings)
listings_imputeMedian %>% select(name, review_scores_rating, reviews_per_month) %>% head()

listings$reviews_per_month[is.na(listings$reviews_per_month)] = 0

# OPTION 2b: You can also impute the mean, but you can only do this for numerical columns.
# Let's try it on the review_scores_rating column
listings_imputeMean <- listings
listings_imputeMean$review_scores_rating[is.na(listings_imputeMean$review_scores_rating)] <- mean(listings_imputeMean$review_scores_rating, na.rm = TRUE)
listings_imputeMean %>% select(review_scores_rating, reviews_per_month) %>% head()

# Option 3 (Advanced): Multiple Imputation (MICE)
# The MICE package uses a multiple imputation method to provide more accurate imputed values for missing data. 
# There is more information on the method and functions here: 
# https://datascienceplus.com/imputing-missing-data-with-r-mice-package/
# Key parameters: The m parameter controls how many versions of imputed data you will create, 
# and maxit controls the number of times you look into each variable defaults to 50.

install.packages("mice")
library(mice)

## We will run a very small example for the sake of time.
listings_small <- listings %>%
  select(bedrooms, bathrooms, price, cleaning_fee, review_scores_rating, reviews_per_month) %>%
  head(50)

temp_imputed <- mice(listings_small, m=5, maxit = 10, method = 'pmm', seed = 500)

# You can get the full imputed dataset using the complete() function.
listings_imputeMICE <- complete(temp_imputed)
listings_imputeMICE %>% head()
