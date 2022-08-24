#' ---
#' title: "Statistical Modeling and Machine Learning"
#' ---


library(modelr) 
library(tidyverse)
library(caTools) 
library(randomForest) 
library(ROCR) 
library(rpart) 
library(rpart.plot) 
library(RColorBrewer) 
library(leaflet) 
library(caret) 
library(e1071) 

#' ## OVERVIEW
#' 
#'In this session we are going to explore three common tasks in Machine Learning:
#'
#'
#' 1. Regression (supervised learning): Linear regression
#' 2. Classification (supervised learning): Classification Trees, Logistic regression 
#' 3. Clustering (unsupervised learning): k-means clustering
#' 
#' 
#' Supervised learning refers to those tasks in which we observe some input x and use it to predict some output y. 
#' 
#' When the target output y is continous we refer to the problem as **regression**, and when the label is instead categorical, we call it a **classification** problem. 
#' 
#' Unsupervised learning, instead, refers to tasks in which we aim to understand and analyze the input x without any target y. 
#' 
#' **Clustering**, for example, is an unsupervised task that groups data points with similar characteristics.
#' 
#' We will discuss some of the most common models for solving these types of problems as well as how to evaluate their performance.
#' 

#'
#'
#' 
#' ## REGRESSION
#'
#' 
#' Given a dataset D of samples (x_i, y_i), where the real number y_i is the output for the real vector x_i, 
#' 
#' the goal of a regression task is to use the dataset to find a function h such that h(x) yields an accurate approximation of the desired output y for any unseen input x.
#' 
#' The function h is usually found by selecting a family of models (e.g. linear models, tree models, neural networks, etc.), 
#' 
#' and solving an optimization problem to find the best model within this family according to some *loss function*. 
#' 
#' Since it might be difficult to know in advance which family works best, we often try several model families and choose the best one at the end. 
#' 
#' 
#' 
#' In our example, we're going to use the Boston Airbnb dataset, where each entry is a property listing. 
#' 
#' The goal is to use the description of each listing (the x) in order to predict its price (the y).
#' 
#' First, set your working directory to wherever you have the .csv file saved, and then load the file.


listings_raw = read.csv("listings.csv", stringsAsFactors=FALSE)

str(listings_raw)



#' ### Data Pre-processing
#' 
#' There are several problems with the data set:  many values are missing, some listings are outliers that can affect our predictions, and most columns are not numeric type.
#' 
#' To have a well defined input x, we need the variables (columns) in it to be interpreted as numbers. 
#' 
#' For this, we need to convert all continous variables to have numeric type and all categorical variables to have factor type.
#' 
#' We can use the data wrangling tools learned earlier to pre-process our data and solve all these issues. 
#' 
#' We have done that in the "process_listings.R" file, which you can open and see in detail. 
#' 
#' In this session, we will simply call the function process_listings() to obtain the already processed data set:


source("process_listings.R")

all_listings <- process_listings("listings.csv")

#' 
#' Note that there are many variables for each listing in the dataset, and it's our job to choose which ones we will include in the input x_i. 
#' 
#' This choice is part of the modeling process, and we will iterate on it.
#' 
#' We will focus on a subset of the columns, and we will later decide which of them are the most relevant for price estimation:


listings <- all_listings %>%
  
  select(price, accommodates, review_scores_rating, property_type, neighbourhood_cleansed, room_type)


#' Let's have a look!

head(listings)


#'
#' ### Ordinary Least Squares
#' 
#' 
#' The first family of models we're going to try is linear models. This means we hypothesise that the output y can be described using a linear combination of the inputs, 
#' 
#' h(x) = w'x$.
#' 
#' Ordinary Least Squares (OLS) finds the best fitting linear model according to the least squares loss function:
#' 
#' min_w  (sum_{i=1,..., n} (w'x_i - y_i)^2)/n
#' 
#' We won't talk about the details behind solving this problem, but the optimal w* can be written in closed form and therefore its exact value can always be found.
#' 
#' Thus, when using OLS, the only choice we need to make is which variables (columns) to include in the input x.
#' 
#' 
#' Which inputs might be predictive of price? Recall the column names:


names(listings)


#' For a very simple model, let's just choose `accommodates` as the only input. To fit this model we use the base R function `lm`:


ols_model <- lm(price ~ accommodates, data = listings)



#' What's happening here? The first argument makes use of R's formula interface. In words, we want the output `price` to be explained by the input `accommodates`.
#' 
#' The second (named) argument supplies the data we're using to build the model -- this is where R looks for the names (`price` and `accommodates`) contained in the formula. 
#' 
#' We will see some more complicated formulae later in the session.
#' 
#' Let's check out the `ols_model` object. It's a list of relevant information generated by the function call, and we can use `$` to view different elements:


names(ols_model)


#' The function `summary` is overloaded for many different objects and often gives a useful snapshot of the model:


summary(ols_model)



#' Let's look at the 'coefficients' section. 
#' 
#' In the 'estimate' column, we see that the point estimates for the model coefficients say that the intercept is 53.65 and the coefficient that multiplies the `accommodates` variable is 38.18. 
#' 
#' **Question**: What is the predicted price for a listing that accommodates 6 people?
#' 


#'
#'  Notice also the star symbols at the end of the '(Intercept)' and 'accommodates' rows indicate that according to a statistical t-test, both coefficients are significantly different from 0.

#'  To visualise the model, let's plot the fitted line. There are some nifty functions in the `modelr` package that make interacting with models easy within the `tidyverse` setting.
#'  
#'  We'll use `modelr::add_predictions`:


listings %>%	
  add_predictions(ols_model, var = "pred") %>%
  ggplot(aes(x = accommodates)) +
  geom_point(aes(y = price)) +
  geom_line(aes(y = pred), color = 'red') +
  labs(title = "OLS Model Fit")



#' Nice. We can also remove the linear trend and check the residuals (the difference between our predictions and the real outputs) directly, which we'll do here using `modelr::add_residuals`.
#' 
#' This is helpful to check whether the residuals looks like random noise rather than an unidentified trend (which would indicate that a linear model is not a good hypothesis for the relationship):
#' 
#' We can also make a box plot of model residuals for each accommodation size.


listings %>%	
  add_residuals(ols_model, var = "resid") %>%
  ggplot(aes(x = as.factor(accommodates), y = resid)) + 
  geom_boxplot() +
  labs(title = "OLS Model Residual Boxplots")



#' Although the residuals seem relatively centered around zero, there does appear to be some right skew. Also, the 9 and 10-person accommodation residuals look less centered. 
#' 
#' Perhaps a linear model doesn't apply so well here!


#' 
#' ### Evaluation
#' 
#' 
#' Now we're going to look at two measures of accuracy for regression models -- in other words, how well a model explains a dataset.
#' 
#' The first is the mean squared error (MSE), which is simply computed as the *mean of the squared residuals*. Recall that the residuals are stored in the `ols_model` object:


mse <- mean(ols_model$residuals ^ 2)
mse

#' **Question**: Why is mse not a great metric?


#' We can also use the 'R squared' coefficient as a more interpretable measure of accuracy, since it falls between 0 and 1. 
#' 
#' It is the *proportion of variance in the data which is explained by the model*, and is calculated as:


rsq <- 1 - mse / mean((listings$price - mean(listings$price)) ^ 2)
rsq


#' The `rmodel::rsquare` function also calculates it for us:


rsquare(ols_model, listings)


#' The R squared value is what we will use to evaluate the performance of our models in this session. 
#' 
#' But, it's important to note that this is definitely not the only choice we could have made!
#' 
#' NOTE:  The loss function is used to fit a model, and the measure of accuracy is used to evaluate how well it explains the data. 
#' 
#' The measure of accuracy usually comes from the task we aim to solve, and the loss function is often something that we will play around with until we find a good model. 



#' 
#' ### Training, Validation and Testing Splits
#' 
#' 
#' Recall that our ultimate goal in this supervised learning task is to be able to predict price (y) from an *unseen* set of inputs x.
#' 
#' When building the OLS model, we used the entire dataset. Simply taking the R squared value on this dataset as a measure of performance is clearly not fair 
#' 
#' -- small residuals in the training set do not guarantee a small residual for a new observation (x, y).
#' 
#' 
#' To address this problem, we often split the dataset into three different chunks:
#' 
#' 1. **Training data**: the data we build our models on.
#' 2. **Validation data**: the data we use to choose one of the models.
#' 3. **Testing data**: the data we use to obtain a final estimate of performance of our model.
#' 
#' The validation set is like an 'intermediate' estimate of performance. If we didn't use the validation set for this purpose, 
#' 
#' the only way of selecting the best model from a model class would be to look at its performance on the testing set, 
#' 
#' and therefore we would be kind of "cheating" by using the "unseen" data to select the model.
#' 
#' `modelr::resample_partition` provides an easy way of creating these data partitions:


set.seed(0)
part <- resample_partition(listings, c(train = .6, val = .2, test = .2))


#' This produces a list of three `resample` objects:


names(part)




#' 
#' ### Model Iteration
#'
#' 
#' Now that we're equipped to build models and evaluate their performance, let's start iterating to find better models.
#' 
#' We've glossed over the precise choice of variables to use in order to explain price, so let's try a few different combinations. 
#' 
#' We will use the training set to train our models, we will use the validation set to choose one of the models and finally we will use the test set to evaluate our chosen model.
#' 

ols_model1 <- lm(price ~ ., data = part$train) 

#' The symbol `.` can also be used to represent all variables, so the above is equivalent to
#' 
#' lm(price ~ accommodates + review_scores_rating + property_type + neighbourhood_cleansed + room_type, data = part$train).
#' 
#' We next check the R squared with respect to the validation set:
#' 
#' 
rsquare(ols_model1, part$val)


#' We will build two more models using only a subset of the columns:

ols_model2 <- lm(price ~ accommodates + review_scores_rating, data = part$train) 
rsquare(ols_model2, part$val)

ols_model3 <- lm(price ~ accommodates + review_scores_rating + property_type + neighbourhood_cleansed + accommodates * room_type, data = part$train) 
rsquare(ols_model3, part$val)


#' **Question**: Does the third formula still correspond to a linear model?
#' 
#' 
#' 
#' 
#' Checking all possible subsets of columns might be intractable when there are many variables, so we want a more systematic way to choose the columns.
#'
#' One option is to look at the summary() of the model to see which variables are not statistically significant and can therefore be removed from the formula.
#' 
#' 
#' 
#' **Question**: Can I always trust the statistically significance results?


#' For now let's stick to those three models. The third model achieves the highest R squared and therefore we will choose this as our final model. 
#' 
#' We can finally evaluate the performance of this model by computing the R squared with respect to the test set:

rsquare(ols_model3, part$test)



#'
#' ## CLASSIFICATION
#' 
#' 
#' 
#' So far we've looked at models which predict a continuous response variable. There are many related models which predict categorical outcomes, 
#' 
#' such as whether an email is spam or not, or which digit a handwritten number is. We'll take a brief look at three of these: logistic regression and classification trees.
#' 
#' 
#' 
#' ### Logistic Regression
#' 
#'
#' Logistic regression is part of the class of generalized linear models (GLMs), which build directly on top of linear regression. 
#' 
#' These models take the linear fit and map it through a non-linear function. Linear regression has the form y_i= w'x_i, whereas a GLM model has the form y_i=f(w'x_i).
#' 
#' For logistic regression, the function f() is given by f(z) = 1/(1+exp(-z))  (the *logistic function*), which looks like this:	


zs <- seq(-10, 10, 0.25)
ys <- exp(zs) / (1 + exp(zs))
plot(zs, ys)


#' Since the function stays between zero and one, it can be interpreted as a mapping from predictor values to a probability of being in one of two classes. 
#' 
#' In OLS regression, the coefficients $w$ were chosen so that they minimize mean squared error:
#' min_w  (sum_{i=1,..., n} (w'x_i - y_i)^2)/n
#' 
#' And we heard that the optimal w* can be written in closed form. 
#' 
#' If we tried to minimize mean squared error for logistic regression, we would *not* be able to obtain w* easily. 
#' 
#' Intuitively, we also might want a different loss function. We want to penalize misclassification (e.g. classifying something that is not spam as spam). 
#' 
#' For logistic regression, the objective is to minimize **logistic loss**:
#' 
#' 
#' min_w  ( sum_{i = 1,...,n} -y_i log(f(w'x_i)) - (1-y_i) log(1-f(w'x_i)) ) / n
#' 
#' 
#' When y_i=1, this function looks like this:


zs <- seq(0, 1, 0.01)
loss <- -log(zs)
plot(zs, loss)


#' And when y_i=0, this function looks like this:


zs <- seq(0, 1, 0.01)
loss <- -log(1-zs)
plot(zs, loss)


#' So when the true value of $y_i$ is 1, the function **heavily** penalizes the values y_i that were predicted to be close 0, and vice versa.
#' 
#' Let's apply this model to the `listings` data. Let's try to predict which listings have elevators in the building by using `price` as a predictor. 
#' 
#' This information is contained in the `amenities` column. 
#' 
#' Let's see what this column looks like:


head(all_listings$amenities)


#' This column also contains information which could be useful for prediction, if we can come up with a clean way of representing the amenities. 
#' 
#' Our goal here is to turn the amenities column into many columns, one for each amenity, and with logical values indicating whether each listing has each amenity. 
#' 
#' This is just a bit tricky, so I'm calling a function called `expand_amenities` that will do this for us. 
#' 
#' We need to `source()` the file that has this function in it, and then we'll call it on the `all_listings` data frame.	


source("expand_amenities.R")
listingsBig <- expand_amenities(all_listings)


#' An indicator for each amenity is now stored in a column called `amentity_x` where `x` is the amenity of interest. Now back to our task of predicting which listings have elevators.
#' 
#' To make sure we're asking a sensible question, we'll only consider apartments priced at $500 or less. 


listingsGLM <- listingsBig %>%
  filter(property_type == "Apartment", price <= 500)

#' We'll also convert categorical variables to factors
listingsGLM$neighbourhood_cleansed=as.factor(listingsGLM$neighbourhood_cleansed)
listingsGLM = listingsGLM %>% mutate_at(vars(contains("amenity")),~as.factor(.))

#'We will use the command `sample.split` from the `caTools` library to split the data. 
#'
#'One nice thing about using `sample.split` for classification is that it preserves the ratio of class labels in the training and testing sets.
#'
#'And don't forget to set the seed if you want to be able to reproduce your results later!

set.seed(123)
spl <- sample.split(listingsGLM$amenity_Elevator_in_Building, SplitRatio = 0.7)
listingsGLMTrain <- subset(listingsGLM, spl == TRUE)
listingsGLMTest <- subset(listingsGLM, spl == FALSE)

#' Notice that we could also create a validation set as we did in regression to choose the best model. 
#' 
#' However, we are going to skip the validation part for this section since it is pretty similar to what we did before. Instead, we will wait to cover trees to explore a new validation strategy.
#'
#' To build a logistic regression model, instead of the `lm()` function, we'll now use `glm()`, but the syntax is almost exactly the same:	


log.reg.model <- glm(amenity_Elevator_in_Building ~ price,
             family = "binomial", data = listingsGLMTrain)
summary(log.reg.model)

#' To evaluate the prediction accuracy of our model, we count up the number of times each of the following occurs:
#' 
#'  * y = 1, prediction = 1 (True Positive)
#'  * y = 0, prediction = 1 (False Positive)
#'  * y = 1, prediction = 0 (False Negative)
#'  * y = 0, prediction = 0 (True Negative)
#'  
#' A table that holds these values is called a "confusion matrix". Then, accuracy = ( # True Positives +  # True Negatives) / (Total # of observations) 

#TESTING ACCURACY
pred_test <- predict(log.reg.model, newdata = listingsGLMTest, type = "response")
confusionMatrixTest <- table(listingsGLMTest$amenity_Elevator_in_Building,
                             ifelse(pred_test > 0.5, "pred = 1", "pred = 0"))
confusionMatrixTest
accTest <- sum(diag(confusionMatrixTest)) / nrow(listingsGLMTest)
print(accTest)

#' We can also add predictions to the data frame and plot these along with the actuals, although the result doesn't look nearly as clean as with linear regression:	


listingsGLMTest %>%
  mutate(pred = predict(log.reg.model, newdata = listingsGLMTest, type = "response")) %>%
  ggplot(aes(x = price)) + 
  geom_line(aes(y = pred)) + 
  geom_point(aes(y = as.numeric(amenity_Elevator_in_Building )-1))



#' 
#' We can now explore out-of-sample performance. Ultimately, we want to predict whether or not a listing has an elevator. 
#' 
#' However, logistic regression gives us something a bit different:  a probability that each listing has an elevator. 
#' 
#' This gives us flexibility in the way we predict. The most natural thing would be to predict that any listing with predicted probability above 0.5 *has*  an elevator, 
#' 
#' and any listing with predicted probability below 0.5 *does not have* an elevator. But what if I use a wheelchair and I want to be really confident that there's going to be an elevator?  
#' 
#' I may want to use a cutoff value of 0.9 rather than 0.5. In fact, we could choose any cutoff value and have a corresponding prediction model.	
#' 
#' 
#' 
#' There's a really nice metric that measures the quality of all cutoffs simultaneously: *AUC*, for "Area Under the receiver operating characteristic Curve." 
#' 
#' That's a mouthful, but the idea is simpler:  For every cutoff, we'll plot the *false positive rate* against the *true positive rate* and then take the area under this curve.
#' 
#' (A *positive* in our case is a listing that has an elevator. So a  *true positive* is a listing that we predict has an elevator and really does have an elevator, 
#'   
#' while a *false positive* is a listing that we predict has an elevator and does *not* actually have an elevator. 
#' 
#' The *true positive rate* is the number of true positives divided by the total number of positives, 
#' 
#' and the *false positive rate* is the numbe of false positives divided by the total number of negatives.)	
#' 
#' 
#' 
#' **Question**: As a sanity check: What is the true positive rate and false positive rate of a random classifier that chooses `has an elevator` with probability of alpha? 
#' 
#' (i.e. a classifier that randomly predicts *positive* alpha % of the time.) What is the AUC for this classifier?
#' 
#' 
#' 
#' 
#' As the cutoff decreases from 1 to 0, the rate of total positives will increase. If the rate of true positives increases faster than the rate of false positives, 
#' 
#' this is one indication that the model is good. This is what AUC measures.	
#' 
#' The `ROCR` package is one implementation that allows us to plot ROC curves and calculate AUC. Here's an example:	


#AUC
pred_test <- predict(log.reg.model, newdata = listingsGLMTest, type = "response")	
pred_obj <- prediction(pred_test, listingsGLMTest$amenity_Elevator_in_Building)
# Creating a prediction object for ROCR	
perf <- performance(pred_obj, 'tpr', 'fpr')	
plot(perf, colorize = T)  # ROC curve
performance(pred_obj, 'auc')@y.values  


#' As you can see, the `performance()` function in the `ROCR` package is versatile and allows you to calculate and plot a bunch of different performance metrics. 
#' 
#' In our case, this model gives an AUC of 0.68. 
#' 
#' The worst possible is 0.5 - random guessing. We're definitely better than random here, and could likely improve by adding more predictors.	
#' 
#' **Exercise**: Add more variables to Logistic Regression. Try to beat the out-of-sample performance for logistic regression of elevators on price by adding new variables.
#' 
#' Compute the out-of-sample AUC of the final model, and plot the ROC curve.




#' Remember, evaluation and iteration are also important when choosing classification models! The same technique we used with regression problems of splitting the data into a training, 
#' 
#' validation, and testing sets should be applied here to decide what columns to include as our input! 



#'
#' ### Classification Trees
#' 
#' 
#' We will briefly explore classification trees (often referred to as CART, for Classification And Regression Trees).
#' 
#' A (binary) classification tree makes predictions by grouping similar observations and then assigning a probability to each group using the proportion of observations 
#' 
#' within that group that belong to the positive class. Groups can be thought of as nodes on a tree, and tree branches correspond to logical criteria on the predictor variables. 
#' 
#' There's a lot of neat math that goes into building the trees, but we won't get into that today. For now let's get familiarized by looking at a simple example. 
#' 
#' We will use the `rpart` library.	
#' 
#' The model construction step follows the same established pattern. We use the modelling function `rpart()`, which takes a formula and a data frame (and optional parameters) as arguments.	


CART.model <- rpart(amenity_Elevator_in_Building ~ price + 
                   neighbourhood_cleansed,
                 data = listingsGLMTrain,
                 cp = 0.05)	


#'We can plot the resulting tree using the `rpart.plot` package:	


prp(CART.model)

#' 
#' Let's calculate the accuracy for our model:

#TRAINING SET
pred_train <- predict(CART.model)[,2]
confusionMatrixTrain <- table(listingsGLMTrain$amenity_Elevator_in_Building, 
                              ifelse(pred_train > 0.5, "pred = 1", "pred = 0"))
confusionMatrixTrain
accTrain <- sum(diag(confusionMatrixTrain)) / nrow(listingsGLMTrain)
print(accTrain)

#TESTING SET
pred_test <- predict(CART.model, newdata = listingsGLMTest)[,2]
confusionMatrixTest <- table(listingsGLMTest$amenity_Elevator_in_Building,
                             ifelse(pred_test > 0.5, "pred = 1", "pred = 0"))
confusionMatrixTest
accTest <- sum(diag(confusionMatrixTest)) / nrow(listingsGLMTest)
print(accTest)

#AUC
cartROCpred = prediction(as.numeric(pred_test), listingsGLMTest$amenity_Elevator_in_Building)
cartAUC = as.numeric(performance(cartROCpred, "auc")@y.values)
cartAUC 

#' What is the baseline out-of-sample accuracy? This is just the frequency of the most common class in the training set.


table(listingsGLMTest$amenity_Elevator_in_Building)
prop.table(table(listingsGLMTest$amenity_Elevator_in_Building))


#' About 70% of the the listings do not have elevators, so if were predicted `no elevator` for every listing, we would get 70% accuracy. This is our naive baseline.
#' 
#' Our tree has an accuracy of about 80%, so it is a significant improvement on the baseline.



#' 
#' #### Tuning the CART model
#' 
#' Notice that in linear regression and logistic regression, the only decision we need to make is which columns to include in our input variables $x$. 
#' 
#' CART however, has many parameters that specify how the decision tree is constructed, and one of the most important is `cp`, the complexity parameter.
#'
#' `cp` is a non-negative parameter which typically takes values like 0.1, 0.1, 0.01, 0.001, etc. (default = 0.01). It is the minimum complexity threshold  
#' 
#' that the CART algorithm uses to decide whether or not to make a split.  So:
#' 
#' 
#' *If `cp` is low => low splitting threshold => big tree
#' 
#' *If `cp` is high => high splitting threshold => small tree
#' 
#' 
#' The parameter `cp` controls the "complexity" of the model, so it important to tune it to avoid over-fitting  or under-fitting. You can think of it like this:
#' 
#' 
#' *If the tree is too big => too many splits => we have an over-fitting problem.
#' 
#' *If the tree is too small => too few splits => we have an under-fitting problem.
#' 
#' 
#' We want to find a tree in the middle, that is "just right". For this, we could use validation as in logistic regression: we train several models with different `cp` 
#' 
#' parameters and then choose the one with the best performance in the validation set. However, we instead are gogint o explore
#' 
#' another very popular way to validate our models called *k-fold cross-validation*. This type of validation splits the training set into $k$ random and equal size subsets 
#' 
#' and evaluates each model by training it using $(k-1)$ of the subsets and validating with the subset that was left out. 
#' 
#' This process is repeated $k$ times, each time leaving out a different fold. The $k$ validation losses are averaged and at the end we choose the model with the lowest average validation loss.
#' 
#' 
#' The rpart package makes it easy to perform tune cp via cross-validation with the "caret" and "e1071" libraries.
#' 
#' First, we create an object that will tell the package how many folds we want. In this case, we'll try 10.

folds = trainControl(method = "cv", number = 10)

# Second, we'll tell the package what values of cp we want to test. We'll try everything from 0.01 to 0.5 by increments of 0.01.

cpValues = expand.grid(.cp = seq(0.001,0.5, 0.01)) 

# Now we're ready to run 10-fold cross-validation of CART.
# The following command has syntax similar to rpart() but instead takes method="rpart" as an argument.
# NOTE: This can take a while for larger datasets.


set.seed(20)
x = train(as.factor(amenity_Elevator_in_Building) ~ price + neighbourhood_cleansed, 
          data = listingsGLMTrain, method="rpart", trControl = folds, tuneGrid = cpValues)
x


# The package build a CART tree with each cp value for each of the 10 folds. In the end, it reports the best cp value.
# Let's build a new CART model that uses this cp value.


CV.CART.model = rpart(amenity_Elevator_in_Building ~ price + neighbourhood_cleansed, 
                  data = listingsGLMTrain, cp = 0.011)

prp(CV.CART.model)

#' We have a more informative tree now! We can also try looking at a more detailed graph of the tree.


rpart.plot(CV.CART.model)

# TESTING ACCURACY
pred_test <- predict(CV.CART.model, newdata = listingsGLMTest)[,2]
confusionMatrixTest <- table(listingsGLMTest$amenity_Elevator_in_Building,
                             ifelse(pred_test > 0.5, "pred = 1", "pred = 0"))
confusionMatrixTest
accTest <- sum(diag(confusionMatrixTest)) / nrow(listingsGLMTest)
print(accTest)

# AUC
cartROCpred = prediction(as.numeric(pred_test), listingsGLMTest$amenity_Elevator_in_Building)
cartAUC = as.numeric(performance(cartROCpred, "auc")@y.values)
cartAUC 

#' 
#' ### Random Forests
#'
#' 
#' We will briefly take a look at random forests, using the `randomForest` package. A random forest is a collection of slightly randomized decision trees 
#' 
#' (hence the name "forest"), and can be used for classification or prediction. 
#' 
#' They often have excellent predictive performance, but can be expensive to train and lack interpretability. Random forests have many hyperparameters 
#' 
#' that can be tuned to achieve the best possible predictive performance. Perhaps the most important hyperparameter is the number of trees to include in the forest. 
#' 
#' More trees results in a longer training time but can improve prediction and decrease overfitting. Other parameters can be seen by inspecting the randomForest command:


?randomForest


#' Let's start by training a random forest model for a classification task. We will perform the same task of predicting whether or not a listing has an elevator, 
#' 
#' using price and neighborhood as predictors. 
#' 
#' We will compare the performance of random forest to what we got using our simple CART model and logistic regression. 
#' 
#' We'll try two models, one using 5 trees and one using 100 trees.

set.seed(123)
rf.model1 <- randomForest(amenity_Elevator_in_Building ~ price+neighbourhood_cleansed,
                   data = listingsGLMTrain, ntree = 5)


#TESTING ACCURACY
pred_test <- predict(rf.model1,newdata = listingsGLMTest)
confusionMatrixTest <- table(listingsGLMTest$amenity_Elevator_in_Building,
                             ifelse(pred_test==TRUE, "pred = 1", "pred = 0"))
accTest <- sum(diag(confusionMatrixTest)) / nrow(listingsGLMTest)
print(accTest)

#AUC
rfROCpred = prediction(as.numeric(pred_test),listingsGLMTest$amenity_Elevator_in_Building)
rfAUC = as.numeric(performance(rfROCpred, "auc")@y.values)
rfAUC

#' Now we will train a random forest with 100 trees:


rf.model2 <- randomForest(amenity_Elevator_in_Building ~ price+neighbourhood_cleansed,
                   data = listingsGLMTrain, ntree = 100)


#TESTING ACCURACY
pred_test <- predict(rf.model2,newdata = listingsGLMTest)
confusionMatrixTest <- table(listingsGLMTest$amenity_Elevator_in_Building,
                             ifelse(pred_test==TRUE, "pred = 1", "pred = 0"))
accTest <- sum(diag(confusionMatrixTest)) / nrow(listingsGLMTest)
print(accTest)

#AUC
rfROCpred = prediction(as.numeric(pred_test),listingsGLMTest$amenity_Elevator_in_Building)
rfAUC = as.numeric(performance(rfROCpred, "auc")@y.values)
rfAUC

#' 
#' So random forest with only 5 trees does not do very well, and 100 trees did not do much better! 
#' 
#' Let's try then adding a lot more variables. We'll add all of the amenities as variables, and train a random forest with 50 trees. 


amenities_string <- listingsGLMTrain %>%
  select(starts_with("amenity"),-amenity_Elevator_in_Building) %>%
  names() %>%
  paste(collapse = " + ")
rf_formula <- as.formula(paste("amenity_Elevator_in_Building ~ price+neighbourhood_cleansed", 
                               amenities_string, sep = " +  "))

rf.model3 <- randomForest(rf_formula, data = listingsGLMTrain, ntree = 50)

#TESTING ACCURACY
pred_test <- predict(rf.model3, newdata = listingsGLMTest)
confusionMatrixTest <- table(listingsGLMTest$amenity_Elevator_in_Building,
                             ifelse(pred_test==TRUE, "pred = 1", "pred = 0"))
accTest <- sum(diag(confusionMatrixTest)) / nrow(listingsGLMTest)
print(accTest)

#AUC
rfROCpred = prediction(as.numeric(pred_test),listingsGLMTest$amenity_Elevator_in_Building)
rfAUC = as.numeric(performance(rfROCpred, "auc")@y.values)
rfAUC


#' The prediction accuracy is now much better, and it doesn't look we are overfitting at all. It would likely be even better if the other hyperparameters of the 
#' 
#' random forest model were properly tuned! 
#' 
#' Although random forest models are not very interpretable and hard to visualize, there is a popular method called *variable importance* that is commonly used 
#' 
#' with random forest models. We will use the function `varImpPlot` form the `randomForest` package.
#' 
#'  This plot shows us which variables the random forest model has determined are the most important for predicting whether or not a listing contains an elevator. 
#'  
#'  It uses a metric called `MeanDecreaseGini` by default, which is the mean decrease in node impurity (you can read more about this metric online), 
#'  
#'  to rank the importance of all of the  predictors used in the model. There are other metrics besides the Gini Importance that can also be used as a metric.


varImpPlot(rf.model3)


#' The variable importance plot shows us that the listing's neighborhood, whether or not it has a gym, its price, and whether it has a doorman are the four most 
#' 
#' important variables for predicting whether a listing will have an elevator. This not only gives us intuition about our random forest model, 
#' 
#' but can also be used to select variables to train other models.
#' 
#' We will now use random forest for the regression task of predicting `price` from `accomodates`.


set.seed(123)
spl <- sample.split(listings$price, SplitRatio = 0.7)
listingsTrain <- subset(listings, spl == TRUE)
listingsTest <- subset(listings, spl == FALSE)


#' For linear regression, the code was:	


lm <- lm(price ~ accommodates, data = listingsTrain)


#' Using Random Forest, we can write


rf <- randomForest(price ~ accommodates,
                   data = listingsTrain, ntree = 100)


#' We can compare the performance of the random forest model to the linear regression model by plotting the predictions of each model:


listingsTrain %>%
  gather_predictions(lm, rf) %>%
  ggplot(aes(x = accommodates)) +	
  geom_point(aes(y = price)) +	
  geom_line(aes(y = pred, color = model))


#' The predictions from these two models look very similar in the case that `accomodates` is less than 7.5. For larger values of `accomodates`,
#' 
#'   the random forest model is able to capture the nonlinear trend.



#' 
#' # CLUSTERING
#'
#' 
#' Thus far, our machine learning task has been to predict labels, which were either continuous-valued (for regression) or discrete-valued (for classification). 
#' 
#'   To do this, we input to the ML algorithms some known (feature, label) examples (the training set), and the ML algorithm outputs a function which enables us to make 
#'   
#'   predictions for some unknown examples (the testing set).
#' 
#' 
#' Next, we consider **Unsupervised Learning**, where we are not given labelled examples, and we simply run ML algorithms on (feature) data, with the purpose of finding 
#' 
#' interesting structure and patterns in the data. 
#' 
#' Let's run one of the widely-used unsupervised learning algorithms, **k-means clustering**, on the `listings` data frame to explore the Airbnb data set.
#' 

# 
# ## k-Means Clustering
# 
# 1. The number of clusters is specified up front.
# 2. Each point is randomly assigned to one of the clusters.
# 3. The centroid of each cluster is computed. 
# 4. Each point is then re-assigned to the cluster whose centroid is closest to the point.
# 5. The cluster centroids are re-computed, and we go back to step 4.
# We terminate when the cluster assignments no longer change or we reach the maximum number of iterations.
#' 
#' First, let's look at help page for the function `kmeans()`:


?kmeans


#' Let's create a new data.frame `listings_numeric` which has the subset of columns that we wish to cluster on.  For the `kmeans()` function, all of these columns must be numeric.


listings_numeric <- all_listings %>%
  select(id, latitude, longitude, accommodates, bathrooms, 
         bedrooms, review_scores_rating, price) %>%
  mutate(price = as.numeric(gsub("\\$|,", "", price))) %>%
  na.omit()
str(listings_numeric)


#' Next, run the **k-means** algorithm on the numeric data.frame, with `k = 5` cluster centers:


set.seed(1234)
kmeans_clust <- kmeans(listings_numeric[,-1:-3],
                       5, iter.max = 1000, nstart = 100)



# Get the cluster assignments
kmeansGroups = kmeans_clust$cluster

# How big are they?
table(kmeansGroups)

#' What are the centers of these 5 groups?  
kmeans_clust$centers

# Break the data into the clusters. 
KmeansCluster1 = subset(listings_numeric, kmeansGroups == 1)
KmeansCluster2 = subset(listings_numeric, kmeansGroups == 2)
KmeansCluster3 = subset(listings_numeric, kmeansGroups == 3)
KmeansCluster4 = subset(listings_numeric, kmeansGroups == 4)
KmeansCluster5 = subset(listings_numeric, kmeansGroups == 5)



#' Finally let's take a look at where the clusters are located; to do this we  will use a package called `leaflet`; additionally to help us get a good color scheme we will use `RColorBrewer`.
#' 
#' To look at color scheme options we can simply type:


display.brewer.all(type="qual") # Type can be set to 'div', 'seq', 'qual', or 'all'

# We can visualize the clusters individually:
leaflet(KmeansCluster1) %>% 
  addTiles() %>% 
  addCircleMarkers(~longitude, ~latitude)

leaflet(KmeansCluster2) %>% 
  addTiles() %>% 
  addCircleMarkers(~longitude, ~latitude)

leaflet(KmeansCluster3) %>% 
  addTiles() %>% 
  addCircleMarkers(~longitude, ~latitude)

leaflet(KmeansCluster4) %>% 
  addTiles() %>% 
  addCircleMarkers(~longitude, ~latitude)

leaflet(KmeansCluster5) %>% 
  addTiles() %>% 
  addCircleMarkers(~longitude, ~latitude)


#' Or we can also visualize the distribution of all clusters from the houses. First we need to add our cluster labels to the data and then we will define a color palette that leaflet can use to help us.


listings_numeric = listings_numeric %>% 
  mutate(clust_label = as.factor(kmeans_clust$cluster))


#' Now we need to define a color palette that to distinguish the clusters; since we have five cluters we will need five distinct colors


pal = colorFactor(palette = "Set1", domain = listings_numeric$clust_label)


#' Now let's plot the houses by cluster


leaflet(listings_numeric) %>% 
  addTiles() %>% 
  addCircleMarkers(~longitude, ~latitude, color = ~pal(clust_label))


#' Can you see where the clusters are?  what is the proper number of clusters? 
#' 
#' In this module, we have covered examples of machine learning methods for linear regression (ordinary and penalized) and classification (supervised and unsupervised).  
#' 
#' This is just the tip of the iceberg.  There are tons more machine learning methods which can be easily implemented in R.