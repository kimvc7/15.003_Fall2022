# Session 1: Introduction to R + Data Wrangling/Visualization
# Thanks to Andrew Zheng for much of this example's content!

# Key Tools: tidyverse packages (dplyr, ggplot2)

## --------------------------------------------------------------------
################################ Agenda ###############################
## --------------------------------------------------------------------
# Part 1: Part 1: Data cleaning and summarization in dplyr
# Part 2: dplyr Analysis Examples
# Part 3: Part 3: Plotting in ggplot2

# ---------------------------------------------------------------------
########### Part 1: Data cleaning and summarization in dplyr #########
# ---------------------------------------------------------------------

# Now that we have a handle on basic R syntax, we're ready to start working with data!

# As you navigate through the world of Analytics, You’ll also need to install some R packages.
# An R package is a collection of functions, data, and documentation that extends the
# capabilities of base R. Using packages is key to the successful use of R.
# R comes with a lot of packages preinstalled in your computer. To see a full list of all the 
# packages installed, click on the "Packages" tab at the bottom panel at the right of your screen.

# How do I install a new package in your machine:
# Option 1: with a command
install.packages("tidyverse")
# Option 2: via an interface: Click Packages at the right bottom panel > Install > type the name in the Packages line.

# Once a package is installed, you still need to load it in each session (but you do not need to re-install it):
library(tidyverse)

#' We just loaded "tidyverse", a master package that contains the most popular data packages,
#' dplyr and ggplot2 (with other useful packages too!)
#' We'll start with dplyr, is a super useful package created by Hadley Wickham that provides
#' several functions for efficiently "slicing-and-dicing" your data. A lot of these functions 
#' are similar to the logic that you use when creating pivot tables. We'll cover the basic 
#' functions: `select`, `filter`, `count`, `summarize`, `mutate`, `group_by`, and `arrange`. 
#' If you are interested in learning more, check out https://cran.r-project.org/web/packages/dplyr/dplyr.pdf.


# Next, load in the rental data (check your working directory!)
# setwd("~/git/mban22_software_tools/session1_R")
raw_listings <- read.csv('data/listings_clean.csv')

# Our traditional data viewing methods don't work well here--there are too many features and observations
str(raw_listings)
head(raw_listings)

# We will explore various ways to look at the data more easily using the dplyr functions.

##### SELECT function and introduction to "chaining" #####
# The select() function allows you to select a specific column (or set of columns)
# Here we select one column and look at the first few rows using head():

head(raw_listings$price)
head(select(raw_listings, price,bedrooms))

# This is fine, but it's a little awkward having to nest our code like that.  
# Luckily, there is a nifty operator included with tidyr called the **chaining operator** 
# which looks like `%>%` and serves like a pipeline from one function to another. 
# Specifically: `x %>% f` is the same as `f(x)`. In other words, the chaining operator feeds in the object 
# on its left as the first argument into the function on its right.
# You will see this syntax a lot with dplyr, so we'll try it out now. Now we can instead do this:
raw_listings %>% select(price, bedrooms) %>% head()

##### MUTATE function #######
#' The mutate() function allows us to create new columns by doing operations on existing ones.
#' In the listings data, we notice that the  prices are actually strings. 
#' If we want to work with these as numbers, we'll need to convert them. 
#' mutate() can help us here! We need to remove the "$" and "," from the price and then
#' convert it to a numeric variable. The gsub() function lets us strip out unwanted characters,
#' and then as.numeric() call lets us make the result into a number.

## Let's start by figuring out how to convert the prices to numbers. What do the prices look like?
example = raw_listings$price[1:5] # choose a small set of values to work with to see how this works
example 
gsub('[$,]', '', example) # gsub lets us  remove the $ and ,: we replace these both with '' (e.g. nothing)
as.numeric(gsub('[$,]', '', example)) # now that the rows "look like" numbers, we can convert them to numbers

## Now we can put this all together to create a new column in the listings column!
raw_listings %>% 
  mutate(price_numeric = as.numeric(gsub('[$,]', '', price))) %>% 
  select(name, price, price_numeric) %>%
  head()

#### COUNT function ####
#' The count() function gives us the count/length of a column.
#' Here we find how many listings there are with non-NA bedrooms files.
raw_listings %>% count(is.na(bedrooms))
count(raw_listings,  is.na(bedrooms))

# we can also look at the count of all bedrooms!
raw_listings %>% count(bedrooms)

#### FILTER function ####
#' The filter() function lets us filter to a subset of rows. We filter on conditions, just like with 
#' the subset() function that we have used in the past.
#' Tip: You can filter using the `%in%` keyword to restrict to row values contained in a set.
## ------------------------------------------------------------------------
raw_listings %>% select(name, neighbourhood, price, bedrooms) %>%
  filter(bedrooms==4) %>% head()

## We could have done the same thing in base R:
head(subset(raw_listings[c("name","neighbourhood","price","bedrooms")], bedrooms == 4))

## Let's try using the %in% operator
## first, what are the unique neighborhoods?
unique(raw_listings$neighbourhood)

## we can restrict our data to look at a subset of this
raw_listings %>% select(name, neighbourhood, price, bedrooms) %>%
  filter(neighbourhood %in% c('Downtown', 'Back Bay', 'Chinatown'))


# EXERCISE: The `%in%` operator.
# Filter the listings to those that accommodate either 2, 4, or 7 AND have at least 2 bedrooms?	

## FILL IN HERE


#' Note that the tidyverse packages generally do not change the dataframe objects they act on. 
#' For example, the code above doesn't change `listings`, but instead returns a new dataframe 
#' that has the same data as `listings`, plus an extra column. 
#' We could store it in a new variable using "=" or "<-".
#' We want to make sure our data has a correct price column and no missing bedroom or bathroom columns. 
#' We'll assign it to a dataframe named `listings`. 

listings <- raw_listings %>%
  filter(!is.na(bedrooms), !is.na(bathrooms)) %>%
  mutate(price = as.numeric(gsub('[$,]', '', price)), # we are now replacing the price column, not creating a new one
         cleaning_fee = as.numeric(gsub('[$,]', '', cleaning_fee)))

head(listings)

##### ARRANGE function #####
## The arrange() function lets us sort the data by a chosen column (or sets of columns)
## We can also sort this information by price using the arrange function. 
## If you want to sort in descending order, wrap the column name in desc()
listings %>%
  select(name, bedrooms, price) %>%
  arrange(desc(price),desc(bedrooms)) %>%
  head()

##### SUMMARISE function (for aggregation) #####
#' Now, how about a summary statistic, like the average price for a listing? 
#' Let's take our clean dataset and `summarize` it to find out.
listings %>% summarise(avg.price = mean(price))

#### GROUP_BY function ####
#' We can also run summary statistics by group, like a pivot table,
#' We first `group_by` a given variable, and then we summarize within each of the groups.
#' This is similar to creating a pivot table in Excel. 
listings %>%
  group_by(neighbourhood) %>%
  summarize(avg.price = mean(price),
            median.price = median(price))

#' EXERCISE: Explore how listings differ by property type (e.g. condo, house). 
#' Sort the property types in descending order by the number of people they accommodate.
#' Which has the highest average accommodation capacity?
#' Notice that once we create a column in the summarize() function, it's available later in the series of operations!

## FILL IN HERE


##### JOIN functions #####
#' Our last topic will be how to **join** two data frames together. 
#' dplyr has several functions for joining: full_join, left_join, inner_join, etc.
#' They all follow the same syntax: full_join(table1, table2, by='name')
#' Here, you would be joining table1 and table2 on the name column. 
#' An inner_join only includes entries that have a matching row in both table1 and table2, 
#' left_join includes all table1 entries and joins table2 entries where there is a match, and
#' full_join will include all entries from both tables, matching where possible.

## Suppose we want to add in the median neighborhood price for each listing. We could do this as follows:
neighborhood_stats <- listings %>%
  group_by(neighbourhood) %>%
  summarise(median_price = median(price))

inner_join(listings, neighborhood_stats, by = 'neighbourhood') %>%
  select(name, neighbourhood, price, median_price)

## --------------------------------------------------------------------
################# Part 2: dplyr Analysis Examples ####################
## --------------------------------------------------------------------
#' Example 1: examining trends by neighborhood.
#' Suppose we're a little worried these averages are skewed by a few outlier listings.
#' Let's compare the average price, median price, and count of listings in each neighborhood. 
#' The `n()` function here just gives a count of how many rows we have in each group.
#' Notice that we can summarize multiple features at once using summarize().
listings %>%
  group_by(neighbourhood) %>%
  summarize(avg.price = mean(price),
            med.price = median(price),
            num = n())

#' We do notice some red flags to our "mean" approach.
#' 
#' PROBLEM 1: if there are a very small number of listings in a neighborhood compared to the rest of the dataset, 
#' we may worry we don't have a representative sample, or that this data point should be discredited somehow 
#' (on the other hand, maybe it's just a small neighborhood, like Bay Village, and it's actually outperforming 
#' expectation).
#' 
#' PROBLEM 2: if the *median* is very different than the *mean* for a particular neighborhood, 
#' it indicates that we have *outliers* skewing the average.  Because of those outliers, as a rule of thumb, 
#' means tend to be a misleading statistic to use with things like rent prices or incomes.

#' EXERCISE: We can address this problem by filtering out 
#' neighborhoods below a threshold count using our new num 
#' variable. Extend the query above to filter to only 
#' neighborhoods with > 200 listings.

## FILL IN HERE


# Example 2: ratings summarized by the number of bedrooms
# What's new here? we are filtering out NA values.
# You will likely come across a lot of datasets with NAs and they can cause headaches.
# In this example, we are grouping listings together if they have the same review score, 
# and taking the median within the group. We also remove any NA values to get valid medians.
by.bedroom.rating <- listings %>%
  filter(!is.na(review_scores_rating)) %>%
  group_by(bedrooms, review_scores_rating) %>%
  summarize(med.price = median(price), listings = n())

## --------------------------------------------------------------------
################## Part 3: Plotting in ggplot2 #######################
## --------------------------------------------------------------------
#' 
#' `ggplot2` provides a unifying approach to graphics, similar to what we've begun to see with tidyr. 
#' (https://www.cs.uic.edu/~wilkinson/TheGrammarOfGraphics/GOG.html)
#' 
#' Every ggplot consists of three main elements:
#' - **Data**: The dataframe we want to plot.
#' - **Aes**thetics: The dimensions we want to plot, e.g. x, y, color, size, shape.
#' - **Geom**etry:  The specific visualization shape. Line plot, scatter plot, bar plot, etc.

## If you have loaded tidyverse(), you don't need to load ggplot2 separately. If not, make sure to load it here.
# install.packages('ggplot2')
# library(ggplot2)

#### Creating a basic plot. ####
# Suppose we want to plot our bedroom rating data from earlier. How should we do that?
#' We must specify all three key elements here: our Data (`by.bedroom.rating`), 
#' our Aesthetic mapping (`x` and `y` to columns of the data), and our desired Geometry (`geom_point`).  
#' We glue everything together with `+` signs.   
by.bedroom.rating %>%
  ggplot(aes(x=review_scores_rating, y=med.price)) +
  geom_point()

## Reminder: before, we would have done this using:
plot(by.bedroom.rating$review_scores_rating, by.bedroom.rating$med.price)

#' We can use the aesthetics to add a lot more detail to our charts. Suppose we want to see these points 
#' broken out by the number of bedrooms. Let's color these points by the number of bedrooms by specifying 
#' which variable to use to determine the color of each point.
by.bedroom.rating %>%
  ggplot(aes(x=review_scores_rating, y=med.price, color=factor(bedrooms))) +
  geom_point()
# Note that `factor` essentially tells ggplot to treat `bedrooms` as categorical rather than numeric.


# We can also use geoms to layer additional plot types on top of each other.
# In the following example, we throw in a linear best-fit line for each bedroom class. 
# Note that the same x, y, and color aesthetics propagate through all the geoms.
by.bedroom.rating %>%
  ggplot(aes(x=review_scores_rating, y=med.price, color=factor(bedrooms))) +
  geom_point() +
  geom_smooth(method = lm)

#### Bar Plots: another type of geometry ####
# We use the (`geom_bar`) geometry for plotting bar charts.

# Let's save the summary of neighborhoods by prices so that we can plot it.
by.neighbor <- listings %>%
  group_by(neighbourhood) %>%
  summarize(med.price = median(price))

# We use `stat='identity'` to tell `geom_bar` that we want the height of the bar to be equal to the `y` value. 
by.neighbor %>%
  ggplot(aes(x=neighbourhood, y=med.price)) +
  geom_bar(stat='identity') + 
  theme(axis.text.x=element_text(angle=60, hjust=1)) # This rotates the labels on the x-axis so we can read them

# We can make this better! Let's add some color and titles, and reorder the entries to be in descending price order.
by.neighbor %>%
  ggplot(aes(x=reorder(neighbourhood, -med.price), y=med.price)) +
  geom_bar(fill='dark blue', stat='identity') +
  theme(axis.text.x=element_text(angle=60, hjust=1)) +
  labs(x='', y='Median price', title='Median daily price by neighborhood')

#### Heatmaps #### 
# A useful geometry for displaying heatmaps in `ggplot` is `geom_tile`. 
# This is typically used when we have data grouped by two different variables, and so we need visualize in 2d.  
# For example, try using `geom_tile` to visualize median price grouped by # bedrooms and bathrooms.
# You can also change the color scale so that it runs between two colors by adjusting a `scale_fill_gradient` theme:

listings %>%
  group_by(bedrooms, bathrooms) %>%
  summarize(med = median(price)) %>%
  ggplot(aes(x=bedrooms, y=bathrooms, fill=med)) +
  geom_tile() + 
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(x='Bedrooms', y='Bathrooms', fill='Median price', 
       title = "Comparison of Price")

#### Plotting Distributions ####
# We can pick out a few of these high end neighborhoods and plot a more detailed view of the distribution of price.
# Two common ways to look at distributions are boxplots and cumulative distribution plots.

# Option 1 - Boxplots: A boxplot shows the 25th and 75th percentiles (top and bottom of the box), 
# the 50th percentile or median (thick middle line), the max/min values (top/bottom vertical lines), 
# and outliers (dots). We see that there are extreme outliers ($3000 per night!!). If we add in
# neighborhood information, we can see that the distributions are quite different 
# (such as Back Bay which has a "heavy tail" of expensive properties) 
listings %>% 
  filter(neighbourhood %in% c('South Boston Waterfront', 'Allston','Back Bay')) %>%
  filter(price<  2000) %>%
  ggplot(aes(x=factor(bedrooms), y=price)) +
  facet_grid(.~neighbourhood) +
  geom_boxplot()

# Option 2 - Cumulative Distribution (CDF) `stat_ecdf` plots the CDF of (i.e. percentiles vs. values) of vectors, 
# which gives you a lot more information about the distribution at the expense of being a bit harder to read.
# Let's plot the distribution of price by number of bedrooms, and use `coord_cartesian` to limit the x-axis range.
listings %>%
  ggplot(aes(price, color=factor(bedrooms))) +
  stat_ecdf() +
  coord_cartesian(xlim = c(0, 1000))

# What do you notice here?:
# - Prices cluster around multiples of $50 (look for the vertical lines). Maybe people should be differentiating on price more!
# - Low-end zero-bedroom units are cheaper than low-end one-bedroom units, but one-bedroom units are cheaper at the high end. 

#### Saving a plot
# You can use the "Plots" frame (lower right) to flip back through all the plots you've created 
# in the session (arrows), "Zoom" in on plots, and "Export" to save (to a PDF, image, or just to the clipboard).

#' EXERCISE: Explore the Airbnb listings data. Suppose you are thinking about investing in a property.
#' You have to decide what neighborhood and property type to buy, what amenities to offer, and 
#' how to price your listing. Use ggplot2 and dplyr to start wrangling the data to inform your decision process.


## --------------------------------------------------------------------
################### Closing Notes: General Tips #######################
## --------------------------------------------------------------------

# 1. How can I figure out the syntax or operations of a function?
# Suppose that you would like to figure out how to take the log base 10.
# You can directly use the help function:
help(log)
# Or you can just use the question mark before the function you are interested in
?log
# Look at the "Arguments" section to look at what kind of inputs the function takes.
# The "Details" section describes also the operations of the function.
# "Examples" will actually show you how to apply it.
# Another option is to just Google it!!
# How do we get then the log base 10?
log10(10)

# 2. What do I do if I get an error?
# First, do not panic! Coding is almost synonymous to debugging!
# You will learn more than what you expect by solving your errors.
# Try to understand the type of error that you encounter. 
# You will see a red message at the bottom of your console describing the error and the expression where it was encountered.

#Here is an example:
log("Maria")     
log(a)
# Object a is not found because it was not defined.
a = 10
log(a)

# Here are some common error messages
# "could not find function" errors, usually caused by typos or not loading a required package
# "Error in if" errors, caused by non-logical data or missing values passed to R's "if" conditional statement
# "Error in eval" errors, caused by references to objects that don't exist
# "cannot open" errors, caused by attempts to read a file that doesn't exist or can't be accessed
# "no applicable method" errors, caused by using an object-oriented function on a data type it doesn't support
# "subscript out of bounds" errors, caused by trying to access an element or dimension that doesn't exist
# package errors caused by being unable to install, compile or load a package.

# A link to the most common errors in R: https://www.r-bloggers.com/common-r-programming-errors-faced-by-beginners/

# 3. Where can I find help?
# There’s lots of information out there to help you decode your warning and error messages. 
# Here are some that I use all the time:
# Typing ? or ?? and the name of the function that’s going wrong in the console will give you help within R itself.
# Googling the error message, warning or package is often very useful
# Stack Overflow or the RStudio community forums can be searched for other people’s (solved!) problems
# Email us with questions if none of the above works!
