## --------------------------------------------------------------------
#################  Introduction to R + Basic Syntax ###################
## --------------------------------------------------------------------

# Use the hashtag symbol '#' at the beginning of the a line to write comments (so yeah this is a comment!)

# 1. How can I execute a command ? 
# To execute a line, simply press CMD+Enter on a Mac or Ctrl+r on a PC or
# click on the "Run" button on the top right of the screen.  
# The result will appear in the console.

# 2. How can I execute multiple commands at the same time? 
# You can run multiple lines of code if you select them and then follow the same commands.

###### Basic R Functions ###### 
# R has many built-in functions that aid in data analysis

# Let's start with some basic functions:
#You can do many mathematical calculations by simply typing the commands
3+5
2^3

abs(-5)
sqrt(2)
log(10)

# R has two equivalent ways of assigning variables, "=" and "<-"
x=5
x <- 5

# Many R functions take vectors as arguments. 
# Let's see how to construct a vector (collection of data)
# Here is an example of a vector of numbers
numbers = c(10, 10, 20, 30, 50)
# To find out what is the type of any given vector you can use the following command
typeof(x)

#Here is another example
name <- c("Maria", "Fred", "Sakura") 
typeof(name)

#We can also find out how big is the vector in hand
length(numbers)
#It has 5 observations

# We can also access the individual numbers via their index
numbers[1]
numbers[5]
numbers[6]

# Some R functions can generate vectors easily.  Here are two examples:

# The following creates a vector of 10 ones. "Repeat 1, 10 times"
rep(1, 10)

# The following creates a vector with the numbers 1 through 50, in order "Sequence from 1 to 50"
seq(1, 50)

# Basic boolean statements
a = 2
a == 4 # checks if a is equal to 4
a <= 4 # checks if a is less than or equal to 4 ## don't confuse with "<-"
a != 4 # checks if a is NOT equal to 4

###### Working Directory ######
# In analytics, you will always work with a dataset of observations that you will need to load into your R session.
# The best practice is to gather all your datasets in a specific folder for a given project and always refer to that
# folder when you would like to load them. We will call this folder the "Working Directory"

# How do I set my working directory?
# At the top of your screen click Session > Set Working Directory > Choose Directory > Select your Folder
# You will see an additional line at your console. 
# For example: "setwd("~/git/mban22_software_tools/R_Basics.R")"
# You can copy that that line at the top of your script to save it for future use.

###### Loading a Dataset ###### 

# Most of the data sets that you will find in practice are in .csv format. 
# You can just look at the end of a given file name to check its format.
# In our class, we will provide you with datasets in .csv format

# 6. How can I import a dataset?
# Option 1: directly through a command
df = read.csv("data/wine.csv")
# Option 2: Through a menu
# Click at top panel at the right of your screen: "Import Dataset" > "From Text (base)" > Select the exact file.

# Look at the structure of the data:
str(df)

# Look at a statistical summary of the data:
summary(df)

# To access a variable in a data frame, you link it to the data frame with the dollar sign.
df$Year
df$WinterRain

# Let's get some basic statistics about the rainfall
# Try applying sum(), median(), mean(), sd(), max(), min() to the data
sum(df$WinterRain)
median(df$WinterRain)
mean(df$WinterRain)
sd(df$WinterRain)
max(df$WinterRain)
min(df$WinterRain)

# We could also pull a statistical summary:
summary(df$WinterRain)

# In one year, there was only 376 mm of winter rain - which year?
which.min(df$Price)
df$Year[19]
# You can also do this in one step:
df$Year[which.min(df$Price)]

# Sorted vector of WinterRain (default: ascending order)
sort(df$WinterRain)

#Suppose that you would like to sort them in a decreasing order, look for the help function and see how that can be done
?sort
sort(df$WinterRain, decreasing = TRUE)

# How can I save a dataset that I was working on?
df_subset <- df[1:10,]
write.csv(df_subset, "test_write.csv", row.names = FALSE)

# How can I clear my working environment?
# Option 1: directly through a commad
rm(list=ls())
# Option 2: Through a menu
# Click at top panel at the right of your screen at the brush icon.


#Great! Now you are all ready for class; see you soon!