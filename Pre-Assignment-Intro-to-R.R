# Pre-assignment: Introduction to R
# Thanks to Andrew Zheng for much of this example's content!

## --------------------------------------------------------------------
########### Introduction to R + Basic Syntax ##################
## --------------------------------------------------------------------

# this is a comment

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
a == 4 # checks if a is equal to 5
a <= 5 # checks if a is less than or equal to 5 ## don't confuse with "<-"
a != 4 # checks if a is NOT equal to 4

# And you are done with the preassignment! 