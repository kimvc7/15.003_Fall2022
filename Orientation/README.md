# Orientation Preassignment (August 23-24)

This is a **long** preassignment that involves lots of software installation and testing. Please leave a total of at least **2 hours** to complete this preassignment. That may seem like a long time, but once you've done it you'll have a powerful suite of software that you can use through your career at MIT and beyond. 

**If You Encounter Problems**
1. If you receive an error message, Google it. 
2. If you tried that, write an email to `kimvc@mit.edu` and `midsumer@mit.edu` describing the problem in as much detail as possible, preferably including screenshots.  


# 1. Data Analysis: R and RStudio

## Install R and RStudio

**We are assuming that you have the latest version of R (4.2.1) installed.** You may need to update your installation if you have an older version.
 
1. **Install R**: Navigate to [`http://cran.wustl.edu`](http://cran.wustl.edu) and follow the instructions for your operating system. 
2. **Download RStudio**: Navigate to [`https://www.rstudio.com/products/rstudio/download/`](https://www.rstudio.com/products/rstudio/download/) and download RStudio Desktop with an Open Source License. 
3. **Test Your Installation**: Open RStudio and type 1+2 into the Console window, and press "Enter." If you see the expected result, you are ready to move on. 

## Install Packages

In the RStudio console, type 
```R
pkgs <- c('tidyverse')
install.packages(pkgs)
```
Once the installation is complete, try to load the package
```R
library(tidyverse)
```
If you encounter any error messages that you are unable to handle, please email us. 


## Introduction to R + Basic Syntax

Donwload the Pre-Assignment folder from Canvas in 15.003_FA22/Files/Pre-Assignment, it should contain a data folder with wine.csv as well as R_Basics.R. 

On the top left corner of R Studio, Click on File -> Open File, and navigate to where you have downloaded the folder to open it. You will see a new window open up on the top left corner with some scripts already written for you. Follow along the instructions in the comment and work through the basic syntax. 



# 2. Optimization: Julia and JuMP

**Please try to complete the steps below before the first day of class.**  We will only be using Julia and Gurobi on the second day, but we have very limited time in class and we will not be able to help you with installation problems during the teaching time. If you have difficulties with the installations below, please email `kimvc@mit.edu` and `midsumer@mit.edu` and include as much information as possible so that we can assist you.

*Note that you will need to be connected to the MIT network to activate the Gurobi installation, but the other steps can be completed from any location.* 

## Install Julia

Julia is programming language developed at MIT. To install Julia, go to [`https://julialang.org/downloads/`](https://julialang.org/downloads/) and download the appropriate version for your operating system. See [`here`](https://julialang.org/downloads/platform/) for more detailed instructions.
We will assume that everyone has installed the most recent version of Julia (v1.7.3). If you have an older version installed, we recommend that you install the newer version as well. Note: Julia just released a newer version this week to v1.8.0, you can find v1.7.3 by scrolling down the page to older releases, or https://julialang.org/downloads/oldreleases/. 

To confirm that Julia is installed, open a Julia window by clicking on the Julia icon in your applications menu (note: mac users should make sure Julia is copied into their applications folder). You should see a prompt at the bottom of the new window that looks like this:

```julia
julia>
```

Type 1+1 after the prompt and press enter/return.
```julia
julia> 1+1
2
```
If you see the response above, Julia is working!

## Install JuMP

JuMP is a Julia package that we will use to create optimization models in class. To install this package, run the following lines in the Julia window:
```julia
julia> using Pkg
julia> Pkg.add("JuMP")
```

This might take quite a while to finish, so don’t worry if it looks like nothing is happening in the Julia window. You will know that the process is complete when you see the command prompt (julia>) appear at the bottom of your screen.

To test if the package is installed correctly, run the following commands
```julia
julia> using JuMP
julia> m = Model()
```
You should see the output below

```julia
A JuMP Model
Feasibility problem with:
Variables: 0
Model mode: AUTOMATIC
CachingOptimizer state: NO_OPTIMIZER
Solver name: No optimizer attached.
```

## Install IJulia and Jupyter

Jupyter is a free, open-source program that will allow us to write and run Julia code in a web browser (instead of typing everything into the command line). IJulia is a Julia package that allows Julia to communicate with the Jupyter software. Instead of installing Jupyter on its own, we can use the IJulia package to install it within Julia.

Run the following lines in a Julia window:

```julia
julia> using Pkg
julia> Pkg.add("IJulia")
julia> using IJulia
```

These lines download and install the IJulia package. 
Now, we will try to open a Jupyter notebook. If Jupyter is not installed, Julia will ask if we want to install it now. Run the following line, and then press enter/return or y to install Jupyter:
```julia
julia> notebook()
install Jupyter via Conda, y/n? [y]: 
```

If this is successful, a Jupyter tab will open in the default browser on your computer. Click “New” in the top right corner to make a new notebook (if a menu appears, select Julia 1.7.3). A new tab will open with a blank Jupyter notebook.


## Install Gurobi
*Note: you must be on the MIT network to activate your academic license. We will leave time at the end of day 1 of orientation for you to complete these steps. If you will not be on campus during orientation, you can use a different solver instead without a license--see notes below.*

Gurobi is a commercial optimization solver that we will use to solve optimization problems in class. Here are the basic steps that you will need to follow to install Gurobi,: 

1. Register for a Gurobi account on the [gurobi website](https://www.gurobi.com). Use your @mit.edu email address, and select the Academic option (not the commercial option).
2. Download the Gurobi Optimizer software [`here`](https://www.gurobi.com/downloads/) and install. You might need to log in to the page first, the current stable version is Gurobi 9.5.2.
3. Create and download an Academic License to use the software [`here`](https://www.gurobi.com/downloads/end-user-license-agreement-academic/).
4. Use the license file to activate the Gurobi software that you installed. Follow the instructions on the license page to run the grbgetkey command. **Note that you must be connected to the MIT SECURE network to do this.** If you are not on campus, please move on to the next section (IJulia) and come back to this step later.

A summary of the Gurobi installation/activation process is available [`here`](https://www.gurobi.com/academia/academic-program-and-licenses/) and detailed installation instructions are available [`here`](https://www.gurobi.com/documentation/quickstart.html). If you get stuck trying to follow these instructions, please email us for assistance.

After installing Gurobi, we need to add a Julia package called "Gurobi" that allows Julia to communicate with the Gurobi software. Run the following lines in your Julia window:
```julia
julia> using Pkg
julia> Pkg.add("Gurobi")
```

###### Gurobi Error in Julia
If you see an error message during this installation, it could be because you did not install/activate Gurobi properly. Please read through the "Installation" information [`here`](https://github.com/JuliaOpt/Gurobi.jl) and see the instructions for setting the GUROBI_HOME environment variable in Julia;
```julia
# On Windows, this might be
ENV["GUROBI_HOME"] = "C:\\Program Files\\gurobi952\\win64"
# ... or perhaps ...
ENV["GUROBI_HOME"] = "C:\\gurobi952\\win64"
using Pkg
Pkg.add("Gurobi")
Pkg.build("Gurobi")

# On Mac, this might be
ENV["GUROBI_HOME"] = "/Library/gurobi952/mac64"
using Pkg
Pkg.add("Gurobi")
Pkg.build("Gurobi")
```

**Note: check the version of Gurobi that you downloaded. The above instructions assume you downloaded version 9.5.2. If you have
a different version, your path may differ (e.g. Gurobi 9.5.2 -> replace gurobi950 with gurobi952). 
If this doesn't work, also check which folder you installed Gurobi in, and update the path accordingly if necessary.**


If the Gurobi package is successfully installed in Julia, run the following lines, you might see a warning of Academic license - for non-commercial use only - expires 2023-08-11, this is normal:
```julia
julia> using JuMP, Gurobi
julia> model = Model(with_optimizer(Gurobi.Optimizer, Presolve=0, OutputFlag=0))
```

You should see this output:

```julia
A JuMP Model
Feasibility problem with:
Variables: 0
Model mode: AUTOMATIC
CachingOptimizer state: EMPTY_OPTIMIZER
Solver name: Gurobi
```
#### Alternative to Gurobi
If you are unable to activate your Gurobi license (i.e. if you are not yet on campus), you can use an open-source solver as a temporary solution. 

Also install the Cbc package, which will be the backend mixed-integer optimization solver for our optimization problems.
```julia
julia> Pkg.add("Cbc")
julia> using JuMP, Cbc
julia> model = Model(with_optimizer(Cbc.Optimizer, Presolve=0, OutputFlag=0))
```

You should see this output: 
```julia
A JuMP Model
Feasibility problem with:
Variables: 0
Model mode: AUTOMATIC
CachingOptimizer state: EMPTY_OPTIMIZER
Solver name: COIN Branch-and-Cut (Cbc)
```



## Final Check

Once you have completed all the steps above, copy and paste the following code into a new Jupyter notebook (next to the "In []:" prompt)

```julia
using JuMP, Gurobi
model = Model(with_optimizer(Gurobi.Optimizer, Presolve=0, OutputFlag=0)) # or Cbc.Optimizer
@variable(model,x>=0)
@objective(model, Min, x)
optimize!(model)
print("The answer is ",JuMP.value(x))
```

Now, click the "Run" button to run this code. You should see this output below:

```julia
The answer is 0.0
```

If you see this output, everything is working correctly. If you see errors, one of the steps above may be incomplete. If you don't see any output, make sure that you have selected the notebook cell where you paste the code and try to run it again. 


# 3. Version Control: Git and GitHub

How can we manage complex, code-based workflows? How can we reliably share code between collaborators without syncing issues? How can we track multiple versions of scripts without going crazy? There are multiple solutions to these problems, but *version control* with git is by far the most common. 

## Install Git

We will be using the command-line interface to Git. First of all, check if you already have git installed (in which case you can skip this step). **Windows** users should look for Git Bash, while **macOS** and **Linux** users should open a terminal and try running the command: `git`

If you don't have git installed, go to the [Git project page](https://www.git-scm.com/) and follow the link on the right side to download the installer for your operating system. Follow the instructions in the README file in the downloaded .zip or .dmg.

**Windows**: During the installation, select to use Git from the Windows command prompt, checkout Windows-style, commit UNIX-style line endings, and add a shortcut to the Desktop for easy access.

**macOS**: if you receive an “unidentified developer” warning, right-click the .pkg file, select Open, then click Open.


## Connect to GitHub.com

[GitHub](https://github.com/) is a hosting service for git that makes it easy to share your code. 


1. Sign up for an account -- remember to keep track of your username and password. Feel free to enter information about yourself and optionally a profile picture. 

2. From the menu at the top right corner of the page, go to Settings, and select SSH and GPG keys.

3. Follow the [GitHub instructions to set up SSH](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh):

	[Check if you already have SSH keys](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh/checking-for-existing-ssh-keys)

	[Generate a SSH key](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) if you don’t have one

	[Add the SSH key to your account](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) 

	[Test the SSH connection](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh/testing-your-ssh-connection)


**If you've made it this far, congratulations!** You now possess a powerful set of tools for analyzing data, solving optimization problems, and collaborating on code. You're ready to go!  




