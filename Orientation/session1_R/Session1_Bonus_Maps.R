
## --------------------------------------------------------------------
############### Part 3b: Plotting Geographical Data ###################
## --------------------------------------------------------------------
#' As our final task, we are going to learn how to plot data on a map! 
#' This is especially useful if you are working with location-specific data
#' to understand spatial/geographical trends.
#' In this example, we will investigate the breakdown of international students at MIT.

## First, we're going to need the ggmap package
# install.packages("ggmap")
# install.packages("mapproj")
# install.packages("map")
library(maps)
data(mapdata)

# Next, we will load in data about international students at MIT.
intlall = read.csv("intlall.csv",stringsAsFactors=FALSE)

# Let's look at the first few rows.
head(intlall)

# Those NAs are really 0s, and we can replace them easily
intlall[is.na(intlall)] = 0

# Now lets look again
head(intlall)  # Much better

# Next step is to load the world map
world_map = map_data("world")
str(world_map)

# Region is the key word in world_map (i.e. the country), which is called Citizenship in intlall. 
# Lets join intlall and world_map using the join command
student_map = inner_join(world_map, intlall, 
                         by = c("region" = "Citizenship"))

# What is the data anyway? Its actually the lat-lon points that define the border of each country.
# We want to make sure that they are properly ordered to plot the countries (so that we don't end up
# with criss-crossed borders.
student_map = student_map[order(student_map$group, student_map$order),]

# To plot a map, we use geom_polygon. The group=group says that we should treat each country as its own polygon 
# (in the data, there is a group identifier for each region that makes up its own polygon.)
ggplot(student_map, aes(x=long, y=lat, group=group)) +
  geom_polygon(fill="white", color="black") +
  coord_map("mercator", xlim=c(-180,180))

# We can also update the fill color to be based on a feature in our data: this gives us a "heatmap map"
ggplot(student_map, aes(x=long, y=lat, group=group)) +
  geom_polygon(aes(fill=Total), color="black") +
  coord_map("mercator", xlim=c(-180,180))

# So where is the missing data? It looks like China is missing from the map. Let's look for it in our data.
table(intlall$Citizenship)  # China (People's Republic Of)
table(map_data("world")$region)  # China

# Lets "fix" that in the intlall dataset
intlall$Citizenship[intlall$Citizenship=="China (People's Republic Of)"] = "China"

# We'll repeat our merge and order from before. We can use arrange() for our ordering, like we learned above!
student_map = inner_join(map_data("world"), intlall, c("region" = "Citizenship")) %>%
  arrange(group,order)

ggplot(student_map, aes(x=long, y=lat, group=group)) +
  geom_polygon(aes(fill=Total), color="black") +
  coord_map("mercator", xlim=c(-180,180))

# We could have also done a left join to keep the outline of countries with no student matches on the map. 
student_map = left_join(map_data("world"), intlall, c("region" = "Citizenship")) %>%
  arrange(group,order)

ggplot(student_map, aes(x=long, y=lat, group=group)) +
  geom_polygon(aes(fill=Total), color="black") +
  coord_map("mercator", xlim=c(-180,180))

# We can try other projections - this one is visually interesting
ggplot(student_map, aes(x=long, y=lat, group=group)) +
  geom_polygon(aes(fill=Total), color="black") +
  coord_map("ortho", orientation=c(20, 30, 0))

# Are these good visualizations?
# We can see some interesting patterns in this picture, and it's clear
# to see the biggest "supplier" countries. This information combined
# with the regional information can give us a good feel for the situation
# but the lack of regional summaries in the map hurts us - Europe and
# South America seem to contribute on the same order as Australia and
# New Zealand which is not accurate at all.

