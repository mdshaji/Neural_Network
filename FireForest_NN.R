# Input Variables (x) = Other Variables
# Output Variable(y) = area

# Loading the dataset
Forest <- read.csv(file.choose())
View(Forest)

# Removing Unnecessary columns months and days
Forest <- Forest[3:30]
View(Forest)

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

summary(Forest)
attach(Forest)

# Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

# Boxplot Representation

boxplot(FFMC, col = "dodgerblue4",main = "FFMC")
boxplot(DMC, col = "dodgerblue4",main = "DMC")
boxplot(DC, col = "dodgerblue4",main = "DC")
boxplot(temp, col = "red", horizontal = T,main = "Temp")

# Histogram Representation

hist(FFMC,col = "orange", main = "FFMC" )
hist(DMC,col = "orange", main = "DMC")
hist(DC,col = "orange", main = "DC")
hist(temp,col = "red", main = "Temp")


# Or make a combined plot
#pairs(Start_up)   #  doesnt work as there is a categorical variable
#Scatter plot for all pairs of variables
plot(Forest)

# correlation matrix
cor(Forest)

# # custom normalization function
normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

# apply normalization to entire data frame
Forest_norm <- as.data.frame(lapply(Forest, normalize))

# create training and test data
Forest_train <- Forest_norm[1:360, ]
Forest_test <- Forest_norm[361:517, ]

## Training a model on the data ----
# train the neuralnet model
library(neuralnet)

# simple ANN with only a single hidden neuron
Forest_model <- neuralnet(formula =  area ~ ., data = Forest_train)


# visualize the network topology
plot(Forest_model)

## Evaluating model performance 

# obtain model results
# results_model <- NULL

results_model <- compute(Forest_model, Forest_test[-9])
# obtain predicted strength values
str(results_model)
predicted_area <- results_model$net.result

# examine the correlation between predicted and actual values
cor(predicted_area, Forest_test$area)

## Improving model performance ----
# a more complex neural network topology with 5 hidden neurons
Forest_model2 <- neuralnet(area ~ ., data = Forest_train, hidden = 5)


# plot the network
plot(Forest_model2)

# evaluate the results as we did before
model_results2 <- compute(Forest_model2, Forest_test[-9])
predicted_area2 <- model_results2$net.result
cor(predicted_area2, Forest_test$area)

## Improving model performance ----
# a more complex neural network topology with Stepmax function
Forest_model3 <- neuralnet(area ~ ., data = Forest_train, stepmax = 1e+05)


# plot the network
plot(Forest_model3)

# evaluate the results as we did before
model_results3 <- compute(Forest_model3, Forest_test[-9])
predicted_area3 <- model_results3$net.result
cor(predicted_area3, Forest_test$area)

 
# a more complex neural network topology with Stepmax function and algorithm
Forest_Model4 <- neuralnet(formula = area ~.,data = Forest_train, stepmax = 1e+05, algorithm = "rprop+", act.fct = "tanh")

# Plot the network
plot(Forest_Model4)

# evaluate the results as we did before
results_model4 <- compute(Forest_Model4, Forest_test[-9])
predicted_area4 <- results_model4$net.result
cor(predicted_area4, Forest_test$area) 
