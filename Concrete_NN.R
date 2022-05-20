##### Neural Networks 
# Input Variables (x) = Other Variables
# Output Variable(y) = Strength

# Load the Concrete data as concrete
concrete <- read.csv(file.choose())
View(concrete)
attach(concrete)

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

summary(concrete)

# Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

# Boxplot Representation

boxplot(cement, col = "dodgerblue4",main = "Cement")
boxplot(slag, col = "dodgerblue",main = "Slag")
boxplot(ash, col = "dodgerblue4",main = "Ash")
boxplot(water, col = "red", horizontal = T,main = "Water")

# Histogram Representation

hist(cement,col = "blue", main = "Cement" )
hist(slag,col = "purple", main = "Slag")
hist(ash,col = "orange", main = "Ash")
hist(water,col = "red", main = "Water")


# Or make a combined plot
#pairs(Start_up)   #  doesnt work as there is a categorical variable
#Scatter plot for all pairs of variables
plot(concrete)

# correlation matrix
cor(concrete)


# custom normalization function
normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

# apply normalization to entire data frame
concrete_norm <- as.data.frame(lapply(concrete, normalize))

# create training and test data
concrete_train <- concrete_norm[1:773, ]
concrete_test <- concrete_norm[774:1030, ]

## Training a model on the data ----
# train the neuralnet model
library(neuralnet)

# simple ANN with only a single hidden neuron
concrete_model <- neuralnet(formula = strength ~ cement + slag +
                              ash + water + superplastic + 
                              coarseagg + fineagg + age,
                              data = concrete_train)


# visualize the network topology
plot(concrete_model)

## Evaluating model performance 

# obtain model results
# results_model <- NULL

results_model <- compute(concrete_model, concrete_test[1:8])
# obtain predicted strength values
str(results_model)
predicted_strength <- results_model$net.result

# examine the correlation between predicted and actual values
cor(predicted_strength, concrete_test$strength)

## Improving model performance ----
# a more complex neural network topology with 5 hidden neurons
concrete_model2 <- neuralnet(strength ~ cement + slag +
                               ash + water + superplastic + 
                               coarseagg + fineagg + age,
                               data = concrete_train, hidden = 5)


# plot the network
plot(concrete_model2)

# evaluate the results as we did before
model_results2 <- compute(concrete_model2, concrete_test[1:8])
predicted_strength2 <- model_results2$net.result
cor(predicted_strength2, concrete_test$strength)
