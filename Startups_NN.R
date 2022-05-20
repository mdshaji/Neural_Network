# Input Variables (x) = R.D Spend , Administration , Marketing Spend , State
# Output Variable(y) = Profit

# Importing the dataset
Startup <- read.csv(file.choose())
colnames(Startup) <- c("RD","Admin","MS","State","Profit") # RD = R.D.Spend , Admin = Administrartion & MS = Marketing Spend
View(Startup)

# Creating dummy variables for State

install.packages("dummies")
library(dummies)

str(Startup)
Startup$State <- as.factor(Startup$State)
Startup$State = as.numeric(Startup$State)
str(Startup)
View(Startup)
attach(Startup)


# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

summary(Startup)

# Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

# Boxplot Representation

boxplot(Startup$RD, col = "dodgerblue4",main = "R.D.Spend")
boxplot(Startup$Admin, col = "dodgerblue4",main = "Administration")
boxplot(Startup$MS, col = "dodgerblue4",main = "Marketing Spend")
boxplot(Startup$Profit, col = "red", horizontal = T,main = "Profit")

# Histogram Representation

hist(Startup$RD,col = "orange", main = "R.D.Spend" )
hist(Startup$Admin,col = "orange", main = "Administration")
hist(Startup$MS,col = "orange", main = "Marketing Spend")
hist(Startup$Profit,col = "red", main = "Profit")

# Scatter plot
plot(RD,Profit) # Plot relation ships between each X with Y
plot(Admin, Profit)
plot(MS,Profit)
#plot(State,Profit) #this shows error as it is a categorical variable

# Or make a combined plot
#pairs(Start_up)   #  doesnt work as there is a categorical variable
#Scatter plot for all pairs of variables
plot(Startup)

cor(RD, Profit) #0.973
cor(Startup) # correlation matrix
#Seems Strong coorelation between profit and R.D.Spend

# custom normalization function
normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

# apply normalization to entire data frame
Startup_norm <- as.data.frame(lapply(Startup, normalize))
View(Startup_norm)

# create training and test data
Startup_train <- Startup_norm[1:30, ]
Startup_test <- Startup_norm[31:50, ]

## Training a model on the data ----
# train the neuralnet model
library(neuralnet)

# simple ANN with only a single hidden neuron
Startup_model <- neuralnet(formula = Profit ~ .,data = Startup_train)


# visualize the network topology
plot(Startup_model)

## Evaluating model performance 

# obtain model results
# results_model <- NULL

results_model <- compute(Startup_model, Startup_test[1:4])
# obtain predicted strength values
str(results_model)
predicted_strength <- results_model$net.result

# examine the correlation between predicted and actual values
cor(predicted_strength, Startup_test$strength)

## Improving model performance ----
# a more complex neural network topology with 5 hidden neurons
Startup_model2 <- neuralnet(Profit ~ .,data = Startup_train, hidden = 5)


# plot the network
plot(Startup_model2)

# evaluate the results as we did before
model_results2 <- compute(Startup_model2, Startup_test[1:4])
predicted_strength2 <- model_results2$net.result
cor(predicted_strength2, Startup_test$strength)

## Improving model performance ----
# a more complex neural network topology with 3 hidden neurons
Startup_model3 <- neuralnet(Profit ~ .,data = Startup_train, hidden = 3)


# plot the network
plot(Startup_model3)

# evaluate the results as we did before
model_results3 <- compute(Startup_model3, Startup_test[1:4])
predicted_strength3 <- model_results3$net.result
cor(predicted_strength3, Startup_test$strength)


# Conclusion
# Model built with 1 hidden neurons give the error as 0.028742
# Model built with 5 hidden neurons give the error as 0.026689
# Model built with 3 hidden neurons give the error as 0.025131

# By comparing all the 3 above models the one with 3 hidden neurons gives less error values which is a right fit model
