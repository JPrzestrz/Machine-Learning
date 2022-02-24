# Load data using pandas
# Before you can train a machine learning model, you must first import data. There are several valid ways to do this, 
# but for now, we will use a simple one-liner from pandas: pd.read_csv(). Recall from the video that the first argument 
# specifies the path or URL. All other arguments are optional.

# In this exercise, you will import the King County housing dataset, which we will use to train a linear model later in the chapter.
# Import pandas under the alias pd
import pandas as pd

# Assign the path to a string variable named data_path
data_path = 'kc_house_data.csv'

# Load the dataset as a dataframe named housing
housing = pd.read_csv(data_path)

# Print the price column of housing
print(housing['price'])

# Setting the data type
# In this exercise, you will both load data and set its type. Note that housing is available and pandas has been imported as pd. 
# You will import numpy and tensorflow, and define tensors that are usable in tensorflow using columns in housing with a given data type. 
# Recall that you can select the price column, for instance, from housing using housing['price'].

# Import numpy and tensorflow with their standard aliases
import numpy as np
import tensorflow as tf

# Use a numpy array to define price as a 32-bit float
price = np.array(housing['price'], np.float32)

# Define waterfront as a Boolean using cast
waterfront = tf.cast(housing['waterfront'], tf.bool)

# Print price and waterfront
print(price)
print(waterfront)

# Loss functions in TensorFlow
# In this exercise, you will compute the loss using data from the King County housing dataset. You are given a target, price, 
# which is a tensor of house prices, and predictions, which is a tensor of predicted house prices. You will evaluate the loss 
# function and print out the value of the loss.

# Import the keras module from tensorflow
from tensorflow import keras
predictions = np.array([0,0,0,0])

# Compute the mean absolute error (mae)
loss = keras.losses.mae(price, predictions)

# Print the mean absolute error (mae)
print(loss.numpy())

# Modifying the loss function
# In the previous exercise, you defined a tensorflow loss function and then evaluated it once for a set of actual and predicted values. 
# In this exercise, you will compute the loss within another function called loss_function(), which first generates predicted values 
# from the data and variables. The purpose of this is to construct a function of the trainable model variables that returns the loss. 
# You can then repeatedly evaluate this function for different variable values until you find the minimum. In practice, you will pass 
# this function to an optimizer in tensorflow. Note that features and targets have been defined and are available. 
# Additionally, Variable, float32, and keras are available.
from tensorflow import Variable, float32
features = np.array([0])
targets = np.array([0])

# Initialize a variable named scalar
scalar = Variable(1.0, float32)

# Define the model
def model(scalar, features = features):
  	return scalar * features

# Define a loss function
def loss_function(scalar, features = features, targets = targets):
	# Compute the predicted values
	predictions = model(scalar, features)
    
	# Return the mean absolute error loss
	return keras.losses.mae(targets, predictions)

# Evaluate the loss function and print the loss
print(loss_function(scalar).numpy())

# Set up a linear regression
# A univariate linear regression identifies the relationship between a single feature and the target tensor. In this exercise, 
# we will use a property's lot size and price. Just as we discussed in the video, we will take the natural logarithms of both 
# tensors, which are available as price_log and size_log.
# In this exercise, you will define the model and the loss function. You will then evaluate the loss function for two different 
# values of intercept and slope. Remember that the predicted values are given by intercept + features*slope. Additionally, note 
# that keras.losses.mse() is available for you. Furthermore, slope and intercept have been defined as variables.
size_log =np.array([8.639411 , 8.887652 , 9.2103405, 7.20786, 7.7782116,6.9810057], dtype=float32)
price_log = np.array([12.309982 , 13.195614 , 12.100712, 12.904459 , 12.8992195, 12.691581 ], dtype=float32)

# Define a linear regression model
def linear_regression(intercept, slope, features = size_log):
	return intercept + slope * features

# Set loss_function() to take the variables as arguments
def loss_function(intercept, slope, features = size_log, targets = price_log):
	# Set the predicted values
	predictions = linear_regression(intercept, slope, features)
    
    # Return the mean squared error loss
	return keras.losses.mse(targets, predictions)

# Compute the loss for different slope and intercept values
print(loss_function(0.1, 0.1).numpy())
print(loss_function(0.1, 0.5).numpy())

# Train a linear model
# In this exercise, we will pick up where the previous exercise ended. The intercept and slope, intercept and slope, have been 
# defined and initialized. Additionally, a function has been defined, loss_function(intercept, slope), which computes the loss 
# using the data and model variables.
# You will now define an optimization operation as opt. You will then train a univariate linear model by minimizing the loss to 
# find the optimal values of intercept and slope. Note that the opt operation will try to move closer to the optimum with each 
# step, but will require many steps to find it. Thus, you must repeatedly execute the operation.

# Initialize an Adam optimizer
opt = keras.optimizers.Adam(0.5)

for j in range(100):
	# Apply minimize, pass the loss function, and supply the variables
	opt.minimize(lambda: loss_function(intercept, slope), var_list=[intercept, slope])

	# Print every 10th value of the loss
	if j % 10 == 0:
		print(loss_function(intercept, slope).numpy())

# Plot data and regression line
plot_results(intercept, slope)

# Multiple linear regression
# In most cases, performing a univariate linear regression will not yield a model that is useful for making accurate predictions. 
# In this exercise, you will perform a multiple regression, which uses more than one feature.
# You will use price_log as your target and size_log and bedrooms as your features. Each of these tensors has been defined and is 
# available. You will also switch from using the the mean squared error loss to the mean absolute error loss: keras.losses.mae(). 
# Finally, the predicted values are computed as follows: params[0] + feature1*params[1] + feature2*params[2]. Note that we've defined 
# a vector of parameters, params, as a variable, rather than using three variables. Here, params[0] is the intercept and params[1] 
# and params[2] are the slopes.

# Define the linear regression model
def linear_regression(params, feature1 = size_log, feature2 = bedrooms):
	return params[0] + feature1*params[1] + feature2*params[2]

# Define the loss function
def loss_function(params, targets = price_log, feature1 = size_log, feature2 = bedrooms):
	# Set the predicted values
	predictions = linear_regression(params, feature1, feature2)
  
	# Use the mean absolute error loss
	return keras.losses.mae(targets, predictions)

# Define the optimize operation
opt = keras.optimizers.Adam()

# Perform minimization and print trainable variables
for j in range(10):
	opt.minimize(lambda: loss_function(params), var_list=[params])
	print_results(params)

# Note that params[2] tells us how much the price will increase in percentage terms if we add one more bedroom. 
# You could train params[2] and the other model parameters by increasing the number of times we iterate over opt.

# Preparing to batch train
# Before we can train a linear model in batches, we must first define variables, a loss function, and an optimization operation. 
# In this exercise, we will prepare to train a model that will predict price_batch, a batch of house prices, using size_batch, a 
# batch of lot sizes in square feet. In contrast to the previous lesson, we will do this by loading batches of data using pandas, 
# converting it to numpy arrays, and then using it to minimize the loss function in steps.
# Variable(), keras(), and float32 have been imported for you. Note that you should not set default argument values for either the 
# model or loss function, since we will generate the data in batches during the training process.

# Define the intercept and slope
intercept = Variable(10.0, float32)
slope = Variable(0.5, float32)

# Define the model
def linear_regression(intercept, slope, features):
	# Define the predicted values
	return intercept + slope * features

# Define the loss function
def loss_function(intercept, slope, targets, features):
	# Define the predicted values
	predictions = linear_regression(intercept, slope, features)
    
 	# Define the MSE loss
	return keras.losses.mse(targets, predictions)