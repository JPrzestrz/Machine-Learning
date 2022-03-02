# The linear algebra of dense layers
# There are two ways to define a dense layer in tensorflow. The first involves the use of low-level, linear algebraic 
# operations. The second makes use of high-level keras operations. In this exercise, we will use the first method to
#  construct the network shown in the image below.

# The input layer contains 3 features -- education, marital status, and age -- which are available as borrower_features. 
# The hidden layer contains 2 nodes and the output layer contains a single node.
# For each layer, you will take the previous layer as an input, initialize a set of weights, compute the product of the 
# inputs and weights, and then apply an activation function. 
from importlib.util import module_for_loader
from tensorflow import Variable, ones, matmul, keras, float32
import numpy as np
import tensorflow as tf
from keras.datasets import mnist 
(train_features, train_labels), (test_features, test_labels) = mnist.load_data()

borrower_features = np.array([[ 2.,  2., 43.]])

# From previous step
bias1 = Variable(1.0)
weights1 = Variable(ones((3, 2)))
product1 = matmul(borrower_features, weights1)
dense1 = keras.activations.sigmoid(product1 + bias1)

# Initialize bias2 and weights2
bias2 = Variable(1.0)
weights2 = Variable(ones((2, 1)))

# Perform matrix multiplication of dense1 and weights2
product2 = matmul(dense1, weights2)

# Apply activation to product2 + bias2 and print the prediction
prediction = keras.activations.sigmoid(product2 + bias2)
print('\n prediction: {}'.format(prediction.numpy()[0,0]))
print('\n actual: 1')

# Our model produces predicted values in the interval between 0 and 1. For the example we considered, the actual value 
# was 1 and the predicted value was a probability between 0 and 1. This, of course, is not meaningful, since we have not 
# yet trained our model's parameters.

# --- 

# The low-level approach with multiple examples
# In this exercise, we'll build further intuition for the low-level approach by constructing the first dense hidden layer 
# for the case where we have multiple examples. We'll assume the model is trained and the first layer weights, weights1, 
# and bias, bias1, are available. We'll then perform matrix multiplication of the borrower_features tensor by the weights1 
# variable. Recall that the borrower_features tensor includes education, marital status, and age. Finally, we'll apply the 
# sigmoid function to the elements of products1 + bias1, yielding dense1.

# Compute the product of borrower_features and weights1
products1 = matmul(borrower_features, weights1)

# Apply a sigmoid activation function to products1 + bias1
dense1 = keras.activations.sigmoid(products1+bias1)

# Print the shapes of borrower_features, weights1, bias1, and dense1
print('\n shape of borrower_features: ', borrower_features.shape)
print('\n shape of weights1: ', weights1.shape)
print('\n shape of bias1: ', bias1.shape)
print('\n shape of dense1: ', dense1.shape)

# Our input data, borrower_features, is 5x3 because it consists of 5 examples for 3 features. The shape of weights1 is 3x2, 
# as it was in the previous exercise, since it does not depend on the number of examples. Additionally, bias1 is a scalar. 
# Finally, dense1 is 5x2, which means that we can multiply it by the following set of weights, weights2, which we defined to 
# be 2x1 in the previous exercise.

# ---

# Using the dense layer operation
# We've now seen how to define dense layers in tensorflow using linear algebra. In this exercise, we'll skip the linear algebra 
# and let keras work out the details. This will allow us to construct the network below, which has 2 hidden layers and 10 features, 
# using less code than we needed for the network with 1 hidden layer and 3 features.

# To construct this network, we'll need to define three dense layers, each of which takes the previous layer as an input, multiplies 
# it by weights, and applies an activation function. Note that input data has been defined and is available as a 100x10 tensor: 
# borrower_features. Additionally, the keras.layers module is available.
from keras import layers

# Define the first dense layer
dense1 = keras.layers.Dense(7, activation='sigmoid')(borrower_features)

# Define a dense layer with 3 output nodes
dense2 = keras.layers.Dense(3, activation='sigmoid')(dense1)

# Define a dense layer with 1 output node
predictions = keras.layers.Dense(1, activation='sigmoid')(dense2)

# Print the shapes of dense1, dense2, and predictions
print('\n shape of dense1: ', dense1.shape)
print('\n shape of dense2: ', dense2.shape)
print('\n shape of predictions: ', predictions.shape)

# Binary classification problems
# In this exercise, you will again make use of credit card data. The target variable, default, indicates whether a credit 
# card holder defaults on his or her payment in the following period. Since there are only two options--default or not--this 
# is a binary classification problem. While the dataset has many features, you will focus on just three: the size of the three 
# latest credit card bills. Finally, you will compute predictions from your untrained network, outputs, and compare those the 
# target variable, default.
from tensorflow import constant
bill_amounts = tf.Variable(np.array([[77479, 77057, 78102],[  326,   326,   326],[13686,  1992,   604],
                                     [  944,  1819,  1133],[    0,     0,     0],[96491, 94043, 97522]]))

default = np.array([[0],[0],[0],[0],[1],[0]])

# Construct input layer from features
inputs = constant(bill_amounts, float32)

# Define first dense layer
dense1 = keras.layers.Dense(3, activation='relu')(inputs)

# Define second dense layer
dense2 = keras.layers.Dense(2,activation='relu')(dense1)

# Define output layer
outputs = keras.layers.Dense(1, activation='sigmoid')(dense2)

# Print error for first five examples
error = default[:5] - outputs.numpy()[:5]
print(error)

# If you run the code several times, you'll notice that the errors change each time. This is because you're using an 
# untrained model with randomly initialized parameters. Furthermore, the errors fall on the interval between -1 and 1 
# because default is a binary variable that takes on values of 0 and 1 and outputs is a probability between 0 and 1.

# Multiclass classification problems
# In this exercise, we expand beyond binary classification to cover multiclass problems. A multiclass problem has targets 
# that can take on three or more values. In the credit card dataset, the education variable can take on 6 different values, 
# each corresponding to a different level of education. We will use that as our target in this exercise and will also expand 
# the feature set from 3 to 10 columns.
# As in the previous problem, you will define an input layer, dense layers, and an output layer. You will also print the 
# untrained model's predictions, which are probabilities assigned to the classes.

borrower_features = np.array([[180201, 181443, 155045, ...,   3100,  92101,   3200],
       [ 12200,   8140,  10176, ...,  15027,    639,    547],
       [  1078,   5543,    500, ...,    617,      0,      0],
       [   291,    763,   2403, ...,      0,      0,    291],
       [ 86076,  87567,  78646, ...,   3116,   2000,   2100],
       [  7791,   5851,    430, ...,   2041,    775,    580]])

# Construct input layer from borrower features
inputs = constant(borrower_features, float32)

# Define first dense layer
dense1 = keras.layers.Dense(10, activation='sigmoid')(inputs)

# Define second dense layer
dense2 = keras.layers.Dense(8, activation='relu')(dense1)

# Define output layer
outputs = keras.layers.Dense(6, activation='softmax')(dense2)

# Print first five predictions
print(outputs.numpy()[:5])

# Notice that each row of outputs sums to one. This is because a row contains the predicted class probabilities 
# for one example. As with the previous exercise, our predictions are not yet informative, since we are using an 
# untrained model with randomly initialized parameters. This is why the model tends to assign similar probabilities 
# to each class.

# The dangers of local minima
# Consider the plot of the following loss function, loss_function(), which contains a global minimum, marked by the dot on the right, 
# and several local minima, including the one marked by the dot on the left.
# In this exercise, you will try to find the global minimum of loss_function() using keras.optimizers.SGD(). You will do this twice, 
# each time with a different initial value of the input to loss_function(). First, you will use x_1, which is a variable with an 
# initial value of 6.0. Second, you will use x_2, which is a variable with an initial value of 0.3. 

# some predefined loss function that we don't know 

# Initialize x_1 and x_2
x_1 = Variable(6.0,float32)
x_2 = Variable(0.3,float32)

# Define the optimization operation
opt = keras.optimizers.SGD(learning_rate=0.01)

for j in range(100):
	# Perform minimization using the loss function and x_1
	opt.minimize(lambda: loss_function(x_1), var_list=[x_1])
	# Perform minimization using the loss function and x_2
	opt.minimize(lambda: loss_function(x_2), var_list=[x_2])

# Print x_1 and x_2 as numpy arrays
print(x_1.numpy(), x_2.numpy())

# Notice that we used the same optimizer and loss function, but two different initial values. When we started at 6.0 
# with x_1, we found the global minimum at 4.38, marked by the dot on the right. When we started at 0.3, we stopped 
# around 0.42 with x_2, the local minimum marked by a dot on the far left.

# Avoiding local minima
# The previous problem showed how easy it is to get stuck in local minima. We had a simple optimization problem in one 
# variable and gradient descent still failed to deliver the global minimum when we had to travel through local minima 
# first. One way to avoid this problem is to use momentum, which allows the optimizer to break through local minima. 
# We will again use the loss function from the previous problem, which has been defined and is available for you as 
# loss_function().
# Several optimizers in tensorflow have a momentum parameter, including SGD and RMSprop. You will make use of RMSprop 
# in this exercise. Note that x_1 and x_2 have been initialized to the same value this time.

# Initialize x_1 and x_2
x_1 = Variable(0.05,float32)
x_2 = Variable(0.05,float32)

# Define the optimization operation for opt_1 and opt_2
opt_1 = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.99)
opt_2 = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0)

for j in range(100):
	opt_1.minimize(lambda: loss_function(x_1), var_list=[x_1])
    # Define the minimization operation for opt_2
	opt_2.minimize(lambda: loss_function(x_2), var_list=[x_2])

# Print x_1 and x_2 as numpy arrays
print(x_1.numpy(), x_2.numpy())

# Recall that the global minimum is approximately 4.38. Notice that opt_1 built momentum, bringing x_1 closer to the global 
# minimum. To the contrary, opt_2, which had a momentum parameter of 0.0, got stuck in the local minimum on the left.

# Initialization in TensorFlow
# A good initialization can reduce the amount of time needed to find the global minimum. In this exercise, we will initialize 
# weights and biases for a neural network that will be used to predict credit card default decisions. To build intuition, we 
# will use the low-level, linear algebraic approach, rather than making use of convenience functions and high-level keras 
# operations. We will also expand the set of input features from 3 to 23. Several operations have been imported from 
# tensorflow: Variable(), random(), and ones().
from tensorflow import random

# Define the layer 1 weights
w1 = Variable(random.normal([23, 7]))

# Initialize the layer 1 bias
b1 = Variable(ones([7]))

# Define the layer 2 weights
w2 = Variable(random.normal([7,1]))

# Define the layer 2 bias
b2 = Variable(0.0)

# Defining the model and loss function
# In this exercise, you will train a neural network to predict whether a credit card holder will default. The features and 
# targets you will use to train your network are available in the Python shell as borrower_features and default. You defined 
# the weights and biases in the previous exercise.
# Note that the predictions layer is defined as , where  is the sigmoid activation, layer1 is a tensor of nodes for the first 
# hidden dense layer, w2 is a tensor of weights, and b2 is the bias tensor.
# The trainable variables are w1, b1, w2, and b2. Additionally, the following operations have been imported for you: 
# keras.activations.relu() and keras.layers.Dropout().

# Define the model
def model(w1, b1, w2, b2, features = borrower_features):
	# Apply relu activation functions to layer 1
	layer1 = keras.activations.relu(matmul(features, w1) + b1)
    # Apply dropout rate of 0.25
	dropout = keras.layers.Dropout(0.25)(layer1)
	return keras.activations.sigmoid(matmul(dropout, w2) + b2)

# Define the loss function
def loss_function(w1, b1, w2, b2, features = borrower_features, targets = default):
	predictions = model(w1, b1, w2, b2)
	# Pass targets and predictions to the cross entropy loss
	return keras.losses.binary_crossentropy(targets, predictions)

# One of the benefits of using tensorflow is that you have the option to customize models down to the linear algebraic-level, 
# as we've shown in the last two exercises. If you print w1, you can see that the objects we're working with are simply tensors.

# Training neural networks with TensorFlow
# In the previous exercise, you defined a model, model(w1, b1, w2, b2, features), and a loss function, 
# loss_function(w1, b1, w2, b2, features, targets), both of which are available to you in this exercise. You will now train the 
# model and then evaluate its performance by predicting default outcomes in a test set, which consists of test_features and 
# test_targets and is available to you. The trainable variables are w1, b1, w2, and b2. Additionally, the following operations 
# have been imported for you: keras.activations.relu() and keras.layers.Dropout().

# Train the model
for j in range(100):
    # Complete the optimizer
	opt.minimize(lambda: loss_function(w1, b1, w2, b2), 
                 var_list=[w1, b1, w2, b2])

# Make predictions with model using test features
model_predictions = model(w1, b1, w2, b2, test_features)

(test_targets) = mnist.load_data()
from sklearn.metrics import confusion_matrix
# Construct the confusion matrix
confusion_matrix(test_targets, model_predictions)

# he diagram shown is called a ``confusion matrix.'' The diagonal elements show the number of correct predictions. 
# The off-diagonal elements show the number of incorrect predictions. We can see that the model performs reasonably-well, 
# but does so by overpredicting non-default. This suggests that we may need to train longer, tune the model's hyperparameters, 
# or change the model's architecture.

#The sequential model in Keras
# In chapter 3, we used components of the keras API in tensorflow to define a neural network, but we stopped short of using its 
# full capabilities to streamline model definition and training. In this exercise, you will use the keras sequential model API 
# to define a neural network that can be used to classify images of sign language letters. You will also use the .summary() 
# method to print the model's architecture, including the shape and number of parameters associated with each layer.
# Note that the images were reshaped from (28, 28) to (784,), so that they could be used as inputs to a dense layer. 
# Additionally, note that keras has been imported from tensorflow for you.

# Define a Keras sequential model
model = keras.Sequential()

# Define the first dense layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the second dense layer
model.add(keras.layers.Dense(8,activation='relu'))

# Define the output layer
model.add(keras.layers.Dense(4,activation='softmax'))

# Print the model architecture
print(model.summary())
m1_inputs = model

# Notice that we've defined a model, but we haven't compiled it. The compilation step in keras allows us to set the 
# optimizer, loss function, and other useful training parameters in a single line of code. Furthermore, the .summary() 
# method allows us to view the model's architecture.

# Compiling a sequential model
# In this exercise, you will work towards classifying letters from the Sign Language MNIST dataset; however, you will 
# adopt a different network architecture than what you used in the previous exercise. There will be fewer layers, but 
# more nodes. You will also apply dropout to prevent overfitting. Finally, you will compile the model to use the adam 
# optimizer and the categorical_crossentropy loss. You will also use a method in keras to summarize your model's 
# architecture. Note that keras has been imported from tensorflow for you and a sequential keras model has been defined as model.

# Define the first dense layer
model.add(keras.layers.Dense(16, activation='sigmoid', input_shape=(784,)))

# Apply dropout to the first layer's output
model.add(keras.layers.Dropout(0.25))

# Define the output layer
model.add(keras.layers.Dense(4,activation='softmax'))

# Compile the model
model.compile('adam', loss='categorical_crossentropy')

# Print a model summary
print(model.summary())
m2_inputs = model

# You've now defined and compiled a neural network using the keras sequential model. Notice that printing the .summary() 
# method shows the layer type, output shape, and number of parameters of each layer.

# Defining a multiple input model
# In some cases, the sequential API will not be sufficiently flexible to accommodate your desired model architecture and 
# you will need to use the functional API instead. If, for instance, you want to train two models with different architectures 
# jointly, you will need to use the functional API to do this. In this exercise, we will see how to do this. We will also use 
# the .summary() method to examine the joint model's architecture.
# Note that keras has been imported from tensorflow for you. Additionally, the input layers of the first and second models 
# have been defined as m1_inputs and m2_inputs, respectively. Note that the two models have the same architecture, but one 
# of them uses a sigmoid activation in the first layer and the other uses a relu.

# For model 1, pass the input layer to layer 1 and layer 1 to layer 2
m1_layer1 = keras.layers.Dense(12, activation='sigmoid')(m1_inputs)
m1_layer2 = keras.layers.Dense(4, activation='softmax')(m1_layer1)

# For model 2, pass the input layer to layer 1 and layer 1 to layer 2
m2_layer1 = keras.layers.Dense(12, activation='relu')(m2_inputs)
m2_layer2 = keras.layers.Dense(4, activation='softmax')(m2_layer1)

# Merge model outputs and define a functional model
merged = keras.layers.add([m1_layer2, m2_layer2])
model = keras.Model(inputs=[m1_inputs, m2_inputs], outputs=merged)

# Print a model summary
print(model.summary())

# Notice that the .summary() method yields a new column: connected to. This column tells you how layers 
# connect to each other within the network. We can see that dense_2, for instance, is connected to the 
# input_2 layer. We can also see that the add layer, which merged the two models, connected to both dense_1 and dense_3.

# Training with Keras
# In this exercise, we return to our sign language letter classification problem. We have 2000 images of four 
# letters--A, B, C, and D--and we want to classify them with a high level of accuracy. We will complete all 
# parts of the problem, including the model definition, compilation, and training.
# Note that keras has been imported from tensorflow for you. Additionally, the features are available as sign_language_features 
# and the targets are available as sign_language_labels.

# Define a sequential model
model=keras.Sequential()

# Define a hidden layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the output layer
model.add(keras.layers.Dense(4,activation='softmax'))

# Compile the model
model.compile('SGD', loss='categorical_crossentropy')

# Complete the fitting operation
(sign_language_features, sign_language_labels)=mnist.load_data()
model.fit(sign_language_features, sign_language_labels, epochs=5)

# You probably noticed that your only measure of performance improvement was the value of the loss function 
# in the training sample, which is not particularly informative. You will improve on this in the next exercise.

# Metrics and validation with Keras
# We trained a model to predict sign language letters in the previous exercise, but it is unclear how successful we were in 
# doing so. In this exercise, we will try to improve upon the interpretability of our results. Since we did not use a validation 
# split, we only observed performance improvements within the training set; however, it is unclear how much of that was due to 
# overfitting. Furthermore, since we did not supply a metric, we only saw decreases in the loss function, which do not have any 
# clear interpretation.
# Note that keras has been imported for you from tensorflow.

# Define sequential model
model = keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(32, activation='sigmoid', input_shape=(784,)))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# Set the optimizer, loss function, and metrics
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Add the number of epochs and the validation split
model.fit(sign_language_features, sign_language_labels, epochs=10, validation_split=0.1)
small_model = model

# With the keras API, you only needed 14 lines of code to define, compile, train, and validate a model. 
# You may have noticed that your model performed quite well. In just 10 epochs, we achieved a classification 
# accuracy of over 90% in the validation sample!

# Overfitting detection
# In this exercise, we'll work with a small subset of the examples from the original sign language letters dataset. 
# A small sample, coupled with a heavily-parameterized model, will generally lead to overfitting. This means that 
# your model will simply memorize the class of each example, rather than identifying features that generalize to 
# many examples.
# You will detect overfitting by checking whether the validation sample loss is substantially higher than the training 
# sample loss and whether it increases with further training. With a small sample and a high learning rate, the model 
# will struggle to converge on an optimum. You will set a low learning rate for the optimizer, which will make it easier 
# to identify overfitting.
# Note that keras has been imported from tensorflow.

# Define sequential model
model = keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(1024,activation='relu', input_shape=(784,)))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# Finish the model compilation
model.compile(optimizer=keras.optimizers.Adam(lr=0.001), 
              loss='categorical_crossentropy', metrics=['accuracy'])

# Complete the model fit operation
model.fit(sign_language_features, sign_language_labels, epochs=50, validation_split=0.5)
large_model = model

# You may have noticed that the validation loss, val_loss, was substantially higher than the training loss, loss. 
# Furthermore, if val_loss started to increase before the training process was terminated, then we may have overfitted. 
# When this happens, you will want to try decreasing the number of epochs.

# Evaluating models
# Two models have been trained and are available: large_model, which has many parameters; and small_model, which has fewer 
# parameters. Both models have been trained using train_features and train_labels, which are available to you. A separate 
# test set, which consists of test_features and test_labels, is also available.
# Your goal is to evaluate relative model performance and also determine whether either model exhibits signs of overfitting. 
# You will do this by evaluating large_model and small_model on both the train and test sets. For each model, you can do this 
# by applying the .evaluate(x, y) method to compute the loss for features x and labels y. You will then compare the four losses 
# generated.

# Evaluate the small model using the train data
small_train = small_model.evaluate(train_features, train_labels)

# Evaluate the small model using the test data
small_test = small_model.evaluate(test_features, test_labels)

# Evaluate the large model using the train data
large_train = large_model.evaluate(train_features, train_labels)

# Evaluate the large model using the test data
large_test = large_model.evaluate(test_features, test_labels)

# Print losses
print('\n Small - Train: {}, Test: {}'.format(small_train, small_test))
print('Large - Train: {}, Test: {}'.format(large_train, large_test))

# Notice that the gap between the test and train set losses is high for large_model, suggesting that overfitting may 
# be an issue. Furthermore, both test and train set performance is better for large_model. This suggests that we may 
# want to use large_model, but reduce the number of training epochs.

# Preparing to train with Estimators
# For this exercise, we'll return to the King County housing transaction dataset from chapter 2. We will again develop 
# and train a machine learning model to predict house prices; however, this time, we'll do it using the estimator API.
# Rather than completing everything in one step, we'll break this procedure down into parts. We'll begin by defining the 
# feature columns and loading the data. In the next exercise, we'll define and train a premade estimator. Note that 
# feature_column has been imported for you from tensorflow. Additionally, numpy has been imported as np, and the Kings 
# County housing dataset is available as a pandas DataFrame: housing.
import pandas as pd
from tensorflow import feature_column
housing = pd.read_csv("kc_house_data.csv")

# Define feature columns for bedrooms and bathrooms
bedrooms = feature_column.numeric_column("bedrooms")
bathrooms = feature_column.numeric_column("bathrooms")

# Define the list of feature columns
feature_list = [bedrooms, bathrooms]

def input_fn():
	# Define the labels
	labels = np.array(housing['price'])
	# Define the features
	features = {'bedrooms':np.array(housing['bedrooms']), 
                'bathrooms':np.array(housing['bathrooms'])}
	return features, labels

# Defining Estimators
# In the previous exercise, you defined a list of feature columns, feature_list, and a data input function, 
# input_fn(). In this exercise, you will build on that work by defining an estimator that makes use of input data.
from tensorflow import estimator 

# Define the model and set the number of steps
model = estimator.DNNRegressor(feature_columns=feature_list, hidden_units=[2,2])
model.train(input_fn, steps=1)

# Define the model and set the number of steps
model = estimator.LinearRegressor(feature_columns=feature_list)
model.train(input_fn, steps=2)