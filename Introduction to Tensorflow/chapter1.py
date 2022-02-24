# Exc 1 

# Import constant from TensorFlow
from tensorflow import constant
import numpy as np

credit_numpy = np.array([[ 2.0000e+00,  1.0000e+00,  2.4000e+01,  3.9130e+03],
       [ 2.0000e+00,  2.0000e+00,  2.6000e+01,  2.6820e+03],
       [ 2.0000e+00,  2.0000e+00,  3.4000e+01,  2.9239e+04]])

# Convert the credit_numpy array into a tensorflow constant
credit_constant = constant(credit_numpy)

# Print constant datatype
print('\n The datatype is:', credit_constant.dtype)
# Print constant shape
print('\n The shape is:', credit_constant.shape)

# Defining variables
# Unlike a constant, a variable's value can be modified. 
# This will be useful when we want to train a model by updating its parameters.
# Let's try defining and printing a variable. We'll then convert the variable to a numpy array, 
# print again, and check for differences. Note that Variable(), which is used to create a variable tensor, 
# has been imported from tensorflow and is available to use in the exercise.

from tensorflow import Variable

# Define the 1-dimensional variable A1
A1 = Variable([1, 2, 3, 4])

# Print the variable A1
print('\n A1: ', A1)

# Convert A1 to a numpy array and assign it to B1
B1 = A1.numpy()

# Print B1
print('\n B1: ', B1)

# Performing element-wise multiplication
# Element-wise multiplication in TensorFlow is performed using two tensors with identical shapes. 
# This is because the operation multiplies elements in corresponding positions in the two tensors. 
# An example of an element-wise multiplication, denoted by the  symbol, is shown below:
# In this exercise, you will perform element-wise multiplication, paying careful attention to the shape of the tensors you multiply. 
# multiply(), constant(), and ones_like()
from tensorflow import multiply, ones_like

# Define tensors A1 and A23 as constants
A1 = constant([1, 2, 3, 4])
A23 = constant([[1, 2, 3], [1, 6, 4]])

# Define B1 and B23 to have the correct shape
B1 = ones_like(A1)
B23 = ones_like(A23)

# Perform element-wise multiplication
C1 = multiply(A1,B1)
C23 = multiply(A23,B23)

# Print the tensors C1 and C23
print('\n C1: {}'.format(C1.numpy()))
print('\n C23: {}'.format(C23.numpy()))
# Notice how performing element-wise multiplication with tensors of ones leaves the original tensors unchanged.

# Making predictions with matrix multiplication.
# In later chapters, you will learn to train linear regression models. 
# This process will yield a vector of parameters that can be multiplied by the input data to generate predictions. 
# In this exercise, you will use input data, features, and a target vector, bill, which are taken 
# from a credit card dataset we will use later in the course.
# The matrix of input data, features, contains two columns: education level and age. 
# The target vector, bill, is the size of the credit card borrower's bill.
# Since we have not trained the model, you will enter a guess for the values 
# of the parameter vector, params. You will then use matmul() to perform matrix 
# multiplication of features by params to generate predictions, billpred, which 
# you will compare with bill. Note that we have imported matmul() and constant().

# Define features, params, and bill as constants
features = constant([[2, 24], [2, 26], [2, 57], [1, 37]])
params = constant([[1000], [150]])
bill = constant([[3913], [2682], [8617], [64400]])

# Compute billpred using features and params
from tensorflow import matmul
billpred = matmul(features,params)

# Compute and print the error
error = bill - billpred
print(error.numpy())
# Understanding matrix multiplication will make things simpler when we start making predictions with linear models.

# Reshaping tensors
# Later in the course, you will classify images of sign language letters using a neural network. 
# In some cases, the network will take 1-dimensional tensors as inputs, but your data will come 
# in the form of images, which will either be either 2- or 3-dimensional tensors, depending on 
# whether they are grayscale or color images.
# The figure below shows grayscale and color images of the sign language letter A. The two images have been imported 
# for you and converted to the numpy arrays gray_tensor and color_tensor. Reshape these arrays into 1-dimensional 
# vectors using the reshape operation, which has been imported for you from tensorflow. Note that the shape of 
# gray_tensor is 28x28 and the shape of color_tensor is 28x28x3.
from tensorflow import reshape
gray_tensor = [0]
color_tensor = [0]

# Reshape the grayscale image tensor into a vector
gray_vector = reshape(gray_tensor, (784, 1))

# Reshape the color image tensor into a vector
color_vector = reshape(color_tensor, (2352, 1))

# Optimizing with gradients
# You are given a loss function, , which you want to minimize. You can do this by computing the slope using the GradientTape() 
# operation at different values of x. If the slope is positive, you can decrease the loss by lowering x. If it is negative, 
# you can decrease it by increasing x. This is how gradient descent works.
# In practice, you will use a high level tensorflow operation to perform gradient descent automatically. In this exercise, 
# however, you will compute the slope at x values of -1, 1, and 0. The following operations are available: 
# GradientTape(), multiply(), and Variable().

from tensorflow import GradientTape

def compute_gradient(x0):
  	# Define x as a variable with an initial value of x0
	x = Variable(x0)
	with GradientTape() as tape:
		tape.watch(x)
        # Define y using the multiply operation
		y = multiply(x,x)
    # Return the gradient of y with respect to x
	return tape.gradient(y, x).numpy()

# Compute and print gradients at x = -1, 1, and 0
print(compute_gradient(-1.0))
print(compute_gradient(1.0))
print(compute_gradient(0.0))

# The slope is positive at x = 1, which means that we can lower the loss by reducing x. 
# The slope is negative at x = -1, which means that we can lower the loss by increasing x. 
# The slope at x = 0 is 0, which means that we cannot lower the loss by either increasing 
# or decreasing x. This is because the loss is minimized at x = 0.

# --- 
# Working with image data
# You are given a black-and-white image of a letter, which has been encoded as a tensor, letter. 
# You want to determine whether the letter is an X or a K. You don't have a trained neural network, 
# but you do have a simple model, model, which can be used to classify letter.
# The 3x3 tensor, letter, and the 1x3 tensor, model, are available in the Python shell. You can 
# determine whether letter is a K by multiplying letter by model, summing over the result, and 
# then checking if it is equal to 1. As with more complicated models, such as neural networks, 
# model is a collection of weights, arranged in a tensor.
from tensorflow import reduce_sum

# Reshape model from a 1x3 to a 3x1 tensor
model = reshape(model, (3, 1))

# Multiply letter by model
output = matmul(letter, model)

# Sum over output and print prediction using the numpy method
prediction = reduce_sum(output)
print(prediction.numpy())