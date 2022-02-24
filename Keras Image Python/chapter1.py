# Images as data: visualizations
# Import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
# Load the image
data = plt.imread('bricks.png')
# green value
#data[:,:,1]=0
# blue -||-
#data[:,:,2]=0
# red  -||-
#data[:,:,0]=0

# green value
# only on 10x10 top left square
data[0:10,0:10,1]=0
# blue -||-
data[0:10,0:10,2]=0
# red  -||-
data[0:10,0:10,0]=1

# Display the image
plt.imshow(data)
plt.show()

# Task 2 
# labels = ["t-shirt","shoe",]
# categories = np.array(["tshirt","dress","shoe",])
# n_categories = 3
# one_labels = np.zeros(len(labels),categories)
# for ii in range(len(labels)): 
#  jj = np.where(categories==labels[ii])
#  one_labels[ii,jj]=1
labels = ['shoe', 'shirt', 'shoe', 'shirt', 'dress', 'dress', 'dress']
# The number of image categories
n_categories = 3
# The unique values of categories in the data
categories = np.array(["shirt", "dress", "shoe"])
# Initialize ohe_labels as all zeros
ohe_labels = np.zeros((len(labels), n_categories))
# Loop over the labels
for ii in range(len(labels)):
    # Find the location of this label in the categories variable
    jj = np.where(categories==labels[ii])
    # Set the corresponding zero to one
    ohe_labels[ii,jj] = 1

predictions = np.array([[0., 0., 1.],[0., 1., 0.],[0., 0., 1.],[1., 0., 0.],[0., 0., 1.],[1., 0., 0.],[0., 0., 1.],[0., 1., 0.]])
test_labels = np.array([[0., 0., 1.],[0., 1., 0.],[0., 0., 1.],[0., 1., 0.],[0., 0., 1.],[0., 0., 1.],[0., 0., 1.],[0., 1., 0.]])
# Calculate the number of correct predictions
number_correct = (predictions*test_labels).sum()
print("number of correct predctions: "+ str(number_correct))
# Calculate the proportion of correct predictions
proportion_correct = number_correct/test_labels.sum()
print("percent of correct predctions: "+ str(proportion_correct*100)+"%")

# add 2 hidden layers -> input shape (784,) --> 10 10, relu relu, output -> 3 softmax
# compile -> optimizer adam, categorical_crossentropy, metrics=['accuracy']
# train data . reshape ((50,784))
# fit train_data, train_labels, validation_split=0.2, epochs = 3 
# model.evaluate[test_data,test_labels]
# Imports components from Keras
# Initializes a sequential model
train_data = np.array([[0., 0., 0., 1.,0., 0., 0., 0.]])
train_labels = np.array([[0., 1., 0.]])
test_data = np.array([[0., 0., 0., 1.,0.,0., 0., 0., 0.]])
test_labels = np.array([[0., 0., 1.]])
model = Sequential()
# First layer
model.add(Dense(10, activation='relu', input_shape=(784,)))
# Second layer
model.add(Dense(10, activation='relu'))
# Output layer
model.add(Dense(3,activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Reshape the data to two-dimensional array
train_data = train_data.reshape((50,784))
# Fit the model
model.fit(train_data, train_labels, validation_split=0.2, epochs=3)
# Reshape test data
test_data = test_data.reshape(10,784)
# Evaluate the model
model.evaluate(test_data, test_labels)