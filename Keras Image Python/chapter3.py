# Going deeper into keras 
# Task 1 
# Adding another convolutional
# Layer with fatten 
img_rows=28
img_cols=28
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
model = Sequential()
# Add a convolutional layer (15 units)
model.add(Conv2D(15,kernel_size=2,activation='relu',input_shape=(img_rows,img_cols,1)))
# Add another convolutional layer (5 units)
model.add(Conv2D(5,kernel_size=2,activation='relu'))
# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

# Continuation of the model
# Task 2 
# Compile model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
# Fit the model to training data 
model.fit(train_data, train_labels, 
          validation_split=0.2, 
          epochs=3, batch_size=10)
# Evaluate the model on test data
model.evaluate(test_data, test_labels, batch_size=10)

# Task 3 
# Summarize 
# CNN model
model = Sequential()
model.add(Conv2D(10, kernel_size=2, activation='relu', 
                 input_shape=(28, 28, 1)))
model.add(Conv2D(10, kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
# Summarize the model 
model.summary()
#    [Out:]
#    Model: "sequential"
#    _________________________________________________________________
#    Layer (type)                 Output Shape              Param #   
#    =================================================================
#    conv2d (Conv2D)              (None, 27, 27, 10)        50        
#    _________________________________________________________________
#    conv2d_1 (Conv2D)            (None, 26, 26, 10)        410       
#    _________________________________________________________________
#    flatten (Flatten)            (None, 6760)              0         
#    _________________________________________________________________
#    dense (Dense)                (None, 3)                 20283     
#    =================================================================
#    Total params: 20,743
#    Trainable params: 20,743
#    Non-trainable params: 0
#    _________________________________________________________________

# Decreasing the number of parameters
# Task 1a 
# Result placeholder
import numpy as np
result = np.zeros((im.shape[0]//2, im.shape[1]//2))
# Pooling operation
for ii in range(result.shape[0]):
    for jj in range(result.shape[1]):
        result[ii, jj] = np.max(im[ii*2:ii*2+2,jj*2:jj*2+2])

# Task 2a 
# Add a convolutional layer
model.add(Conv2D(15, kernel_size=2, activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))
# Add a pooling operation
model.add(MaxPool2D(2))
# Add another convolutional layer
model.add(Conv2D(5,kernel_size=2,activation='relu'))
# Flatten and feed to output layer
model.add(MaxPool2D(2))
model.add(Flatten())
model.add(Dense(3,activation='softmax'))
model.summary()
# Total params 923 ! 

# Task 3a
# Compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Fit to training data
model.fit(train_data,train_labels,epochs=3,batch_size=10,validation_split=0.2)
# Evaluate on test data 
model.evaluate(test_data,test_labels,batch_size=10)