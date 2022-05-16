# Classification of points (images) lying on one side of the curve - the decision boundary:
#
#      class 0
#
#                      (4,4) o-------------------------------------
#                           /
#                          /
#                         /
# -----------------------o (1,1)
#                               
#      class 1
#         
#
import time
import os.path
import pdb
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf
print(tf.__version__)

num_of_features = 2
num_of_epochs = 1500
num_to_show = 50

sess = tf.compat.v1.InteractiveSession()

x = tf.compat.v1.placeholder(tf.float32, shape=[None, num_of_features])  # place for input vectors
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])               # place for desired output of ANN

IW = ....................                                    # 1-st level weights - trainable version
b1 = ....................                                    # 1-st level biases -||-

my_tensor = tf.constant([.........],dtype=float)
IW_a = tf.Variable(my_tensor)                                # 1-st level weights - analytical version
b1_a = tf.Variable(tf.constant([........],dtype=float))      # 1-st level biases -||- 

h1_a = .................................                     # analytical version
h1 = ..................................                      # output values from 1-st level


LW21 = ..................           # 2-nd level weights values
b2 = ....................                                          # 2-nd level bias values

LW21_a = ..................           # 2-nd level weights values - analytical version
b2_a = ....................  

# LW32 = .................           # maybe 3-d layer
# b3 = ...................

y_a = ......................                   # output flom ANN - analytical version  
y = ........................                   # output from ANN (single value using sigmoidal act.funct in range (0,1))



mean_square_error = tf.reduce_mean(tf.reduce_sum((y_ - y)*(y_ - y), reduction_indices=[1]))          # MSE loss function 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y) + y*tf.log(y_+0.001), reduction_indices=[1])) # full cross-entropy loss function

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(mean_square_error)   # training method, step value, loss function 
                                                                              # You can choose loss function

init = tf.global_variables_initializer() 

sess = tf.Session()
sess.run(init)

# the training process:
# ........................................................
# ........................................................


# drawing decision boundary:
X1, X2 = np.meshgrid(np.linspace(0, 8, 120), np.linspace(0,8, 120))  # grid of points in 2D plane
P = np.stack((X1.flatten(),X2.flatten()), axis=1)                    # points formated for ANN input
Y = sess.run(y_a, feed_dict={x: P})                                    # ANN outputs for flatten grid  points
Z = np.reshape(Y,X1.shape)                                           # reshaping to shape of grid 
plt.contourf(X1, X2, Z, levels=[0.5, 1.0])                           # curve for level=0.5 - a decision boundary, shaded class 1 area
plt.title('analytical method')
plt.show()


# drawing 3D mesh:
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap=cm.viridis)  
plt.title('analytical method')       
plt.show()


# drawing 3D mesh:
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
Y = sess.run(y, feed_dict={x: P})                                    # ANN outputs for flatten grid  points
Z = np.reshape(Y,X1.shape)
ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap=cm.viridis)  
plt.title('learning method')       
plt.show()