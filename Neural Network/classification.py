# example of 2D plane points classification problem (JD, 2020)
import time
import os.path
import pdb
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# training data:
#points = np.array([[1,5],[3,5],[1,3],[2,3],[3,1],[5,3],[7,1],[3,3],[2,1],[5,5]], dtype=float)
#class_labels = np.transpose(np.array([[1,1,1,1,0,0,0,0,1,1]], dtype = float))
points = np.array([[0,6],[0,7],[2,2],[4,3],[4,4],[9,6],[9,7],[1,4],[2,5],[5,1],[6,7],[7,4],[7,6],[8,3]], dtype=float)
class_labels = np.transpose(np.array([[1,0,1,1,0,0,0,0,0,0,1,1,1,1]], dtype = int))

num_of_examples, num_of_features = points.shape
range_min = np.amin(points,axis=0)       # for plotting 
range_max = np.amax(points,axis=0)
print ("number of examples = "+str(num_of_examples)+ ", number of features = "+str(num_of_features) )

ind0 = np.where(class_labels == 0)       # indexes of examples which belong to class 0
ind1 = np.where(class_labels == 1)

line1 = plt.plot(np.transpose(points[ind0[0],0]), np.transpose(points[ind0[0],1]), 'ro', label = 'class 0')
line2 = plt.plot(np.transpose(points[ind1[0],0]), np.transpose(points[ind1[0],1]), 'bs', label = 'class 1')
plt.title("points in 2d plane (close to continue ...)") 
plt.xlabel("x1") 
plt.ylabel("x2") 
plt.legend()
plt.show()


nneu = [10]                     # number of neurons in the 1-st level of ANN
                                         # number of neurons in the last layer == 1 
num_of_epochs = 1500
num_to_show = 50

sess = tf.compat.v1.InteractiveSession()

x = tf.compat.v1.placeholder(tf.float32, shape=[None, num_of_features])  # place for input vectors
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])               # place for desired output of ANN


IW = tf.Variable(tf.compat.v1.truncated_normal([num_of_features, nneu[0]], stddev=0.1))  # 1-st level weights initialized with normal distribution
b1 = tf.Variable(tf.constant(0.1, shape=[nneu[0]]))                            # 1-st level biases -||- 

h1 = tf.nn.tanh(tf.matmul(x, IW) + b1)                                    # output values from 1-st level (using hyperbolic tangent activation func.)


LW21 = tf.Variable(tf.compat.v1.truncated_normal([nneu[0],1], stddev=0.1))               # 2-nd level weights values
b2 = tf.Variable(tf.zeros([1]))                                           # 2-nd level bias values

#LW32 = ...                                                     # 3-nd level weights values
#b3 = ...                                                       # 3-nd level bias values

y = tf.nn.sigmoid(tf.matmul(h1, LW21) + b2)                               # output from ANN (single value using sigmoidal act.funct in range (0,1))


mean_square_error = tf.reduce_mean(tf.reduce_sum((y_ - y)*(y_ - y), axis=1))          # MSE loss function 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.math.log(y) + y*tf.math.log(y_+0.001), axis=1)) # full cross-entropy loss function

train_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(mean_square_error)   # training method, step value, loss function 
                                                                              # You can choose loss function

init = tf.compat.v1.global_variables_initializer() 

sess = tf.compat.v1.Session()
sess.run(init)

# the training process:
for epoch in range(num_of_epochs+1):
    sess.run(train_step, feed_dict={x: points, y_: class_labels})     # ses.run using dictionary with whole training data
    if epoch % num_to_show == 0:
        wrong_prediction = tf.greater(tf.abs(y-y_),0.5)               # vector of classification errors
        error = tf.reduce_mean(tf.cast(wrong_prediction, tf.float32)) # mean classification error

        print("\nafter "+str(epoch)+" epoch")
        print("MSE error = " + str(sess.run(mean_square_error, feed_dict={x: points, y_: class_labels}))) 
        print("Cross Entropy error = " + str(sess.run(cross_entropy, feed_dict={x: points, y_: class_labels}))) 
        print("training classification error = " + str(sess.run(error, feed_dict={x: points, y_: class_labels}))) 


# drawing points:
line1 = plt.plot(np.transpose(points[ind0[0],0]), np.transpose(points[ind0[0],1]), 'ro', label = 'class 0')
line2 = plt.plot(np.transpose(points[ind1[0],0]), np.transpose(points[ind1[0],1]), 'bs', label = 'class 1')
plt.title("points in 2d plane - decision boundary") 
plt.xlabel("x1") 
plt.ylabel("x2") 
plt.legend()

# drawing decision boundary:
X1, X2 = np.meshgrid(np.linspace(range_min[0]- 0.3, range_max[0]+0.3, 120), np.linspace(range_min[1]- 0.3, range_max[1]+0.3, 80))  # grid of points in 2D plane
P = np.stack((X1.flatten(),X2.flatten()), axis=1)                    # points formated for ANN input
Y = sess.run(y, feed_dict={x: P})                                    # ANN outputs for flatten grid  points
Z = np.reshape(Y,X1.shape)                                           # reshaping to shape of grid 
plt.contourf(X1, X2, Z, levels=[0.5, 1.0])                                 # curve for level=0.5 - a decision boundary
plt.show()


# drawing 3D mesh:
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap=cm.viridis)         
plt.show()
