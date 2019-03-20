#####################################################################################
# University of Applied Sciences Munich
# Schöttl: Machine Learning and Deep Learning (c) 2019
# DL1: Linear Models, First Steps in TF
#####################################################################################


import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
tf.enable_eager_execution()

def plot_1d_features(X,y,w=-999):
    plt.plot(X,y,'.')
    if w!=-999:
        x0 = 0
        x1 = 100
        y0 = w[0]*x0
        y1 = w[0]*x1
        plt.plot([x0, x1], [y0, y1], 'r')
    # plt.pause(0.1)    # comment in this line if the plot doesn't show up
    plt.show()

# Load the data set. It contains home prices of districts in/around Boston.
# The used feature in X is the proportion of residential land zoned for lots over 25,000 sq.ft. 
# The response is the mean value (median) of homes in this district in $1000's
(X_train,y_train), (X_test,y_test) = tf.keras.datasets.boston_housing.load_data()
X_train = X_train[:,[1]]  # only use feauture 1
X_test = X_test[:,[1]]    # only use feauture 1

###########################
# 1. Explore the dataset
# TODO: 
# - print the shapes of the training features and responses
# - print the rank of the training features
# - define a variable N which contains the number of samples of the data set
# - comment in (and out afterwards) the plot command at the end of this section

print(X_train.shape)
print(y_train.shape)
print(len(X_train.shape))
N = X_train.shape[0]
#plot_1d_features(X_train, y_train)
###########################

###########################
# 2. Training:
# TODO:
# - calculate the pseudo-inverse P=(X'*X)^-1*X' of the training feature matrix
# - calculate the weights w and print it
# - comment in (and out afterwards) the plot command at the end of this section
y_train = tf.expand_dims(y_train,1)   # y_train is rank 1 (size [N]), extend the rank (shape [N,1]) to perform matrix calculation 
X_train_t = tf.transpose(X_train)
P = tf.linalg.inv(X_train_t @ X_train) @ X_train_t
w = P @ y_train


#plot_1d_features(X_train, y_train, w)
###########################


###########################
# 3. inference:
# TODO:
# - use the test data X_test and infer the response y_pred (hint: use y_pred = y_pred[:,0] 
#   after the calculation to get out a vector instead of a Nx1-matrix)
# - print the predicted and the true house prices y_test. Compare (and don't be too disappointed).
y_pred = X_test * w
plt.plot(X_test, y_test, 'b.', label='acutal')
plt.plot(X_test, y_pred[:,0], 'r.', label='predicted')
plt.legend(loc='lower right')
plt.title('Test run')
plt.show()

###########################


###########################
# Further work:
# TODO:
# repeat the steps above for the whole dataset with 13 features 
(X_train,y_train), (X_test,y_test) = tf.keras.datasets.boston_housing.load_data()


###########################