#############################################################################################
# University of Applied Sciences Munich                                                     #
# Department of Electrical Engineering and Information Sciences                             #
# Machine Learning and Deep Learning, (c) Alfred Schöttl                                    #
#-------------------------------------------------------------------------------------------#
# Assigment DL2b                                                                            #
#############################################################################################

import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

# Load the data set. It contains home prices of districts in/around Boston.
# The used feature in X is the proportion of residential land zoned for lots over 25,000 sq.ft. 
# The response is the mean value (median) of homes in this district in $1000's
(X_train,y_train), (X_test,y_test) = tf.keras.datasets.boston_housing.load_data()
# we change the dataset so that y contains binary data (0/1) only:
# class 0: affordable districts
# class 1: expensive districts
y_train = (y_train>25).astype(np.int)
y_test = (y_test>25).astype(np.int)

n = X_train.shape[1]            # number of features


###########################
# 1. Define the model and the loss function
# TODO: fill in the logistic model equation and the formula for the total loss function
#       use tf.nn.sigmoid and the binary cross entropy formula
def model(X,w):
    y = 
    return y[:,0]

def loss(y_true, y_pred):
    l = 
    return l
###########################


###########################
# 2. Define a weights variable with shape [n] and zero inital values
w = tf.get_variable("w", [n], initializer=tf.zeros_initializer()) 
###########################


alpha = 0.00001
n_epochs = 100


###########################
# 3. Training by steepest descent:
for k in range(n_epochs):
    with tf.GradientTape() as tape:
        y_pred = model(X_train, w)
        l = loss(y_train, y_pred)
    print('loss:', l)
    dldw = tape.gradient(l,w)
    w.assign(w - alpha*dldw)
###########################


###########################
# 4. Predict with the trained model:
X_new = X_test[[0],:]
# TODO: use new samples (here: the first test sample only) to predict
y_pred = 
print('Predicted value for features ', X_new, ':', y_pred)
###########################


alpha = 0.0001
n_epochs = 10

###########################
# 5. Training by Keras:
# TODO: Repeat steps 1-4 with Keras. There are slight differences in the
# implementation, so don't be surprised if the result is different!
def model2():
    inputs = 
    y_pred = 
    mdl = 
    return mdl

mdl = model2()
# TODO: train the model


X_new = X_test[[0],:]
# TODO: predict
y_pred = 
print('Predicted value for features ', X_new, ':', y_pred)
###########################

