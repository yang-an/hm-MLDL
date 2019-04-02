########################################################################
# University of Applied Sciences Munich                                 
# Department of Electrical Engineering and Information Sciences         
# Machine Learning and Deep Learning, (c) Alfred Schöttl                
#-----------------------------------------------------------------------
# Assigment DL2b                                                        
########################################################################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.enable_eager_execution()

# Alpha parameter for gradient descent
ALPHA = 1e-5
# Number of optimization runs
N_EPOCHS = 20

# Load the data set. It contains home prices of districts in/around
# Boston. The used feature in X is the proportion of residential land
# zoned for lots over 25,000 sq.ft. The response is the mean value
# (median) of homes in this district in $1000's
(X_train,y_train), (X_test,y_test) = \
    tf.keras.datasets.boston_housing.load_data()
# we change the dataset so that y contains binary data (0/1) only:
# class 0: affordable districts
# class 1: expensive districts
y_train = (y_train>25).astype(np.int)
y_test = (y_test>25).astype(np.int)

n = X_train.shape[1]            # number of features

###########################
# 1. Define the model and the loss function
# TODO: fill in the logistic model equation and the formula for the
#       total loss function
#       use tf.nn.sigmoid and the binary cross entropy formula
def model(X,w):
    y = tf.nn.sigmoid(X @ tf.expand_dims(w, 1))
    return y[:, 0]

def loss(y_true, y_pred):
    l = tf.reduce_mean(
        -y_true * tf.log(y_pred) - (1-y_true) * tf.log(1 - y_pred),
        0
    )
    return l
###########################

###########################
# 2. Define a weights variable with shape [n] and zero inital values
w = tf.get_variable("w", [n], initializer=tf.zeros_initializer()) 
###########################

###########################
# 3. Training by steepest descent:
losses = []

for k in range(N_EPOCHS):
    with tf.GradientTape() as tape:
        y_pred = model(X_train, w)
        l = loss(y_train, y_pred)
    dldw = tape.gradient(l,w)
    w.assign(w - ALPHA * dldw)

    print('[#{:2d}] loss: {}'.format(k, float(l)))
    losses.append(l)
###########################

###########################
# 4. Predict with the trained model:
# TODO: use new samples to predict
y_pred = model(X_test, w)
#print('Predicted value for features ', X_new, ':', float(y_pred))
###########################

###########################
# 5. Training by Keras:
# TODO: Repeat steps 1-4 with Keras. There are slight differences in the
# implementation, so don't be surprised if the result is different!
def model2():
    inputs = tf.keras.layers.Input(shape=[n]) 
    y_pred = tf.keras.layers.Dense(1, activation='sigmoid')(inputs)
    mdl = tf.keras.Model(inputs=inputs, outputs=y_pred) 
    return mdl

mdl = model2()
# TODO: train the model
opt = tf.keras.optimizers.SGD(ALPHA)
mdl.compile(optimizer=opt, loss='binary_crossentropy')

history = mdl.fit(X_train, y_train, epochs=N_EPOCHS)

X_new = X_test[[0],:]
# TODO: predict
y_pred_keras = mdl.predict(X_test)
#print('Predicted value for features ', X_new, ':', y_pred)

###########################
# Plots
plt.clf()

plt.subplot(3, 1, 1)
plt.title('Gradient descent')
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.plot(losses)

plt.subplot(3, 1, 2)
plt.title('Keras SGD')
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.plot(history.history['loss'])

index = range(len(y_test))
plt.subplot(3, 1, 3)
plt.title('Predicion vs true value')
plt.plot(index, y_pred, 'r+', label='gradDesc')
plt.plot(index, y_pred_keras, 'b+', label='keras')
plt.plot(index, y_test, '.', label='true')

plt.legend()
plt.tight_layout()
plt.get_current_fig_manager().window.showMaximized()
plt.show()