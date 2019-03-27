#############################################################################################
# University of Applied Sciences Munich                                                     #
# Department of Electrical Engineering and Information Sciences                             #
# Machine Learning and Deep Learning, (c) Alfred Schöttl                                    #
#-------------------------------------------------------------------------------------------#
# Assigment DL2a                                                                            #
#############################################################################################

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
tf.enable_eager_execution()

# Load the data set. It contains home prices of districts in/around Boston.
# The used feature in X is the proportion of residential land zoned for lots over 25,000 sq.ft. 
# The response is the mean value (median) of homes in this district in $1000's
(X_train,y_train), (X_test,y_test) = tf.keras.datasets.boston_housing.load_data()
n = X_train.shape[1]            # number of features

###########################
# 1. Define the model and the loss function
# TODO: fill in the linear model equation and the formula for the total loss function
def model(X,w):
	y = X @ tf.expand_dims(w, 1)
	return y
	
def loss(y_true, y_pred):
	l = tf.transpose(y_true - y_pred) @ (y_true - y_pred) / (2 * len(X_train))
	return l
###########################


###########################
# 2. Define a weights variable with shape [n] and zero inital values
# TODO: use tf.get_variable to define a variable
w = tf.get_variable('weights', n, initializer=tf.zeros_initializer())
###########################

alpha = 0.000001
n_epochs = 100

###########################
# 3. Training by steepest descent:
# TODO: for each loop:
losses = []

for k in range(n_epochs):
    #  - compute the gradient dl/dw
	with tf.GradientTape() as tape:
		y_pred = model(X_train, w)
		l = loss(tf.expand_dims(y_train.astype('float32'), 1), y_pred)
	
	dldw = tape.gradient(l, w)
		
    #  - modify the weights according to the steepest descent algorithm
	w.assign(w - alpha * dldw)
	
	losses.append(float(l))

##########################

###########################
# 4. Predict with the trained model:
X_new = X_test[[0],:]
# TODO: use new samples (here: the first test sample only) to predict
y_pred = X_new @ tf.expand_dims(w, 1)
print('Predicted value for features ', X_new[0], ':', float(y_pred))
###########################

plt.subplot(2, 1, 1)
plt.title('Value of loss function')
plt.plot(losses, '.')
#plt.show()

###########################
# 5. Training and prediction by Keras:
# TODO: Repeat steps 1-4 with Keras. There are slight differences in the
# implementation, so don't be surprised if the result is different!
alpha = 0.000001
n_epochs = 100

def model2():
    inputs = tf.keras.layers.Input(shape=[n]) 
    y_pred = tf.keras.layers.Dense(1, activation='linear')(inputs)
    mdl = tf.keras.Model(inputs=inputs, outputs=y_pred) 
    return mdl

mdl = model2()
# TODO: train the model
opt = tf.keras.optimizers.SGD(alpha)
mdl.compile(optimizer=opt, loss='mean_squared_error')
mdl.fit(X_train, y_train, epochs=n_epochs)

X_new = X_test[[0], :]
# TODO: predict
y_pred2 = mdl.predict(X_new)

#print('Predicted value for features ', X_new, ':', y_pred2)
plt.subplot(2, 1, 2)
plt.plot(np.array(y_pred), 'r.', label='gradDesc')
plt.plot(y_pred2, 'b.', label='keras')
plt.plot(y_test[0], '+', label='true')
plt.legend()
plt.show()
############################
#
#
############################
## 6. Change parameters. 
## TODO: Play around and see what happens if you change the training 
## parameters alpha and n_epochs.