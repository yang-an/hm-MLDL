########################################################################
# University of Applied Sciences Munich                                 
# Department of Electrical Engineering and Information Sciences         
# Machine Learning and Deep Learning, (c) Alfred Schöttl                
#-----------------------------------------------------------------------
# Assigment DL2a                                                        
########################################################################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.enable_eager_execution()

ALPHA = 1e-6
N_EPOCHS = 100

# Load the data set. It contains home prices of districts in/around
# Boston. The used feature in X is the proportion of residential land
# zoned for lots over 25,000 sq.ft. The response is the mean value
# (median) of homes in this district in $1000's
(X_train,y_train), (X_test,y_test) = \
	tf.keras.datasets.boston_housing.load_data()

n = X_train.shape[1]            # Use second feature only

###########################
# 1. Define the model and the loss function
# TODO: fill in the linear model equation and the formula for the total
# 		loss function
def model(X,w):
	y = X @ tf.expand_dims(w, 1)
	return y[:, 0]
	
def loss(y_true, y_pred):
	l = tf.reduce_mean((y_true - y_pred) ** 2, 0)
	return l
###########################


###########################
# 2. Define a weights variable with shape [n] and zero inital values
# TODO: use tf.get_variable to define a variable
w = tf.get_variable('weights', n, initializer=tf.zeros_initializer())
###########################

###########################
# 3. Training by steepest descent:
losses = []

# TODO: for each loop:
for k in range(N_EPOCHS):
    #  - compute the gradient dl/dw
	with tf.GradientTape() as tape:
		y_pred = model(X_train, w)
		l = loss(y_train, y_pred)
	
	dldw = tape.gradient(l, w)
    #  - modify the weights according to the steepest descent algorithm
	w.assign(w - ALPHA * dldw)
	
	print('[#{:2d}] loss: {}'.format(k, float(l)))
	losses.append(float(l))
##########################

###########################
# 4. Predict with the trained model:
# TODO: use new samples to predict
y_pred = model(X_test, w)
#print('Predicted value for features ', X_new[0], ':', float(y_pred))
###########################

###########################
# 5. Training and prediction by Keras:
# TODO: Repeat steps 1-4 with Keras. There are slight differences in the
# implementation, so don't be surprised if the result is different!

def model2():
    inputs = tf.keras.layers.Input(shape=[n]) 
    y_pred = tf.keras.layers.Dense(1, activation='linear')(inputs)
    mdl = tf.keras.Model(inputs=inputs, outputs=y_pred) 
    return mdl

mdl = model2()
# TODO: train the model
opt = tf.keras.optimizers.SGD(ALPHA)
mdl.compile(optimizer=opt, loss='mean_squared_error')
history = mdl.fit(X_train, y_train, epochs=N_EPOCHS)

# TODO: predict
y_pred_keras = mdl.predict(X_test)
#print('Predicted value for features ', X_new, ':', y_pred2)

############################
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
############################
## 6. Change parameters. 
## TODO: Play around and see what happens if you change the training 
## parameters alpha and n_epochs.