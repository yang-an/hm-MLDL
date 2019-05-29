#############################################################################################
# University of Applied Sciences Munich                                                     #
# Department of Electrical Engineering and Information Sciences                             #
# Machine Learning and Deep Learning, (c) Alfred Schöttl                                    #
#-------------------------------------------------------------------------------------------#
# Assigment DL10a                                                                           #
#############################################################################################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.enable_eager_execution()

batch_size = 128
n_epochs = 30

def plot2(x1, x2):
    x = np.hstack([x1,x2])
    plt.clf()
    fig = plt.figure(1)
    plt.imshow(x)
    plt.pause(0.000001)

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

# Preprocessing
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

def model():
################################
# 1. Define the autoencoder for an flattened image with shape [784].
#>>
#
#################################
	inputs = tf.keras.layers.Input(shape=[X_train.shape[1]])
	enc1 = tf.keras.layers.Dense(200, activation='relu')(inputs)
	code = tf.keras.layers.Dense(20, activation='relu')(enc1)
	dec1 = tf.keras.layers.Dense(200, activation='relu')(code)
	outputs = tf.keras.layers.Dense(X_train.shape[1], activation='sigmoid')(dec1)
	
	mdl = tf.keras.Model(inputs=inputs, outputs=outputs)
	mdl.summary()
	return mdl

mdl = model()
#################################
# 2. Compile the model and define an appropriate loss function
#>>
#
##################################
mdl.compile(
	optimizer=tf.keras.optimizers.Adam(),
	loss='binary_crossentropy',
)

def train():
	n_batches = X_train.shape[0]//batch_size
	for epoch in range(n_epochs):
		print('-'*40, 'Epoch %d' % (epoch+1), '-'*40)
		for k in range(n_batches):
			batch = X_train[k*batch_size:(k+1)*batch_size]
			X, y = batch, batch
			mdl_loss = mdl.train_on_batch(X, y)
			if k%10==0:
				print('batch {}: loss: {:.4f}'.format(k, mdl_loss))
				#################################
				# 3. Generate the output of the autoencoder for X and 
				# display input and output for the first image in the batch by the function plot2
				#>>
				#
				#################################
				y_pred = mdl.predict(X)
				plot2(
					np.reshape(X[0], [28, 28]),
					np.reshape(y_pred[0], [28, 28])
				)
train()