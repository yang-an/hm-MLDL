import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.enable_eager_execution()

N_LAYER1 = 1000
N_LAYER2 = 100
N_CLASSES = 10
N_EPOCHS = 10

def plot_digit(x):
    xr = np.reshape(x,[28,28])
    plt.imshow(x)
    plt.show()

###########################
# 5. Train the Keras model using the Adam optimizer
# TODO: Change the dataset to fashion MNIST
(x_train,y_train), (x_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()
###########################

###########################
# 1. Investigate the data: 
# TODO: 
# - play with the following index to display different digits
# - comment out the line afterwards

###########################

###########################
# 2. Preprocess the data: 
# TODO:
# - Rescale the gray-valued pixel values to the scale 0..1
# - flatten the data
# - recode the labels to one-hot-coding
x_train = np.reshape(x_train / 255, [-1,784]) 
y_train = tf.keras.utils.to_categorical(y_train)
###########################

n = x_train.shape[1]            # number of features
n_classes = 10

###########################
# 3. Define the Keras model
# TODO: define a function model() which returns the Keras model.
# Which activiation function (= non-linearity) shall be used?
def model():
	inputs = tf.keras.layers.Input(shape=[n]) 
	y_pred = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(inputs)
	mdl = tf.keras.Model(inputs=inputs, outputs=y_pred)
	return mdl
	
def dnn():
	x0 = tf.keras.layers.Input(shape=[n])
	x1 = tf.keras.layers.Dense(N_LAYER1, activation='sigmoid')(x0)
	x2 = tf.keras.layers.Dense(N_LAYER2, activation='sigmoid')(x1)
#	x3 = tf.keras.layers.Dense(N_LAYER2, activation='sigmoid')(x2)
	y_pred = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x2)
	mdl = tf.keras.Model(inputs=x0, outputs=y_pred)
	return mdl
###########################

###########################
# 4. Train the Keras model using the Adam optimizer
# TODO: include the code for training. Which loss function shall be used?
mdl = dnn()
opt = tf.keras.optimizers.Adam()
mdl.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
history = mdl.fit(x_train, y_train, epochs=N_EPOCHS)
###########################

open('DL4_arch.yaml', 'wt').write(mdl.to_yaml())
mdl.save_weights('DL4_weights.h5')

