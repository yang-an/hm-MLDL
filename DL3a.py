import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.enable_eager_execution()

def plot_digit(x):
    xr = np.reshape(x,[28,28])
    plt.imshow(x)
    # plt.pause(0.1)    # comment in this line if the plot doesn't show up
    plt.show()

# Load the mnist data ans split it into a training and a test data set.
(x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()

###########################
# 1. Investigate the data: 
# TODO: 
# - play with the following index to display different digits
# - comment out the line afterwards
#plot_digit(x_train[0])
###########################

###########################
# 2. Preprocess the data: 
# TODO:
# - Rescale the gray-valued pixel values to the scale 0..1
# - flatten the data
# - recode the labels to one-hot-coding
# uncomment the next two lines
x_train = x_train/255.0
x_train = np.reshape(x_train, [-1,784]) 
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
	y_pred = tf.keras.layers.Dense(n_classes, activation='softmax')(inputs)
	mdl = tf.keras.Model(inputs=inputs, outputs=y_pred)
	return mdl
	
###########################

alpha = 0.05
n_epochs = 20

###########################
# 4. Train the Keras model
# TODO: include the code for training. Which loss function shall be used?
mdl = model()
opt = tf.keras.optimizers.SGD(alpha)
mdl.compile(optimizer=opt, loss='categorical_crossentropy')
history = mdl.fit(x_train, y_train, epochs=n_epochs)
###########################

# save the model:
open('DL3_arch.yaml', 'wt').write(mdl.to_yaml())
mdl.save_weights('DL3_weights.h5')

plt.plot(history.history['loss'])
plt.show()