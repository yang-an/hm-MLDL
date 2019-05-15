import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
import numpy as np
import matplotlib.pyplot as plt

tf.enable_eager_execution()

# Neurons in convolutional layers
N_C0 = 8
N_C1 = 16
WINDOW_SHAPE = (3, 3)

# Neurons in deep layers
N_D0 = 200
N_D1 = 100

N_CLASSES = 10
N_EPOCHS = 20
BATCH_SIZE = 128

def plot_img(x):
    xr = np.reshape(x,[32,32,3])
    plt.imshow(x)
    plt.show()

(x_train,y_train), (x_test,y_test) = tf.keras.datasets.cifar10.load_data()
classes = {0:'Airplane', 1:'Automobile', 2:'Bird', 3:'Cat', 4:'Deer', 
           5:'Dog', 6:'Frog', 7:'Horse', 8:'Ship', 9:'Truck'}


###########################
# 1. Investigate the data: 
# TODO: 
# - print the shape of the data

###########################

###########################
# 2. Preprocess the data: 
# TODO:
# - Rescale the pixel values to the scale 0..1
# - recode the labels to one-hot-coding

# Omit one-hot coding with logits
x_train = x_train / 255
x_test = x_test / 255

###########################

###########################
# 3. Define the Keras model
# TODO: define a function model() which returns the Keras model.
# Which activiation function (= non-linearity) shall be used?
def model():
	
	mdl = Sequential()

	mdl.add(
		Conv2D(
			filters=N_C0,
			kernel_size=3,
			input_shape=x_train.shape[1:],
			padding='valid',
			activation='relu'
		)
	)
	mdl.add(
		MaxPool2D(
			pool_size=3,
			strides=2,
			padding='valid'
		)
	)
	mdl.add(
		Conv2D(
			filters=N_C1,
			kernel_size=3,
			padding='valid',
			activation='relu'
		)
	)
	mdl.add(
		MaxPool2D(
			pool_size=3,
			strides=2,
			padding='valid'
		)
	)
	mdl.add(
		Flatten()
	)
	mdl.add(
		Dense(
			units=N_D0,
			activation='sigmoid')
	)
	mdl.add(
		Dense(
			units=N_D1,
			activation='sigmoid')
	)
	mdl.add(
		Dense(
			units=N_CLASSES
		)
	)
	
	mdl.summary()
	
	return mdl
###########################


###########################
# 4. Train the Keras model using the Adam optimizer, including validation data,
# checking the accuracy, and storing the results to a history vector
# TODO: include the code for training. Which loss function shall be used?
mdl = model()

mdl.compile(
	optimizer=tf.keras.optimizers.Adam(),
	loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
	metrics=['acc']
)
	
history = mdl.fit(
	x_train,
	y_train, 
	batch_size=BATCH_SIZE,
	epochs=N_EPOCHS,
	validation_data=(x_test, y_test)
)
###########################

###########################
# 5. Plot the train history and use your trained model
# TODO: plot the loss history
#       then use it as you like (save the weights or predict the class of some images)

plt.subplot(2, 1, 1)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], 'r', label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(history.history['acc'], label='acc')
plt.plot(history.history['val_acc'], 'r', label='val_acc')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()

plt.tight_layout()
plt.show()
