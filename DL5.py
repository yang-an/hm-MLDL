import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.enable_eager_execution()

def plot_img(x, label):
	xr = np.reshape(x,[32,32,3])
	plt.imshow(x)
	plt.title(label)
	plt.show()

(x_train,y_train), (x_test,y_test) = tf.keras.datasets.cifar10.load_data()
classes = {0:'Airplane', 1:'Automobile', 2:'Bird', 3:'Cat', 4:'Deer', 
           5:'Dog', 6:'Frog', 7:'Horse', 8:'Ship', 9:'Truck'}
		   

N_CLASSES = 10
N_EPOCHS = 30

# Width of fully connected layers
N_LAYERS = [800, 400, 200, 100, 50]

BATCH_SIZE = 128

###########################
# 1. Investigate the data: 
# TODO: 
# - play with the following index to display different digits
# - comment out the lines afterwards
for ndx in range(10):
	pass
	#plot_img(x_train[ndx], classes[y_train[ndx,0]])
###########################

###########################
# 2. Preprocess the data: 
# TODO:
# - Rescale the pixel values to the scale 0..1
# - flatten the data
# - recode the labels to one-hot-coding
# Hint: you can print out the shape of a numpy array a by print(a.shape)

n_features = np.prod(x_train.shape[1:])

x_train = x_train / 255
x_train = np.reshape(x_train, [-1, n_features])
y_train = tf.keras.utils.to_categorical(y_train)

x_test = x_test / 255
x_test = np.reshape(x_test, [-1, n_features])
y_test = tf.keras.utils.to_categorical(y_test)

###########################

###########################
# 3. Define the Keras model
# TODO: define a function model() which returns the Keras model.
# Which activiation function (= non-linearity) shall be used?
def model():
	
	layers = [tf.keras.layers.Input(shape=[n_features]),]
	
	# Generate hidden layers
	for n_layer in N_LAYERS:
		layers.append(
			tf.keras.layers.Dense(n_layer, activation='relu')(layers[-1])
		)
	
	y_pred = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(layers[-1])
	mdl = tf.keras.Model(inputs=layers[0], outputs=y_pred)
	return mdl
###########################

###########################
# 4. Train the Keras model using the Adam optimizer, including validation data,
# checking the accuracy, and storing the results to a history vector
# TODO: include the code for training. Which loss function shall be used?
mdl = model()
opt = tf.keras.optimizers.Adam()
mdl.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
history = mdl.fit(
	x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
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
