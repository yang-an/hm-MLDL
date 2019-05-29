#############################################################################################
# University of Applied Sciences Munich                                                     #
# Department of Electrical Engineering and Information Sciences                             #
# Machine Learning and Deep Learning, (c) Alfred Schöttl                                    #
#-------------------------------------------------------------------------------------------#
# Assigment DL10b                                                                           #
#############################################################################################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.enable_eager_execution()

z_width = 100
batch_size = 128
n_epochs = 30

def plot_generated_samples():
    noise = np.random.normal(0, 1, size=(10, z_width))
    generated_images = gen.predict(noise)
    generated_images = generated_images.reshape(10, 28, 28)
    plt.figure(1, figsize=(12, 2))
    plt.clf()
    for i in range(generated_images.shape[0]):
        plt.subplot(1, 10, i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.pause(0.00001)

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

# Preprocessing
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

################################
# 1. Define the generator and the discriminator model.
#    The generator takes an input of shape [z_width] and outputs a tensor of shape [784].
#    The discriminator takes an input of [784] and outputs a tensor of shape [1].
def model():
#>> 
#
################################
	g_input = tf.keras.layers.Input(shape=[z_width])
	g_x1 = tf.keras.layers.Dense(200, activation='relu')(g_input)
	g_output = tf.keras.layers.Dense(X_train.shape[1], activation='sigmoid')(g_x1)
	gen = tf.keras.Model(inputs=g_input, outputs=g_output)
	
	d_input = tf.keras.layers.Input(shape=[X_train.shape[1]])
	d_x1 = tf.keras.layers.Dense(80, activation='relu')(d_input)
	d_output = tf.keras.layers.Dense(1, activation='sigmoid')(d_x1)
	discr = tf.keras.Model(inputs=d_input, outputs=d_output)
	
	gen.summary()
	discr.summary()
	
	return gen, discr

gen, discr = model()

################################
# 2. Compile the discriminator and generator.
#    Hints: 
#    - The discriminator returns the probability of the image being artificial.
#      It can be trained by the built-in loss function 'binary_crossentropy'.
#    - You will need to write your own loss function 
#                       def gen_loss(y_true, y_pred) 
#      for the generator. The first argument y_true of the loss function is the true label vector
#      (containing 0/1 for real/artifical data). In our case, it only contains 1s.
#      The second argument y_pred is the output of the generator, it is recommended to rename it to img:
#                       def gen_loss(y_true, img) 
#    - In gen_loss, call the discriminator and define the loss as the binary crossentropy between 1-y_true and 
#      the output of the discriminator (to foster to deceive the discriminator). Use the function 
#      tf.keras.losses.binary_crossentropy().
#>>
discr.compile(
	optimizer=tf.keras.optimizers.Adam(),
	loss='binary_crossentropy'
)

def gen_loss(y_true, img):
	y_pred = discr(img)
	loss = tf.keras.losses.binary_crossentropy(1-y_true, y_pred)
	return loss
	
gen.compile(
	optimizer=tf.keras.optimizers.Adam(),
	loss=gen_loss
)

#
################################

def train():
	n_batches = X_train.shape[0]//batch_size
	for k in range(n_epochs):
		print('-'*40, 'Epoch %d' % (k+1), '-'*40)
		for step in range(n_batches):
			image_batch = X_train[step*batch_size:(step+1)*batch_size]
			noise = np.random.normal(0, 1, size=(batch_size, z_width))
			generated_images = gen.predict(noise)
		
			X = np.concatenate([image_batch, generated_images])
			y = np.zeros(2*batch_size)
			y[:batch_size] = 1.0     # 1: natural, 0: artificial
		
			# Train discriminator on generated images
			discr.trainable = True
			discr_loss = discr.train_on_batch(X, y)
		
			# Train generator
			discr.trainable = False
			################################
			# 3. Train the generator
			#    Hints:
			#    - You will need two arguments for the method train_on_batch:
			#      The first is the input to the generator. It should be a Gaussian noise, the data type is float32
			#      The second are the labels (just zeros for artificial images in this case)
			#>>
			y_pred = discr.predict(X)
			gen_loss = gen.train_on_batch(noise, np.zeros(batch_size))
			#
			################################
		print('loss: discriminator {:.4f}, generator {:.4f}'.format(discr_loss, gen_loss))
		plot_generated_samples()

train()