#############################################################################################
# University of Applied Sciences Munich                                                     #
# Department of Electrical Engineering and Information Sciences                             #
# Machine Learning and Deep Learning, (c) Alfred Schöttl                                    #
#-------------------------------------------------------------------------------------------#
# Assigment DL8a                                                                            #
#############################################################################################

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
import numpy as np
tf.enable_eager_execution()

# Neurons in convolutional layers
N_C0 = 8
N_C1 = 16
WINDOW_SHAPE = (3, 3)

# Neurons in deep layers
N_D0 = 200
N_D1 = 100

N_CLASSES = 2
N_EPOCHS = 20
BATCH_SIZE = 16

###########################
# 1. Define a data generator to manipulate rgb images with augmentation. Scale the pixel
#    values in the range 0..1
#
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

#
###########################

###########################
# 2. Define two data generators which reads in the files for training and test. They shall
#    use the generator from 1. The image shape is 224, 224, 3
#
train_generator = datagen.flow_from_directory(
	directory='train',
	target_size=(224, 224),
	color_mode="rgb",
	batch_size=BATCH_SIZE,
	class_mode="categorical", 
	shuffle=True
) 
test_generator = datagen.flow_from_directory(
	directory='test',
	target_size=(224, 224),
	color_mode="rgb",
	batch_size=BATCH_SIZE,
	class_mode="categorical", 
	shuffle=True
) 
###########################


###########################
# 3. Define a convolutional model to distinguish between cats and dogs.
#
def model():
	mdl = Sequential()
		mdl.add(		Conv2D(			filters=N_C0,			kernel_size=3,			input_shape=train_generator.image_shape,			padding='valid',
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
			activation='sigmoid'
		)
	)
	mdl.add(
		Dense(
			units=N_D1,
			activation='sigmoid'
		)
	)
	mdl.add(
		Dense(
			units=N_CLASSES,
			activation='softmax'
		)
	)
	
	mdl.summary()
	
	return mdl

###########################


###########################
# 4. Train the model. Hint: get the model, compile, fit_generator
mdl = model()

mdl.compile(
	optimizer=tf.keras.optimizers.Adam(),
	loss='categorical_crossentropy',
	metrics=['acc']
)

mdl.fit_generator(
	train_generator,
	steps_per_epoch = 2000//BATCH_SIZE,
	epochs = 10,
	validation_data = test_generator,
	validation_steps = 1000 // BATCH_SIZE
)

###########################