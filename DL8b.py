#############################################################################################
# University of Applied Sciences Munich                                                     #
# Department of Electrical Engineering and Information Sciences                             #
# Machine Learning and Deep Learning, (c) Alfred Schöttl                                    #
#-------------------------------------------------------------------------------------------#
# Assigment DL8b                                                                            #
#############################################################################################

import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

import matplotlib.pyplot as plt

batch_size = 16


###########################
# 1. Use the same generators as in assignment 8a.
#
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
	directory='train',
	target_size=(224, 224),
	color_mode="rgb",
	batch_size=batch_size,
	class_mode="categorical", 
	shuffle=True
) 
test_generator = datagen.flow_from_directory(
	directory='test',
	target_size=(224, 224),
	color_mode="rgb",
	batch_size=batch_size,
	class_mode="categorical", 
	shuffle=True
)
#
###########################


###########################
# 2. Load the VGG model, do not include the last (dense) layers. Print the summary.
#
mdl = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False)
mdl.summary()

###########################


###########################
# 3. We start with the output of the last layer 'block5_pool'. 
#    Get the output and add your part of the net. 
#    Combine the layers to a new model mdl2 and print its summary.
#
x1 = mdl.get_layer('block5_pool').output
x2 = tf.keras.layers.GlobalAveragePooling2D()(x1)
x3 = tf.keras.layers.Flatten()(x2)
x4 = tf.keras.layers.Dense(10, activation='sigmoid')(x3)
y_pred = tf.keras.layers.Dense(2, activation='softmax')(x4)

mdl2 = tf.keras.models.Model(inputs=mdl.input,outputs=y_pred)
mdl2.summary()
###########################


###########################
# 4. Freeze the VGG layers to exclude them from training. 
#    Then, train the net and compare the results to your solution in assignment 8a.

for layer in mdl.layers:
	layer.trainable = False
	
mdl2.compile(
	optimizer=tf.keras.optimizers.Adam(),
	loss='categorical_crossentropy',
	metrics=['acc']
)

history = mdl2.fit_generator(
	train_generator,
	steps_per_epoch = 2000 // batch_size,
	epochs = 10,
	validation_data = test_generator,
	validation_steps = 1000 // batch_size
)
###########################

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