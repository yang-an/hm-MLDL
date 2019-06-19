#############################################################################################
# University of Applied Sciences Munich                                                     #
# Department of Electrical Engineering and Information Sciences                             #
# Machine Learning and Deep Learning, (c) Alfred Schöttl                                    #
#-------------------------------------------------------------------------------------------#
# Assigment DL11                                                                            #
#############################################################################################

import tensorflow as tf
import numpy as np
tf.enable_eager_execution()
import matplotlib.pyplot as plt

def read_data():
    x_train = np.loadtxt('train-features.txt', dtype=np.float32)
    y_train = np.loadtxt('train-labels.txt', dtype=np.int64)
    x_test = np.loadtxt('test-features.txt', dtype=np.float32)
    y_test = np.loadtxt('test-labels.txt', dtype=np.int64)
    return (x_train, y_train), (x_test, y_test)

(x_train,y_train), (x_test,y_test) = read_data()
n_classes = 11
seq_len = x_train.shape[1]

#################################
# 1. Preprocess the data (convert the labels to one-hot-coding)
#>>
#
x_train = np.expand_dims(x_train, 2)
y_train = tf.keras.utils.to_categorical(y_train)

x_test = np.expand_dims(x_test, 2)
y_test = tf.keras.utils.to_categorical(y_test)

##################################

def model(model_type='lstm'):
	################################
	# 2. Define the LSTM
	#>>
	x0 = tf.keras.layers.Input([seq_len, 1])
	
	if model_type == 'lstm':
		x1 = tf.keras.layers.LSTM(50, return_sequences=True)(x0)
		x2 = tf.keras.layers.LSTM(50)(x1)
	elif model_type == 'gru':
		x1 = tf.keras.layers.GRU(50, return_sequences=True)(x0)
		x2 = tf.keras.layers.GRU(50)(x1)
	else:
		raise ValueError('unknown network type')
	
	d1 = tf.keras.layers.Dense(25, activation='sigmoid')(x2)
	d2 = tf.keras.layers.Dense(n_classes, activation='softmax')(d1)
	
	mdl = tf.keras.Model(x0, d2)
	mdl.summary()
	
	return mdl

################################
# 3. Get the model and train it. Use the categorical cross-entropy as loss function
#    and also show the accuracy of the training and test data
#>>
hist = {}

for t in ['lstm', 'gru']:
	m = model(t)
	m.compile(
		optimizer=tf.keras.optimizers.Adam(),
		loss='categorical_crossentropy',
		metrics = ['acc']
	)
	hist[t] = m.fit(
		x_train, y_train,
		epochs=10,
		validation_data=(x_test, y_test)
	)
################################

plt.subplot(2, 2, 1)
plt.plot(hist['lstm'].history['loss'], label='loss')
plt.plot(hist['lstm'].history['val_loss'], 'r', label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('LSTM')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(hist['lstm'].history['acc'], label='acc')
plt.plot(hist['lstm'].history['val_acc'], 'r', label='val_acc')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(hist['gru'].history['loss'], label='loss')
plt.plot(hist['gru'].history['val_loss'], 'r', label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('GRU')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(hist['gru'].history['acc'], label='acc')
plt.plot(hist['gru'].history['val_acc'], 'r', label='val_acc')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()

plt.tight_layout()
plt.show()