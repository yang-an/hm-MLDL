import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.enable_eager_execution()

N_SAMPLES = 400

def plot_digit(x):
    xr = np.reshape(x,[28,28])
    plt.imshow(x)
    plt.show()

# Load the trained model
mdl = tf.keras.models.model_from_yaml(open('DL3_arch.yaml', 'rt').read())
mdl.load_weights('DL3_weights.h5')

# Load the mnist data and split it into a training and a test data set.
(x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()

###########################
# 1. Preprocess the test data: 
# - Rescale the gray-valued pixel values to the scale 0..1
# - flatten the data
# uncomment the next line
x_test_new = x_test/255.0
x_test_new = np.reshape(x_test, [-1,784]) 
###########################

###########################
# 2. Predict the test data:
# - predict the first 10 samples
# - print the true and the predicted label
# Tips: 
# - use M[:10,:] to get the first 10 rows of a matrix
# - use np.argmax(M,1) to obtain the vectoe with the indices 
#   of the greatest entry in each row
x_test_new = x_test_new[:N_SAMPLES, :]
y_test = y_test[:N_SAMPLES]
y_pred = mdl.predict(x_test_new)
y_pred_labels = np.argmax(y_pred, 1)

#print(y_pred_labels)
#print(y_test)

###########################

plt.plot(y_pred_labels, 'rx', label='pred')
plt.plot(y_test, '.', label='true')
plt.legend()
plt.show()
