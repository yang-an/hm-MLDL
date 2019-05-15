#############################################################################################
# University of Applied Sciences Munich                                                     #
# Department of Electrical Engineering and Information Sciences                             #
# Machine Learning and Deep Learning, (c) Alfred Schöttl                                    #
#-------------------------------------------------------------------------------------------#
# Assigment DL7                                                                             #
#############################################################################################

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2
tf.enable_eager_execution()

# Helper functions to load the image and display the result
def load_image(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # opencv stores images in BGR format, we convert it to standard RGB
    return img

def plot(x, txt=''):
	plt.imshow(x)
	plt.axis('off')
	if txt:
		txt = ', '.join(['{:.3f}: {}'.format(s[2], s[1]) for s in txt])
		plt.title(txt)
	
	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()
	plt.show()

# The standard image size of VGG is 224x224
sh = (224,224,3)


###########################
# 1. Load the trained vgg16 model (call the constructor VGG16 in package tf.keras.applications.vgg16 
#    with the option weights='imagenet'). The model is trained on over one million images with 1000 classes.
#    You may use the summary() method to print out the model structure.
#>>


if 'mdl' not in dir():
	mdl = vgg = tf.keras.applications.vgg16.VGG16(weights='imagenet')
	mdl.summary()

###########################


###########################
# 2. Define the necessary preprocessing steps to classify one image
#    - use cv2.resize to resize the image
#    - expand the dimensions to create a batch of images with one sample
#    - call preprocess_input of the vgg16 package to do the VGG-specific preprocessing
#    - return the modified image
#>>
def preprocess(img):
	ret = cv2.resize(img, sh[:2])
	ret = np.expand_dims(ret, 0)
	ret = tf.keras.applications.vgg16.preprocess_input(ret)
	return ret
###########################

###########################
# 3. Use it!
#    - load an image (try different color images)
#    - perform the preprocessing
#    - predict the result (we are using imagenet classification. There are 1000 classes within the imagenet data!)
#      (since we do not train here, you do not need to compile or fit the model!)
#    - obtain the names of the 3 most likely classes by calling decode_predictions(preds, top=3) of the vgg16 package
#    - plot the result
#>>
images = ['DL7_testimg.jpg', 'unicorn.png', 'bread.png']


for file in os.listdir(os.getcwd()):
	filename = os.fsdecode(file)
	if filename.endswith('.png') or filename.endswith('.jpg'):
		img = load_image(filename)
		proc_img = preprocess(img)
		pred = mdl.predict(proc_img)
		txt = tf.keras.applications.vgg16.decode_predictions(pred, top=5)[0]

		plot(img, txt)
		
		break

###########################
 
