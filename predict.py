from keras.preprocessing import image as image_utils
from keras.models import load_model
import numpy as np
import os, sys

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

print("Loading network...")
model = load_model('vocaloid.h5')

# Run the python script like this:
# python predict.py data\000_hatsune_miku\face_93_104_36.png
print("Loading and preprocessing image...")
labels = os.listdir('data')
file = sys.argv[1]
image = image_utils.load_img(file, target_size=(150, 150))
image = image_utils.img_to_array(image)
image = np.expand_dims(image, axis=0)

print("Classifying image...")
preds = model.predict_classes(image)
print(labels[preds[0]])

preds = model.predict(image)
print(["{0:2.2f}%".format(i*100) for i in preds[0]])