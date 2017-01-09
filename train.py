from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# percentage of GPU memory used
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))

# dimensions of our input images
img_width, img_height = 150, 150
img_size = img_width * img_height * 3

train_data_dir = 'data'
validation_data_dir = 'data'
model_path = "model.h5"
nb_train_samples = 1000
nb_validation_samples = 100
nb_epoch = 1000

import os.path
import glob
folder = glob.glob(train_data_dir+"/*")

# each folder is a class
# images of the same class is put in the same folder
num_of_class = len(folder)

# This model comes from keras blog
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

model = Sequential()

# input layer should match the size of input image
model.add(Convolution2D(32, 3, 3, input_shape=(150, 150, 3)))
model.add(Activation('tanh'))
# halve the image size
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

#Deeeeeep learning
model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='tanh'))

# prevent overfitting
model.add(Dropout(0.5))

# output layer should have n classes
model.add(Dense(num_of_class, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# provide various training data
# it should recognize the image regardless of image transforms
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        )

# validation data for estimating accuracy
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        directory=train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        directory=validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical')

# save to model when val_loss has improved
checkpointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=False)

if os.path.isfile(model_path):
    del model
    model = load_model(model_path)

model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples,
        callbacks=[checkpointer])

#model.save('model.h5')
