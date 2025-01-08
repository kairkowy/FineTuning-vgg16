import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Activation,Dropout, Flatten, Dense, BatchNormalization
from keras.applications import VGG16
from keras import backend as K
import matplotlib.pyplot as plt
import pickle

# dimensions of our images.
img_width, img_height = 150, 150

cus_model_path = 'repo/cus_vgg16_1a.keras'
tun_model_path = 'repo/tun_vgg16_2a1.keras'
tun_model_weight_path = 'repo/tun_vgg16_2a1.weights.h5'

train_data_dir = 'data/tun_train'
validation_data_dir = 'data/tun_validation'
nb_train_samples = 1000
nb_validation_samples = 200
epochs = 20
batch_size = 32
#batch_size = 16

# define image shpae
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# get model from VGG16 network and build
base_model = VGG16(include_top=False, input_shape=input_shape, weights='imagenet')

base_model.summary()

# frozen all low conc layer fro base model
base_model.trainable = False

for layer in base_model.layers:
  print(layer, layer.trainable)

# get layer for top of base model and add classification layers
last_layer = base_model.get_layer('block5_pool')

last_output = last_layer.output
x = Flatten()(last_output)

x = Dense(256, activation='relu', name='FC2')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

tun_model = Model(inputs=base_model.input, outputs=x)

tun_model.summary()
# compile custom model with a lr/momentum optimizer
# and a very slow learning rate.
opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001, momentum=0.9)

tun_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# prepare data augmentation configuration

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# learning custom model and save model
hist = tun_model.fit(train_generator,
            epochs = epochs,
            batch_size = batch_size,
            validation_data = validation_generator)

#tun_model.save(tun_model_path)
#tun_model.save_weights(tun_model_weight_path)

#tun_model.summary()
## Save and Load the Trainning history  for Visualization

with open('repo/TrainHistory_vgg16_2a1.pick',"wb") as file_pi:
      pickle.dump(hist.history, file_pi)

## Visualize the Training/Validation Data
# training accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()

