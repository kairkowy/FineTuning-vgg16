import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense
from keras import applications


# dimensions of our images.
img_width, img_height = 150, 150

#eval_model_path = 'repo/cus_vgg16_1a.keras'
eval_model_path = 'repo/tun_vgg16_2a2.keras'
#eval_model_path = 'repo/tun_vgg16_2a1.keras'
val_data_dir = 'data/validation'
test_data_dir = 'data/test'

batch_size = 32

cus_model = load_model(eval_model_path)

cus_model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
cus_model.summary()

test_datagen = ImageDataGenerator(rescale=1./255)

val_generator = test_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')


val_mse, val_mae = cus_model.evaluate(val_generator)

