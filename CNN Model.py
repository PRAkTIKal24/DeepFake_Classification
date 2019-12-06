# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:35:44 2019

@author: koesa
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
import matplotlib.pyplot as plt


dataset_Dir = 'C:\Dataset'

# Network parameters
batch_size = 32
numEpochs = 20
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

# Modify the data for use
image_data = ImageDataGenerator(rescale=1./255,
                                validation_split=0.4)
# load training data
trainingData_gen = image_data.flow_from_directory(batch_size=batch_size,
                                                           directory=dataset_Dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           color_mode="rgb",
                                                           class_mode='binary',
                                                           seed=9,
                                                           subset="training")

# load validation data
validationData_gen = image_data.flow_from_directory(batch_size=batch_size,
                                                           directory=dataset_Dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           color_mode="rgb",
                                                           class_mode='binary',
                                                           seed=9,
                                                           subset="validation")
# Load VGG Model without classification layers
# Training tip: setting weights = None will perform random intialization of weights
VGG_Model = VGG16(include_top = False,
                  input_shape=(IMG_HEIGHT,
                         IMG_WIDTH,
                         IMG_CHANNELS),
                 weights="imagenet",
                 )

# build new top layers
flat1 = Flatten()(VGG_Model.outputs)
dense1 = Dense(1024, activation='relu')(flat1)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(512, activation='relu')(drop1)
drop2 = Dropout(0.5)(dense2)
output = Dense(2, activation='sigmoid')

# define new model
model = Model(inputs = VGG_Model.inputs, outputs = output)

# summarize model
model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the network
history = model.fit_generator(
    trainingData_gen,
    epochs=numEpochs,
    validation_data=validationData_gen,
    steps_per_epoch=200,
    verbose=2)

# Visualize the results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(numEpochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()