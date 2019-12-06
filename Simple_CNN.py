#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:45:53 2019

@author: revant
"""


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

dataset_Dir = '/home/revant/storage/Courses/RBE/RBE595/DeepFakeClassifier/dataset'

batch_size = 32
numEpochs = 10
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

image_data = ImageDataGenerator(rescale=1./255,
                                validation_split=0.4)

trainingData_gen = image_data.flow_from_directory(batch_size=batch_size,
                                                           directory=dataset_Dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           color_mode="rgb",
                                                           class_mode='binary',
                                                           seed=9,
                                                           subset="training")
validationData_gen = image_data.flow_from_directory(batch_size=batch_size,
                                                           directory=dataset_Dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           color_mode="rgb",
                                                           class_mode='binary',
                                                           seed=9,
                                                           subset="validation")
model = Sequential()
model.add(Conv2D(64, kernel_size = 3, padding= "same", activation = 'relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Dropout(0.5))
model.add(Conv2D(128, kernel_size=3, padding= "same", activation='relu'))
model.add(Conv2D(128, kernel_size=3, padding= "same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Dropout(0.5))
model.add(Conv2D(512, kernel_size=3, padding= "same", activation='relu'))
model.add(Conv2D(512, kernel_size=3, padding= "same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Dropout(0.5))
model.add(Conv2D(1024, kernel_size=3, padding= "same", activation='relu'))
model.add(Conv2D(1024, kernel_size=3, padding= "same", activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'tanh'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
history = model.fit_generator(
    trainingData_gen,
    epochs=numEpochs,
    validation_data=validationData_gen,
    steps_per_epoch=20,
    verbose=2, validation_steps=2)

#print(history.history['val_acc'])
