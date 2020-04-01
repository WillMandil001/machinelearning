import os
import cv2
import make_data
import numpy as np
from numpy import asarray

from keras import layers
from keras import models
from keras.preprocessing.image import load_img

image_shape = 100

# 1. Build CNN model.
print("Building CNN model")
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(image_shape, image_shape, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(layers.Dense(2, activation='softmax'))
model.summary()

# 2. Compile the model
print("Compiling the model")
model.compile(loss= 'categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 3. Make the data for training and testing
train_images, train_labels, test_images, test_labels, val_images, val_labels = make_data.make_data(image_shape)

# 4. Train the model.
print("Training the model")
model.fit(np.array(train_images), np.array(train_labels), batch_size=64, epochs=10, verbose=1, validation_data=(np.array(val_images), np.array(val_labels)))

# 5. Evaluation
print("Evaluating the model's performance")
test_loss, test_acc = model.evaluate(np.array(test_images), np.array(test_labels))
print('Test accuracy:', test_acc)