import os
import cv2
from PIL import Image
import numpy as np
from numpy import asarray
import make_data
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator


# 1. Load images for train and test
# TRAIN:
print("Creating training images")
f = []
for (dirpath, dirnames, filenames) in os.walk("train/"):
    f.extend(filenames)
    break
train_images = []
train_labels = []
for image in f:
    img = load_img("train/" + image, color_mode="grayscale")
    # im = Image.open("train/" + image).convert('L')  # Load as grayscale
    resized = img.resize((50, 50))  # Reduce size
    image_array = asarray(resized)
    train_images.append(image_array)
    train_labels.append(image[0])
    # train_images.append(image_array)

#TEST:
print("Creating test images")
# f = []
# for (dirpath, dirnames, filenames) in os.walk("test/"):
#     f.extend(filenames)
#     break
# test_images = []
# test_labels = []
# for image in f:
#     im = Image.open("test/" + image).convert('L')  # Load as grayscale
#     resized = im.resize((50,50))  # Reduce size
#     image_array = asarray(resized)
#     test_images.append([image_array, image[0]])
#     test_labels.append(image[0])
#     # test_images.append(image_array)
# print(np.shape(test_images))
# print(np.shape(test_labels))

# 2. Pre-processing the image.
print("Pre-processing the image")

# 3. Build CNN model.
print("Building CNN model")
model = models.Sequential()
model.add(layers.Conv2D(32,(5,5),activation='relu', input_shape=(50,50,1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(2, activation='softmax'))
model.summary()

print("Compiling the model")
model.compile(loss= 'categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. Train the model.
print("Training the model")
model.fit(np.array(train_images), np.array(train_labels), batch_size=100, epochs=5, verbose=1)

# 5. Evaluate the model's performance:
print("Evaluating the model's performance")
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)


