import os
import cv2
import numpy as np
from numpy import asarray
from PIL import Image

from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator

# 1. Load images for train and test
# TRAIN:
def make_data(image_shape):
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
        resized = img.resize((image_shape, image_shape))  # Reduce size
        image_array = asarray(resized)
        train_images.append(image_array)
        if image[0] == "c":
            train_labels.append(np.array([1, 0]))
        elif image[0] == "d":
            train_labels.append(np.array([0, 1]))
    train_images = asarray(train_images)
    shape_ = train_images.shape
    train_images = train_images.reshape((shape_[0], shape_[1], shape_[2], 1))
    train_images = train_images.astype('float32') / 255

    # TEST:
    print("Creating test images")
    f = []
    for (dirpath, dirnames, filenames) in os.walk("test/"):
        f.extend(filenames)
        break
    test_images = []
    test_labels = []
    for image in f:
        img = load_img("test/" + image, color_mode="grayscale")
        # im = Image.open("train/" + image).convert('L')  # Load as grayscale
        resized = img.resize((image_shape, image_shape))  # Reduce size
        image_array = asarray(resized)
        test_images.append(image_array)
        if image[0] == "c":
            test_labels.append(np.array([1, 0]))
        elif image[0] == "d":
            test_labels.append(np.array([0, 1]))
    test_images = asarray(test_images)
    shape_ = test_images.shape
    test_images = test_images.reshape((shape_[0], shape_[1], shape_[2], 1))
    test_images = test_images.astype('float32') / 255

    # VALIDATION:
    print("Creating validation images")
    f = []
    for (dirpath, dirnames, filenames) in os.walk("validation/"):
        f.extend(filenames)
        break
    val_images = []
    val_labels = []
    for image in f:
        img = load_img("validation/" + image, color_mode="grayscale")
        # im = Image.open("train/" + image).convert('L')  # Load as grayscale
        resized = img.resize((image_shape, image_shape))  # Reduce size
        image_array = asarray(resized)
        val_images.append(image_array)
        if image[0] == "c":
            val_labels.append(np.array([1, 0]))
        elif image[0] == "d":
            val_labels.append(np.array([0, 1]))
    val_images = asarray(val_images)
    shape_ = val_images.shape
    val_images = val_images.reshape((shape_[0], shape_[1], shape_[2], 1))
    val_images = val_images.astype('float32') / 255
    return train_images, train_labels, test_images, test_labels, val_images, val_labels