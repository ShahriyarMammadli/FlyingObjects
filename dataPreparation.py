# Binary Beats
# Import required libraries
import os
import cv2
import helperFunctions as hf
import pickle
from tensorflow.keras import layers
import tensorflow as tf

# Function to analyze the data
def analyzeData(path):
    objects = os.listdir(path)
    # Ignore the system files
    objects = [elem for elem in objects if not '.' in elem]
    imageSizes = []

    # Loop over the object folders
    for obj in objects:
        images = os.listdir(path + '\\' + obj)
        images = [img for img in images if img.endswith('.jpg')]
        # Progress bar
        print("Processing " + obj + " images.")
        # Iterate over the images
        for index, imgName in enumerate(images):
            # Read an image
            img = cv2.imread(path + '\\' + obj + '\\' + imgName)
            imageSizes.append(img.shape)
            hf.progressBar(index + 1, len(images))
        print("\n")

    # Transpose the list
    imageSizes = list(zip(*imageSizes))
    imageSizes = [list(elem) for elem in imageSizes]
    # Display the histograms
    hf.histPlot(list(imageSizes[0]), "Histogram of the heights of images")
    hf.histPlot(list(imageSizes[1]), "Histogram of the widths of images")

    imageSizes[0].sort()
    imageSizes[1].sort()
    print("Sorted list of the heights of images")
    print(imageSizes[0])
    print("Sorted list of the widths of images")
    print(imageSizes[1])

# Readand split the data
def prepareData(path, imgHeight, imgWidth, batchSize):
    # Read training data
    trainData = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="training",
        seed=2020,
        image_size=(imgHeight, imgWidth),
        batch_size=batchSize,
    )
    # Read validation data
    valData = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="validation",
        seed=2020,
        image_size=(imgHeight, imgWidth),
        batch_size=batchSize,
    )
    # Standardize the data
    normalizationLayer = layers.experimental.preprocessing.Rescaling(1. / 255)
    normalized_ds = trainData.map(lambda x, y: (normalizationLayer(x), y))
    return trainData, valData