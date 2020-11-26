# Binary Beats
# Import required libraries
import sys
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf

# Progress bar to print the current state of loadings
def progressBar(current,max):
    # Number of bar elements
    barCount = 50
    ratio = current/max
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(barCount * ratio):{barCount}s}] {int(100 * ratio)}%")
    sys.stdout.flush()

# Plot histogram
def histPlot(data, title):
    # Set the bin size
    plt.hist(data, bins=30)
    plt.title(title)
    plt.xlabel('Size')
    plt.ylabel('Count')
    # Display the plot
    plt.show()

# Build a resnet-50 model
# def buildResnet50(trainData, valData, imgHeight, imgWidth, batchSize):
#     callbacks = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')
#
#     resnet50_model = ResNet50(include_top=False,
#                              input_tensor=None, input_shape=(imgHeight, imgWidth, 3))
#
#     resnet50_model.summary()
#
#     resnet50_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
#     history = resnet50_model.fit_generator(trainData,
#                                            epochs=30,
#                                            verbose=1,
#                                            validation_data=valData,
#                                            callbacks=[callbacks]
#                                            )

def modelCNN(imgHeight, imgWidth, numClasses):
    model = Sequential([
      layers.experimental.preprocessing.Rescaling(1./255, input_shape=(imgHeight, imgWidth, 3)),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(numClasses)
    ])
    return model

def buildCNN(trainData, valData, imgHeight, imgWidth, numClasses):
    model = modelCNN(imgHeight, imgWidth, numClasses)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    epochs = 10
    history = model.fit(
        trainData,
        validation_data=valData,
        epochs=epochs
    )