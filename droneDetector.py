# Shahriyar Mammadli
# Import required files
import dataPreparation as dp
import helperFunctions as hf
# Set the parameters
# The height and width values are based on the analyze of the scales of images...
# ...that is done below
imgHeight = 224
imgWidth = 224
# Set the batch size
batchSize = 32
# Number of different classes in the dataset
numClasses = 6
# Path to read the input images
path = "C:\\Users\\smammadli\\Desktop\\Store\\DataSets\\Raw\\FlyingObjects-Drones\\FlyingObjects"
# Analysis of the data scales
# dp.analyzeData(path)

# Split the data
trainData, valData = dp.prepareData(path, imgHeight, imgWidth, batchSize)

# Train the model
# hf.buildResnet50(trainData, valData, imgHeight, imgWidth, batchSize)

model = hf.buildCNN(trainData, valData, imgHeight, imgWidth, numClasses)
