import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import ImageEnhance

imageHeight = 256
imageWidth = 256
channels = 1

def loadImagesFromDirectory(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (imageWidth, imageHeight))
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)

healthyImages, healthyLabels = loadImagesFromDirectory(r'C:\Users\Marcus Connelly\Downloads\noiseremovedhealthy', 0)
malignantImages, malignantLabels = loadImagesFromDirectory(r'C:\Users\Marcus Connelly\Downloads\noiseremovedmalignant', 1)

allImages = np.concatenate((healthyImages, malignantImages))
allLabels = np.concatenate((healthyLabels, malignantLabels))

allImages = allImages.reshape(-1, imageHeight, imageWidth, channels)

xTrain, xTest, yTrain, yTest = train_test_split(allImages, allLabels, test_size=0.2, random_state=42, stratify=allLabels)

xTrain, xVal, yTrain, yVal = train_test_split(xTrain, yTrain, test_size=0.2, random_state=42, stratify=yTrain)

xTrain = xTrain / 255.0
xVal = xVal / 255.0
xTest = xTest / 255.0

trainDataGen = ImageDataGenerator(
    rotation_range=2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.05,
    horizontal_flip=True,
)

valDataGen = ImageDataGenerator()
testDataGen = ImageDataGenerator()

trainDataGen.fit(xTrain)

trainGenerator = trainDataGen.flow(xTrain, yTrain, batch_size=64)
valGenerator = valDataGen.flow(xVal, yVal, batch_size=64)
testGenerator = testDataGen.flow(xTest, yTest, batch_size=64, shuffle=False)

inputs = tf.keras.Input(shape=(imageHeight, imageWidth, channels))

y = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(inputs)
y = tf.keras.layers.MaxPooling2D((2, 2))(y)
y = tf.keras.layers.Dropout(0.3)(y)

y = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(y)
y = tf.keras.layers.MaxPooling2D((2, 2))(y)
y = tf.keras.layers.Dropout(0.3)(y)

y = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(y)
y = tf.keras.layers.MaxPooling2D((2, 2))(y)
y = tf.keras.layers.Dropout(0.3)(y)

y = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(y)
y = tf.keras.layers.MaxPooling2D((2, 2))(y)
y = tf.keras.layers.Dropout(0.3)(y)

y = tf.keras.layers.Flatten()(y)

y = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01))(y)
y = tf.keras.layers.Dropout(0.3)(y)

outputs = tf.keras.layers.Dense(2, activation='softmax')(y)

model = tf.keras.Model(inputs=inputs, outputs=outputs)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

classWeight = {0: 1.25, 1: 0.8333}
earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
model.fit(trainGenerator, epochs=65, validation_data=valGenerator, callbacks=[earlyStopping], class_weight=classWeight)

loss, accuracy = model.evaluate(testGenerator)
print(f"Test accuracy: {accuracy:.2f}")

model.save('lung_cancer_model.h5')

newHealthyDir = r'C:\Users\Marcus Connelly\Downloads\sample'
newHealthyImages, newHealthyLabels = loadImagesFromDirectory(newHealthyDir, 0)

newHealthyImages = newHealthyImages.reshape(-1, imageHeight, imageWidth, channels) / 255.0

newLoss, newAccuracy = model.evaluate(newHealthyImages, newHealthyLabels)
print(f"Accuracy on the new healthy dataset: {newAccuracy:.2f}")
