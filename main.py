### <------- IMPORTS -------> ###

# External imports
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist

# Internal imports
import imageProcessing as impro


# Loading the MNIST dataset
(training_images, training_labels), (validation_images, validation_labels) = mnist.load_data()

preprocessing = impro.ImageProcessing(training_images, training_labels)


# Normalize the data
training_images = preprocessing.normalize(rescale = 1./255.)


# Plot function
indexes = [0, 1, 2, 9]
preprocessing.plot(indexes=indexes)


# Convert to numpy array
training_images = np.array(training_images)
training_labels = np.array(training_labels)


# Create the model architecture
model = tf.keras.models.Sequential([
	tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
	tf.keras.layers.MaxPooling2D((2, 2)),
	tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
	tf.keras.layers.MaxPooling2D((2, 2)),

	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(tf.keras.optimizers.RMSprop(learning_rate=0.001),
		  tf.keras.losses.sparse_categorical_crossentropy,
		  metrics=['accuracy'])

print(training_images)

#history = model.fit(training_images, training_labels, 10)