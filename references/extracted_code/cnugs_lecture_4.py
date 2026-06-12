# Extracted code cells from cnugs_lecture_4.ipynb


# %% [cell 2]
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import matplotlib.pyplot as plt
height = 5
plt.rcParams["figure.figsize"] = [1.618*height, height]

# %% [cell 4]
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# %% [cell 5]
# Plot 6 images from the training set
plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(train_images[i], cmap='gray')
    plt.title(f"Label: {train_labels[i]}")
    plt.axis('off')

# %% [cell 6]
# Build a MLP model
model = keras.Sequential()
model.add(layers.Flatten(input_shape=(28, 28)))  # Flatten the input images
model.add(layers.Dense(512, activation='relu'))  # Hidden layer with 128
model.add(layers.Dense(512, activation='relu'))  # Hidden layer with 128
model.add(layers.Dense(512, activation='relu'))  # Hidden layer with 128
model.add(layers.Dense(10, activation='softmax'))  # Output layer with 10 classes

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %% [cell 7]
# Train the model
history = model.fit(train_images, train_labels, epochs=20, validation_split=0.2)

# %% [cell 8]
# Plot loss and val loss
plt.figure()
plt.plot(history.history['loss'][1:], label='Training Loss')
plt.plot(history.history['val_loss'][1:], label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
