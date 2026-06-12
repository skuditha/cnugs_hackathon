# Extracted code cells from cnugs_lecture_4_soln.ipynb


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
history = model.fit(train_images, train_labels, epochs=10, validation_split=0.2)

# %% [cell 8]
# Plot loss and val loss
plt.figure()
plt.plot(history.history['loss'][1:], label='Training Loss')
plt.plot(history.history['val_loss'][1:], label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend();

# %% [cell 11]
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

# Example of early stopping with callbacks
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(train_images,
                    train_labels,
                    epochs=50,
                    validation_split=0.2,
                    callbacks=[early_stopping])

# %% [cell 12]
# Plot loss and val loss
plt.figure()
plt.plot(history.history['loss'][1:], label='Training Loss')
plt.plot(history.history['val_loss'][1:], label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend();

# %% [cell 14]
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

# %% [cell 15]
# Plot confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
predictions = model.predict(test_images)
predictions = predictions.argmax(axis=1)
cm = confusion_matrix(test_labels, predictions)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True Label')
plt.title("Confusion Matrix")
plt.show()

# %% [cell 17]
from google.colab import drive
drive.mount('/content/drive')
import os

# %% [cell 18]
# Install Keras Tuner
!pip install -q -U keras-tuner

# %% [cell 19]
TUNING_DIR = "/content/drive/MyDrive/mnist_tuning_lecture4"

os.makedirs(TUNING_DIR, exist_ok=True)

# %% [cell 20]
# Helper function that builds our model
def build_model(hp):

    neurons = hp.Choice("neurons", values=[256, 512, 1024]) # First hyperparameter
    n_layers = hp.Choice("n_layers", values=[1, 2, 3]) # Second hyperparameter

    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))  # Flatten the input images
    for i in range(n_layers):
      model.add(layers.Dense(neurons, activation='relu'))  # Hidden layer
    model.add(layers.Dense(10, activation='softmax'))  # Output layer with 10
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# %% [cell 21]
import keras_tuner as kt

# %% [cell 22]
import keras_tuner as kt

tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=5,
    executions_per_trial=2,
    directory=TUNING_DIR,
    overwrite=False,
    project_name="n_neuron_search")

tuner.search(train_images,
             train_labels,
             epochs=10,
             validation_split=0.2,
             verbose=1)

best_hp = tuner.get_best_hyperparameters(1)[0]

print(f"Best # neurons per layer: {best_hp}")

# %% [cell 23]
print(f"Best # neurons per layer: {best_hp.get("neurons")}")

# %% [cell 25]
# Build a CNN model
model = keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# # Compile the model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# Compile the model with adam with the learning rate set 1E-4
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1E-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %% [cell 26]
# Model description
model.summary()

# %% [cell 27]
# Create an early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Train the model
history = model.fit(train_images, train_labels, epochs=100, validation_split=0.2, callbacks=[early_stopping])

# %% [cell 28]
# Plot history
plt.figure()
plt.plot(history.history['loss'][1:], label='Training Loss')
plt.plot(history.history['val_loss'][1:], label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot()

# %% [cell 29]
# Look at accuracy
model.evaluate(test_images, test_labels)

# %% [cell 31]
# load training data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# %% [cell 32]
# Normalization step, optional but improves results
import numpy as np
np.max(train_images)
if np.max(train_images) > 1:
  train_images = train_images / 255
  test_images = test_images / 255

# %% [cell 33]
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# %% [cell 34]
# Plot 25 images
plt.figure(figsize=(20, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# %% [cell 35]
train_images.shape

# %% [cell 36]
model = keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %% [cell 37]
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(train_images, train_labels, epochs=100, validation_split=0.2)#, callbacks=[early_stopping])

# %% [cell 38]
# Plot loss and val loss
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend();
