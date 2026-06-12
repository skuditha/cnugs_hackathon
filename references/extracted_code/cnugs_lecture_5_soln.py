# Extracted code cells from cnugs_lecture_5_soln.ipynb


# %% [cell 2]
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
# Let's add some noise
clean_images = train_images.astype("float32") / 255.0
clean_images = clean_images[..., None]
clean_test_images = test_images.astype("float32") / 255.0
clean_test_images = clean_test_images[..., None]

noise_sigma = 0.3
noisy_images = clean_images + noise_sigma * np.random.normal(size=clean_images.shape)
noisy_images = np.clip(noisy_images, 0, 1)
noisy_test_images = clean_test_images + noise_sigma * np.random.normal(size=clean_test_images.shape)
noisy_test_images = noisy_test_images[..., None]

# %% [cell 7]
# Plot 6 images from the noisy training set
plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(noisy_images[i], cmap='gray')
    plt.title(f"Label: {train_labels[i]}")
    plt.axis('off')

# %% [cell 8]
# let's build a denoising autoencoder
denoiser = keras.Sequential()
denoiser.add(layers.Input(shape=(28, 28, 1)))
denoiser.add(layers.Conv2D(32, 3, padding="same", activation="relu"))
denoiser.add(layers.Conv2D(32, 3, padding="same", activation="relu"))
denoiser.add(layers.Conv2D(1, 3, padding="same", activation="linear"))

denoiser.compile(optimizer=keras.optimizers.Adam(1e-3),loss="mse")

history = denoiser.fit(
    noisy_images,
    clean_images,
    validation_split=.2,
    epochs=10,
    batch_size=128
)

# %% [cell 9]
# Let's see if it worked and take 5 noisy input images and see if it removes the noise
cleaned_test_images = denoiser.predict(noisy_test_images[:5])

# Plot the images side by side noisy vs clean
plt.figure(figsize=(8,5))
for i in range(5):
    plt.subplot(3, 5, i + 1)
    plt.imshow(noisy_test_images[i].squeeze(), cmap='gray')
    plt.title(f"Noisy")
    plt.axis('off')
    plt.subplot(3, 5, i + 6)
    plt.imshow(cleaned_test_images[i].squeeze(), cmap='gray')
    plt.title(f"Clean")
    plt.axis('off')
    plt.subplot(3, 5, i + 11)
    plt.imshow(clean_test_images[i].squeeze(), cmap='gray')
    plt.title(f"Original")
    plt.axis('off')

# %% [cell 10]
# Get accuracy
test_loss = denoiser.evaluate(np.squeeze(noisy_test_images, axis=-1), clean_test_images)
print(f"Test loss: {test_loss}")

# %% [cell 11]
# Regression example. Let's take the MNIST data and create a regression for the total ink used
total_ink = np.sum(train_images, axis=(1, 2))
total_ink_test = np.sum(test_images, axis=(1, 2))

# %% [cell 12]
total_ink

# %% [cell 13]
plt.figure(figsize=(6, 4))
plt.hist(total_ink, bins=50)
plt.xlabel("Total ink")
plt.ylabel("Number of images")
plt.title("Distribution of MNIST total ink")
plt.show()

# %% [cell 14]
model = keras.Sequential()
model.add(layers.Input(shape=(28, 28, 1)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(1, activation="linear"))

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"]
)

model.summary()

# %% [cell 15]
history = model.fit(train_images,total_ink, epochs=10, batch_size=128, validation_split=.2)

# %% [cell 16]
# Evaluate the model
test_loss, test_mae = model.evaluate(test_images, total_ink_test)
print(f"Test loss: {test_loss}")
print(f"Test MAE: {test_mae}")

# %% [cell 17]
# Show the first ten images with total ink prediction and actual total ink in the title
predictions = model.predict(test_images[:10])
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[i], cmap='gray')
    plt.title(f"{predictions[i][0]:.0f}/{total_ink_test[i]:.0f}")
    plt.axis('off')
