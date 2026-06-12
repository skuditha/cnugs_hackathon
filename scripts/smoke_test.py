"""Small smoke test for the helper package.

Run from repo root:

    python scripts/smoke_test.py

This intentionally avoids TensorFlow training so it can run quickly in a plain
Python environment.
"""

import numpy as np

from cnugs_ml.data import add_gaussian_noise, describe_array, ensure_channel_dim, normalize_images
from cnugs_ml.regression import chi2_ndf, fit_curve, sine_model
from cnugs_ml.classical import load_iris_dataframe, train_knn_classifier
from sklearn.model_selection import train_test_split


raw = np.random.randint(0, 256, size=(10, 28, 28), dtype=np.uint8)
x = ensure_channel_dim(normalize_images(raw))
noisy = add_gaussian_noise(x, sigma=0.1, seed=42)
describe_array("smoke images", noisy)
assert noisy.shape == (10, 28, 28, 1)
assert noisy.min() >= 0 and noisy.max() <= 1

xx = np.linspace(0, 2 * np.pi, 50)
yy = 1.2 * np.sin(xx) + 0.05 * np.random.randn(50)
popt, pcov, perr = fit_curve(sine_model, xx, yy)
assert len(popt) == 2
print("curve fit:", popt, perr, chi2_ndf(yy, sine_model(xx, *popt), len(popt)))

df = load_iris_dataframe()
X = df.drop("type", axis=1)
y = df["type"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
model = train_knn_classifier(X_train, y_train, n_neighbors=3)
print("kNN accuracy:", model.score(X_test, y_test))

print("smoke test passed")
