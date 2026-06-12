# CNUGS AI/ML Hackathon Starter Repository

This repository is a lightweight collaborative starter kit built from the CNUGS AI/ML Bootcamp notebooks. It is designed for a team hackathon where you need to move fast, avoid notebook chaos, and produce a valid submission early.

## Recommended team workflow

1. **Clone this repo in Colab or locally.**
2. **Do not all edit the same notebook.** Copy a baseline notebook for each experiment.
3. **Keep reusable code in `src/cnugs_ml/`.**
4. **Track every experiment in `experiments/experiment_log.csv`.**
5. **Use one final integration notebook** for the final submission.

## Fast Colab setup

```python
from google.colab import drive
drive.mount("/content/drive")

!git clone <YOUR_REPO_URL> cnugs_hackathon_repo
%cd cnugs_hackathon_repo
!pip install -q -r requirements.txt
!pip install -q -e .
```

If the dataset is shared through Google Drive, set paths like this:

```python
DATA_DIR = "/content/drive/MyDrive/CNUGS_Hackathon/data"
OUTPUT_DIR = "/content/drive/MyDrive/CNUGS_Hackathon/predictions"
```

## Repository layout

```text
cnugs_hackathon_repo/
├── notebooks/
│   ├── 00_quick_data_check.ipynb
│   ├── 01_image_classifier_baseline.ipynb
│   ├── 02_denoising_autoencoder_baseline.ipynb
│   ├── 03_image_regression_baseline.ipynb
│   └── 04_tabular_knn_curvefit_baseline.ipynb
├── src/cnugs_ml/
│   ├── data.py
│   ├── models.py
│   ├── plots.py
│   ├── train.py
│   ├── regression.py
│   ├── classical.py
│   └── submission.py
├── experiments/
│   └── experiment_log.csv
├── scripts/
│   └── smoke_test.py
├── references/
│   ├── original_notebooks/
│   └── extracted_code/
├── data/
├── models/
└── outputs/
```

## What this repo is optimized for

The bootcamp notebooks covered exactly the patterns this repo emphasizes:

- NumPy array inspection, reshaping, normalization, and image preprocessing.
- Matplotlib plotting and quick visual checks.
- Pandas loading/cleaning for tabular data.
- Scipy `curve_fit` for regression.
- Scikit-learn kNN for simple classification baselines.
- Keras/TensorFlow MLPs, CNNs, early stopping, loss curves, confusion matrices.
- Denoising autoencoder and image-to-image reconstruction patterns.
- Image regression, such as predicting total ink in MNIST.

## Hackathon principle

Get a valid baseline first. Then improve one piece at a time.

The most common failure modes are:

- Bad image shape: `(N,H,W)` vs `(N,H,W,1)`.
- Bad pixel scale: `0-255` vs `0-1`.
- Wrong target/output pairing: classifier vs regressor vs image-to-image model.
- Training on the test set accidentally.
- Changing too many things at once.

Use `00_quick_data_check.ipynb` before modeling anything.
