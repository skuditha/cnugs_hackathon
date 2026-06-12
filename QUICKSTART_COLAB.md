# Quickstart in Google Colab

## 1. Clone and install

```python
from google.colab import drive
drive.mount("/content/drive")

!git clone <YOUR_REPO_URL> cnugs_hackathon_repo
%cd cnugs_hackathon_repo
!pip install -q -r requirements.txt
!pip install -q -e .
```

## 2. Define shared paths

```python
DATA_DIR = "/content/drive/MyDrive/CNUGS_Hackathon/data"
OUTPUT_DIR = "/content/drive/MyDrive/CNUGS_Hackathon/predictions"
MODEL_DIR = "/content/drive/MyDrive/CNUGS_Hackathon/model_checkpoints"
```

## 3. Start with data checks

Open:

```text
notebooks/00_quick_data_check.ipynb
```

Confirm:

- data shape,
- dtype,
- min/max values,
- image samples,
- label balance or target distribution,
- expected submission format.

## 4. Then copy a baseline

Use the closest matching notebook:

| Problem type | Notebook |
|---|---|
| Image classification | `01_image_classifier_baseline.ipynb` |
| Denoising / image reconstruction | `02_denoising_autoencoder_baseline.ipynb` |
| Image-to-number regression | `03_image_regression_baseline.ipynb` |
| Tabular/classical ML/curve fitting | `04_tabular_knn_curvefit_baseline.ipynb` |
