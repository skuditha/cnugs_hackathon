# Extracted code cells from cnugs_lecture_2_2.ipynb


# %% [cell 0]
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')
import numpy as np

# %% [cell 1]
df = pd.read_csv("/content/drive/My Drive/nvda.csv")

# %% [cell 2]
df

# %% [cell 3]
# Date is a str, need to convert to datetime
df["Date"] = pd.to_datetime(df["Date"])

# %% [cell 4]
df["Date"]

# %% [cell 5]
df.dtypes

# %% [cell 6]
df["Volume"] = df["Volume"].str.replace(",","", regex=True)
df["Volume"] = df["Volume"].astype(np.int64)

# %% [cell 8]
# # If you make the date the index
# df["11-29-2025:11-30-2025"]

plt.plot(df["Date"], df["Close"])
plt.xlabel("Date")
plt.ylabel("NVDA Closing Price [$ USD]");

# %% [cell 9]
# Practice creating one column
df["Delta"] = df["High"] - df["Low"]

# %% [cell 10]
df

# %% [cell 11]
# Practice deleting column
df.drop("Delta", axis=1)
