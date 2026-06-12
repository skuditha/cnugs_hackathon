# Extracted code cells from cnugs_lecture_3_1_soln.ipynb


# %% [cell 1]
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
height = 5
plt.rcParams["figure.figsize"] = [1.618*height, height]

# %% [cell 3]
x = np.linspace(0, np.pi*2, 100)
y = 1.2*np.sin(x) + 0.1*np.random.randn(100)
plt.plot(x, y, "o")
plt.show()

# %% [cell 4]
def func(x, a, b):
  return a*np.sin(x) + b

# %% [cell 5]
from scipy.optimize import curve_fit

fit_soln = curve_fit(func, x, y)
popt, pcov = fit_soln # curve fit returns a tuple

# %% [cell 6]
# Print
print(f"a = {popt[0]:.3f} ± {np.sqrt(pcov[0][0]):.3f}")
print(f"b = {popt[1]:.3f} ± {np.sqrt(pcov[1][1]):.3f}")

# %% [cell 7]
plt.plot(x, y, 'o')
# plt.plot(x, func(x, popt[0], popt[1]), 'r-');
plt.plot(x, func(x, *popt), 'r-');

# %% [cell 8]
# Calculate and print chi^2/ndf
y_fit = func(x, *popt)
chi_sq = np.sum((y - y_fit)**2)
ndf = len(x) - len(popt)
print(f"chi^2/ndf = {chi_sq/ndf:.3f}")

# %% [cell 10]
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
df = pd.read_csv(url, names=col_names)

# %% [cell 11]
df

# %% [cell 12]
# Split into 3 dataframes
df_setosa = df[df['type'] == 'Iris-setosa']
df_versicolor = df[df['type'] == 'Iris-versicolor']
df_virginica = df[df['type'] == 'Iris-virginica']

# %% [cell 13]
# Plot Sepal Length vs Sepal Width
plt.scatter(df_setosa['sepal_length'], df_setosa['sepal_width'], label='Setosa')
plt.scatter(df_versicolor['sepal_length'], df_versicolor['sepal_width'], label='Versicolor')
plt.scatter(df_virginica['sepal_length'], df_virginica['sepal_width'], label='Virginica')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()

# %% [cell 14]
# Plot petal length vs petal width
plt.scatter(df_setosa['petal_length'], df_setosa['petal_width'], label='Setosa')
plt.scatter(df_versicolor['petal_length'], df_versicolor['petal_width'], label='Versicolor')
plt.scatter(df_virginica['petal_length'], df_virginica['petal_width'], label='Virginica')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()
plt.show();

# %% [cell 15]
# Sepal length vs petal length
plt.scatter(df_setosa['sepal_length'], df_setosa['petal_length'], label='Setosa')
plt.scatter(df_versicolor['sepal_length'], df_versicolor['petal_length'], label='Versicolor')
plt.scatter(df_virginica['sepal_length'], df_virginica['petal_length'], label='Virginica')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend()
plt.show();

# %% [cell 16]
# Convert type from string to int
species = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
df['type'] = df['type'].map(species)

# %% [cell 17]
data = df.drop('type', axis=1) # Remove the label
labels = df['type'];

# %% [cell 18]
df

# %% [cell 19]
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data,
                                                                    labels,
                                                                    test_size=0.2,
                                                                    random_state=27)

# %% [cell 20]
# Import kNN Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# %% [cell 21]
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_data, train_labels)

# %% [cell 22]
knn.score(data, labels)

# %% [cell 23]
scores = []
n_neighbors = []
for n in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(train_data, train_labels)
    scores.append(knn.score(test_data, test_labels))
    n_neighbors.append(n)

# %% [cell 24]
# Plot the score
plt.plot(n_neighbors, scores)
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
