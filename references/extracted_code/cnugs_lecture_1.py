# Extracted code cells from cnugs_lecture_1.ipynb


# %% [cell 1]
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# %% [cell 2]
# Data from p74 of Visual Display of Quantitative Information.
x = list(range(1951, 1960))
y = [264, 231, 274, 241, 322, 286, 283, 247, 243]

# Plot line, line masks, then dots.
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, y, linestyle='-', color='black')
ax.grid()
# ax.scatter(x, y, color='white', s=100, zorder=2)
# ax.scatter(x, y, color='black', s=20, zorder=3)

# %% [cell 3]
# Anscombe's quartet #1.
x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]

# Plot the dots.
fig, ax = plt.subplots()
_ = ax.scatter(x, y, color='black', s=20)

# %% [cell 4]
# Data from p128 of Visual Display of Quantitative Information.
y = [9, 12, 6.5, 7, 3, 18, 13, 8.5, 6, 11, 5, 9.5]
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']

# Plot the bars.
fig, ax = plt.subplots()
x = list(range(len(y)))
ax.bar(x, y, color='#7a7a7a', width=0.6)
