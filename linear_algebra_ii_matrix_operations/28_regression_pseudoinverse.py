import numpy as np
import matplotlib.pyplot as plt

# if more (n, or rows of X) than (m, or columns of X) we can solve overdetermined situation
x1 = [0, 1, 2,  3, 4, 5, 6, 7.] # dosage of drug to treat alzhiemer's disease
y = [1.86, 1.31, .62, .33, .09, -.67, -1.23, -1.37] # patient's "forgetfulness score"

title = "Clinical Trial"
xlabel = "Drug dosage (mL)"
ylabel = "Forgetfulness"

# fig, ax = plt.subplots()
# plt.title(title)
# plt.xlabel(xlabel)
# plt.ylabel(ylabel)
# ax.scatter(x1, y)
# # plt.show()

# although one predictor (x1), need second (x0) to allow y-intercept (m = 2).
x0 = np.ones(8)

# concatenate x0 and x1 into matrix X
X = np.concatenate((np.matrix(x0).T, np.matrix(x1).T), axis=1)

# calculate weights (w) using w = X⁺y
w = np.dot(np.linalg.pinv(X), y)

# first weight corresponds to y-intercept of line denoted as b
b = np.asarray(w).reshape(-1)[0]

# second weight corresponds to slope of line denoted as m
m = np.asarray(w).reshape(-1)[1]

# use wieghts to plot line
fig, ax = plt.subplots()

plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)

ax.scatter(x1, y)

# create regression line
x_min, x_max = ax.get_xlim()
y_min, y_max = m*x_min + b, m*x_max + b

ax.set_xlim([x_min, x_max])
ax.plot([x_min, x_max],[y_min, y_max])
plt.show()
