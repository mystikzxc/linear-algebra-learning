from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()

# get iris data shape and columns
iris_shape = iris.data.shape
iris_features = iris.get("feature_names")

# check 6 iris flowers
iris_6 = iris.data[0:6, :]

# return 2 principal components
pca = PCA(n_components=2)

# fit the pca model into 2 columns
X = pca.fit_transform(iris.data)

X_shape = X.shape
X_6 = X[0:6, :]

# create scatter plot using X
plt.scatter(X[:, 0], X[:, 1])
# plt.show()

# show iris label shape check 6 iris targetss
iris_target = iris.target.shape
target_6 = iris.target[0:6]

# show unique elements of iris and convert into array
unique_elements, count_elements = np.unique(iris.target, return_counts=True)
iris_unique = np.asarray((unique_elements, count_elements))

# show target names
iris_names = list(iris.target_names)

# create scatter plot with colour
plt.scatter(X[:, 0], X[:, 1], c=iris.target)
plt.show()