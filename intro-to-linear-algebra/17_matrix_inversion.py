import numpy as np
import torch
import tensorflow as tf
X = np.array([[4, 2], [-5, -3]])

Xinv = np.linalg.inv(X)

y = np.array([4, -7])

# y=X^-1y
w = np.dot(Xinv, y)

# shows y=Xw
y = np.dot(X, w)

# matrix inversion using pytorch and tensorflow
Xinv_pt = torch.inverse(torch.tensor([[4, 2], [-5, -3.]])) # use float type
Xinv_tf = tf.linalg.inv(tf.Variable([[4, 2], [-5, -3.]])) # use float type

# matrix inversion with no solution
X_no1 = np.array([[-4, 1], [-8, 2]]) # lines are parralel / don't overlap (row 2 is multiple of row 1)

# matrix inversion with no solution
X_no2 = np.array([[-4, 1], [-4, 1]])  # same lines twice

# returns a singular "matrix error"
X_inv_no = np.linalg.inv(X_no2)

print(X_inv_no)