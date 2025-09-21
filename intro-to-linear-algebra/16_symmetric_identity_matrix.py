import numpy as np
import torch
import tensorflow as tf

# symmetric matrix
X = np.array([[0, 1, 2], [1, 7, 8], [2, 8, 9]])

X_T = X.T

X_is_sym = X == X.T

# identity matrix
I_pt = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
x_pt = torch.tensor([25, 2, 5])

I_tf = tf.Variable([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
x_tf = tf.Variable([25, 2, 5])

mul_Ix_pt = torch.matmul(I_pt, x_pt)

mul_Ix_tf = tf.linalg.matvec(I_tf, x_tf)

print(mul_Ix_tf)