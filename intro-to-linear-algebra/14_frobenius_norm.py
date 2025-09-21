import numpy as np
import torch
import tensorflow as tf

X = np.array([[1, 2], [3, 4]])
X_pt = torch.tensor([[1, 2], [3, 4.]])
X_tf = tf.Variable([[1, 2], [3, 4.]])

# calculate frobenius norm of matrix X
X_fnorm = (1**2 + 2**2 + 3**2 + 4**2) ** (1/2)

# calculate frobenius norm of matrix X using numpy
X_fnorm_np = np.linalg.norm(X) 

# calculate frobenius norm of matrix X using pytorch
X_fnorm_pt = torch.norm(X_pt) # torch.norm() requires float type

X_fnorm_tf = tf.norm(X_tf) # tf.norm() requires float type

print(X_fnorm_tf)