import numpy as np
import torch
import tensorflow as tf

X = np.array([[25, 2], [5, 26], [3, 7]])
X_pt = torch.tensor([[25, 2], [5, 26], [3, 7]])
X_tf = tf.Variable([[25, 2], [5, 26], [3, 7]])

op_X_pt = X_pt*2+2 # Python operators overloaded, can use torch.mul() or torch.add()
op_X_tf = X_tf*2+2 # Python operators overloaded, can use tf.multiply() or tf.add()

A = X+2
A_pt = X_pt+2
A_tf = X_tf+2

# Hadamard product
hadamard_AX = A * X
hadamard_AX_pt = A_pt * X_pt
hadamard_AX_tf = A_tf * X_tf

print(hadamard_AX)