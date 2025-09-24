import numpy as np
import torch
import tensorflow as tf

A = np.array([[3, 4], [5, 6], [7, 8]])
B = np.array([1, 2])

A_pt = torch.tensor([[3, 4], [5, 6], [7, 8]])
B_pt = torch.tensor([1, 2])

A_tf = tf.Variable([[3, 4], [5, 6], [7, 8]])
B_tf = tf.Variable([1, 2])

dot_AB = np.dot(A, B) # even technically dot products are between vectors only

mul_AB_pt = torch.matmul(A_pt, B_pt) # like np.dot(), automatically infers dims in order to perform dot product, matvec, or matrix multiplication

mul_AB_tf = tf.linalg.matvec(A_tf, B_tf) # tf.linalg.matvec() for matrix by vector multiplication

C = np.array([[1, 9], [2, 0]])

# matrix multiplication of AC
dot_AC = np.dot(A, C)

# matrix multiplication is not commutative (A*C != C*C) // will throw a dim error
# print(np.dot(C, A))

C_pt = torch.from_numpy(C) # converts numpy array to pytorch
C_tf = tf.convert_to_tensor(C, dtype="int32") # converts numpy array to tensorflow

# way to create matrix with transposition
C_pt = torch.tensor([[1, 2], [9, 0]]).T

mul_AC_pt = torch.matmul(A_pt, C_pt)
mul_AC_tf = tf.matmul(A_tf, C_tf) # tf.matmul() for matrix by matrix multiplication

print(mul_AC_tf)