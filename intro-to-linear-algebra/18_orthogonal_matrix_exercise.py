import numpy as np

x1 = np.array([1, 0, 0])
x2 = np.array([0, 1, 0])
x3 = np.array([0, 0, 1])

# get dot product of x1 and x2 to check if orthogonal (= 0)
dot_x1_x2 = np.dot(x1, x2)

# print(f"dot product (x1 and x2): {dot_x1_x2}")

# get norms of each vector x1, x2, x3 to check for unit norm (= 1)
norm_x1 = np.linalg.norm(x1)
norm_x2 = np.linalg.norm(x2)
norm_x3 = np.linalg.norm(x3)

# print(f"norm (x1): {norm_x1}\nnorm (x2): {norm_x2}\nnorm (x3): {norm_x3}")

K = np.array([[(2/3), (1/3), (2/3)], [(-2/3), (2/3), (1/3)], [(1/3), (2/3), (-2/3)]])

# split K matrix columns
k1 = K[:, 0]
k2 = K[:, 1]
k3 = K[:, 2]

# get dot product of k1 and k2
dot_k1_k2 = np.dot(k1, k2)
print(f"dot product (k1 and k2): {dot_k1_k2}")

# get norms of each vector k1, k2, k3
norm_k1 = np.linalg.norm(k1)
norm_k2 = np.linalg.norm(k2)
norm_k3 = np.linalg.norm(k3)

print(f"norm (k1): {norm_k1}\nnorm (k2): {norm_x2}\nnorm (k3): {norm_k3}")