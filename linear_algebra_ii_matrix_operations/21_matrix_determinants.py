import numpy as np
import torch

X = np.array([[4, 2], [-5, -3]])
det_X = np.linalg.det(X)

N = np.array([[-4, 1], [-8, 2]])
det_N = np.linalg.det(N)

# line results in "singular matrix" error since N doesn't have indipendent columns
# Ninv = np.linalg.inv(N)

N_pt = torch.tensor([[-4, 1], [-8, 2.]])
det_N_pt = torch.det(N_pt)

print(det_N_pt)