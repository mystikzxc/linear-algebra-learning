import numpy as np
import torch

# calculate pseudoinverse A⁺ of matrix A
# A⁺ = VD⁺Uᵀ

A = np.array([[-1, 2], [3, -2], [5, 7]])

# using SVD to get U, d and Vᵀ
U, d, VT = np.linalg.svd(A)

# create D⁺, first invert non-zero values of d
D = np.diag(d)

# manually invert 
col1_D = 1/8.669
col2_D = 1/4.104
# then take transpose of resulting matrix

# becuase D is diagonal matrix, can invert D
Dinv = np.linalg.inv(D)

# D⁺ must have same dimensions A⁺ for VD⁺Uᵀ matrix multiplication
Dplus = np.concatenate((Dinv, np.array([[0, 0]]).T), axis=1)

# calculate A⁺ for VD⁺Uᵀ
Aplus = np.dot(VT.T, np.dot(Dplus, U.T))

# can calculate using numpy pinv() method
Aplus_np = np.linalg.pinv(A)

# using torch to calculate pseudoinverse A⁺ of matrix A
A = torch.tensor([[-1, 2], [3, -2], [5, 7.]])

# using svd to get U, d and Vᵀ
U, d, VT = torch.linalg.svd(A)

# convert d to diagonal matrix and invert
D = torch.diag(d)
Dinv = torch.inverse(D)

# D⁺ must have same dimensions as A⁺
Dplus = torch.concatenate((Dinv, torch.tensor([[0, 0]]).T), axis=1)

# calculate A⁺ for VD⁺Uᵀ
Aplus = np.dot(VT.T, np.dot(Dplus, U.T))
print(Aplus)

# calculate using torch.linalg.pinv()
Aplus_pt = torch.linalg.pinv(A)
print(Aplus_pt)