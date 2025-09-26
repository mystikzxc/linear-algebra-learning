import numpy as np
import torch

# A = UDV^T
# U (orthogonal m x m matrix) columns left-singular vector of A
# V (orthogonal n x n matrix) columns right-singular vector of A
# D (diagonal m x n matrix) elements along diagonal are singular values of A

A = np.array([[-1, 2], [3, -2], [5, 7]])

U, d, VT = np.linalg.svd(A) # V is already transposed

D = np.concatenate((np.diag(d), [[0, 0]]), axis=0) # D must have same dimensions as A for UDV^T matrix multiplication

# confirm UDV^T
svd = np.dot(U, np.dot(D, VT))

# P = torch.tensor([[-1, 2], [3, -2], [5, 7.]])

# svd = torch.linalg.svd(P)
# U = svd.U
# VT = svd.Vh
# d = svd.S

# D = torch.concat((torch.diag(d), torch.tensor([[0, 0]])), axis=0)

# svd_P = torch.matmul(U, torch.matmul(D, VT))
# print(svd_P)

# SVD and eigendecomposition are related
# Left-singular vectors of A = eigenvectors of AA^T
# Right-singular vectors of A = eigenvectors pf A^TA
# Non-zero singular vectors of A = square roots of eigenvectors of AA^T = square roots of A^TA
P = torch.tensor([[25, 2, -5], [2, -2, 1], [-5, 1, 4.]])

U, d, VT = torch.linalg.svd(P)
D = torch.diag(d)

# # confirm A = UDV^T
svd_P = torch.matmul(U, torch.matmul(D, VT))

# # Left-singular vectors of A = eigenvectors of AA^T
PPT = torch.matmul(P, P.T)
lambdas_PPT, V_PPT = torch.linalg.eig(PPT)
print(f"U = {U}")
print(f"AAT = {V_PPT}")

# Right-singular vectors of A = eigenvectors pf A^TA
PTP = torch.matmul(P.T, P)
lambdas_PTP, V_PTP = torch.linalg.eig(PTP)
print(f"V = {VT}")
print(f"ATA = {V_PTP}")

# # Non-zero singular vectors of A = square roots of eigenvectors AA^T = square roots of eigenvectors A^TA
print(f"d = {d}")
print(f"sqrt_AAT = {torch.sqrt(lambdas_PPT)}")
print(f"sqrt_ATA = {torch.sqrt(lambdas_PTP)}")