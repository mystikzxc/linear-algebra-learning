import numpy as np
import torch

# eigendecomposition of A
# A = VΛV⁻¹
# V is concatenation of eigenvectors A
# Λ (uppercase lambda) is matrix diag(λ)
A = np.array([[4, 2], [-5, -3]])

lambdas, V = np.linalg.eig(A)

Vinv = np.linalg.inv(V)

Lambda = np.diag(lambdas)

# confirm A = VΛV⁻¹
eig_decom = np.dot(V, np.dot(Lambda, Vinv))

# if A is symmetric matrix
# A = QΛQᵀ
# Q is analagous to V from previous equation except it's an orthogonal matrix
A = np.array([[2, 1], [1, 2]])

lambdas, Q = np.linalg.eig(A)

Lambda = np.diag(lambdas)

# confirm A = Q∧Qᵀ
eig_decom = np.dot(Q, np.dot(Lambda, Q.T))

# can demonstrate Q is orthogonal matrix QᵀQ = QQᵀ = I
QTQ = np.dot(Q.T, Q)
QQT = np.dot(Q, Q.T)

# using pytorch to decompose matrix P, P = V∧V⁻¹
P = torch.tensor([[25, 2, -5], [3, -2, 1], [5, 7, 4.]])

eigens = torch.linalg.eig(P)
lambdaP = eigens.eigenvalues
V = eigens.eigenvectors

Vinv = torch.inverse(V)
Lambda = torch.diag(lambdaP)

# confirm P = V∧V⁻¹
eig_decom = torch.matmul(V, torch.matmul(Lambda, Vinv))

# using pytorch to decompose symmetric matrix S, S = Q∧Qᵀ
P = torch.tensor([[25, 2, -5], [2, -2, 1], [-5, 1, 4.]])

eigens = torch.linalg.eig(P)
lambdas = eigens.eigenvalues
Q = eigens.eigenvectors

Lambda = torch.diag(lambdas)

# confirm A = Q∧Qᵀ
eig_decom = torch.matmul(Q, torch.matmul(Lambda, Q.T))

# can demonstrate Q is orthogonal matrix QᵀQ = QQᵀ = I
QTQ = torch.matmul(Q.T, Q)
QQT = torch.matmul(Q, Q.T)
