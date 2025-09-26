import numpy as np
import torch

# eigendecomposition of A
# A = VAV-1
# V is concatenation of eigenvectors A
# A (uppercase lambda) is matrix diag(lambda)
A = np.array([[4, 2], [-5, -3]])

lambdas, V = np.linalg.eig(A)

Vinv = np.linalg.inv(V)

Lambda = np.diag(lambdas)

# confirm A = VAV-1
eig_decom = np.dot(V, np.dot(Lambda, Vinv))

# if A is symmetric matrix
# A = QAQ^T
# Q is analagous to V from previous equation except it's an orthogonal matrix
A = np.array([[2, 1], [1, 2]])

lambdas, Q = np.linalg.eig(A)

Lambda = np.diag(lambdas)

# confirm A = QAQ^T
eig_decom = np.dot(Q, np.dot(Lambda, Q.T))

# can demonstrate Q is orthogonal matrix Q^TQ = QQ^T = I
QTQ = np.dot(Q.T, Q)
QQT = np.dot(Q, Q.T)

# using pytorch to decompose matrix P, P = VAV-1
P = torch.tensor([[25, 2, -5], [3, -2, 1], [5, 7, 4.]])

eigens = torch.linalg.eig(P)
lambdaP = eigens.eigenvalues
V = eigens.eigenvectors

Vinv = torch.inverse(V)
Lambda = torch.diag(lambdaP)

# confirm P = VAV-1
eig_decom = torch.matmul(V, torch.matmul(Lambda, Vinv))

# using pytorch to decompose symmetric matrix S, S = QAQ^T
P = torch.tensor([[25, 2, -5], [2, -2, 1], [-5, 1, 4.]])

eigens = torch.linalg.eig(P)
lambdas = eigens.eigenvalues
Q = eigens.eigenvectors

Lambda = torch.diag(lambdas)

# confirm A = QAQ^T
eig_decom = torch.matmul(Q, torch.matmul(Lambda, Q.T))

# can demonstrate Q is orthogonal matrix Q^TQ = QQ^T = I
QTQ = torch.matmul(Q.T, Q)
QQT = torch.matmul(Q, Q.T)
print(QTQ)
print(QQT)