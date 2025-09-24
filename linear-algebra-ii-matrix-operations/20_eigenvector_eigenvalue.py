import numpy as np
import matplotlib.pyplot as plt
import torch

A = np.array([[-1, 4], [2, -2]])

# plot vector using plot_vectors()
def plot_vectors(vectors, colors):
    plt.figure()
    plt.axvline(x=0, color="lightgray")
    plt.axhline(y=0, color="lightgrey")

    for i in range(len(vectors)):
        x = np.concatenate([[0, 0], vectors[i]])
        plt.quiver([x[0]], [x[1]], [x[2]], [x[3]],
                   angles="xy", scale_units="xy", scale=1, color=colors[i],)

# lambdas to receive eigenvalues and V to receive eigenvectors
lambdas, V = np.linalg.eig(A)

# print(V) # each column is a separate eigenvector v

# print(lambdas) # each eigenvector has a coresponding eigenvalue

# confirm Av = lambdav for first eigenvector
v = V[:,0]

lamduh = lambdas[0] # lambda is a python function

# Av = lambdav
Av = np.dot(A, v)
lam_v = lamduh * v

# plot_vectors([Av, v], ['blue', 'lightblue'])
# plt.xlim(-1, 2)
# plt.ylim(-1, 2)
# plt.show()

# show for second eigenvector of A
v2 = V[:,1]

lamduh2 = lambdas[1]

Av2 = np.dot(A, v2)
lam_v2 = lamduh2 * v2

# plot_vectors([Av, v, Av2, v2],
#              ['blue', 'lightblue', 'green', 'lightgreen'])
# plt.xlim(-1, 4)
# plt.ylim(-3, 2)
# plt.show()

# using pytorch to get eigenvector and eigenvalue
A_pt = torch.tensor([[-1, 4.], [2, -2.]]) # must be float for torch.eig()

eigens = torch.linalg.eig(A_pt)

# specify first eigenvector and eigenvalue
v_pt = eigens.eigenvectors[:,0].float()
lambda_pt = eigens.eigenvalues[0].float()

Av_pt = torch.matmul(A_pt, v_pt)
lam_v_pt = lambda_pt * v_pt

# specify second eigenvector and eigenvalue
v2_pt = eigens.eigenvectors[:,1].float()
lambda2_pt = eigens.eigenvalues[1].float()

Av2_pt = torch.matmul(A_pt, v2_pt)
lam2_v_pt = lambda2_pt * v2_pt

# plot_vectors([Av_pt.numpy(), v_pt.numpy(), Av2_pt.numpy(), v2_pt.numpy()],
#              ['blue', 'lightblue', 'green', 'lightgreen'])
# plt.xlim(-1, 4)
# plt.ylim(-3, 2)
# plt.show()

# eigenvectors in > 2 dimensions
X = np.array([[25, 2, 9], [5, 26, -5], [3, 7, -1]])

# lambdas_X (one eigenvalue per column) // V_X (corresponding eigenvalue per eigenvector)
lambdas_X, V_X = np.linalg.eig(X)

# confirm Xv = lambdav
v_x = V_X[:,0]
lambda_x = lambdas_X[0]

X_v = np.dot(X, v_x) # matrix multiplication
lambda_v = lambda_x * v_x

# using pytorch for eigenvectors > 2 dimensions
X_pt = torch.tensor([[25, 2, 9.], [5, 26, -5.], [3, 7, -1.]])

# function to perform matrix multiplication
def matrix_multiply(X, v):
    return torch.matmul(X, v)

def check_equal(Xv, lambdav):
    return Xv.int() == lambdav.int()

# get eigenvectors and eigenvalues
eigens_X = torch.linalg.eig(X_pt)

x_pt = eigens_X.eigenvectors[:,0].float() # first
lambda_x_pt = eigens_X.eigenvalues[0].float()
x_v = matrix_multiply(X_pt, x_pt)
lambda_v_pt = lambda_x_pt * x_pt

x2_pt = eigens_X.eigenvectors[:,1].float() # second
lambda_x2_pt = eigens_X.eigenvalues[1].float()
x2_v = matrix_multiply(X_pt, x2_pt)
lambda_v2_pt = lambda_x2_pt * x2_pt

x3_pt = eigens_X.eigenvectors[:,2].float() # third
lambda_x3_pt = eigens_X.eigenvalues[2].float()
x3_v = matrix_multiply(X_pt, x3_pt)
lambda_v3_pt = lambda_x3_pt * x3_pt

# show Xv and lambdav and check if they are equal
print(f"FIRST:\nXv: {x_v} lambdav: {lambda_v_pt}\nXv == lambdav: {check_equal(x_v, lambda_v_pt)}")
print(f"SECOND:\nXv: {x2_v} lambdav: {lambda_v2_pt}\nXv == lambdav: {check_equal(x2_v, lambda_v2_pt)}")
print(f"THIRD\nXv: {x3_v} lambdav: {lambda_v3_pt}\nXv == lambdav: {check_equal(x3_v, lambda_v3_pt)}")