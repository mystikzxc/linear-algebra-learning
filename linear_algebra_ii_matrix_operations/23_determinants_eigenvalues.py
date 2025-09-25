import numpy as np
import matplotlib.pyplot as plt

def plot_vectors(vectors, colors):
    plt.figure()
    plt.axvline(x=0, color='lightgrey')
    plt.axhline(y=0, color='lightgrey')

    for i in range(len(vectors)):
        x = np.concatenate([[0, 0], vectors[i]])
        plt.quiver(x[0], x[1], x[2], x[3],
                   angles='xy', scale_units='xy', scale=1, color=colors[i])
        
def vectorfy(mtrx, clmn):
    return np.array(mtrx[:,clmn]).reshape(-1)

X = np.array([[1, 2, 4], [2, -1, 3], [0, 5, 1]])

lambdas, V = np.linalg.eig(X)

# det(X) == product of eigen values
product_lambdas = np.prod(lambdas)

abs_det_X = np.abs(np.linalg.det(X))

# using matrix B (basis vectors) to explore the impact of applying matrices to |det(X)|
B = np.array([[1, 0], [0, 1]])

# plot_vectors([vectorfy(B, 0), vectorfy(B, 1)],
#              ['lightblue', 'lightgreen'])
# plt.xlim(-1, 3)
# plt.ylim(-1, 3)
# plt.show()

# apply matrix N to B. N is (singular, columns linearly dependent.)
N = np.array([[-4, 1], [-8, 2]])

det_N = np.linalg.det(N) # 0, can't be inverted

NB = np.dot(N, B)

# plot_vectors([vectorfy(B, 0), vectorfy(B, 1), vectorfy(NB, 0), vectorfy(NB, 1)],
#              ['lightblue', 'lightgreen', 'blue', 'green'])
# plt.xlim(-6, 6)
# plt.ylim(-9, 3)
# plt.show()

lambdas_N, N_V = np.linalg.eig(N) # if any eigenvalues == 0, then product of eigenvalus and determinant == 0

# applying I2 to B
I = np.array([[1, 0], [0, 1]])

det_I = np.linalg.det(I)

IB = np.dot(I, B)

# plot_vectors([vectorfy(B, 0), vectorfy(B, 1), vectorfy(IB, 0), vectorfy(IB, 1)],
#               ['lightblue', 'lightgreen', 'blue', 'green'])
# plt.xlim(-1, 2)
# plt.ylim(-1, 2)
# plt.show()

lambdas_I, I_V = np.linalg.eig(I)

# apply J to B
J = np.array([[-0.5, 0], [0, 2]])

det_J = np.linalg.det(J)
abs_det_J = np.abs(np.linalg.det(J))

JB = np.dot(J, B)

# plot_vectors([vectorfy(B, 0), vectorfy(B, 1), vectorfy(JB, 0), vectorfy(JB, 1)],
#              ['lightblue', 'lightgreen', 'blue', 'green'])
# plt.xlim(-1, 3)
# plt.ylim(-1, 3)
# plt.show()

lambdas_J, V_J = np.linalg.eig(J)

# applying matrix D, scales vector by double on x and y axis
D = I*2

det_D = np.linalg.det(D)

DB = np.dot(D, B)

# plot_vectors([vectorfy(B, 0), vectorfy(B, 1), vectorfy(DB, 0), vectorfy(DB, 1)],
#              ['lightblue', 'lightgreen', 'blue', 'green'])
# plt.xlim(-1, 3)
# plt.ylim(-1, 3)
# plt.show()

lambdas_D, V_D = np.linalg.eig(D)
print(lambdas_D)