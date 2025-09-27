import numpy as np
import matplotlib.pyplot as plt

v = np.array([3, 1])

# plot vector using plot_vectors()
def plot_vectors(vectors, colors):
    plt.figure()
    plt.axvline(x=0, color="lightgray")
    plt.axhline(y=0, color="lightgrey")

    for i in range(len(vectors)):
        x = np.concatenate([[0, 0], vectors[i]])
        plt.quiver(x[0], x[1], x[2], x[3],
                   angles="xy", scale_units="xy", scale=1, color=colors[i])

# plot_vectors([v], ["lightblue"])
# plt.xlim(-1, 5)
# plt.ylim(-1, 5)
# plt.show()

# apply identity matrix to v
I = np.array([[1, 0], [0, 1]])
Iv = np.dot(I, v)

is_equal = v == Iv

# plot_vectors([Iv], ["blue"])
# plt.xlim(-1, 5)
# plt.ylim(-1, 5)
# plt.show()

# matrix E that flips over the x axis
E = np.array([[1, 0], [0, -1]])
Ev = np.dot(E, v)

# plot_vectors([v, Ev], ["lightblue", "blue"])
# plt.xlim(-1, 5)
# plt.ylim(-3, 5)
# plt.show()

# matrix F that flips over the y axis
F = np.array([[-1, 0], [0, 1]])
Fv = np.dot(F, v)

# plot_vectors([v, Fv], ["lightblue", "blue"])
# plt.xlim(-4, 4)
# plt.ylim(-1, 5)
# plt.show()

# multiple affine transformations
A = np.array([[-1, 4], [2, -2]])
Av = np.dot(A, v)

# plot_vectors([v, Av], ["lightblue", "blue"])
# plt.xlim(-1, 5)
# plt.ylim(-1, 5)
# plt.show()

# another example of applying A
v2 = np.array([2, 1])

# plot_vectors([v2, np.dot(A, v2)], ["lightgreen", "green"])
# plt.xlim(-1, 5)
# plt.ylim(-1, 5)
# plt.show()

# concatenate vectors into matrix v
# convert array to 2D to transpose into column e.g
# v_T = np.matrix(v).T

v3 = np.array([-3, -1])
v4 = np.array([-1, 1])

V = np.concatenate((np.matrix(v).T,
                    np.matrix(v2).T,
                    np.matrix(v3).T,
                    np.matrix(v4).T),
                    axis=1)

IV = np.dot(I, V)
AV = np.dot(A, V)

# function to convert column of matrix to 1D vector
def vectorfy(mtrx, clmn):
    return np.array(mtrx[:, clmn]).reshape(-1)

v1_ved = vectorfy(V, 0)

is_equal = v1_ved == v


plot_vectors([vectorfy(V, 0), vectorfy(V, 1), vectorfy(V, 2), vectorfy(V, 3),
              vectorfy(AV, 0), vectorfy(AV, 1), vectorfy(AV, 2), vectorfy(AV, 3)],
              ['lightblue', 'lightgreen', 'lightgray', 'orange',
               'blue', 'green', 'gray', 'red'])
plt.xlim(-4, 6)
plt.ylim(-5, 5)
plt.show()