import numpy as np

x = np.array([25, 2, 5])

# Calculate L2 Norm of x
x_l2norm = (25**2 + 2**2 + 5**2) ** (1/2)

# Using numpy to calculate L2 Norm of x
x_l2norm_np = np.linalg.norm(x)

# Calculate L1 Norm of x
x_l1norm = np.abs(25) + np.abs(2) + np.abs(5)

# Calculate Squared L2 Norm of x
x_sl2norm = (25**2 + 2**2 + 5**2)

# Using numpy to calculate Squared L2 Norm of x
x_sl2norm_np = np.dot(x, x)

# Calculate the Max Norm of x
x_maxnorm = np.max([np.abs(25), np.abs(2), np.abs(5)])

print(x_maxnorm)