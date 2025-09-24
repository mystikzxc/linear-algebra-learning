import numpy as np
import torch
import tensorflow as tf

x = np.array([25, 2, 5])
y = np.array([0, 1, 2])

x_pt = torch.tensor([25, 2, 5])
y_pt = torch.tensor([0, 1, 2])

x_tf = tf.Variable([25, 2, 5])
y_tf = tf.Variable([0, 1, 2])

# Calculate dot product manually
dot_xy = 25*0 + 2*1 + 5*2

# Calculate dot product using numpy
dot_xy_np = np.dot(x, y)

# Calculate dot product using pytorch // can also use numpy method
dot_xy_pt = torch.dot(x_pt, y_pt)

# Calculate dot product using tensorflow
dot_xy_tf = tf.reduce_sum(tf.multiply(x_tf, y_tf))

print(dot_xy_tf)