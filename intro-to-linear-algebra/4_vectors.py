import numpy as np
import torch
import tensorflow as tf

x = np.array([25, 2, 5])

x_t = x.T

y = np.array([[25, 2, 5]])

y_t = y.T

z = np.zeros(3)

x_pt = torch.tensor([25, 2, 5])

x_tf = tf.Variable([25, 2, 5])

print(x_tf)