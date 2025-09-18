import numpy as np
import torch
import tensorflow as tf

X = np.array([[25, 2], [5, 26], [3, 7]])

X_pt = torch.tensor([[25, 2], [5, 26], [3, 7]])

X_tf = tf.Variable([[25, 2], [5, 26], [3, 7]])

print(X_tf[2,:])