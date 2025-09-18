import numpy as np
import torch
import tensorflow as tf

X = np.array([[25, 2], [5, 26], [3,7]])
X_T = X.T

X_pt = torch.tensor([[25, 2], [5, 26], [3,7]])
X_pt_T = X_pt.T

X_tf = tf.Variable([[25, 2], [5, 26], [3,7]])
X_tf_T = tf.transpose(X_tf)
print(X_tf_T)