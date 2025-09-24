import numpy as np
import torch
import tensorflow as tf

X = np.array([[25, 2], [5, 26], [3, 7]])
X_pt = torch.tensor([[25, 2], [5, 26], [3, 7]])
X_tf = tf.Variable([[25, 2], [5, 26], [3, 7]])

X_sum = X.sum()
X_pt_sum = torch.sum(X_pt)
X_tf_sum = tf.reduce_sum(X_tf)

# Sum can be done allong specific axis
X_rows_sum = X.sum(axis=0) # Sums all rows
X_cols_sum = X.sum(axis=1) # Sums all columns
X_pt_rows_sum = torch.sum(X_pt, 0)
X_tf_cols_sum = tf.reduce_sum(X_tf, 1)

print(X_tf_cols_sum)

