import tensorflow as tf

x_tf = tf.Variable(25, dtype=tf.int16)

y_tf = tf.Variable(3, dtype=tf.int16)

tf_sum = tf.add(x_tf, y_tf)

tf_float = tf.Variable(25, dtype=tf.float16)
print(tf_float)