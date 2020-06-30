import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(2, 3, 4), name="Hole")
diag_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(2, 3), name="Hole")
op_ = tf.compat.v1.matrix_set_diag(in_, diag_)
