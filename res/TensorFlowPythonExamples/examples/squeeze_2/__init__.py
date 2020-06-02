import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1, 4, 1, 1), name="Hole")
op_ = tf.compat.v1.squeeze(in_, (0, 2))
