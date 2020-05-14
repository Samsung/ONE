import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(4, 3), name="Hole")
op_ = tf.compat.v1.split(in_, [1, 2, 1])
