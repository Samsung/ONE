import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(3, 2, 3), name="Hole")
op_ = tf.compat.v1.slice(in_, [1, 0, 0], [1, 1, 3])
