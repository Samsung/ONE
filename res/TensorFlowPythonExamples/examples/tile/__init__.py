import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(2, 3), name="Hole")
multiples_ = tf.compat.v1.constant([1, 2], name="Hole")
op_ = tf.compat.v1.tile(in_, multiples_)
