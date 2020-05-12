import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.int32, shape=(), name="Hole")
op_ = tf.compat.v1.fill((3, 4), in_)
