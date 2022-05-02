import tensorflow as tf

tf.compat.v1.disable_eager_execution()

in_ = tf.compat.v1.placeholder(dtype=tf.int32, shape=(), name="Hole")
op_ = tf.compat.v1.fill((3, 4), in_)
