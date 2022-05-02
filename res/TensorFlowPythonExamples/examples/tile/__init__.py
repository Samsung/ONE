import tensorflow as tf

tf.compat.v1.disable_eager_execution()

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(2, 3), name="Hole")
multiples_ = tf.compat.v1.constant([1, 2], name="Hole")
op_ = tf.compat.v1.tile(in_, multiples_)
