import tensorflow as tf

tf.compat.v1.disable_eager_execution()

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1, 8, 8, 4), name="Hole")
op_ = tf.compat.v1.reduce_sum(in_, -1, True)
