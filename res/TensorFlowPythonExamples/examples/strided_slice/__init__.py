import tensorflow as tf

tf.compat.v1.disable_eager_execution()

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(3, 2, 3), name="Hole")
op_ = tf.compat.v1.strided_slice(in_, [1, 0, 0], [2, 1, 3], [1, 1, 1])
