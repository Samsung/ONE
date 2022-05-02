import tensorflow as tf

tf.compat.v1.disable_eager_execution()

tensor_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(2, 3), name="Hole")
paddings_ = tf.compat.v1.constant([[1, 1], [2, 2]], name="Hole")
op_ = tf.compat.v1.pad(tensor_, paddings_)
