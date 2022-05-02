import tensorflow as tf

tf.compat.v1.disable_eager_execution()

tensor_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(2, 3), name="Hole")
paddings_ = tf.constant([[1, 1], [2, 2]], name="Hole")
op_ = tf.pad(tensor_, paddings_, "REFLECT")
