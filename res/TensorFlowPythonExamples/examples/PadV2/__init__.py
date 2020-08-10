import tensorflow as tf
import numpy as np

input_ = tf.compat.v1.placeholder(shape=[1, 1, 1, 1], dtype=tf.float32)
paddings_ = tf.compat.v1.constant(np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.int32))
constant_values_ = tf.compat.v1.constant(1, shape=(), dtype=tf.float32)
op_ = tf.compat.v1.pad(input_, paddings=paddings_, constant_values=constant_values_)
