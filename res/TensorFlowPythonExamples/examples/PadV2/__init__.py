import tensorflow as tf
import numpy as np

input_ = np.ones(shape=[1, 1, 1, 1], dtype=np.float32)
paddings_ = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.int32)
constant_values_ = 1
op_ = tf.raw_ops.PadV2(input=input_, paddings=paddings_, constant_values=constant_values_)
