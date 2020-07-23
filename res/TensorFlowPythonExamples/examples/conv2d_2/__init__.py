import tensorflow as tf
import numpy as np

in_ = tf.compat.v1.placeholder(tf.float32, shape=(1, 32, 32, 3), name="Hole")

filters = np.random.uniform(low=-1., high=1, size=[5, 5, 3, 32]).astype(np.float32)
strides = (1, 2, 2, 1)
dilations = (1, 2, 2, 1)

op_ = tf.compat.v1.nn.conv2d(in_,
                             filters,
                             strides,
                             "VALID",
                             data_format="NHWC",
                             dilations=dilations)
