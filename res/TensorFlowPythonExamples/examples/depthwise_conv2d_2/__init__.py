import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

in_ = tf.compat.v1.placeholder(tf.float32, shape=(1, 32, 32, 4), name="Hole")

filters = np.array(np.random.uniform(low=-1., high=1, size=[3, 3, 4, 1]),
                   dtype=np.float32)
strides = (1, 2, 2, 1)
dilations = (2, 2)

op_ = tf.compat.v1.nn.depthwise_conv2d(in_,
                                       filters,
                                       strides,
                                       "VALID",
                                       data_format="NHWC",
                                       dilations=dilations)
