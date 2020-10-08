import tensorflow as tf
import numpy as np

in_ = tf.compat.v1.placeholder(tf.float32, shape=(1, 32, 32, 3), name="Hole")

filters = np.random.uniform(low=-1., high=1, size=[5, 5, 3, 32]).astype(np.float32)

op_ = tf.compat.v1.nn.atrous_conv2d(in_, filters, 2, "VALID")
