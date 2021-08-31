import tensorflow as tf
import numpy as np

print("TF version=", tf.__version__)

tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.Session()

in_ = tf.compat.v1.placeholder(tf.float32, shape=(1, 32, 32, 3), name="Hole")

strides = (1, 2, 2, 1)

filters_1 = np.random.uniform(low=-1., high=1, size=[5, 5, 3, 2]).astype(np.float32)
op_1_ = tf.compat.v1.nn.conv2d(in_, filters_1, strides, "VALID", data_format="NHWC")

filters_2 = np.random.uniform(low=-1., high=1, size=[5, 5, 3, 16]).astype(np.float32)
op_2_ = tf.compat.v1.nn.conv2d(in_, filters_2, strides, "VALID", data_format="NHWC")

filters_3 = np.random.uniform(low=-1., high=1, size=[5, 5, 3, 32]).astype(np.float32)
op_3_ = tf.compat.v1.nn.conv2d(in_, filters_3, strides, "VALID", data_format="NHWC")
