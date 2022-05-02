import tensorflow as tf

tf.compat.v1.disable_eager_execution()

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1, 1, 2, 3), name="Hole")
op_ = tf.nn.bias_add(in_, bias=[1.0, 1.0, -1.0], data_format="NHWC")
