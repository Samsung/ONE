import tensorflow as tf

tf.compat.v1.disable_eager_execution()

in_ = tf.compat.v1.placeholder(tf.float32, shape=(1, 8, 8, 1), name="Hole")
op_ = tf.compat.v1.nn.avg_pool2d(in_, (2, 2), 1, "VALID")
