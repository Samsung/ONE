import tensorflow as tf

tf.compat.v1.disable_eager_execution()

x_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1, 4, 4, 20), name="Hole")
op_ = tf.compat.v1.nn.lrn(x_, 5, 1.0, 1.0, 0.5)
