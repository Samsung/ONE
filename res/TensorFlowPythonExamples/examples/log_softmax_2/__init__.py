import tensorflow as tf

tf.compat.v1.disable_eager_execution()

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(2, 3, 4, 5), name="Hole")
op_ = tf.compat.v1.nn.log_softmax(in_, axis=1)
