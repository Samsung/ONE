import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1, 1), name="Hole")
op_ = tf.compat.v1.nn.softmax(in_)
