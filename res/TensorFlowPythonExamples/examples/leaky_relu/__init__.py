import tensorflow as tf

in_ = tf.placeholder(dtype=tf.float32, shape=(1, 1), name="Hole")
op_ = tf.nn.leaky_relu(in_)
