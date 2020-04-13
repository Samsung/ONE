import tensorflow as tf

in_ = tf.placeholder(dtype=tf.float32, shape=(1, 1), name="Hole")
elu_ = tf.nn.elu(in_)
