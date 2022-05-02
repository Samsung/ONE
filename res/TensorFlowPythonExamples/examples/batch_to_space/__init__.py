import tensorflow as tf

tf.compat.v1.disable_eager_execution()

in_ = tf.compat.v1.placeholder(tf.float32, shape=[4, 1, 1, 1], name="Hole")
cr_ = tf.constant([[0, 0], [0, 0]], name="Hole")
op_ = tf.batch_to_space(in_, cr_, 2)
