import tensorflow as tf

tf.compat.v1.disable_eager_execution()

mean = tf.compat.v1.constant([1., 2., 3.])
variance = tf.compat.v1.constant([4., 5., 6.])
offset = tf.compat.v1.constant([7., 8., 9.])
scale = tf.compat.v1.constant([10., 11., 12.])

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(3, 3), name="Hole")
bn_ = tf.nn.batch_normalization(in_, mean, variance, offset, scale, 1e-5)
