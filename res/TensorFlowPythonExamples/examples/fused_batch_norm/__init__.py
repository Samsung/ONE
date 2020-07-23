import tensorflow as tf

scale = tf.compat.v1.constant([1., 2., 3.])
offset = tf.compat.v1.constant([4., 5., 6.])
mean = tf.constant([1., 2., 3.])
variance = tf.constant([4., 5., 6.])

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(3, 3, 3, 3), name="Hole")
fbn_ = tf.compat.v1.nn.fused_batch_norm(in_,
                                        scale,
                                        offset,
                                        mean,
                                        variance,
                                        is_training=False)
