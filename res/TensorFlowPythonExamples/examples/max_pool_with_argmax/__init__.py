import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1, 4, 4, 1), name="Hole")
op_ = tf.compat.v1.nn.max_pool_with_argmax(in_,
                                           ksize=[1, 2, 2, 1],
                                           strides=[1, 1, 1, 1],
                                           padding="VALID")
