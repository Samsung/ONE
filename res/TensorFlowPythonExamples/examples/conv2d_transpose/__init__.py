import tensorflow as tf

input_ = tf.compat.v1.placeholder(tf.float32, shape=(1, 8, 8, 1), name="Hole")
kernel_ = tf.compat.v1.placeholder(tf.float32, shape=(3, 3, 1, 1), name="Hole")
op_ = tf.compat.v1.nn.conv2d_transpose(input_,
                                       kernel_,
                                       output_shape=[1, 8, 8, 1],
                                       strides=[1, 1, 1, 1],
                                       padding='SAME')
