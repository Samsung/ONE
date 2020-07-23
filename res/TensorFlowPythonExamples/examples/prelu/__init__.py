import tensorflow as tf

input_tensor = tf.compat.v1.placeholder(dtype=tf.float32,
                                        name="input",
                                        shape=[1, 4, 4, 3])
prelu = tf.keras.layers.PReLU(shared_axes=[1, 2])
op_ = prelu(input_tensor)
