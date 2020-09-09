import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=[28, 28, 3], name="Hole")
op_ = tf.compat.v1.keras.layers.LSTM(1, time_major=False, return_sequences=True)(in_)
