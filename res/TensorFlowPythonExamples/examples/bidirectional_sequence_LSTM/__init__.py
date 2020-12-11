import tensorflow as tf

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=[28, 28, 3], name="Hole")

op_uni_ = tf.compat.v1.keras.layers.LSTM(1, time_major=False, return_sequences=True)
op_bidi_ = tf.compat.v1.keras.layers.Bidirectional(op_uni_)(in_)
