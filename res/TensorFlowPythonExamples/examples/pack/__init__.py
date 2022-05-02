import tensorflow as tf

tf.compat.v1.disable_eager_execution()

in_1 = tf.compat.v1.placeholder(dtype=tf.float32, shape=(2, 3, 4), name="Hole")
in_2 = tf.compat.v1.placeholder(dtype=tf.float32, shape=(2, 3, 4), name="Hole")
op_ = tf.compat.v1.stack([in_1, in_2])
