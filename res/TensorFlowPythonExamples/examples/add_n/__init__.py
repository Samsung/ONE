import tensorflow as tf

in1_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(4, 4), name="Hole")
in2_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(4, 4), name="Hole")
in3_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(4, 4), name="Hole")
op_ = tf.compat.v1.math.add_n([in1_, in2_, in3_])
