import tensorflow as tf

tf.compat.v1.disable_eager_execution()

in1_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(2, 3, 4), name="Hole1")
in2_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(2, 2, 4), name="Hole2")
concat_ = tf.compat.v1.concat([in1_, in2_], axis=-2)

# note that tflite file also contain axis = -2
