import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# example 1 where input has all known dims and axis is const

in_ = tf.compat.v1.placeholder(dtype=tf.int32, shape=(2, 3), name="Hole")
axis_ = tf.compat.v1.placeholder(dtype=tf.int32, shape=(), name="Axis")
expand_dim_ = tf.compat.v1.expand_dims(in_, axis_, name="ExpandDims")

# note: the code above will produce tflite file where output of ExpandDims is int32 scalar,
#       meaning that TFLC cannot decide the shape
