import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# example 3 where input has all known dim and axis is not const

in_ = tf.compat.v1.placeholder(dtype=tf.int32, shape=(2, 3), name="Hole")
expand_dim_ = tf.compat.v1.expand_dims(in_, 1, name="ExpandDims")

# note: the code above will produce tflite file where ExpandDims op is
#       replaced with "Reshape" op and the output shape is [2, 1, 3]
