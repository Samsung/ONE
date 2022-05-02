import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# example 2 where input has unknown dim and axis is const

in_ = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None, None), name="Hole")
expand_dim_ = tf.compat.v1.expand_dims(in_, 1, name="ExpandDims")

# note: the code above will produce tflite file where output shape of ExpandDims is [1, 1, 1]
#       (Tflite converter replace "None" to "1")
#       axis can be negative integer.
