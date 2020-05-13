import tensorflow as tf

# example 1 where input has all known dims and axis is const

in_ = tf.compat.v1.placeholder(dtype=tf.int32, shape=(2, 3), name="Hole")
axis_ = tf.compat.v1.placeholder(dtype=tf.int32, shape=0, name="Axis")
expand_dim_ = tf.compat.v1.expand_dims(in_, axis_, name="ExpandDims")

# note: the code above will produce tflite file where output of ExpandDims is int32 scalar,
#       meaning that TFLC cannot decide the shape
'''
# example 2 where input has unknown dim and axis is const

in_ = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None, None), name="Hole")
expand_dim_ = tf.compat.v1.expand_dims(in_, 1, name="ExpandDims")

# note: the code above will produce tflite file where output shape of ExpandDims is [1, 1, 1]
       (Tflite converter replace "None" to "1")
       axis can be negative integer.
'''
'''
# example 3 where input has all known dim and axis is not const

in_ = tf.compat.v1.placeholder(dtype=tf.int32, shape=(2, 3), name="Hole")
expand_dim_ = tf.compat.v1.expand_dims(in_, 1, name="ExpandDims")

# note: the code above will produce tflite file where ExpandDims op is
#       replaced with "Reshape" op and the output shape is [2, 1, 3]
'''
