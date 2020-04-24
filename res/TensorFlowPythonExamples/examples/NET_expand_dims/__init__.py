import tensorflow as tf
'''
imaginary graph to test dynamic tensor and unknown dims

placeholder (shape = [None,None])    if input is of shape [1, 10]
  |     |
  |  expand_dims                        shape will be [1, 10, 1]
  |    |
  |  shape                              shape will be [1, 10, 1]
  |   |
 reshape                                shape will be [1, 10, 1]
    |
 squeeze                                shape will be [10]

'''

in_ = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None, None), name="Hole")  # [1, 10]
expand_dim_ = tf.compat.v1.expand_dims(in_, -1)  # [1, 10, 1]
shape_ = tf.compat.v1.shape(expand_dim_)
reshape_ = tf.compat.v1.reshape(in_, shape_)  # [1, 10, 1]
squeeze_ = tf.compat.v1.squeeze(reshape_)  # [10]
