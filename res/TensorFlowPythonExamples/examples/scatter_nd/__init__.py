# This example is a modified example from TF API documentation

import tensorflow as tf

indices = tf.compat.v1.constant([[0], [2]])
updates = tf.compat.v1.constant([[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                                 [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8,
                                                                             8]]])
shape = tf.constant([4, 4, 4])
sc_ = tf.compat.v1.scatter_nd(indices, updates, shape)
