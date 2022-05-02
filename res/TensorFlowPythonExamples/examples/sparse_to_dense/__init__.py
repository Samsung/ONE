import tensorflow as tf

tf.compat.v1.disable_eager_execution()

in_ = tf.compat.v1.sparse_placeholder(tf.float32, name="Hole")
op_ = tf.compat.v1.sparse_tensor_to_dense(in_)
