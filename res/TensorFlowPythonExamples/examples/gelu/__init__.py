# NOTE please use TF2.4.0-dev or above to use gelu op
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(4, 4), name="Hole")
op_ = tf.nn.gelu(in_, approximate=False, name="Output")
