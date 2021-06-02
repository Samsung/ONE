import tensorflow as tf

tf.compat.v1.disable_eager_execution()

in_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(2, 3), name="Hole")
op_, _, _ = tf.compat.v1.quantization.quantize(in_, -1.0, 1.0, tf.quint8)
