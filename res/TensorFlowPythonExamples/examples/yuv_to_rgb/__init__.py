import tensorflow as tf

in_ = tf.placeholder(dtype=tf.float32, shape=(1, 16, 16, 3), name="Hole")
op_ = tf.image.yuv_to_rgb(in_)
