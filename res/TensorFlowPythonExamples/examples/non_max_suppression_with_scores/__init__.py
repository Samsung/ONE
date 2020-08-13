import tensorflow as tf

max_output_size = tf.compat.v1.constant(4)

in_boxes_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(8, 4), name="Hole")
in_scores_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(8), name="Hole")

# non_max_suppression_with_scores requires TF 1.15+
non_max_suppression_with_scores_ = tf.compat.v1.image.non_max_suppression_with_scores(
    in_boxes_, in_scores_, max_output_size)
