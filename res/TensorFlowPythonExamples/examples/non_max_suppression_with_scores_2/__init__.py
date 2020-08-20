import tensorflow as tf

max_output_size = tf.compat.v1.constant(6)
iou_threshold = tf.compat.v1.constant(0.5)
score_threshold = tf.compat.v1.constant(0.6)
soft_nms_sigma = tf.compat.v1.constant(0.5)

in_boxes_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(12, 4), name="Hole")
in_scores_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(12), name="Hole")

# non_max_suppression_with_scores requires TF 1.15+
non_max_suppression_with_scores_ = tf.compat.v1.image.non_max_suppression_with_scores(
    in_boxes_, in_scores_, max_output_size, iou_threshold, score_threshold,
    soft_nms_sigma)
