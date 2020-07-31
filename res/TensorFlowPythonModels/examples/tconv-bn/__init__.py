import tensorflow as tf
import numpy as np

input_ = tf.compat.v1.placeholder(tf.float32, shape=(1, 2, 2, 1), name="Hole")
W = np.ones(9).reshape((3, 3, 1, 1))
filter_ = tf.compat.v1.constant(W, dtype=tf.float32)
tconv_ = tf.compat.v1.nn.conv2d_transpose(
    input_, filter_, output_shape=(1, 4, 4, 1), strides=[1, 1, 1, 1], padding='VALID')

scale_ = tf.compat.v1.constant([1.0177339315414429], dtype=tf.float32)
offset_ = tf.compat.v1.constant([0.015628524124622345], dtype=tf.float32)
mean_ = tf.compat.v1.constant([1.027155211195349693], dtype=tf.float32)
variance_ = tf.compat.v1.constant([0.25580066442489624], dtype=tf.float32)
bn_out, _, _ = tf.compat.v1.nn.fused_batch_norm(
    tconv_,
    scale_,
    offset_,
    mean=mean_,
    variance=variance_,
    epsilon=0.0010000000474974513,
    is_training=False)
'''
python ../../compiler/tf2tfliteV2/tf2tfliteV2.py --v1 \
-i tconv-bn.pbtxt \
-o tconv-bn.tflite \
-I Hole -O FusedBatchNorm
'''
