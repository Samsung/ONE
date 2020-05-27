import tensorflow as tf

input_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 2, 3), name="Hole")
axis_ = tf.compat.v1.placeholder(dtype=tf.int32, shape=(1), name="HoleAxis")
inshape_ = tf.compat.v1.shape(input_)
reshape_ = tf.compat.v1.reshape(input_, inshape_)
exdims01_ = tf.compat.v1.expand_dims(reshape_, axis_, name="ExpDims_01")
output_ = tf.compat.v1.expand_dims(exdims01_, 1, name="ExpDims_02")

# tf2tfliteV2 will convert .pbtxt model to .tflite with two EXPAND_DIMS
'''
python ../../compiler/tf2tfliteV2/tf2tfliteV2.py \
--v1 \
--input_path ./UnknownExpandDims.pbtxt \
--output_path ./UnknownExpandDims.tflite \
--input_arrays Hole,HoleAxis \
--output_arrays ExpDims_02
'''
