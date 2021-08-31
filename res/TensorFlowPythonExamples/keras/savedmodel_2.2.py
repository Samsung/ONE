import tensorflow as tf

print(tf.version.VERSION)

# this does not store SignatureDef and output order is correect

model = tf.saved_model.load("test_saved_model")
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec((1, 3, 3, 32), tf.float32))

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.allow_custom_ops = True
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()
open("test_saved_model-fromsaved.tflite", "wb").write(tflite_model)
