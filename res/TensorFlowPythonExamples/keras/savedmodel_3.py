import tensorflow as tf

print(tf.version.VERSION)

model = tf.keras.models.load_model('test_saved_model')
run_model = tf.function(lambda x: model(x))

print(model.inputs[0].shape)
print(model.inputs[0].dtype)

concrete_func = run_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

converter.allow_custom_ops = True
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()
open("test_saved_model-fromkeras.tflite", "wb").write(tflite_model)
