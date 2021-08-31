import tensorflow as tf

print(tf.version.VERSION)

# this does not store SignatureDef and output order is correct

model = tf.keras.models.load_model("test_saved_model")
infer = model.signatures["serving_default"]
print(infer)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.allow_custom_ops = True
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()
open("test_saved_model-fromsaved.tflite", "wb").write(tflite_model)
