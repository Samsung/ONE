import tensorflow as tf

print(tf.version.VERSION)

# This saves with SignatureDef but the order is incorrect
# anyway, we have tflite with SignatureDef

converter = tf.lite.TFLiteConverter.from_saved_model(
    "test_saved_model", signature_keys=['serving_default'])
converter.allow_custom_ops = True
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()
open("test_saved_model-fromsaved.tflite", "wb").write(tflite_model)
