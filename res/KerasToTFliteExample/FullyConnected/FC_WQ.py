import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(16, 32)),
    tf.keras.layers.Dense(32, activation="relu", use_bias=True, bias_initializer="random_uniform"),
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

model.summary()

run_model = tf.function(lambda x: model(x))

# This is important, let's fix the input size.
BATCH_SIZE = 1
X = 16
Y = 32
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([BATCH_SIZE, X, Y], model.inputs[0].dtype))

# model directory.
MODEL_DIR = "keras_fc"
model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT
                           ]  # For Weight Quantization(a.k.a Hybrid Quantization)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
]
converted_model = converter.convert()
save_to = "fc_wq.tflite"
if save_to is not None:
    with open(save_to, 'wb') as tf_lite_file:
        tf_lite_file.write(converted_model)
