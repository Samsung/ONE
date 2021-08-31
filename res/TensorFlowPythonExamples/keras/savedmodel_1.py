import tensorflow as tf

print(tf.version.VERSION)

input = tf.keras.layers.Input(shape=(3, 3, 32), name="input")

o1 = tf.keras.layers.Conv2D(
    2, (1, 1), activation='relu', input_shape=(1, 3, 3, 32), name="c1")(input)
o2 = tf.keras.layers.Conv2D(
    16, (1, 1), activation='relu', input_shape=(1, 3, 3, 32), name="c2")(input)
o3 = tf.keras.layers.Conv2D(
    32, (1, 1), activation='relu', input_shape=(1, 3, 3, 32), name="c3")(input)

model = tf.keras.Model(inputs=input, outputs=[o2, o3, o1])
model.summary()

#tf.keras.models.save_model(model, "test_saved_model")
model.save("test_saved_model")
