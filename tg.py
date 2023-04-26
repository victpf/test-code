import tensorflow as tf

# Load the training data.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Define the model.
model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model.
model.fit(x_train, y_train, epochs=10)

# Evaluate the model.
loss, accuracy = model.evaluate(x_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Save the model.
model.save('mnist_model.h5')
