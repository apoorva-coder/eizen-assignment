from tensorflow import keras
import tensorflow as tf
import numpy as np


def bi_tempered_logistic_loss(y_true, y_pred, t1=1.2, t2=1.8, label_smoothing=0.2):
    """
    Computes the bi-tempered logistic loss for TensorFlow/Keras.

    :param y_true: the true labels
    :param y_pred: the predicted labels
    :param t1: the temperature parameter (default: 0.8)
    :param t2: the temperature parameter (default: 1.2)
    :param label_smoothing: the amount of label smoothing (default: 0.1)
    :return: the bi-tempered logistic loss
    """
    # Apply label smoothing
    y_true = tf.cast(y_true, tf.float32)

    # apply label smoothing
    y_true = y_true * (1 - label_smoothing) + label_smoothing / y_true.shape[1]

    # calculate the tempered exponential
    temp1 = tf.math.exp((tf.math.log(y_pred + 1e-10)) / t1)
    temp2 = tf.math.exp((tf.math.log(1 - y_pred + 1e-10)) / t2)
    tempered_exp = (temp1 + temp2) ** (-1)

    # calculate the final loss
    loss = tf.reduce_sum(y_true * tempered_exp, axis=-1)

    return loss

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# Preprocess the data
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

# Define the model architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model with the custom loss function
model.compile(optimizer="adam", loss=bi_tempered_logistic_loss, metrics=["accuracy"])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

