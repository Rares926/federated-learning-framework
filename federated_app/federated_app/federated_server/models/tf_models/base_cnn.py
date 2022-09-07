from typing import Tuple, cast
import tensorflow as tf
import flwr as fl

class CNN(tf.keras.Model):

    def __init__(self, input_shape: Tuple[int, int, int]):
        super(CNN, self).__init__()
        # build model layers
        self.input = tf.keras.Input(shape=input_shape),
        self.conv1 = tf.keras.layers.Conv2D(
            filters=6,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='valid',
            activation='relu',
        )
        self.pool = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='valid',
            activation='relu'
        )
        self.flat = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(
            units=120,
            activation='relu',
        )
        self.fc2 = tf.keras.layers.Dense(
            units=84,
            activation='relu',
        )
        self.fc3 = tf.keras.layers.Dense(
            units=10
        )

    def call(self):
        x = self.input()
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def get_weights_for_initialize(self):
        weights = self.get_weights()
        return cast(fl.common.Weights, weights)
