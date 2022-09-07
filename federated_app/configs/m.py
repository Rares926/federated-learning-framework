import tensorflow as tf

class Model:

    DEFAULT_MODEL = [tf.keras.layers.Dense(128, activation='relu')]

    def get_model():
        return Model.DEFAULT_MODEL
