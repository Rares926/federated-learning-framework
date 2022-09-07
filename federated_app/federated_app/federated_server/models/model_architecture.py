from collections import namedtuple
from unicodedata import name
import tensorflow as tf
import importlib.util

class ModelArchitecture:

    def __init__(self):
        self.model = tf.keras.models.Sequential()
        self.inner_model = None

    def build_model(self, classes: int, num_features: int, model_path: str):

        if model_path != None:  # noqa: E711
            spec = importlib.util.spec_from_file_location("model", model_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            self.inner_model = mod.Model.get_model()
        else:
            raise Exception("No model given!")

        self.model.add(tf.keras.layers.Input([num_features], name="input"))

        layer_enumerator = 0

        for inner_layer in self.inner_model:
            inner_layer._name = str(layer_enumerator)
            self.model.add(inner_layer)
            layer_enumerator += 1

        self.model.add(tf.keras.layers.Dense(classes, name=str(layer_enumerator)))

    def get_model(self):
        return self.model
