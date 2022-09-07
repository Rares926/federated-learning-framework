from ...federated_dataset.data_structures.image_shape import ImageShape
from ...federated_server.server.data_structures.feature_extractor_data import FeatureExtractorData
import tensorflow as tf

class FeatureExtractorBuilder:
    def __init__(self, feature_extractor_data: FeatureExtractorData, input_shape: ImageShape) -> None:
        self.feature_extractor_data = feature_extractor_data
        self.input_shape = input_shape

    def get_opt(self):
        """Dispatch method"""
        method_name = 'opt_' + str(self.feature_extractor_data.name)
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: "Invalid optimizier")
        # Call the method as we return it
        return method()

    def opt_MobileNetV2(self):
        return tf.keras.applications.MobileNetV2(input_shape=(self.input_shape.width, self.input_shape.height, self.input_shape.channels),
                                                 alpha=1.0,
                                                 include_top=False,
                                                 weights='imagenet')

    def opt_DenseNet169(self):
        return tf.keras.applications.Xception(input_shape=(self.input_shape.width, self.input_shape.height, self.input_shape.channels),
                                              include_top=False,
                                              weights='imagenet')

    def opt_Xception(self):
        return tf.keras.applications.DenseNet169(input_shape=(self.input_shape.width, self.input_shape.height, self.input_shape.channels),
                                                 include_top=False,
                                                 weights='imagenet')
