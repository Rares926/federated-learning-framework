
from enum import Enum

class FeatureExtractorData(Enum):

    MobileNetV2 = (5, 1280)
    DenseNet169 = (5, 1664)
    Xception = (5, 2048)

    @classmethod
    def str2enum(cls, feature_extractor: str):
        feature_extractor = feature_extractor.lower()
        if feature_extractor == 'mobilenetv2':
            return cls.MobileNetV2
        elif feature_extractor == 'densenet121':
            return cls.DenseNet169
        elif feature_extractor == 'xception':
            return cls.Xception
        else:
            raise Exception("Invalid feature extractor name")
