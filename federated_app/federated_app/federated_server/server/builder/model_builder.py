from federated_app.model_convertor.builder.feature_extractor_builder import FeatureExtractorBuilder
from ..Helpers.dict_helper import DICTHelper
from ...models.model_architecture import ModelArchitecture
from ....federated_dataset.data_structures.image_shape import ImageShape
from ...Utils.Helpers.downsample_helper import Downsample
from ..data_structures.feature_extractor_data import FeatureExtractorData

class ModelBuilder:

    def __init__(self) -> None:

        self.model: ModelArchitecture = ModelArchitecture()
        self.classes: int = None
        self.num_features: int = None
        self.feature_extractor_data = None
        self.input_shape: ImageShape = None
        self.feature_extractor: FeatureExtractorBuilder = None

    def build_model_params(self, model_dict: str):

        model_dict = DICTHelper.clean_dict_keys(model_dict)

        self.classes = int(model_dict['num_classes'])
        self.input_shape = ImageShape(model_dict['input_shape'])

        if not {'feature_extractor'} <= model_dict.keys():
            self.feature_extractor_data = FeatureExtractorData.MOBILENETV2
        else:
            self.feature_extractor_data = FeatureExtractorData.str2enum(model_dict['feature_extractor'])

        downsampled_input_shape = Downsample.downsample_image_sizes(self.input_shape.get_sizes(),
                                                                    self.feature_extractor_data.value[0])

        self.num_features = downsampled_input_shape[0] ** 2 * self.feature_extractor_data.value[1]
        self.model.build_model(self.classes, self.num_features, model_dict['model'])

        self.feature_extractor = FeatureExtractorBuilder(self.feature_extractor_data, self.input_shape)
