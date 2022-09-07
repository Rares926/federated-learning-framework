# Imports
# Internal imports
from ..data_structures.image_format import ImageFormat
from ..data_structures.image_shape import ImageShape
from ..data_structures.resize_method import ResizeMethod
from ..data_structures.ratio import Ratio

# Typing imports
class DatasetParams:
    def __init__(self):
        self.image_shape = None
        self.image_format = None
        self.resize_method = None
        self.ratios = None
        self.resize_after_crop = None

    def build_dataset_params(self, network_data: dict):
        if not {'input_shape', 'input_format', 'resize'} <= network_data.keys():
            raise Exception("Invalid network params format")
        if not {'width', 'height', 'depth'} <= network_data['input_shape'].keys():
            raise Exception("Invalid input shape params")
        if not {'channels', 'data_type'} <= network_data['input_format'].keys():
            raise Exception("Invalid input format params")
        if not {'method'} <= network_data['resize'].keys():
            raise Exception("Invalid resize method params")

        self.image_shape = ImageShape(network_data['input_shape'])
        self.image_format = ImageFormat(network_data['input_format'])
        self.resize_method = ResizeMethod.str2enum(network_data['resize']['method'])

        if self.resize_method == ResizeMethod.CROP:
            if not {'params'} <= network_data['resize'].keys():
                raise Exception("Missing crop ratio params")
            if not {'tl_ratio', 'br_ratio', 'resize_after_crop'} <= network_data['resize']['params'].keys():
                raise Exception("Invalid crop ratio params")
            ratio_sum = network_data['resize']['params']['tl_ratio'] + network_data['resize']['params']['br_ratio']
            if ratio_sum > 1:
                raise Exception("Crop ratio sum must not exceed 1")
            self.ratios = Ratio(network_data['resize']['params']['tl_ratio'], network_data['resize']['params']['br_ratio'])
            self.resize_after_crop = ResizeMethod.str2enum(network_data['resize']['params']['resize_after_crop'], True)

    def get_network_params(self):
        return self.image_shape, self.image_format, self.resize_method, self.ratios, self.resize_after_crop
