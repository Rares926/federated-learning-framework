# Imports
import os
# Internal imports
from ..dataset.dataset_params import DatasetParams
from ...federated_server.Utils.Helpers.json_helper import JsonHelper
# Typing imports

# Typing imports


class DatasetBuilder:
    def __init__(self) -> None:

        self.dataset_path = None
        self.image_shape = None
        self.image_format = None
        self.resize_method = None
        self.ratios = None
        self.resize_after_crop = None

        self.federated_factor = 1.0
        self.split_percentage = 0.9

        self.new_dataset_path = None

    def arg_parse(self, path: str):
        raw_data = JsonHelper.read_json(path)

        if not {'dataset_path', 'input_data'} <= raw_data.keys():
            raise Exception("Invalid config file format")

        if not {'new_dataset_path'} <= raw_data.keys():
            self.new_dataset_path = os.path.join(raw_data['dataset_path'], "new_dataset")
        else:
            self.new_dataset_path = raw_data['new_dataset_path']

        self.dataset_path = raw_data["dataset_path"]

        if {'federated_factor'} <= raw_data.keys():
            self.federated_factor = raw_data['federated_factor']

        if {'split_percentage'} <= raw_data.keys():
            self.split_percentage = raw_data['split_percentage']

        dataset_params = DatasetParams()
        dataset_params.build_dataset_params(raw_data['input_data'])
        self.image_shape, self.image_format, self.resize_method, self.ratios, self.resize_after_crop = dataset_params.get_network_params()
