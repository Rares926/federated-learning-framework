# Imports
import os
import tensorflow as tf
from ...federated_server.server.builder.model_builder import ModelBuilder
from ..optimizer.optimizer import Optimizer
from ..helpers.optimizer_helper import OptimizerHelper

from ...federated_server.server.data_structures.training_params import TrainingParams
from ...federated_server.server.Helpers.dict_helper import DICTHelper
from ...federated_dataset.data_structures.image_shape import ImageShape
# Internal imports
from ...federated_server.Utils.Helpers.json_helper import JsonHelper
# Typing imports

# Typing imports


class ConvertedModelBuilder:
    def __init__(self) -> None:
        self.training_data: TrainingParams = TrainingParams()
        self.optimizer: tf.keras.optimizers = None
        self.training_model: ModelBuilder() = ModelBuilder()

    def arg_parse(self, path: str):
        raw_data = JsonHelper.read_json(path)
        clean_raw_data = DICTHelper.clean_dict_keys(raw_data)

        if not {'training_params', 'model_params'} <= clean_raw_data.keys():
            raise Exception("Invalid config file format")

        self.training_data.build_training_params(clean_raw_data['training_params'])

        optimizer_params = Optimizer(self.training_data.learning_rate)

        if {'optimizer'} <= clean_raw_data.keys():
            optimizer_params.build_optimizer_params(clean_raw_data['optimizer'])
            self.optimizer = optimizer_params.get_opt()
        else:
            self.optimizer = OptimizerHelper.get_base_optimizer(self.training_data.learning_rate)

        self.training_model.build_model_params(raw_data['model_params'])
