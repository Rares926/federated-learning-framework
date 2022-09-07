
from federated_app.federated_server.server.Helpers.strategy_name_helper import StrategyName
from ..data_structures.client_info import ClientInfo
from ..data_structures.strategy_type import StrategyType
from ..Helpers.dict_helper import DICTHelper
from ..data_structures.aggregation_params import AggregationParams
from ...strategy.strategy import Strategy
import numpy as np
import os

class StrategyBuilder:
    def __init__(self) -> None:
        self.name = "FedAvg"
        self.type = 0
        self.client_info = ClientInfo()
        self.aggregation_params = AggregationParams()
        self.training_model = None
        self.evaluation_fn = None
        self.initial_weights = None

    def build_strategy_params(self, strategy_dict: str):

        strategy_dict = DICTHelper.clean_dict_keys(strategy_dict)

        if not {'name', 'type', 'client_info'} <= strategy_dict.keys():
            raise Exception("Invalid strategy format")

        if {'name', 'type'} <= strategy_dict.keys():
            self.type = StrategyType.Type.str2enum(str(strategy_dict["type"]))
            self.name = StrategyName.get_strategy_name(strategy_dict["name"])

        self.client_info.build_params(strategy_dict['client_info'])

        self.aggregation_params.build_params(strategy_dict["aggregation_params"])

    def set_init_weights(self, init_weights_path):

        if init_weights_path is None:
            w = None
        else:
            arr = os.listdir(init_weights_path)
            if len(arr) == 0:
                w = None
            else:
                init_path = os.path.join(init_weights_path, arr[-1])
                w = np.load(init_path)

        if self.type == StrategyType.Type.DESKTOP:
            from flwr.common import weights_to_parameters
            if w is None:
                self.initial_weights = weights_to_parameters(self.training_model.get_weights())
            else:
                self.initial_weights = weights_to_parameters(w)
        else:
            from ...Utils.weights_converter import WeightsConverter

            if w is None:
                self.initial_weights = WeightsConverter.weights_to_parameters(self.training_model.get_weights())
            else:
                list_w = [w[key] for key in w]
                self.initial_weights = WeightsConverter.weights_to_parameters(list_w)

    def set_eval_fn(self, eval_fn):
        self.evaluation_fn = eval_fn

    def set_model(self, training_model):
        self.training_model = training_model

    def get_strategy(self):
        strategy = Strategy(self.name,
                            self.type,
                            self.client_info,
                            self.aggregation_params,
                            self.training_model,
                            self.evaluation_fn,
                            self.initial_weights)

        return strategy.get_str()
