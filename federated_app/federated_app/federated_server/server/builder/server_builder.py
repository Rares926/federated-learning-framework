from federated_app.federated_server.server.builder.model_builder import ModelBuilder
from ...models.model_architecture import ModelArchitecture
from ..builder.strategy_builder import StrategyBuilder
from ..data_structures.training_params import TrainingParams

from ...Utils.Helpers.json_helper import JsonHelper
from ...Utils.ip_validation import IPAddress
from ...models.model import head_MLP
from ...Utils.Helpers.display_helper import Display
from .evaluation_builder import EvaluationBuilder

class ServerBuilder:

    def __init__(self) -> None:

        self.server_address = "[::]:8999"
        self.training_data = TrainingParams()
        self.strategy = None
        self.rounds = 10
        self.training_model = None
        self.evaluation = None

    def arg_parse(self, path: str):

        raw_data = JsonHelper.read_json(path)

        if not {'rounds', 'training_params', 'strategy'} <= raw_data.keys():
            raise Exception("Invalid config file format")

        if {'server_address'} <= raw_data.keys():
            self.server_address = IPAddress.validReturnIPAddress(raw_data['server_address'])

        self.rounds = int(raw_data["rounds"])

        self.training_data.build_training_params(raw_data['training_params'])

        tmp_training_model = ModelBuilder()
        if not {'model_params'} <= raw_data.keys():
            self.training_model = head_MLP
        else:
            if not {'num_classes', 'input_shape', 'model'} <= raw_data['model_params'].keys():
                raise Exception("Invalid config file format")

            tmp_training_model.build_model_params(raw_data['model_params'])
            self.training_model = tmp_training_model.model.get_model()

        Display.open()
        Display.model_info(tmp_training_model, self.training_model)

        if{'centralized_test_dataset', 'pb_model'} <= raw_data.keys():
            evaluation = EvaluationBuilder(self.training_model, raw_data["pb_model"], raw_data['centralized_test_dataset'])
            self.evaluation = evaluation.get_eval_fn()

        init_weights_path = None
        if{'saved_weights_path'} <= raw_data.keys():
            init_weights_path = raw_data['saved_weights_path']

        strategy = StrategyBuilder()
        strategy.set_model(self.training_model)
        strategy.set_init_weights(init_weights_path)
        strategy.set_eval_fn(self.evaluation)
        strategy.build_strategy_params(raw_data["strategy"])
        self.strategy = strategy.get_strategy()

        Display.strategy_info(strategy)
        Display.close()
