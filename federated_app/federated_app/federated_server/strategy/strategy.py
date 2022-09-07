
from federated_app.federated_server.server.builder.evaluation_builder import EvaluationBuilder
from ..Utils.weights_converter import WeightsConverter
from ..server.data_structures.client_info import ClientInfo
from ..server.data_structures.aggregation_params import AggregationParams
from ..server.data_structures.strategy_type import StrategyType

class Strategy:

    def __init__(self,
                 strategy_name: str,
                 strategy_type: str,
                 client_info: ClientInfo,
                 aggregation_params: AggregationParams,
                 training_model,
                 evaluation_fn,
                 initial_weights):

        self.name = strategy_name
        self.type = strategy_type
        self.client_info = client_info
        self.aggregation_params = aggregation_params
        self.model = training_model
        self.evaluation_fn = evaluation_fn
        self.initial_weights = initial_weights

    # - To be adjusted
    def build_strategy_params(self, strategy_data, fit_config):

        self.params = strategy_data
        self.fit_config = fit_config
        # self.params = StrategyHelper.set_optimizer_value(strategy_data["params"], self.name)

    def get_str(self):

        """Dispatch method"""
        method_name = 'strat_' + str(self.name)

        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: "Invalid strategy")

        # Call the method as we return it
        return method()

    @staticmethod
    def fit_config(rnd: int):
        """Return training configuration dict for each round.

        Keep batch size fixed at 32, perform two rounds of training with one
        local epoch, increase to two local epochs afterwards.
        """
        config = {
            "batch_size": 32,
            "local_epochs": 1,
        }

        return config

    def strat_FedAvg(self):

        if self.type == StrategyType.Type.DESKTOP:
            from flwr.server.strategy.fedavg import FedAvg as FA
        else:
            from ..strategy.FedAvgAndroid import FedAvgAndroid as FA

        return FA(fraction_fit=self.client_info.fraction_fit,
                  fraction_eval=self.client_info.fraction_eval,
                  min_fit_clients=self.client_info.min_fit_clients,
                  min_eval_clients=self.client_info.min_eval_clients,
                  min_available_clients=self.client_info.min_available_clients,
                  eval_fn=self.evaluation_fn,
                  on_fit_config_fn=Strategy.fit_config,
                  accept_failures=self.client_info.accept_failures,
                  initial_parameters=self.initial_weights
                  )

    def strat_FedYogi(self):

        if self.type == StrategyType.Type.DESKTOP:
            from flwr.server.strategy.fedyogi import FedYogi as FY
        else:
            from ..strategy.FedYogiAndroid import FedYogiAndroid as FY

        return FY(fraction_fit=self.client_info.fraction_fit,
                  fraction_eval=self.client_info.fraction_eval,
                  min_fit_clients=self.client_info.min_fit_clients,
                  min_eval_clients=self.client_info.min_eval_clients,
                  min_available_clients=self.client_info.min_available_clients,
                  eval_fn=self.evaluation_fn,
                  on_fit_config_fn=Strategy.fit_config,
                  accept_failures=self.client_info.accept_failures,
                  initial_parameters=self.initial_weights,
                  eta=self.aggregation_params.eta,
                  eta_l=self.aggregation_params.eta_l,
                  beta_1=self.aggregation_params.beta_1,
                  beta_2=self.aggregation_params.beta_2,
                  tau=self.aggregation_params.tau
                  )

    def strat_FedAdam(self):
        if self.type == StrategyType.Type.DESKTOP:
            from flwr.server.strategy.fedadam import FedAdam as FA
        else:
            from ..strategy.FedAdamAndroid import FedAdamAndroid as FA

        return FA(fraction_fit=self.client_info.fraction_fit,
                  fraction_eval=self.client_info.fraction_eval,
                  min_fit_clients=self.client_info.min_fit_clients,
                  min_eval_clients=self.client_info.min_eval_clients,
                  min_available_clients=self.client_info.min_available_clients,
                  eval_fn=self.evaluation_fn,
                  on_fit_config_fn=Strategy.fit_config,
                  accept_failures=self.client_info.accept_failures,
                  initial_parameters=self.initial_weights,
                  eta=self.aggregation_params.eta,
                  eta_l=self.aggregation_params.eta_l,
                  beta_1=self.aggregation_params.beta_1,
                  beta_2=self.aggregation_params.beta_2,
                  tau=self.aggregation_params.tau
                  )
