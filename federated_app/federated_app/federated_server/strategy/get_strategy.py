from tokenize import String

from flwr.common import (
    Parameters,
    Scalar,
    Weights
)
from typing import (
    Callable,
    Dict,
    Optional,
    Tuple
)
from ..models.model import head
from ..Utils.weights_converter import WeightsConverter

def get_strategy(strategy_name: String,
                 android_strategy: bool,
                 fraction_fit: float = 0.1,
                 fraction_eval: float = 0.1,
                 min_fit_clients: int = 2,
                 min_eval_clients: int = 2,
                 min_available_clients: int = 2,
                 eval_fn: Optional[Callable[[Weights],
                                   Optional[Tuple[float,
                                            Dict[str, Scalar]]]]] = None,
                 on_fit_config_fn: Optional[Callable[[int],
                                            Dict[str, Scalar]]] = None,
                 on_evaluate_config_fn: Optional[Callable[[int],
                                                 Dict[str, Scalar]]] = None,
                 accept_failures: bool = True,
                 eta: float = 1e-2,
                 eta_l: float = 0.0316,
                 beta_1: float = 0.9,
                 beta_2: float = 0.99,
                 tau: float = 1e-3,
                 model_used=head
                 ):

    if not android_strategy:
        if strategy_name == "FedAvg":
            from flwr.server.strategy import FedAvg
            print("SERVER STARTED, STRATEGY USED : FEDERATED AVERAGE ")
            return FedAvg(fraction_fit=fraction_fit,
                          fraction_eval=fraction_eval,
                          min_fit_clients=min_fit_clients,
                          min_eval_clients=min_eval_clients,
                          min_available_clients=min_available_clients,
                          eval_fn=eval_fn,
                          on_fit_config_fn=on_fit_config_fn,
                          on_evaluate_config_fn=on_evaluate_config_fn,
                          accept_failures=accept_failures)
        elif strategy_name == "FedYogi":
            from flwr.server.strategy import FedYogi
            print("SERVER STARTED, STRATEGY USED : FEDERATED YOGI ")
            return FedYogi(fraction_fit=fraction_fit,
                           fraction_eval=fraction_eval,
                           min_fit_clients=min_fit_clients,
                           min_eval_clients=min_eval_clients,
                           min_available_clients=min_available_clients,
                           eval_fn=eval_fn,
                           on_fit_config_fn=on_fit_config_fn,
                           on_evaluate_config_fn=on_evaluate_config_fn,
                           accept_failures=accept_failures,
                           eta=eta,
                           eta_l=eta_l,
                           beta_1=beta_1,
                           beta_2=beta_2,
                           tau=tau)
        elif strategy_name == "FedAdam":
            from flwr.server.strategy import FedAdam
            pass
    else:
        if strategy_name == "FedAverageAndroid":
            print("SERVER STARTED, STRATEGY USED : FEDERATED AVERAGE ANDROID ")
            from flwr.server.strategy import FedAvgAndroid
            return FedAvgAndroid(fraction_fit=fraction_fit,
                                 fraction_eval=fraction_eval,
                                 min_fit_clients=min_fit_clients,
                                 min_eval_clients=min_eval_clients,
                                 min_available_clients=min_available_clients,
                                 eval_fn=eval_fn,
                                 on_fit_config_fn=on_fit_config_fn,
                                 on_evaluate_config_fn=on_evaluate_config_fn,
                                 accept_failures=accept_failures)
        elif strategy_name == "FedYogiAndroid":
            from ..strategy.FedYogiAndroid import FedYogiAndroid
            print("SERVER STARTED, STRATEGY USED : FEDERATED YOGI ANDROID ")
            return FedYogiAndroid(fraction_fit=fraction_fit,
                                  fraction_eval=fraction_eval,
                                  min_fit_clients=min_fit_clients,
                                  min_eval_clients=min_eval_clients,
                                  min_available_clients=min_available_clients,
                                  eval_fn=eval_fn,
                                  on_fit_config_fn=on_fit_config_fn,
                                  on_evaluate_config_fn=on_evaluate_config_fn,
                                  accept_failures=accept_failures,
                                  eta=eta,
                                  eta_l=eta_l,
                                  beta_1=beta_1,
                                  beta_2=beta_2,
                                  tau=tau,
                                  initial_parameters=WeightsConverter.weights_to_parameters(model_used.get_weights()))
        elif strategy_name == "FedAdamAndroid":
            from ..strategy.FedAdamAndroid import FedAdamAndroid
