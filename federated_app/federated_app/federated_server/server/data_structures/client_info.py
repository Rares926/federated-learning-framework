

from dataclasses import dataclass
from ..Helpers.dict_helper import DICTHelper
from ...strategy.Helpers.strategy_helper import StrategyHelper

@dataclass
class ClientInfo:
    fraction_fit: float = None
    fraction_eval: float = None
    min_fit_clients: int = None
    min_eval_clients: int = None
    min_available_clients: int = None
    accept_failures: bool = True

    def build_params(self, client_info: dict):

        client_info = DICTHelper.clean_dict_keys(client_info)

        # ! Something does not work here
        combined_client_info = StrategyHelper.get_client_info_params(client_info)

        self.fraction_fit = float(combined_client_info['fraction_fit'])
        self.fraction_eval = float(combined_client_info['fraction_eval'])
        self.min_fit_clients = int(combined_client_info['min_fit_clients'])
        self.min_eval_clients = int(combined_client_info['min_eval_clients'])
        self.min_available_clients = int(combined_client_info['min_available_clients'])
        self.accept_failures = bool(combined_client_info['accept_failures'])

    def get_params(self):
        return self.fraction_fit, self.fraction_eval, self.min_fit_clients, self.min_eval_clients, self.min_available_clients, self.accept_failures  # noqa: E501
