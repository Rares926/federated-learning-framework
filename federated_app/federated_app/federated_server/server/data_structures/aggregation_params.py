

from dataclasses import dataclass
from ..Helpers.dict_helper import DICTHelper
from ...strategy.Helpers.strategy_helper import StrategyHelper

@dataclass
class AggregationParams:
    eta: float = None
    eta_l: float = None
    beta_1: float = None
    beta_2: float = None
    tau: float = None

    def build_params(self, aggregation_params: dict):

        aggregation_params = DICTHelper.clean_dict_keys(aggregation_params)

        combined_aggregation_params = StrategyHelper.get_aggregation_params(aggregation_params)

        self.eta = float(combined_aggregation_params['eta'])
        self.eta_l = float(combined_aggregation_params['eta_l'])
        self.beta_1 = float(combined_aggregation_params['beta_1'])
        self.beta_2 = float(combined_aggregation_params['beta_2'])
        self.tau = float(combined_aggregation_params['tau'])
