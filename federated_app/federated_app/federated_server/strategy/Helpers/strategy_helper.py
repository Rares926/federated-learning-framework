
from ...server.Helpers.dict_helper import DICTHelper

class StrategyHelper:

    STR_TO_BOOL = {
        "True": True,
        "False": False
    }

    DEFAULT_CLIENT_INFO = {"fraction_fit": 1,
                           "fraction_eval": 1,
                           "min_fit_clients": 1,
                           "min_eval_clients": 1,
                           "min_available_clients": 1,
                           "accept_failures": True}

    DEFAULT_AGGREGATION_PARAMS = {"eta": 0.01,
                                  "eta_l": 0.0316,
                                  "beta_1": 0.9,
                                  "beta_2": 0.99,
                                  "tau": 0.001}

    def __init__(self):
        pass

    @staticmethod
    def get_client_info_params(client_info: dict):
        client_info = DICTHelper.combine_dict_params(StrategyHelper.DEFAULT_CLIENT_INFO,
                                                     client_info)
        return client_info

    @staticmethod
    def get_aggregation_params(aggregation_params: dict):
        aggregation_params = DICTHelper.combine_dict_params(StrategyHelper.DEFAULT_AGGREGATION_PARAMS,
                                                            aggregation_params)
        return aggregation_params
