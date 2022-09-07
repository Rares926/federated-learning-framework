# Internal framework inputs

from ...federated_server.server.Helpers.dict_helper import DICTHelper
import tensorflow as tf

class OptimizerHelper:

    STR_TO_BOOL = {"True": True,
                   "False": False}

    DEFAULT_OPTIMIZER_VALUES = {
        "Adam":
        {
            "lr": 0.001,
            "grad_clip": 0.5,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-07,
            "decay": 0.0,
            "amsgrad": False
        },

        "SGD":
        {
            "lr": 0.01,
            "momentum": 0.9,
            "nesterov": True
        },

        "Adadelta":
        {
            "lr": 0.001,
            "rho": 0.95,
            "epsilon": 1e-07
        },

        "Adagrad":
        {
            "lr": 0.001,
            "initial_accumulator_value": 0.1,
            "epsilon": 1e-07

        },

        "RMSprop":
        {
            "lr": 0.001,
            "rho": 0.9,
            "momentum": 0.0,
            "epsilon": 1e-07,
            "centered": "False"
        }


    }

    def __init__(self):
        pass

    @staticmethod
    def set_optimizer_value(optimizer_params: dict, name: str):

        if not optimizer_params:
            return OptimizerHelper.DEFAULT_OPTIMIZER_VALUES[name]

        optimizer_params = DICTHelper.clean_dict_keys(optimizer_params)
        opt_tmp = DICTHelper.combine_dict_params(OptimizerHelper.DEFAULT_OPTIMIZER_VALUES,
                                                 optimizer_params)

        return opt_tmp

    @staticmethod
    def set_learning_rate(name, lr=None):
        if lr is None:
            return OptimizerHelper.DEFAULT_OPTIMIZER_VALUES[name]["lr"]
        elif len(lr) == 1:
            if not{"value"} <= lr.keys():
                raise Exception("Invalid LR format")
            return lr["value"]

    @staticmethod
    def get_base_optimizer(lr: float):
        return tf.keras.optimizers.SGD(learning_rate=lr)
