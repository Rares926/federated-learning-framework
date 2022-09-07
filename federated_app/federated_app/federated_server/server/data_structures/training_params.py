
from ..Helpers.dict_helper import DICTHelper

class TrainingParams:

    def __init__(self) -> None:
        self.batch = 32
        self.learning_rate = 0.001
        self.epochs = 1

    def __str__(self):
        print(f"The model will train on device on batches of size {self.batch}, "
              f"with a constant learning_rate of {self.learning_rate} "
              f"for a number of {self.epochs} epochs")

    def build_training_params(self, training_params: dict):
        training_params = DICTHelper.clean_dict_keys(training_params)
        dict_params = training_params.keys()
        for key in dict_params:
            if any(key in s for s in ["b", "batch", "batches"]):
                self.batch = training_params[key]
            elif any(key in s for s in ["lr", "learning_rate", "learning rate"]):
                self.learning_rate = training_params[key]
            elif any(key in s for s in ["e", "epochs", "epoch"]):
                self.epochs = training_params[key]
            else:
                raise Exception("Invalid training params params")
