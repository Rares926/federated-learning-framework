

from federated_app.federated_server.server.builder.model_builder import ModelBuilder
from federated_app.federated_server.server.builder.strategy_builder import StrategyBuilder


class Display:

    @staticmethod
    def open():
        open_line = "*" * 100
        print(open_line)
        print("FEDERATED SERVER STARTED")
        print(open_line)

    @staticmethod
    def close():
        close_line = "*" * 100
        print(close_line)

    @staticmethod
    def model_info(model_builder: ModelBuilder, model):

        if model is not None:
            model.summary()
            print("\nFEATURE EXTRACTOR USED: MobilenetV2")
        else:
            model_builder.model.model.summary()
            print(f"\nFEATURE EXTRACTOR USED: {model_builder.feature_extractor_data.name}")

    @staticmethod
    def strategy_info(strategy: StrategyBuilder, verbose: int = 0):
        print(f"\nAGGREGATION STRATEGY USED: {strategy.name} \n")

    @staticmethod
    def server_info(server):
        pass
