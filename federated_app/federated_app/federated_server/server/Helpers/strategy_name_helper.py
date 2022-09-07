

class StrategyName:

    # ! sa elimin din nume spatiile si caracterele ciudate
    @staticmethod
    def get_strategy_name(strategy_name: str):
        strategy_name = strategy_name.lower()
        if any(strategy_name in s for s in ["fedavg", "federated_average", "federated average", "federatedaverage"]):
            return "FedAvg"
        elif any(strategy_name in s for s in ["fedyogi", "federated_yogi", "federated yogi", "federatedyogi"]):
            return "FedYogi"
        elif any(strategy_name in s for s in ["fedadam", "federated_adam", "federated adam", "federatedadam"]):
            return "FedAdam"
        else:
            raise Exception("Invalid strategy name ")
