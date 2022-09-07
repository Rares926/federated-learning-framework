import json
import tensorflow as tf
import matplotlib.pyplot as plt

# flake8: noqa

class ResultsVisualization:

    def __init__(self, result_path) -> None:

        self.training_results_dict = None

        with open(result_path) as json_file:
            self.training_results_dict = json.load(json_file)

    def show_accuracy(self):
        accuracy = [r[1] for r in self.training_results_dict["accuracy"]]


        epochs = [r[0] for r in self.training_results_dict["accuracy"]]

        plt.plot(epochs, accuracy, 'g', label='Validation accuracy')
        plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        plt.xticks(epochs)
        plt.title('Centralized validation accuracy')
        plt.xlabel('Aggregation rounds')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def show_loss(self):

        loss_val = [r[1] for r in self.training_results_dict["loss"]]
        epochs = [r[0] for r in self.training_results_dict["loss"]]

        plt.plot(epochs, loss_val, 'g', label='loss')
        plt.xticks(epochs)
        plt.title('Centralized loss')
        plt.xlabel('Aggregation rounds')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def show_mae(self):
        pass

    def show_binary_crossentropy(self):

        binary_crossentropy = [r[1] for r in self.training_results_dict["binary_crossentropy"]]
        epochs = [r[0] for r in self.training_results_dict["binary_crossentropy"]]

        plt.plot(epochs, binary_crossentropy, 'b', label='Binary crossentropy')
        plt.xticks(epochs)

        plt.title('Centralized validation accuracy')
        plt.xlabel('Aggregation rounds')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def show_data_distribution_validation_dataset(self):
        # Pass the x and y cordinates of the bars to the
        # function. The label argument gives a label to the data.
        plt.bar(["bars_and_rebars","channel_bars","grapes", "pills", "tubes", "vials", "logs"],[39,33,20,33,30,24,33])
        plt.legend()

        # The following commands add labels to our figure.
        plt.xlabel('Clase')
        plt.ylabel('Numar imagini')
        # plt.title('Vertical Bar chart')

        plt.show()

    def show_metrics(self):
        accuracy = [r[1] for r in self.training_results_dict["accuracy"]]
        epochs = [r[0] for r in self.training_results_dict["accuracy"]]

        plt.figure("Accuracy")
        plt.plot(epochs, accuracy, 'g', label='Validation accuracy')
        plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        plt.xticks(epochs)
        plt.title('Centralized validation accuracy')
        plt.xlabel('Aggregation rounds')
        plt.ylabel('Accuracy')
        plt.legend()

        loss_val = [r[1] for r in self.training_results_dict["loss"]]
        epochs = [r[0] for r in self.training_results_dict["loss"]]
        plt.figure("Loss")
        plt.plot(epochs, loss_val, 'g', label='loss')
        plt.xticks(epochs)
        plt.title('Centralized loss')
        plt.xlabel('Aggregation rounds')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

if __name__ == '__main__':
    model = ResultsVisualization("Z:/Federated Learning Projects/federated-learning-framework/federated_app/centralized_training_result.json") 
    # model.show_accuracy()
    # model.show_binary_crossentropy()
    # model.show_loss()
    model.show_metrics()
    # model.show_data_distribution_validation_dataset()