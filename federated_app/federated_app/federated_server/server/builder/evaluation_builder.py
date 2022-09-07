import tensorflow as tf
import flwr as fl
from typing import Optional, Tuple
import cv2 as cv
import numpy as np
import json

class EvaluationBuilder():

    def __init__(self, training_model=None, model_pb=None, test_data_path=None) -> None:
        self.head_model = training_model
        self.model_pb = model_pb
        self.test_path = test_data_path
        self.head_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                                metrics=["accuracy",
                                         "binary_crossentropy"])

        self.centralized_result = {"loss": [], "accuracy": [], "binary_crossentropy": []}
        self.round = 1

    @staticmethod
    def one_hot_encode(val, size):
        encoded_value = np.zeros(size)
        encoded_value[val] = 1
        return encoded_value

    def get_eval_fn(self):

        model = tf.saved_model.load(self.model_pb)
        model.update_static_values(224, 7 * 7 * 1280, 6)

        # interpreter = tf.lite.Interpreter(model_path="Z:/Federated Learning Projects/federated-learning-framework/federated_app/model.tflite")
        # load_signature = interpreter.get_signature_runner('load')

        images = []
        labels = []
        class_maping = {"bars_and_rebars": 0,
                        "channel_bars": 1,
                        "logs": 2,
                        "pills": 3,
                        "tubes": 4,
                        "vials": 5}

        centralized_partition_data_path = self.test_path + "/centralized_partition_test.txt"
        with open(centralized_partition_data_path) as file_in:

            for line in file_in:

                img_path = self.test_path + "/" + line.strip()
                img = cv.imread(img_path, cv.IMREAD_COLOR).astype(np.float32)
                img = img / 255
                images.append(img)

                class_in_text = line.split("/")
                one_hot_encoded = EvaluationBuilder.one_hot_encode(class_maping[class_in_text[1]], 6)
                labels.append(one_hot_encoded)

        # X = np.array(images)
        # we might have to one hot encode this classes
        Y = np.array(labels)
        # ! aici unde e 2 e practic numarul de clase si tre sa rescriu asta sa nu mai fie harcodat
        reshaped_Y = Y.reshape(-1, 6)

        def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
            # weights = WeightsConverter.parameters_to_weights(weights) weights are deserialized in the evaluate strategy method
            for index, layer in enumerate(self.head_model.layers):
                w, _ = layer.get_weights()
                layer_shape = w.shape
                weights[index * 2] = weights[index * 2].reshape(layer_shape)

            self.head_model.set_weights(weights)  # Update head model with the latest parameters

            bootlenecks = []
            for image in images:
                input_image = np.expand_dims(image, axis=0)
                bootleneck = model.load(input_image)["bottleneck"]
                # bootleneck = load_signature(feature=image)["bottleneck"]
                bootlenecks.append(bootleneck[0])
            bootlenecks = np.array(bootlenecks)
            print(bootlenecks.shape, reshaped_Y.shape)

            loss, accuracy, binary_crossentropy = self.head_model.evaluate(bootlenecks,
                                                                           reshaped_Y,
                                                                           verbose=0)

            self.centralized_result["loss"].append((self.round, loss))
            self.centralized_result["accuracy"].append((self.round, accuracy))
            self.centralized_result["binary_crossentropy"].append((self.round, binary_crossentropy))
            self.round += 1

            with open('centralized_training_result.json', 'w') as fp:
                json.dump(self.centralized_result, fp)

            return loss, {"accuracy": accuracy,
                          "binary_crossentropy": binary_crossentropy}

        return evaluate
