import tensorflow as tf
import numpy as np


class TransferLearningModel(tf.Module):
    IMG_SIZE = None
    NUM_FEATURES = None
    NUM_CLASSES = None

    def __init__(self, training_model, feature_extractor, optimizer):
        """
        Initializes a transfer learning model instance.

        Parameters:

        """
        # - head model
        self.head_model = training_model

        # - base model
        self.base_model = feature_extractor

        # - loss function
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        # - optimizer
        self.optimizer = optimizer

        self.head_model.compile(optimizer=self.optimizer,
                                loss=self.loss_fn)

    @tf.function()
    def update_static_values(self, img_size, num_features, num_classes):
        TransferLearningModel.IMG_SIZE = img_size
        TransferLearningModel.NUM_FEATURES = num_features
        TransferLearningModel.NUM_CLASSES = num_classes

    @tf.function(input_signature=[tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32), ])
    # * TESTED
    def load(self, feature):
        """
        Generates and loads bottleneck features from the given image batch.

        Parameters:
            feature: A tensor of image feature batch to generate the bottleneck from.
        Returns:
            Map of the bottleneck.
        """

        # - Preprocesses a tensor or Numpy array encoding a batch of images.
        x = tf.keras.applications.mobilenet_v2.preprocess_input(
            tf.multiply(feature, 255))

        # - reshapes the base_model output to 1,1*1*1280(1 is image size downsampled five times
        # - and 1280 is the number of features extracted)
        base_model_output = self.base_model(x, training=False)
        bottleneck = tf.reshape(
            base_model_output, (-1, TransferLearningModel.NUM_FEATURES))

        return {'bottleneck': bottleneck}

    # - passes the bottleneck features trought the head model
    # * TESTED
    @tf.function(input_signature=[
        tf.TensorSpec([None, NUM_FEATURES], tf.float32),
        tf.TensorSpec([None, NUM_CLASSES], tf.float32), ])
    def train(self, bottleneck, label):
        """
        Runs one training step with the given bottleneck features and labels.

        Parameters:
            bottleneck: A tensor of bottleneck features generated from the base model.
            label: A tensor of class labels for the given batch.
        Returns:
            Map of the training loss.
        """

        with tf.GradientTape() as tape:
            logits = self.head_model(bottleneck)
            prediction = tf.nn.softmax(logits)

            loss = self.head_model.loss(prediction, label)

        gradients = tape.gradient(loss, self.head_model.trainable_variables)

        self.head_model.optimizer.apply_gradients(
            zip(gradients, self.head_model.trainable_variables))

        result = {"loss": loss}
        for grad in gradients:
            result[grad.name] = grad
        return result

    # * TESTED
    @tf.function(input_signature=[tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32)])
    def infer(self, image):
        """
        Invokes an inference on the given image.

        Parameters:
                feature: A tensor of image feature batch to invoke an inference on.
        Returns:
                Map of the softmax output.
        """

        x = tf.keras.applications.mobilenet_v2.preprocess_input(
            tf.multiply(image, 255))

        base_model_output = self.base_model(x, training=False)

        bottleneck = tf.reshape(
            base_model_output, (-1, TransferLearningModel.NUM_FEATURES))

        # logits = self.head_model(bottleneck, training=False)

        # result = tf.nn.softmax(logits)

        with tf.GradientTape() as tape:
            logits = self.head_model(bottleneck)
            prediction = tf.nn.softmax(logits)

        return {'output': prediction}

    # * TESTED
    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def save(self, checkpoint_path: str):
        """
        Saves the trainable weights to the given checkpoint file.

        Parameters:
                checkpoint_path (String) : A file path to save the model.
        Returns:
                Map of the checkpoint file path.
        """

        tensor_names = [weight.name for weight in self.head_model.weights]
        tensors_to_save = [weight.read_value() for weight in self.head_model.weights]
        tf.raw_ops.Save(
            filename=checkpoint_path,
            tensor_names=tensor_names,
            data=tensors_to_save,
            name='save')

        return {'checkpoint_path': checkpoint_path}

    # * TESTED
    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def restore(self, checkpoint_path):
        """
        Restores the serialized trainable weights from the given checkpoint file.

        Paramaters:
            checkpoint_path (String) : A path to a saved checkpoint file.
        Returns:
            Map of restored weights and biases.
        """
        restored_tensors = {}
        for tensor in self.head_model.weights:
            restored = tf.raw_ops.Restore(file_pattern=checkpoint_path,
                                          tensor_name=tensor.name,
                                          dt=tensor.dtype,
                                          name='restore')
            tensor.assign(restored)
            restored_tensors[tensor.name] = restored

        return restored_tensors

    # * TESTED
    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def extract_weights(self, checkpoint_path):
        """
        Extracts the traininable weights of the head model as a list of tensors.

        Paramaters:

        Returns:
            Map of extracted weights and biases.
        """
        tmp_dict = {}
        tensor_names = [weight.name for weight in self.head_model.weights]
        tensors_to_save = [weight.read_value() for weight in self.head_model.weights]
        for index, layer in enumerate(tensors_to_save):
            tmp_dict[tensor_names[index]] = layer

        return tmp_dict

    # * tested
    @tf.function(input_signature=[tf.TensorSpec(shape=[NUM_FEATURES, 128], dtype=tf.float32),
                                  tf.TensorSpec(shape=[128, ], dtype=tf.float32),
                                  tf.TensorSpec(shape=[128, NUM_CLASSES], dtype=tf.float32),
                                  tf.TensorSpec(shape=[NUM_CLASSES, ], dtype=tf.float32)])
    def initialize_weights(self, weights1, bias1, weights2, bias2):
        """
        Initializes weights of the head model.

        Paramaters:
            weights : Numpy arrays used for initialization.
        Returns:
            NONE
        """

        restored_tensors = {}
        tensor_names = [weight.name for weight in self.head_model.weights]
        for i, tensor in enumerate(self.head_model.weights):
            if i == 0:
                tensor.assign(weights1)
                restored_tensors[tensor.name] = tensor
            elif i == 1:
                tensor.assign(bias1)
                restored_tensors[tensor.name] = tensor
            elif i == 2:
                tensor.assign(weights2)
                restored_tensors[tensor.name] = tensor
            elif i == 3:
                tensor.assign(bias2)
                restored_tensors[tensor.name] = tensor

        return restored_tensors
