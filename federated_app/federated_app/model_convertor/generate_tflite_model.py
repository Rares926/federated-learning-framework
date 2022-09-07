import tensorflow as tf
import numpy as np
import os

IMG_SIZE = 224
NUM_CLASSES = 11
NUM_FEATURES = 7 * 7 * 1280
BATCH_SIZE = 32
SIGNATURE_DICT = {"dense_1/kernel:0": tf.TensorSpec(shape=[NUM_FEATURES, 128], dtype=tf.float32, name="item1"),
                  "dense_1/bias:0": tf.TensorSpec(shape=[128, ], dtype=tf.float32, name="item2"),
                  "dense_2/kernel:0": tf.TensorSpec(shape=[128, 10], dtype=tf.float32, name="item2"),
                  "dense_2/bias:0": tf.TensorSpec(shape=[10, ], dtype=tf.float32, name="item2")}


class TransferLearningModel(tf.Module):

    def __init__(self, learning_rate=0.01):
        """
        Initializes a transfer learning model instance.

        Parameters:
            learning_rate (float) : A learning rate for the optimzer.
        """

        # - head model
        # ? DEBATABLE IF THE INPUT SHAPE SHOULD BE DECLARED ON THE FIRST LAYER
        self.head_model = tf.keras.Sequential([
            tf.keras.layers.Input([NUM_FEATURES]),
            tf.keras.layers.Dense(128, activation='relu', name='0'),
            tf.keras.layers.Dense(NUM_CLASSES, name='1')])

        # - base model
        self.base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            alpha=1.0,
            include_top=False,
            weights='imagenet')

        # ? from_logits = True or False
        # - loss function
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        # - optimizer
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

        self.head_model.compile(optimizer=self.optimizer,
                                loss=self.loss_fn)

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

        # - Preprocesses a tensor or Numpy array while first multiplying the normalized values with 255
        x = tf.keras.applications.mobilenet_v2.preprocess_input(
            tf.multiply(feature, 255))

        # - reshapes the base_model output to 1,1*1*1280(1 is image size downsampled five times
        # - and 1280 is the number of features extracted)
        base_model_output = self.base_model(x, training=False)
        bottleneck = tf.reshape(
            base_model_output, (-1, NUM_FEATURES))

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
            # ? loss=self.loss_fn(prediction,label)

        gradients = tape.gradient(loss, self.head_model.trainable_variables)

        self.head_model.optimizer.apply_gradients(
            zip(gradients, self.head_model.trainable_variables))
        # ? self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

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
            base_model_output, (-1, NUM_FEATURES))

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

def convert_and_save(saved_model_dir='saved_model_new'):
    """
    Converts and saves the TFLite Transfer Learning model.

    Parameters:
        saved_model_dir: A directory path to save a converted model.
    Returns:
        NONE
    """
    transfer_learning_model = TransferLearningModel()

    tf.saved_model.save(
        transfer_learning_model,
        saved_model_dir,
        signatures={
            'load': transfer_learning_model.load.get_concrete_function(),
            'train': transfer_learning_model.train.get_concrete_function(),
            'infer': transfer_learning_model.infer.get_concrete_function(),
            'save': transfer_learning_model.save.get_concrete_function(),
            'restore': transfer_learning_model.restore.get_concrete_function(),
            'extract': transfer_learning_model.extract_weights.get_concrete_function(),
            'initialize': transfer_learning_model.initialize_weights.get_concrete_function(),

        })

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]

    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()

    model_file_path = os.path.join('model.tflite')
    with open(model_file_path, 'wb') as model_file:
        model_file.write(tflite_model)

if __name__ == '__main__':
    model = TransferLearningModel()
    convert_and_save()
