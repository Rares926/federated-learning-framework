import os
import sys
import tensorflow as tf

from jsonargparse import ArgumentParser
from jsonargparse.util import usage_and_exit_error_handler

from .builder.converted_model_builder import ConvertedModelBuilder
from .transfer_learning_model import TransferLearningModel

class ModelGenerator():

    def __init__(self, model_args) -> None:
        self.model_args = model_args

    def convert_and_save(self, saved_model_dir='saved_model_new'):
        """
        Converts and saves the TFLite Transfer Learning model.

        Parameters:
            saved_model_dir: A directory path to save a converted model.
        Returns:
            NONE
        """

        transfer_learning_model = TransferLearningModel(optimizer=self.model_args.optimizer,
                                                        training_model=self.model_args.training_model.model.get_model(),
                                                        feature_extractor=self.model_args.training_model.feature_extractor.get_opt())

        transfer_learning_model.update_static_values(img_size=self.model_args.training_model.input_shape.width,
                                                     num_features=self.model_args.training_model.num_features,
                                                     num_classes=self.model_args.training_model.classes)

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

def run():
    try:
        parser = ArgumentParser(prog="generate_model",
                                error_handler=usage_and_exit_error_handler,
                                description="Convert and generate a tflite model.")

        parser.add_argument("--model_generator_file",
                            "-model",
                            required=True,
                            help="The path of the model generator file (must be JSON format)")

        program_args = parser.parse_args()
        model_args = ConvertedModelBuilder()
        model_args.arg_parse(program_args.model_generator_file)

        generated_model = ModelGenerator(model_args)
        generated_model.convert_and_save()

    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(ex)

if __name__ == "__main__":
    run()
