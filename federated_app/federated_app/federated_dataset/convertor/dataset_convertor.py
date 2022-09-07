# Imports
import os
import sys
from jsonargparse import ArgumentParser
from jsonargparse.util import usage_and_exit_error_handler

# Internal imports
from ..builder.dataset_builder import DatasetBuilder
from ...federated_server.Utils.Helpers.io_helper import IOHelper
from ..data_structures.image_loader import ImageLoader
from ..utils.helpers.workspace_helper import WorkspaceHelper

# Typing imports

class DatasetConvertor():
    def __init__(self, dataset: DatasetBuilder) -> None:
        self.dataset = dataset

    def convert_dataset(self, dataset_root_dir: str, converted_dataset_dir: str):

        IOHelper.create_directory(converted_dataset_dir)

        image_loader = ImageLoader(self.dataset.image_shape,
                                   self.dataset.image_format,
                                   self.dataset.resize_method,
                                   self.dataset.ratios,
                                   self.dataset.resize_after_crop,
                                   normalize=False)

        workspace_creator = WorkspaceHelper(dataset_root_dir,
                                            converted_dataset_dir,
                                            image_loader)

        workspace_creator.build_labels()
        workspace_creator.createFolders()
        workspace_creator.splitData(self.dataset.split_percentage)
        workspace_creator.federateData(self.dataset.federated_factor)


def run():
    try:
        parser = ArgumentParser(prog="datasetconvertor",
                                error_handler=usage_and_exit_error_handler,
                                description="Convert a dataset into a federated dataset")

        parser.add_argument("--dataset_convertor_file",
                            "-config",
                            required=True,
                            help="The path of the dataset convertor file (must be JSON format)")

        program_args = parser.parse_args()

        dataset_args = DatasetBuilder()
        dataset_args.arg_parse(program_args.dataset_convertor_file)

        convertor = DatasetConvertor(dataset_args)

        convertor.convert_dataset(dataset_args.dataset_path, dataset_args.new_dataset_path)

    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(ex)

if __name__ == "__main__":
    run()
