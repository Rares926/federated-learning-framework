import json

# Internal framework imports
from .io_helper import IOHelper

# Typing imports imports


class JsonHelper:
    """
    Class used to read/write a JSON file
    """
    def __init__(self):
        pass

    @staticmethod
    def read_json(input_json_file_path: str, check_if_file_exists: bool = True, custom_error_message: str = None) -> dict:
        if check_if_file_exists and not IOHelper.file_exists(input_json_file_path):
            error_message = 'The specified JSON file path does not exist' if custom_error_message is None else custom_error_message
            raise FileNotFoundError("{}: {}".format(error_message, input_json_file_path))

        json_data = None
        try:
            with open(input_json_file_path) as input_json_file:
                json_data = json.load(input_json_file)
        except ValueError:
            raise Exception("Decoding JSON has failed!")

        return json_data

    @staticmethod
    def write_json(output_json_file_path: str, json_data: dict) -> None:
        with open(output_json_file_path, 'w') as output_json_file_path:
            json.dump(json_data, output_json_file_path)
