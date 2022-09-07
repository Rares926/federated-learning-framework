# Imports
from enum import Enum

# Internal imports
# Typing imports

class ImageFormat:

    class ChannelsFormat(Enum):
        UNDEFINED = 1
        BGR = 2
        RGB = 3
        GRAYSCALE = 4

        @classmethod
        def str2enum(cls, channel_format_string, error_if_undefined=False):
            channel_format_string = channel_format_string.lower()
            if channel_format_string == 'bgr':
                return cls.BGR
            elif channel_format_string == 'rgb':
                return cls.RGB
            elif channel_format_string == 'gray':
                return cls.GRAYSCALE
            elif error_if_undefined:
                raise Exception('Error: Undefined color space!')
            return cls.UNDEFINED

    class DataType(Enum):
        UNDEFINED = 1
        UINT8 = 2
        FLOAT = 3

        @classmethod
        def str2enum(cls, data_type_string, error_if_undefined=False):
            data_type_string = data_type_string.lower()

            if data_type_string == 'uint8':
                return cls.UINT8
            elif data_type_string == 'float':
                return cls.FLOAT
            elif error_if_undefined:
                raise Exception('Error: Undefined aspect ratio mode type!')
            return cls.UNDEFINED

    def __init__(self, image_format_data: dict):
        self.channels_format = self.ChannelsFormat.str2enum(image_format_data['channels'])
        self.data_format = self.DataType.str2enum(image_format_data['data_type'])

    def __str__(self) -> str:
        print("The data type is {} in {} format".format(self.data_format, self.channels_format))
