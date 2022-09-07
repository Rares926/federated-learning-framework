# Imports
import cv2 as cv
import numpy as np

# Internal imports
from ..data_structures.image_shape import ImageShape
from ..data_structures.image_format import ImageFormat
from ..data_structures.resize_method import ResizeMethod
from ..utils.image_preprocessing import ImageProcessing
from ..data_structures.ratio import Ratio

# Typing imports

class ImageLoader:
    def __init__(self, image_shape: ImageShape,
                 image_format: ImageFormat,
                 resize_method: ResizeMethod,
                 ratios: Ratio,
                 resize_after_crop: ResizeMethod,
                 normalize: bool = True):

        self.image_shape = image_shape
        self.image_format = image_format
        self.resize_method = resize_method
        self.ratios = ratios
        self.resize_after_crop = resize_after_crop
        self.normalize = normalize

    def load_image(self, image_path: str):
        image = cv.imread(image_path)

        if self.image_format.channels_format == ImageFormat.ChannelsFormat.RGB:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        elif self.image_format.channels_format == ImageFormat.ChannelsFormat.GRAYSCALE:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        if self.image_format.data_format == ImageFormat.DataType.FLOAT:
            image = image.astype(np.float32)

        if self.resize_method == ResizeMethod.CROP:
            image = ImageProcessing.crop(image, self.image_shape, self.ratios, self.resize_after_crop)
        if self.resize_method == ResizeMethod.STRETCH:
            image = ImageProcessing.stretch(image, self.image_shape)
        if self.resize_method == ResizeMethod.LETTERBOX:
            image = ImageProcessing.letterbox(image, self.image_shape)

        if self.normalize:
            image = ImageProcessing.normalize(image)

        return image
