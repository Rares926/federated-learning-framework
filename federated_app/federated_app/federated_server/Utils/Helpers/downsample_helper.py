from typing import Tuple


class Downsample:

    @staticmethod
    def downsample_image_sizes(img_size: Tuple[int, int], downsample_factor: int):
        downsample_factor = 2**downsample_factor
        result = [x // downsample_factor for x in img_size]
        return result
