# Imports
# Internal imports
# Typing imports

class ResizeMethod:
    NONE = 1
    CROP = 2
    STRETCH = 3
    LETTERBOX = 4

    @classmethod
    def str2enum(cls, resize_method_string, error_if_none=False):
        resize_method_string = resize_method_string.lower()

        if resize_method_string == "crop":
            return cls.CROP
        elif resize_method_string == "stretch":
            return cls.STRETCH
        elif resize_method_string == "letterbox":
            return cls.LETTERBOX
        elif error_if_none:
            raise Exception("Error: No resize method!")
        return cls.NONE
