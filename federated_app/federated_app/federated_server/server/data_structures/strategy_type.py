from enum import Enum

class StrategyType:
    class Type(Enum):
        DESKTOP = 0
        ANDROID = 1

        @classmethod
        def str2enum(cls, type: str):
            low_type = type.lower()
            if low_type == 'android' or low_type == "a":
                return cls.ANDROID
            elif low_type == 'desktop' or low_type == "d":
                return cls.DESKTOP
            elif low_type == '0':
                return cls.DESKTOP
            elif low_type == '1':
                return cls.ANDROID
            else:
                return cls.DESKTOP
