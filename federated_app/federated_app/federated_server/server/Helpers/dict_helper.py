

class DICTHelper:

    STR_TO_BOOL = {
        "True": True,
        "False": False
    }

    SPECIAL_CHARS = "_,.*-"

    def __init__(self):
        pass

    @staticmethod
    def set_dict_keys_to_lower(d):
        d = {key.lower(): item for key, item in d.items()}
        return d

    @staticmethod
    def remove_special_chars(d):
        removed_special_chars = {key.strip(DICTHelper.SPECIAL_CHARS): item for key, item in d.items()}
        return removed_special_chars

    @staticmethod
    def remove_dict_keys_spaces(d):
        removed_spaces = {key.replace(' ', ''): item for key, item in d.items()}
        return removed_spaces

    @staticmethod
    def clean_dict_keys(d):
        dict_to_lower = DICTHelper.set_dict_keys_to_lower(d)
        removed_spaces = DICTHelper.remove_dict_keys_spaces(dict_to_lower)
        removed_special_chars = DICTHelper.remove_special_chars(removed_spaces)
        return removed_special_chars

    @staticmethod
    def combine_dict_params(const_dict: dict, config_dict: dict):

        tmp_dict = {}
        for key in const_dict:

            if key in config_dict:
                if isinstance(config_dict[key], str):
                    tmp_dict[key] = DICTHelper.STR_TO_BOOL[config_dict[key]]
                else:
                    tmp_dict[key] = config_dict[key]
            else:
                tmp_dict[key] = const_dict[key]

        return tmp_dict
