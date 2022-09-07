import os
import ntpath
import shutil
from pathlib import Path

# Internal framework imports

# Typing imports imports
from typing import List, Tuple


class IOHelper:
    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')
    VIDEO_EXTENSIONS = ('.mp4', '.wmv', '.mov', '.avi', '.mkv')

    def __init__(self):
        pass

    @staticmethod
    def ensure_valid_path_name(path: str) -> Path:
        return Path(path)

    @staticmethod
    def dir_exists(dir_path: str) -> bool:
        return os.path.isdir(dir_path) and os.path.exists(dir_path)

    @staticmethod
    def check_if_dir_exists(dir_path: str, custom_error_message: bool = None) -> None:
        if not IOHelper.dir_exists(dir_path):
            error_message = 'The specified dir path does not exist' if custom_error_message is None else custom_error_message
            raise Exception("{}: {}".format(error_message, dir_path))

    @staticmethod
    def file_exists(file_path: str) -> bool:
        return os.path.isfile(file_path) and os.path.exists(file_path)

    @staticmethod
    def check_if_file_exists(file_path: str, custom_error_message: bool = None) -> None:
        if not IOHelper.file_exists(file_path):
            error_message = 'The specified file path does not exist' if custom_error_message is None else custom_error_message
            raise Exception("{}: {}".format(error_message, file_path))

    @staticmethod
    def get_subdirs(root_dir: str, full_path: bool = False) -> List[str]:
        if full_path:
            sub_dirs = [os.path.join(root_dir, d)
                        for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        else:
            sub_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        return sub_dirs

    @staticmethod
    def get_files(dir_path: str,
                  extensions: Tuple[str],
                  full_path: bool = False,
                  recursively: bool = False) -> List[str]:
        if full_path:
            files = [os.path.join(dir_path, f)
                     for f in os.listdir(dir_path)
                     if os.path.isfile(os.path.join(dir_path, f)) and f.lower().endswith(extensions)]
        else:
            files = [f for f in os.listdir(dir_path)
                     if os.path.isfile(os.path.join(dir_path, f)) and f.lower().endswith(extensions)]

        if recursively:
            for subdir in IOHelper.get_subdirs(dir_path):
                files.extend(IOHelper.get_files(os.path.join(dir_path, subdir), extensions, recursively))

        return files

    @staticmethod
    def get_image_files(dir_path: str, full_path: bool = False, recursively: bool = False) -> List[str]:
        return IOHelper.get_files(dir_path, IOHelper.IMAGE_EXTENSIONS, full_path, recursively)

    @staticmethod
    def get_image_files_without_extension(dir_path: str,
                                          full_path: bool = False,
                                          recursively: bool = False) -> List[str]:
        img_files = IOHelper.get_files(dir_path, IOHelper.IMAGE_EXTENSIONS, full_path, recursively)
        return [IOHelper.get_filename_without_extension(file) for file in img_files]

    @staticmethod
    def get_filename(path: str) -> str:
        return Path(path).name.strip()

    @staticmethod
    def get_filename_without_extension(path: str) -> str:
        filename = IOHelper.get_filename(path)
        filename_without_extension, _ = os.path.splitext(filename)
        return filename_without_extension

    @staticmethod
    def get_epoch_from_checkpoint_path(path: str) -> int:
        epoch = int(IOHelper.get_filename_without_extension(path)[-3:])
        return epoch

    @staticmethod
    def get_extension(path: str) -> str:
        return Path(path).suffix.strip()

    @staticmethod
    def get_parent_path(path: str) -> Path:
        return Path(path).parent

    @staticmethod
    def create_directory(dir_path: str, show_confirmation: bool = False) -> None:
        if not IOHelper.dir_exists(dir_path):
            os.makedirs(dir_path)
            if show_confirmation:
                print("Created {} directory".format(dir_path))

    @staticmethod
    def copyfile(src: str, dst: str) -> None:
        shutil.copyfile(src, dst)

    # TODO: Add dir_exists_ok parameter when switch to min python version == 3.8
    @staticmethod
    def copytree(src: str, dst: str) -> None:
        shutil.copyfile(src, dst)

    @staticmethod
    def movefile(src: str, dst: str) -> None:
        shutil.move(src, dst)

    @staticmethod
    def move_directory(src: str, dst: str) -> None:
        shutil.move(src, dst)

    @staticmethod
    def renamefile(src: str, dst: str) -> None:
        os.rename(src, dst)

    @staticmethod
    def deletefile(file_path: str) -> None:
        if IOHelper.file_exists(file_path):
            os.remove(file_path)
        else:
            print("Cannot remove file because it doesn't exist: ", file_path)

    @staticmethod
    def deletedirectory(dir_path: str) -> None:
        if IOHelper.dir_exists(dir_path):
            shutil.rmtree(dir_path)
        else:
            print("Cannot delete directory because it doesn't exist ", dir_path)

    @staticmethod
    def get_path_leaf(path: str) -> str:
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)

    @staticmethod
    def is_empty_file(file_path: str) -> bool:
        return os.stat(file_path).st_size == 0

    @staticmethod
    def is_image_file(file_path: str) -> bool:
        extension = IOHelper.get_extension(file_path)
        return extension.lower() in IOHelper.IMAGE_EXTENSIONS
