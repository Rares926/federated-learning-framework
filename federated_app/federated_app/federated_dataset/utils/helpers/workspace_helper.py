# Imports
import os
import cv2 as cv
# Internal imports
from ...data_structures.image_loader import ImageLoader
from ....federated_server.Utils.Helpers.io_helper import IOHelper
from ....federated_server.Utils.Helpers.json_helper import JsonHelper
# Typing imports
from typing import Dict


class WorkspaceHelper:
    def __init__(self, dataset_directory: str, workspace_directory: str, image_loader: ImageLoader):
        self.dataset_directory = dataset_directory
        self.workspace_directory = workspace_directory
        self.image_loader = image_loader
        self.labels: Dict[str, Dict[str, str]] = None

    def createFolders(self):
        IOHelper.deletedirectory(os.path.join(self.workspace_directory, 'inputData'))
        directories = ('train', 'test')
        for item in directories:
            output_path = os.path.join(self.workspace_directory, item)
            IOHelper.create_directory(output_path, True)
            for key in self.labels.keys():
                class_output_path = os.path.join(output_path, self.labels[key]["name"])
                IOHelper.create_directory(class_output_path)

    def build_labels(self):  # 1
        labels = {}

        dir_list = IOHelper.get_subdirs(self.dataset_directory)
        count = len(dir_list)

        for i in range(count):
            content = {}
            content['name'] = dir_list[i]
            content['uid'] = "class_" + format(i, '03d')
            labels[str(i)] = content

        JsonHelper.write_json(os.path.join(self.workspace_directory, 'data.json'), labels)

        self.labels = labels

    def splitData(self, quotient: float):

        log_train = {}
        log_train_path = os.path.join(self.workspace_directory, 'train_log.json')
        log_test = {}
        log_test_path = os.path.join(self.workspace_directory, 'test_log.json')
        if quotient >= 1 or quotient <= 0:
            raise Exception("Split quotient out of bounds!")
        for key in self.labels:
            # ! eventual aplic un random pe lista asta
            list = IOHelper.get_image_files(os.path.join(self.dataset_directory, self.labels[key]['name']))
            number_of_files = len(list)
            to_be_trained = int(quotient * number_of_files)

            for photo in range(to_be_trained):
                source = os.path.join(self.dataset_directory, self.labels[key]['name'], list[photo])
                new_name = self.labels[key]['uid'] + 'P' + str(photo) + '.jpg'
                destination = os.path.join(self.workspace_directory,
                                           'train',
                                           self.labels[key]['name'],
                                           new_name)
                image = self.image_loader.load_image(source)
                log_train[list[photo]] = new_name
                cv.imwrite(destination, image)
                # IOHelper.copyfile(source, destination)

            for photo in range(to_be_trained, number_of_files):
                source = os.path.join(self.dataset_directory, self.labels[key]['name'], list[photo])
                new_name = self.labels[key]['uid'] + 'P' + str(photo) + '.jpg'
                destination = os.path.join(self.workspace_directory,
                                           'test',
                                           self.labels[key]['name'],
                                           new_name)
                image = self.image_loader.load_image(source)
                log_test[list[photo]] = new_name
                cv.imwrite(destination, image)
                # IOHelper.copyfile(source, destination)
            JsonHelper.write_json(log_train_path, log_train)
            JsonHelper.write_json(log_test_path, log_test)

    def federateData(self, federated_factor: float):
        # this may be changed to extract the fedQuotient directly from the json config file
        fedQuotient = int(1 // federated_factor)

        centralized_partition_data = self.workspace_directory + "/centralized_partition_test.txt"
        f_centralized_test = open(centralized_partition_data, "a")

        for partition in range(fedQuotient):

            train_name = self.workspace_directory + "/partition_" + str(partition) + "_train.txt"
            f_train = open(train_name, "a")

            # ! this could be changed to extract the names from the labels dict
            cls_list = os.listdir(os.path.join(self.workspace_directory, 'train'))

            for cls in cls_list:
                img_list = os.listdir(os.path.join(self.workspace_directory, 'train', cls))
                nr_of_img = len(img_list)
                partition_size = (nr_of_img // fedQuotient)

                if partition == 0:
                    start = 0
                    end = partition_size
                else:
                    start = partition * partition_size
                    end = start + partition_size

                for train_img in range(start, end):
                    path = "train/" + cls + "/" + img_list[train_img] + "\n"
                    f_train.write(path)

            test_name = self.workspace_directory + "/partition_" + str(partition) + "_test.txt"
            f_test = open(test_name, "a")

            for cls in cls_list:
                img_list = os.listdir(os.path.join(self.workspace_directory, 'test', cls))
                nr_of_img = len(img_list)
                partition_size = (nr_of_img // fedQuotient)

                if partition == 0:
                    start = 0
                    end = partition_size
                else:
                    start = partition * partition_size
                    end = start + partition_size

                for test_img in range(start, end):
                    path = "test/" + cls + "/" + img_list[test_img] + "\n"
                    f_test.write(path)
                    f_centralized_test.write(path)
