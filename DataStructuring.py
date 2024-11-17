import csv
import os
import re


class DataStruct:
    @staticmethod
    def format_packages_to_tuples(main_data_folder_path, main_xml_folder_path):
        # returns a tuple with 3 elements (WNID, IntKey, Name)
        data = []
        # this for is creating tuples (WNID, IntKey)
        for i, folder in enumerate(os.listdir(main_data_folder_path)):
            data.append((folder[1:], i))

        # this for is appending a class name to each tuple
        # for is scanning main folder which contains a lot of another folders
        # in every folder is a lot xml files, for is scanning one file
        # and getting a class name from them
        for i, folder in enumerate(os.listdir(main_xml_folder_path)):
            if folder[1:] != data[i][0]:
                raise Exception("Incorrect (index)argument")
            file = os.listdir(main_xml_folder_path + fr"\{folder}")[i]
            with open(main_xml_folder_path + fr"\{folder}\{file}", "r") as xml_file:
                content = xml_file.read()
                name = re.findall(r"<name>(.*?)</name>", content)[0]
                data[i] = (data[i][0], data[i][1], name)
        print("Formed an indexed list")
        return data

    @staticmethod
    def create_wnid_int_name_file(indexed_file_path, main_data_folder_path, main_xml_folder_path):
        # indexed_file_path should be a csv file
        if indexed_file_path[-4:] != ".csv":
            raise Exception("incorrect csv file extension")
        with open(indexed_file_path, "w", newline="\n") as file:
            writer = csv.writer(file)
            data = Convertor.format_packages_to_tuples(main_data_folder_path,
                                                       main_xml_folder_path)
            writer.writerows(data)
        print("Created an indexed csv file!")

    @staticmethod
    def create_images_paths(csv_file, dict_wnid_int, main_folder_path, delete_previous=False):
        # this function create a "csv" file which contains all
        # image paths and their int _value. Main folder must
        # contain a lot of folders and in every folder must be
        # certain amount of images
        if csv_file[-4:] != ".csv":
            raise Exception("Incorrect csv file extension")
        if delete_previous:
            os.remove(csv_file)
        with open(csv_file, "a", newline="\n") as csv_thread:
            writer = csv.writer(csv_thread)
            for folder in os.listdir(main_folder_path):
                for file in os.listdir(main_folder_path + fr"\{folder}"):
                    writer.writerow((main_folder_path + fr"\{folder}\{file}", dict_wnid_int[folder[1:]]))
        print("Created csv file with images paths!")

    @staticmethod
    def read_wnid_int_name_file(wnid_int_name_path):
        with open(wnid_int_name_path, "r", newline="\n") as file_thread:
            return list(csv.reader(file_thread, delimiter=","))

    @staticmethod
    def read_images_paths(images_file_path):
        with open(images_file_path, "r", newline="\n") as file_thread:
            return list(csv.reader(file_thread, delimiter=""))

