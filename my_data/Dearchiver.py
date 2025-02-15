import os
import tarfile


class Dearchiver:
    @staticmethod
    def dearchive(main_folder_path):
        # this function is dearchivating all "tar" archives in current folder
        # into current folder with deleting old tar archives
        for folder in os.listdir(main_folder_path):
            # TODO: check what try-catch do!
            folder_path = main_folder_path + fr"\{folder}"
            try:
                if tarfile.is_tarfile(folder_path):
                    with tarfile.TarFile(folder_path, 'r') as tar_ref:
                        tar_ref.extractall(main_folder_path + fr"\{folder.split('.', 1)[0]}")
                    os.remove(folder_path)
            except PermissionError: pass
        print("Dearchivation completed!")