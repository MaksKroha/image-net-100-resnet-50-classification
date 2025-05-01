import os
import tarfile
from src.utils.logger import exception_logger

# this function is dearchivating all "tar" archives in current folder
# into current folder with deleting old tar archives
@exception_logger
def dearchive(dir):
    for folder in os.listdir(dir):
        folder_path = dir + fr"\{folder}"
        try:
            if tarfile.is_tarfile(folder_path):
                with tarfile.TarFile(folder_path, 'r') as tar_ref:
                    tar_ref.extractall(dir + fr"\{folder.split('.', 1)[0]}")
                os.remove(folder_path)
        except PermissionError:
            pass
