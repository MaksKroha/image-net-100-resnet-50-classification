import os
import tarfile
from src.utils.logger import log_exception

# this function is dearchivating all "tar" archives in current folder
# into current folder with deleting old tar archives
def dearchive(dir):
    for folder in os.listdir(dir):
        folder_path = dir + fr"\{folder}"
        try:
            if tarfile.is_tarfile(folder_path):
                with tarfile.TarFile(folder_path, 'r') as tar_ref:
                    tar_ref.extractall(dir + fr"\{folder.split('.', 1)[0]}")
                os.remove(folder_path)
        except PermissionError as e:
            log_exception(e)
            pass
