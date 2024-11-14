import os
import tarfile

# r allows to write \w... without special indicating
MAIN_FOLDER = r"F:\imageNetILSVRC2012\train_task_3"
foldersNum = len(os.listdir(MAIN_FOLDER))
for ind, file in enumerate(os.listdir(MAIN_FOLDER)):
    file_path = os.path.join(MAIN_FOLDER, file)
    try:
        if tarfile.is_tarfile(file_path):
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            with tarfile.TarFile(file_path, 'r') as tar_ref:
                tar_ref.extractall(MAIN_FOLDER + fr"\{file_name}")
            os.remove(file_path)
            print(f"{ind + 1}/{foldersNum} is dearchived")
    except PermissionError: pass
print("dearchivation was completed!")