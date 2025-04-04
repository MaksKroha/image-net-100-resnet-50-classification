import os
import csv
import json
from src.utils.logger import log_exception

# appending all jpg or jpeg files path's to csv file
# csv file format - <path, label>
def format_paths_into_csv_name_label(root_dir, output_csv, labels_json):
    # root_dir - folder which contains images that will be added to csv_paths
    # output_csv - "path/name.csv" csv file in which all images will be storing
    # labels_json - "path/name.json" json file with labels,
    #        json format - <"n012345": {"index": 0, "name": "class name"}>
    try :
        with open(labels_json, mode='r') as labels_json_file:
            json_data = json.load(labels_json_file)

        with open(output_csv, mode='a', newline='\n', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)

            # iterating through all dirs/files
            for file in os.listdir(root_dir):
                # checking if file have right extension
                if file.endswith(".jpg") or file.endswith(".jpeg"):
                    # checking if have enough length and if in json_data
                    if len(file) > 15 and file[:9] in json_data: # n02011460_0.jpg
                        writer.writerow([f"{root_dir}/{file}", json_data[file[:9]]["index"]])
    except Exception as e:
        log_exception(str(e))


