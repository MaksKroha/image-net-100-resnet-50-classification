import os
import csv

def format_paths_into_csv_name_label(root_dir, output_csv, images_num):
    with open(output_csv, mode='a', newline='\n', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)

        for root, _, files in os.walk(root_dir):
            for i, file in enumerate(files):
                if i == images_num:
                    break
                if file.lower().endswith(".jpg"):
                    if file.lower().startswith("cat"):
                        label = 0
                    else:
                        label = 1
                    image_path = os.path.join(root, file)
                    writer.writerow([image_path, label])
            break

