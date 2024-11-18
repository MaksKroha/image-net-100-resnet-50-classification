import csv
import numpy


class DataLoader:
    @staticmethod
    def shuffle_csv_file_by_rows(csv_file_path, elements_num, shuffle_iterations):
        data = numpy.array([])
        with open(csv_file_path, "r", newline="\n") as csv_thread:
            reader = csv.reader(csv_thread, delimiter=",")
            data = numpy.array([el for el in reader])
            for _ in range(shuffle_iterations):
                numpy.random.shuffle(data)
        with open(csv_file_path, "w", newline="\n") as csv_thread:
            writer = csv.writer(csv_thread, delimiter=",")
            writer.writerows(data)
