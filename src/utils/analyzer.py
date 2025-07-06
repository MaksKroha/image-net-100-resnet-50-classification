import matplotlib.pyplot as plt
from IPython.display import clear_output
from src.utils.logger import exception_logger
import matplotlib.colors as mcolors


def is_valid_color(color_name):
    return color_name.lower() in mcolors.CSS4_COLORS


class Analyzer:
    def __init__(self, xlabel, ylabel, title, train_color, test_color):
        self.__train_accuracy = []
        self.__test_accuracy = []
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.train_color = train_color if is_valid_color(train_color) else "red"
        self.test_color = test_color if is_valid_color(test_color) else "blue"

    @exception_logger
    def add_train_val(self, value: float, epoch: int):
        if len(self.__train_accuracy) < epoch + 1:
            self.__train_accuracy.append([])
        else:
            if isinstance(value, float):
                self.__train_accuracy[epoch].append(value)
            else:
                raise TypeError('loss must be a float')

    @exception_logger
    def add_test_val(self, value: float, epoch: int):
        if len(self.__test_accuracy) < epoch + 1:
            self.__test_accuracy.append([])
        else:
            if isinstance(value, float):
                self.__test_accuracy[epoch].append(value)
            else:
                raise TypeError('loss must be a float')

    def get_train_accuracy(self):
        return self.__train_accuracy

    def get_test_accuracy(self):
        return self.__test_accuracy

    @exception_logger
    def show_accuracy(self):
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.cla()
        if self.__train_accuracy:
            lengths = [len(epoch) for epoch in self.__train_accuracy]
            accuracy = [sum(epoch) / length for epoch, length in zip(self.__train_accuracy, lengths)]
            ax.plot(accuracy, 'o-', color=self.train_color)

        if self.__test_accuracy:
            lengths = [len(epoch) for epoch in self.__test_accuracy]
            accuracy = [sum(epoch) / length for epoch, length in zip(self.__test_accuracy, lengths)]
            ax.plot(accuracy, 'o-', color=self.test_color)
        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.grid(True)
        ax.legend()
        fig.canvas.draw()
        plt.show()