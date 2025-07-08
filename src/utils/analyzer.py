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
    def add_train_val(self, value: float):
        if isinstance(value, float):
            self.__train_accuracy.append(value)
        else:
            raise TypeError('loss must be a float')

    @exception_logger
    def add_test_val(self, value: float):
        if isinstance(value, float):
            self.__test_accuracy.append(value)
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
            ax.plot(self.__train_accuracy, 'o-', color=self.train_color)

        if self.__test_accuracy:
            ax.plot(self.__test_accuracy, 'o-', color=self.test_color)
        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.grid(True)
        ax.legend()
        fig.canvas.draw()
        plt.show()