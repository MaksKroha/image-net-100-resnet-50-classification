import matplotlib.pyplot as plt
from IPython.display import clear_output
from src.utils.logger import exception_logger

class Analyzer:
    def __init__(self):
        self.__accuracy = []

    @exception_logger
    def add_loss(self, value: float):
        if isinstance(value, float):
            self.__accuracy.append(value)
        else:
            raise TypeError('loss must be a float')

    def get_accuracy(self):
        return self.__accuracy

    @exception_logger
    def show_accuracy(self):
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.cla()
        ax.plot(self.get_accuracy(), 'o-', label='Accuracy')
        ax.set_title('mean log loss in real time')
        ax.set_xlabel('Batch number')
        ax.set_ylabel('Value')
        ax.grid(True)
        ax.legend()
        fig.canvas.draw()
        plt.show()