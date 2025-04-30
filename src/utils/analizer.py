import matplotlib.pyplot as plt
from IPython.display import clear_output

class Analizer:
    def __init__(self):
        self.accuracy = []
    
    def add_loss(self, value):
        self.accuracy.append(value)

    def get_accuracy(self):
        return self.accuracy
    
    def show_accuracy(self):
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(8, 5))  # Явно створюємо фігуру та осі
        ax.cla()  # Очищаємо лише осі, а не всю фігуру
        ax.plot(self.get_accuracy(), 'o-', label='Точність')
        ax.set_title('Точність у реальному часі в Colab')
        ax.set_xlabel('Пакет')
        ax.set_ylabel('Значення')
        ax.grid(True)
        ax.legend()
        fig.canvas.draw()  # Примусово малюємо полотно
        plt.show()