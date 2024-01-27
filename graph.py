import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных из файла diagnostic_backup.npz
data = np.load('diagnostic_backup.npz')
params = data['params']
functional = data['functional']

# Ограничение количества отображаемых точек
num_points = 20  # Замените это на желаемое количество точек

params = params[:num_points, :]
functional = functional[:num_points]

# Получение количества подэлементов в каждом элементе params
num_subplots = len(params[0])

# Вычисление количества строк и столбцов в субплотах
num_cols = min(num_subplots, 4)  # Максимальное количество столбцов (в данном случае 3)
num_rows = -(-num_subplots // num_cols)  # Округление вверх для количества строк

# Создание субплотов в виде решетки
fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 3*num_rows))

# Построение графиков для каждого подэлемента в params
for i, ax in enumerate(axs.flat):
    if i < num_subplots:
        # График значения величины из params
        ax.plot(params[:, i], label=f'Param {i+1}')
        ax.set_ylabel(f'Param {i+1}')

        # Создание второй координатной сетки для functional
        ax2 = ax.twinx()

        # График из functional
        ax2.plot(functional, color='r', label='Functional')
        ax2.set_ylabel('Functional', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

# # Отображение легенды
# axs[0, 0].legend(loc='upper left')
# ax2.legend(loc='upper right')

# Отображение графиков
plt.tight_layout()
plt.show()
