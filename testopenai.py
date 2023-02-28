import numpy as np
import matplotlib.pyplot as plt

# Задаем два массива данных
x = np.arange(0, 10, 0.1)
y1 = np.sin(x)
y2 = np.sin(x) + np.random.normal(0, 0.1, len(x))  # добавляем шум

# Очищаем данные от шума с помощью фильтра скользящего среднего
window_size = 11
y2_filtered = np.convolve(y2, np.ones(window_size) / window_size, mode='same')

# Строим графики
plt.figure()
plt.plot(x, y1, label='y1')
plt.plot(x, y2, label='y2 with noise')
plt.plot(x, y2_filtered, label='y2 filtered')
plt.legend()
plt.show()