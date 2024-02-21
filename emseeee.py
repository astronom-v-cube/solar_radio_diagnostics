import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

# Определение вашей функции
def my_function(x, y):
    return x**2 + y**2

# Определение логарифма апостериорного распределения
def log_likelihood(params):
    x, y = params
    return -0.5 * np.sum((my_function(x, y) - data) ** 2)

def log_prior(params):
    x, y = params
    # Пример равномерного prior в пределах [-10, 10] для обоих параметров
    if -15.0 < x < 15.0 and -15.0 < y < 15.0:
        return 0.0
    return -np.inf

def log_probability(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params)

# Генерация некоторых фиктивных данных
np.random.seed(42)
data = my_function(np.random.uniform(-15, 15, 1000), np.random.uniform(-15, 15, 1000)) + np.random.normal(0, 0.15, 1000)

# Настройка emcee
nwalkers = 10000
ndim = 2
nsteps = 500

# Инициализация позиций путем выбора случайных точек в пространстве параметров
initial = np.random.uniform(-15, 15, (nwalkers, ndim))

# Запуск MCMC
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, threads=12)
sampler.run_mcmc(initial, nsteps, progress=True)

# Получение цепей Маркова
samples = sampler.get_chain(discard=100, thin=50, flat=True)

print(samples)

# # Определение минимума
# min_params = samples[np.argmax(log_probability(samples))]

# Визуализация с помощью библиотеки corner
fig = corner.corner(samples, labels=['x', 'y'], truths=[0, 0], range=[(-15, 15), (-15, 15)], plot_contours=True, fill_contours=True, levels=[-0.2, -0.5, -0.8, -0.95], color='k', reverse=True)
plt.show()


