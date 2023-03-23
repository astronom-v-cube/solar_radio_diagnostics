import numpy as np
import corner
import matplotlib.pyplot as plt

data = np.random.normal(1, 3, size=(300000, 3))
corner.corner(data, range = [(1, 3), (1, 3), (1, 3)]) 

plt.show()