import numpy as np
import corner
import matplotlib.pyplot as plt

data = np.random.normal(1, 300, size=(3000000, 4))
ranges = [(0, 800), (0, 900), (0, 750), (0, 350)]

fig_no_bug = corner.corner(data, range = ranges) 
plt.show()

fig_bug = plt.figure()
corner.corner(data, range = ranges, fig = fig_bug) 
plt.show()