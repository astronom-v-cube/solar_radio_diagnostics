import corner
import numpy as np
import matplotlib.pyplot as plt

# Generate some data
ndim, nsamples = 5, 10000
np.random.seed(0)
data = np.random.randn(ndim * nsamples).reshape([nsamples, ndim])

# Set the ranges for each parameter
ranges = [(-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2)]

# Set the parameter truths
truths = [True, False, False, True, False]

# Create the corner plot
figure = corner.corner(data, range=ranges, truths=truths)

# Save the figure to a file
figure.savefig('corner_plot.png')

# Show the saved image
plt.imshow(plt.imread('corner_plot.png'))
plt.axis('off')
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# ndim, nsamples = 3, 10000
# np.random.seed(0)
# data = np.random.randn(ndim * nsamples).reshape([nsamples, ndim])

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=1)

# plt.show()
