import corner
import matplotlib.pyplot as plt
import numpy as np
import time
from askcarl.lightgmm import LightBaggingGMM
from askcarl.plot import plot_gmm_corner

ndim = 10

# Example GMM
X = np.vstack([
    np.random.multivariate_normal([0] * ndim, np.eye(ndim), 3000 * ndim),
    np.random.multivariate_normal([3] * ndim, np.eye(ndim), 3000 * ndim)
])

# Corner-style plot with confidence contours

t0 = time.time()
corner.corner(X, plot_datapoints=False, plot_density=False, levels=[0.393, 0.675, 0.864])
print('corner:', time.time() - t0)
plt.savefig('plotcorner1.pdf')
plt.close()

t0 = time.time()
lgmm = LightBaggingGMM(n_gmms=20, n_components=10)
lgmm.fit(X)
gmm = lgmm.to_sklearn()

fig, axes = plot_gmm_corner(gmm, levels=[0.393, 0.675, 0.864])
print('LightBaggingGMM:', time.time() - t0)
plt.savefig('plotcorner.pdf')
