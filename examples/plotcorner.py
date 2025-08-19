import corner
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

from askcarl.lightgmm import LightGMM

# Example GMM
X = np.vstack([
    np.random.multivariate_normal([0, 0], np.eye(2), 3000),
    np.random.multivariate_normal([3, 3], np.eye(2), 3000)
])

corner.corner(X)
plt.savefig('plotcorner1.pdf')
plt.close()

lgmm = LightGMM(n_components=2)
lgmm.fit(X)
gmm = lgmm.to_sklearn()
#gmm = GaussianMixture(n_components=10)
#gmm.fit(X)

# Corner-style plot with confidence contours
from askcarl.plot import plot_gmm_corner

fig, axes = plot_gmm_corner(gmm)
plt.savefig('plotcorner.pdf')
