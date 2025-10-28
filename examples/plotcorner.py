import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import corner
import numpy as np
import time
from askcarl.lightgmm import LightBaggingGMM
from askcarl.plot import plot_gmm_corner, plot_gmm_corner_pdf

ndim = 10
print(f'ndim={ndim}')

# Example GMM
X = np.vstack([
    np.random.multivariate_normal(0 + np.arange(ndim), np.eye(ndim), 3000 * ndim),
    np.random.multivariate_normal(3 + np.arange(ndim), np.eye(ndim), 3000 * ndim)
])

t0 = time.time()
lgmm = LightBaggingGMM(n_gmms=20, n_components=10)
lgmm.fit(X)
print(f'LightBaggingGMM fit of {len(X)} data points: {time.time() - t0:.2f}s')

gmm = lgmm.to_sklearn()
t0 = time.time()
plot_gmm_corner_pdf('plotcorner_custom.pdf', gmm, levels=[0.393, 0.675, 0.864], scale=1.0)
print(f'custom corner pdf: {time.time() - t0:.2f}s')

# Corner-style plot with confidence contours

t0 = time.time()
Y, _ = lgmm.sample(100000)
fig = corner.corner(Y, weights=np.ones(len(Y))/len(Y), plot_datapoints=False, plot_density=False, levels=[0.393, 0.675, 0.864])
corner.corner(X, weights=np.ones(len(X))/len(X), plot_datapoints=False, plot_density=False, levels=[0.393, 0.675, 0.864], fig=fig, color='red')
plt.savefig('plotcorner1.pdf')
plt.close()
print(f'sample + corner: {time.time() - t0:.2f}s')

#mplstyle.use('fast')
#mpl.use("module://mplcairo.base")
#mpl.use('Agg')
#mpl.rcParams['path.simplify'] = True
#mpl.rcParams['path.simplify_threshold'] = 1.0

for i in range(100):
    t0 = time.time()
    fig, axes = plot_gmm_corner(gmm, levels=[0.393, 0.675, 0.864])
    plt.savefig('plotcorner.pdf')
    plt.savefig('plotcorner.pdf')
    plt.close()
    print(f'GMM plot: {time.time() - t0:.2f}s')
    break
