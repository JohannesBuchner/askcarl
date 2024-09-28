import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.mixture import GaussianMixture
import askcarl

colors = ["navy", "turquoise", "darkorange"]

iris = datasets.load_iris()
X = iris.data[:,:3]
y = iris.target

n_classes = len(np.unique(y))

# Try GMMs using different types of covariances.
gmm = GaussianMixture(
    n_components=n_classes, random_state=123,
    means_init=np.array([X[y == i].mean(axis=0) for i in range(n_classes)]),
)

# Train the other parameters using the EM algorithm.
gmm.fit(X)

fig, axs = plt.subplots(
    2, 2, figsize=(8, 8), sharex=True, sharey=True,
    gridspec_kw=dict(hspace=0.02, wspace=0.02),
)
for i1, i2, ax in [(0, 1, axs[0,0]), (0, 2, axs[1,0]), (1, 2, axs[1,1])]:
    for n, color in enumerate(colors):
        covariances = [[gmm.covariances_[n][i,j] for i in (i1, i2)] for j in (i1, i2)]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(
            [gmm.means_[n,i1], gmm.means_[n,i2]], v[0], v[1], angle=180 + angle, color=color
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ell.set_linewidth(1)
        ax.add_artist(ell)
        #ax.set_aspect("equal", "datalim")

    for n, color in enumerate(colors):
        data = X[y == n]
        ax.scatter(
            data[:, i1], data[:, i2], s=0.8, color=color,
            label=iris.target_names[n] # if (i1,i2) == (0,1) else None,
        )
    if (i1, i2) != (1, 2):
        ax.set_ylabel(iris.feature_names[i2])
    ax.set_xlabel(iris.feature_names[i1])
    ax.legend()

#axs[0,0].set_xlim(3.5, 8)
axs[0,0].set_xlim(axs[0,0].get_xlim())
axs[1,0].set_ylim(axs[1,0].get_ylim())
axs[1,1].set_xlim(axs[1,1].get_xlim())

#plt.savefig('iris.png')

print(iris.feature_names)
mix = askcarl.GaussianMixture.from_sklearn(gmm)
x = np.array([
   [5, 3.5, 1.5],
   [6, 5.0, 5.0],
   [5.5, 3, np.inf],
])
mask = np.array([
   [True, True, True],
   [True, True, True],
   [False, True, False],
], dtype=bool)

resp = np.array([g.pdf(x, mask) for g in mix.components])
print(resp)
print(resp.argmax(axis=0))

p = mix.pdf(x, mask)
print(p)

for axmask, ax in [(np.array([0,1]), axs[0,0]), (np.array([0,2]), axs[1,0]), (np.array([1,2]), axs[1,1])]:
    i1, i2 = np.arange(3)[axmask]
    for i, (xi, maski, color) in enumerate(zip(x, mask, ['lime','k','r'])):
        if not maski[i1] and np.isfinite(xi[i1]):
            marker = '<-'
        elif not maski[i2] and np.isfinite(xi[i2]):
            marker = 'v-'
        #elif np.isfinite(xi[i1]) or np.isfinite(xi[i1]):
        #    marker = '-'
        else:
            marker = 'x-'
        if not np.isfinite(xi[i1]):
            ax.plot(ax.get_xlim(), [xi[i2], xi[i2]], marker, color=color, ms=10, mew=4)
        elif not np.isfinite(xi[i2]):
            ax.plot([xi[i1], xi[i1]], ax.get_ylim(), marker, color=color, ms=10, mew=4)
            #ax.vlines(xi[i1], *ax.get_ylim(), colors=[color])
        else:
            ax.plot(xi[i1], xi[i2], marker, color=color, ms=10, mew=4)
    
        if (i1, i2) == (0,1):
            axs[0,1].text(
                3, 5 - i,
                f'{resp[i,0]:.2f},{resp[i,1]:.2f},{resp[i,2]:.2f}$\\rightarrow$',
                color=color)
            axs[0,1].text(
                5.5, 5 - i, f"{iris.target_names[resp[i].argmax(axis=0)]}",
                color=colors[resp[i].argmax(axis=0)])

axs[0,1].text(3, 6, 'probabilities', size=16)

plt.savefig('iris.png')
plt.close()
