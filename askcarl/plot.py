"""Visualisation of GMM as a corner plot."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def confidence_contours(pdf, levels=[0.393, 0.675, 0.864]):
    """
    Compute PDF thresholds for given enclosed probability levels.

    Parameters
    ----------
    pdf: array
        probability density function as an array.
    levels: list
        cumulative probability levels (like corner.py).

    Returns
    -------
    thresholds: array
        Threshold values.
    """
    flat = pdf.flatten()
    idx = np.argsort(flat)[::-1]
    # high -> low
    sorted_pdf = flat[idx]
    cumsum = np.cumsum(sorted_pdf)
    # normalize to 1
    cumsum /= cumsum[-1]

    thresholds = []
    for lev in levels:
        # Find minimum pdf value that encloses fraction=lev
        thresh = sorted_pdf[np.searchsorted(cumsum, lev)]
        thresholds.append(thresh)
    return thresholds


def plot_gmm_corner(
    gmm,
    limits=None,
    bins=100,
    levels=[0.393, 0.675, 0.864],
    fig=None,
    axes=None,
    color="k",
):
    """
    Analytic corner plot from a Gaussian Mixture model.

    Parameters
    ----------
    gmm: object
        scikit-learn GaussianMixture
    limits: None or list
        axes limits
    bins: int
        number of bins
    levels: list
        Contour levels enclose given confidence intervals (as in corner.py).
    fig: object
        if provided, plot into this matplotlib figure, otherwise a new one figure is created.
    axes: object
        if provided, plot into these matplotlib axes, otherwise a new one figure is created.
    color: str
        color.

    Returns
    -------
    fig: object
        first return value of matplotlib.subplots
    axes: object
        second return value of matplotlib.subplots
    """
    n_dim = gmm.means_.shape[1]
    if limits is None:
        # default limits: mean plus-minus 5 std for each dimension
        limits = []
        for d in range(n_dim):
            mean_d = np.average(gmm.means_[:, d], weights=gmm.weights_)
            var_d = np.average(
                [cov[d, d] for cov in gmm.covariances_], weights=gmm.weights_
            )
            std_d = np.sqrt(var_d)
            limits.append((mean_d - 5 * std_d, mean_d + 5 * std_d))

    if fig is None or axes is None:
        fig, axes = plt.subplots(
            n_dim,
            n_dim,
            figsize=(2.5 * n_dim, 2.5 * n_dim),
            gridspec_kw=dict(hspace=0, wspace=0),
        )

    for i in range(n_dim):
        for j in range(n_dim):
            ax = axes[i, j]

            if i == j:
                # 1D marginal
                x = np.linspace(*limits[i], bins)
                pdf = np.zeros_like(x)
                for w, mean, cov in zip(gmm.weights_, gmm.means_, gmm.covariances_):
                    pdf += w * multivariate_normal.pdf(x, mean=mean[i], cov=cov[i, i])
                ax.plot(x, pdf, "-", color=color)
                ax.set_ylim(0, None)
                ax.set_yticks([])
                ax.set_xlim(limits[i])

            elif j < i:
                # 2D marginal
                x = np.linspace(*limits[j], bins)
                y = np.linspace(*limits[i], bins)
                X, Y = np.meshgrid(x, y)
                pos = np.dstack((X, Y))
                pdf = np.zeros_like(X)

                for w, mean, cov in zip(gmm.weights_, gmm.means_, gmm.covariances_):
                    mean_2d = [mean[j], mean[i]]
                    cov_2d = cov[[j, i]][:, [j, i]]
                    pdf += w * multivariate_normal.pdf(pos, mean=mean_2d, cov=cov_2d)

                # Compute contour thresholds
                thresholds = confidence_contours(pdf, levels=levels)

                ax.contour(X, Y, pdf, levels=sorted(thresholds), colors=color)
                ax.set_xlim(limits[j])
                ax.set_ylim(limits[i])
            else:
                ax.axis("off")

            if i == n_dim - 1:
                ax.set_xlabel(f"x{j}")
            if j == 0 and i != 0:
                ax.set_ylabel(f"x{i}")

    return fig, axes
