"""Visualisation of GMM as a corner plot."""

import matplotlib
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
    bins=40,
    levels=[0.393, 0.675, 0.864],
    fig=None,
    axes=None,
    color="k",
    labels=None,
    truths=None,
    truths_kw={"color": "tab:blue", "lw": 1},
    max_err_frac=0.00,
    max_err_frac_eps=20,
    overlap_prevention_factor=2.0,
    linewidth=1,
):
    """Make analytic corner plot from a Gaussian Mixture model.

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
    labels: list
        name for each parameter.
    truths: list
        list of true values for each parameter.
    truths_kw: dict
        arguments passed for styling lines of true parameters.
    max_err_frac: float
        Maximum ratio of ellipse misclassifion area to ellipse area,
        to use an ellipse instead of a contour.
    max_err_frac_eps: float
        number to add to the total when dividing the number
        of false positive and false negative classifications .
    overlap_prevention_factor: float
        Expansion factor for ellipses, when checking for overlapping ellipses.
        Should be >= 1.
    linewidth: float
        Width of the line.

    Returns
    -------
    fig: object
        first return value of matplotlib.subplots
    axes: object
        second return value of matplotlib.subplots
    """
    n_dim = gmm.means_.shape[1]
    tickformatter = matplotlib.ticker.NullFormatter()
    # min_f1 = 0.8  # performance threshold to accept ellipse approximation

    if labels is None:
        labels = ["x%d" % i for i in range(n_dim)]
    else:
        labels = list(labels)
        if not len(labels) == n_dim:
            raise ValueError(
                f"number of labels ({len(labels)}) should be the same as number of dimensions ({n_dim})"
            )
    if truths is None:
        truths = [np.nan for i in range(n_dim)]
    else:
        truths = list(truths)
        if not len(truths) == n_dim:
            raise ValueError(
                f"number of labels ({len(truths)}) should be the same as number of dimensions ({n_dim})"
            )

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

    grids = [np.linspace(lo, hi, bins) for lo, hi in limits]

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

            if j > i:
                ax.axis("off")
                continue
            if i == n_dim - 1:
                # bottom row
                ax.set_xlabel(labels[j])
            else:
                # ax.set_xticklabels([])
                # ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                ax.xaxis.set_major_formatter(tickformatter)
                pass
            if j == 0 and i != 0:
                # left column (except first row)
                ax.set_ylabel(labels[i])
                # ax.yaxis.set_major_locator(ticklocator)
            else:
                # ax.set_yticklabels([])
                ax.yaxis.set_major_formatter(tickformatter)
                pass
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))

            if i == j:
                # 1D marginal
                x = grids[i]
                pdf = np.zeros_like(x)
                for w, mean, cov in zip(gmm.weights_, gmm.means_, gmm.covariances_):
                    pdf += w * multivariate_normal.pdf(x, mean=mean[i], cov=cov[i, i])
                ax.plot(x, pdf, "-", color=color)
                ax.set_ylim(0, None)
                ax.set_yticks([])
                ax.set_xlim(limits[i])
                ylo, yhi = ax.get_ylim()
                ax.set_ylim(ylo, yhi)

                if np.isfinite(truths[i]):
                    ax.vlines(truths[i], ylo, yhi, **truths_kw)
                ax.set_title(labels[i])
            elif j < i:
                # 2D marginal
                x = grids[j]
                y = grids[i]
                X, Y = np.meshgrid(x, y)
                pos = np.dstack((X, Y))
                pdf = np.zeros_like(X)

                for w, mean, cov in zip(gmm.weights_, gmm.means_, gmm.covariances_):
                    mean_2d = [mean[j], mean[i]]
                    cov_2d = cov[[j, i]][:, [j, i]]
                    pdf += w * multivariate_normal.pdf(pos, mean=mean_2d, cov=cov_2d, allow_singular=True)

                # Compute contour thresholds
                thresholds = confidence_contours(pdf, levels=levels)
                ax.contour(
                    X,
                    Y,
                    pdf,
                    levels=sorted(thresholds),
                    linewidths=[linewidth] * len(thresholds),
                    colors=color,
                )

                ax.set_xlim(limits[j])
                ax.set_ylim(limits[i])
                if np.isfinite(truths[i]):
                    ax.vlines(truths[j], *limits[i], **truths_kw)
                if np.isfinite(truths[j]):
                    ax.hlines(truths[i], *limits[j], **truths_kw)

    return fig, axes
