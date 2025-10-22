"""Visualisation of GMM as a corner plot."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse
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


def _nearest_index(val, grid):
    """Find nearest array index.

    Parameters
    ----------
    val: float
        value to find
    grid: array
        array of values

    Returns
    -------
    int
        position in grid.
    """
    idx = np.searchsorted(grid, val)
    if idx <= 0:
        return 0
    if idx >= len(grid):
        return len(grid) - 1
    # choose nearest of idx-1 and idx
    return idx if abs(grid[idx] - val) < abs(grid[idx - 1] - val) else idx - 1


def _ellipse_params_from_cov(cov2d, r):
    """Get Ellipse parameters from a covariance.

    Parameters
    ----------
    cov2d: array
        Covariance matrix
    r: float
        scale radius

    Returns
    -------
    width: float
        Ellipse width
    height: float
        Ellipse height
    angle: float
        angle of ellipse
    """
    # r is Mahalanobis radius (sqrt of D^2 threshold)
    evals, evecs = np.linalg.eigh(cov2d)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    width = 2.0 * r * np.sqrt(evals[0])
    height = 2.0 * r * np.sqrt(evals[1])
    angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
    return width, height, angle

    # Helper to build an ellipse mask on the current grid


def _ellipse_mask(mean_2d, cov_2d, scale, x, y):
    """Mask points on a grid using an ellipse.

    Parameters
    ----------
    mean_2d: arrary
        Center
    cov_2d: array
        Covariance
    scale: float
        sigma level
    x: array
        x positions
    y: array
        y positions

    Returns
    -------
    array
        boolean mask of points which are inside ellipse.
    """
    c00, c01, c11 = cov_2d[0, 0], cov_2d[0, 1], cov_2d[1, 1]
    det = c00 * c11 - c01 * c01
    if det <= 0 or not np.isfinite(det):
        return None
    inv00 = c11 / det
    inv01 = -c01 / det
    inv11 = c00 / det
    dx = x[None, :] - mean_2d[0]
    dy = y[:, None] - mean_2d[1]
    D2 = inv00 * (dx * dx) + 2.0 * inv01 * (dy * dx) + inv11 * (dy * dy)
    return D2 <= scale


def _eigminmax_sym2(S):
    """Compute eigenvalue range.

    Parameters
    ----------
    S: array
        symmetric matrix

    Returns
    -------
    lam_min: float
        smallest eigenvalue
    lam_max: float
        largest eigenvalue
    """
    # Fast and stable for 2x2 symmetric
    a, b = S[0, 0], S[0, 1]
    c = S[1, 1]
    tr = a + c
    det = a * c - b * b
    disc = max(tr * tr - 4.0 * det, 0.0)
    rdisc = np.sqrt(disc)
    lam_max = 0.5 * (tr + rdisc)
    lam_min = 0.5 * (tr - rdisc)
    return lam_min, lam_max


def ellipses_overlap_analytic(mu1, Sigma1, s1, mu2, Sigma2, s2, jitter=1e-12):
    """Check if Ellipses overlap.

    Return True if ellipses (x-mu1)^T Sigma1^{-1} (x-mu1) <= s1 and
    (x-mu2)^T Sigma2^{-1} (x-mu2) <= s2 overlap.

    Parameters
    ----------
    mu1: array
        Mean of Gaussian 1
    Sigma1: array
        Covariance of Gaussian 1
    s1: float
        positive scalar for Gaussian 1 (D^2 thresholds)
    mu2: array
        Mean of Gaussian 1
    Sigma2: array
        Covariance of Gaussian 1
    s2: float
        positive scalar for Gaussian 1 (D^2 thresholds)
    jitter: float
        small number for numerical stability

    Returns
    -------
    ellipses_overlap_analytic: bool
        Whether the ellipses are overlapping.
    """
    mu1 = np.asarray(mu1, float)
    mu2 = np.asarray(mu2, float)
    S1 = np.asarray(Sigma1, float).copy()
    S2 = np.asarray(Sigma2, float).copy()
    s1 = float(s1)
    s2 = float(s2)

    # Early-outs with bounding circles
    r1 = np.sqrt(s1)
    r2 = np.sqrt(s2)
    lam1_min, lam1_max = _eigminmax_sym2(S1)
    lam2_min, lam2_max = _eigminmax_sym2(S2)
    if lam1_min <= 0 or lam2_min <= 0:
        return False  # degenerate; treat as non-overlap

    R1_out = r1 * np.sqrt(lam1_max)
    R1_in = r1 * np.sqrt(lam1_min)
    R2_out = r2 * np.sqrt(lam2_max)
    R2_in = r2 * np.sqrt(lam2_min)

    d = np.linalg.norm(mu1 - mu2)
    if d > R1_out + R2_out:
        return False  # definitely disjoint
    if d <= R1_in + R2_in:
        return True  # definitely overlap

    # Exact check via trust-region reduction
    # Ellipse form: (x-mu)^T A (x-mu) <= 1 with A = Sigma^{-1} / s
    # Cholesky of A1 to map E1 -> unit disk
    # Use solves to avoid explicit inverses; 2x2 is cheap anyway.
    try:
        A1 = np.linalg.inv(S1) / s1
        A2 = np.linalg.inv(S2) / s2
        # Ensure SPD (jitter if needed)
        # Small jitter to handle borderline numerics
        A1 = 0.5 * (A1 + A1.T) + jitter * np.eye(2)
        A2 = 0.5 * (A2 + A2.T) + jitter * np.eye(2)
        W = np.linalg.cholesky(A1)  # W^T W = A1
    except np.linalg.LinAlgError:
        # Add jitter and retry once
        try:
            W = np.linalg.cholesky(A1 + 10 * jitter * np.eye(2))
        except Exception:
            return False

    # Transform E2 into y-space where E1 is ||y|| <= 1
    # x = W^{-1} y + mu1
    dmu = mu1 - mu2
    Winv = np.linalg.inv(W)
    B = Winv.T @ A2 @ Winv
    # SPD
    g = Winv.T @ (A2 @ dmu)
    c_const = float(dmu.T @ A2 @ dmu)
    # We need min_{||y||<=1} q(y) where q(y) = y^T B y + 2 g^T y + c_const - 1
    # If min q <= 0, overlap exists.

    # Unconstrained minimizer
    try:
        y0 = -np.linalg.solve(B, g)
    except np.linalg.LinAlgError:
        # Add tiny jitter to B
        y0 = -np.linalg.solve(B + 10 * jitter * np.eye(2), g)

    if (y0 @ y0) <= 1.0 + 1e-14:
        qmin = float(y0.T @ (B @ y0) + 2.0 * g.T @ y0 + (c_const - 1.0))
        return qmin <= 0.0

    # Boundary case: minimize on ||y||=1 via scalar root-find
    def y_norm_sq(lam):
        """Compute distance.

        Parameters
        ----------
        lam: float
            scale

        Returns
        -------
        norm: float
            distance
        """
        M = B + lam * np.eye(2)
        y = -np.linalg.solve(M, g)
        return float(y @ y)

        # Find lam_hi so that ||y(lam_hi)|| <= 1

    lam_lo, lam_hi = 0.0, 1.0
    n_hi = y_norm_sq(lam_hi)
    it_guard = 0
    while n_hi > 1.0 and lam_hi < 1e12 and it_guard < 60:
        lam_hi *= 2.0
        n_hi = y_norm_sq(lam_hi)
        it_guard += 1
    if lam_hi >= 1e12:
        # Very ill-conditioned; fall back to conservative answer (assume overlap)
        return True

    # Bisection to solve ||y(λ)|| = 1
    for _ in range(40):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        if y_norm_sq(lam_mid) > 1.0:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid
    lam = lam_hi
    y = -np.linalg.solve(B + lam * np.eye(2), g)
    qmin = float(y.T @ (B @ y) + 2.0 * g.T @ y + (c_const - 1.0))
    return qmin <= 0.0


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
    max_err_frac=0.05,
    overlap_prevention_factor=1.5,
    linewidth=1,
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
    labels: list
        name for each parameter.
    truths: list
        list of true values for each parameter.
    truths_kw: dict
        arguments passed for styling lines of true parameters.
    max_err_frac: float
        Maximum ratio of ellipse misclassifion area to ellipse area,
        to use an ellipse instead of a contour.
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
    eps = 1e-12
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
                    pdf += w * multivariate_normal.pdf(pos, mean=mean_2d, cov=cov_2d)

                # Compute contour thresholds
                thresholds = confidence_contours(pdf, levels=levels)
                accepted_thresholds = set()

                for t in thresholds:
                    mask = pdf > t
                    P = int(mask.sum())
                    if P == 0:
                        continue
                    candidate_ellipsoids = []

                    for mean, cov in zip(gmm.means_, gmm.covariances_):
                        mean_2d = [mean[j], mean[i]]
                        cov_2d = cov[[j, i]][:, [j, i]]
                        # Skip if the component mean is not in this region or already covered
                        ix = _nearest_index(mean_2d[0], x)
                        iy = _nearest_index(mean_2d[1], y)
                        if not mask[iy, ix]:
                            continue
                        # scale ellipsoid so that it contains members
                        # Precompute inverse covariance elements and D^2 grid
                        c00, c01, c11 = cov_2d[0, 0], cov_2d[0, 1], cov_2d[1, 1]
                        det = c00 * c11 - c01 * c01
                        if not np.isfinite(det) or det <= 0:
                            continue
                        inv00 = c11 / det
                        inv01 = -c01 / det
                        inv11 = c00 / det

                        dx = x[None, :] - mean_2d[0]  # (1, Nx)
                        dy = y[:, None] - mean_2d[1]  # (Ny, 1)
                        # Mahalanobis D^2 over the grid
                        D2 = inv00 * (dx * dx) + 2.0 * inv01 * (dy * dx) + inv11 * (dy * dy)
                        # Distances of member pixels
                        D2_members = D2[mask]
                        if D2_members.size == 0:
                            continue

                        last_performance = 1.0
                        last_scale = 0.0

                        max_scale = float(D2_members.max())
                        # Scan increasing D^2 threshold; stop when F1 starts decreasing
                        for scale in np.arange(0.25, max_scale + 0.25, 0.25):
                            ell_mask = D2 <= scale
                            tp = np.count_nonzero(ell_mask & mask)
                            fp = np.count_nonzero(ell_mask & (~mask))
                            fn = P - tp
                            errfrac_new = (fp + fn) / (P + eps)

                            if errfrac_new <= last_performance:
                                last_performance = errfrac_new
                                last_scale = scale
                            else:
                                break

                        if last_scale > 0.0 and last_performance > 0.0:
                            candidate_ellipsoids.append(
                                (mean_2d, cov_2d, last_scale, last_performance)
                            )

                    occupied = np.zeros_like(mask, dtype=bool)
                    # Prune overlapping candidates greedily by F1
                    # Sort by increasing error (smaller is better)
                    candidate_ellipsoids.sort(key=lambda c: c[3])
                    selected = []
                    errfrac_curr = 1.0  # with no ellipses: FP=0, FN=P => (FP+FN)/P = 1
                    for mean_2d_new, cov_2d_new, scale_new, _ in candidate_ellipsoids:
                        # Strict analytic overlap rejection
                        overlaps = False
                        for m_sel, C_sel, s_sel in selected:
                            if ellipses_overlap_analytic(
                                mean_2d_new,
                                cov_2d_new,
                                scale_new * overlap_prevention_factor,
                                m_sel,
                                C_sel,
                                s_sel,
                            ):
                                overlaps = True
                                break
                        if overlaps:
                            continue

                        ell_mask = _ellipse_mask(
                            mean_2d_new, cov_2d_new, scale_new, x, y
                        )
                        if ell_mask is None:
                            continue

                        # Reject any overlap with already selected ellipses
                        if np.any(ell_mask & occupied):
                            continue
                        occupied |= ell_mask

                        # Prospective union and its misclassification error
                        union_new = occupied | ell_mask
                        tp = np.count_nonzero(union_new & mask)
                        fp = np.count_nonzero(union_new & (~mask))
                        fn = (
                            P - tp
                        )  # equivalent to np.count_nonzero((~union_new) & mask)
                        errfrac_new = (fp + fn) / (P + eps)

                        # Accept only if it improves the union error
                        if errfrac_new < errfrac_curr - 1e-12:
                            selected.append((mean_2d_new, cov_2d_new, scale_new))
                            occupied = union_new
                            errfrac_curr = errfrac_new

                    if not selected:
                        continue

                    # Final acceptance based on union misclassification fraction
                    print(t, "error fraction:", errfrac_curr)
                    if errfrac_curr <= max_err_frac:
                        patches = []
                        for mean_2d, cov_2d, scale in selected:
                            r = np.sqrt(scale)
                            w, h, ang = _ellipse_params_from_cov(cov_2d, r)
                            patches.append(
                                Ellipse(
                                    xy=mean_2d, width=w, height=h, angle=ang, fill=False
                                )
                            )
                        pc = PatchCollection(
                            patches,
                            edgecolor=color,
                            facecolor="none",
                            linewidth=linewidth,
                        )
                        ax.add_collection(pc)
                        accepted_thresholds.add(t)

                # Draw remaining levels (those not approximated by ellipses)
                remaining_thresholds = [
                    tt for tt in thresholds if tt not in accepted_thresholds
                ]
                if remaining_thresholds:
                    ax.contour(
                        X,
                        Y,
                        pdf,
                        levels=sorted(remaining_thresholds),
                        linewidths=[linewidth * 2] * len(remaining_thresholds),
                        colors=color,
                    )

                ax.set_xlim(limits[j])
                ax.set_ylim(limits[i])
                if np.isfinite(truths[i]):
                    ax.vlines(truths[j], *limits[i], **truths_kw)
                if np.isfinite(truths[j]):
                    ax.hlines(truths[i], *limits[j], **truths_kw)

    return fig, axes
