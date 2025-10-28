"""Visualisation of GMM as a corner plot."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from contourpy import contour_generator
from matplotlib.colors import to_rgb as _mpl_to_rgb
from reportlab.lib.colors import Color, black, navy, purple
from reportlab.pdfgen import canvas

_TWO_PI = 2.0 * np.pi


def _to_rl_color(color):
    """Convert to ReportLab color.

    Parameters
    ----------
    color: object
        ReportLab color or matplotlib-readable rgb tuple.

    Returns
    -------
    color: object
        ReportLab color
    """
    # Accept reportlab Color, matplotlib string/tuple
    if isinstance(color, Color):
        return color
    rgb = _mpl_to_rgb(color) if isinstance(color, (str, tuple, list)) else (0, 0, 0)
    return Color(*rgb)


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
                means_1d = gmm.means_[:, i]
                vars_1d = np.array([cov[i, i] for cov in gmm.covariances_], dtype=float)
                pdf = _mixture_pdf_1d(x, gmm.weights_, means_1d, vars_1d)
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
                means_2d = np.stack([gmm.means_[:, j], gmm.means_[:, i]], axis=1)
                covs_2d = np.array(
                    [cov[[j, i]][:, [j, i]] for cov in gmm.covariances_], dtype=float
                )
                pdf = _mixture_pdf_2d_grid(x, y, gmm.weights_, means_2d, covs_2d)

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


def _mixture_pdf_1d(x, weights, means, variances):
    """Get marginal posterior from 1d GMM.

    Parameters
    ----------
    x: array
        positions
    weights: array
        GMM weights
    means: array
        list of mean of each component
    variances: list
        variance of each component.

    Returns
    -------
    pdf: array
        probability
    """
    inv_var = 1.0 / variances
    norm = weights * np.sqrt(inv_var / _TWO_PI)
    dx = x[None, :] - means[:, None]
    return (norm[:, None] * np.exp(-0.5 * (dx * dx) * inv_var[:, None])).sum(axis=0)


def _mixture_pdf_2d_grid(x, y, weights, means_2d, covs_2d):
    """Get conditional posterior from 2d GMM.

    Parameters
    ----------
    x: array
        x positions
    y: array
        y positions
    weights: array
        GMM weights
    means_2d: array
        list of means of each component
    covs_2d: list
        covariance matrices of components.

    Returns
    -------
    pdf: array
        probability
    """
    Ny = y.shape[0]
    Nx = x.shape[0]
    pdf = np.zeros((Ny, Nx), dtype=float)
    for w, m, C in zip(weights, means_2d, covs_2d):
        mu_x, mu_y = m[0], m[1]
        c00, c01, c11 = C[0, 0], C[0, 1], C[1, 1]
        det = c00 * c11 - c01 * c01
        if det <= 0:
            continue
        inv00 = c11 / det
        inv01 = -c01 / det
        inv11 = c00 / det
        norm = w / (_TWO_PI * np.sqrt(det))
        dx = x[None, :] - mu_x  # (1,Nx)
        dy = y[:, None] - mu_y  # (Ny,1)
        expo = -0.5 * (inv00 * (dx * dx) + 2.0 * inv01 * (dx * dy) + inv11 * (dy * dy))
        pdf += norm * np.exp(expo)
    return pdf


def _map_xy(xx, yy, rect, xlim, ylim):
    """Transform linearly.

    Parameters
    ----------
    xx: float
        x value
    yy: float
        y value
    rect: tuple
        in pixel units, (x0, y0, w, h), with bottom-left origin (ReportLab)
    xlim: tuple
        x value range within rectangle
    ylim: tuple
        y value range within rectangle

    Returns
    -------
    px: float
        positions of xx in pixels
    py: float
        positions of yy in pixels
    """
    x0, y0, w, h = rect
    px = x0 + (xx - xlim[0]) * (w / (xlim[1] - xlim[0]))
    py = y0 + (yy - ylim[0]) * (h / (ylim[1] - ylim[0]))
    return px, py


def _draw_contours_pdf(
    canvas,
    rect,
    xlim,
    ylim,
    level_segs,
    color=black,
    lw=1.0,
    clip=True,
):
    """Draw contour lines.

    Parameters
    ----------
    canvas: object
        ReportLab canvas object
    rect: tuple
        panel rectangle
    xlim: tuple
        x axis range
    ylim: tuple
        y axis range
    level_segs: list
        plot segments of level
    color: str
        color.
    lw: float
        line width
    clip: bool
        whether to clip the contour at the border
    """
    x0, y0, w, h = rect
    canvas.saveState()
    if clip:
        pclip = canvas.beginPath()
        pclip.rect(x0, y0, w, h)
        canvas.clipPath(pclip, stroke=0, fill=0)
    canvas.setStrokeColor(color)
    canvas.setLineWidth(lw)
    for seg in level_segs:
        # Map to pixel coordinates first
        pts_px = np.empty_like(seg)
        for k in range(seg.shape[0]):
            pts_px[k, 0], pts_px[k, 1] = _map_xy(seg[k, 0], seg[k, 1], rect, xlim, ylim)
        # Draw path
        p = canvas.beginPath()
        p.moveTo(pts_px[0, 0], pts_px[0, 1])
        for k in range(1, pts_px.shape[0]):
            p.lineTo(pts_px[k, 0], pts_px[k, 1])
        canvas.drawPath(p, stroke=1, fill=0)
    canvas.restoreState()


def _draw_vline_pdf(canvas, rect, xlim, ylim, xval, color=black, lw=1.0):
    """Draw vertical line.

    Parameters
    ----------
    canvas: object
        ReportLab canvas object
    rect: tuple
        panel rectangle
    xlim: tuple
        x axis range
    ylim: tuple
        y axis range
    xval: float
        value where to draw x value.
    color: str
        color.
    lw: float
        line width
    """
    x0, y0, w, h = rect
    px = x0 + (xval - xlim[0]) * (w / (xlim[1] - xlim[0]))
    canvas.saveState()
    canvas.setStrokeColor(color)
    canvas.setLineWidth(lw)
    canvas.line(px, y0, px, y0 + h)
    canvas.restoreState()


def _draw_hline_pdf(canvas, rect, xlim, ylim, yval, color=black, lw=1.0):
    """Draw horizontal line.

    Parameters
    ----------
    canvas: object
        ReportLab canvas object
    rect: tuple
        panel rectangle
    xlim: tuple
        x axis range
    ylim: tuple
        y axis range
    yval: float
        value where to draw y line.
    color: str
        color.
    lw: float
        line width
    """
    x0, y0, w, h = rect
    py = y0 + (yval - ylim[0]) * (h / (ylim[1] - ylim[0]))
    canvas.saveState()
    canvas.setStrokeColor(color)
    canvas.setLineWidth(lw)
    canvas.line(x0, py, x0 + w, py)
    canvas.restoreState()


def _compute_rects(ndim, width, height, margin):
    """Compute rectangles for each panel.

    Parameters
    ----------
    ndim: int
        number of axes
    width: int
        width in pixels
    height: int
        height in pixels
    margin: int
        margin

    Returns
    -------
    rects: list
        list of lists, containing (x, y, width, height) of each panel.
    """
    plotW = width - 2 * margin
    plotH = height - 2 * margin
    cellW = plotW / ndim
    cellH = plotH / ndim
    rects = [[None] * ndim for _ in range(ndim)]
    for i in range(ndim):
        for j in range(ndim):
            if j > i:
                continue
            x0 = margin + j * cellW
            y0 = height - margin - (i + 1) * cellH  # bottom-left origin
            rects[i][j] = (x0, y0, cellW, cellH)
    return rects


def _build_axes_form(
    canvas, rects, labels, limits, tick_count, tickfontsize, labelfontsize, fontname="Helvetica"
):
    """Set up panels.

    Parameters
    ----------
    canvas: object
        ReportLab canvas object
    rects: list
        list of list giving position of each panel
    labels: list
        name for each parameter.
    limits: None or list
        axes limits
    tick_count: int
        maximum number of ticks to draw
    tickfontsize: float
        font size for tick labels
    labelfontsize: float
        font size for axis labels.
    fontname: str
        font name
    """
    ndim = len(rects)
    c = canvas
    c.beginForm("axes")
    c.setStrokeColor(black)
    c.setLineWidth(0.6)
    ticks = []
    ticklabels = []
    tickfmts = []
    for i in range(ndim):
        hi = limits[i][1]
        lo = limits[i][0]
        sigfig = int(np.ceil(-np.log10(hi - lo) + 1))
        if sigfig > 3:
            fmt = "%.1e"
        elif sigfig >= 1:
            fmt = f"%.{sigfig}f"
        elif sigfig > -3:
            fmt = "%d"
        else:
            fmt = "%.1e"

        tsm1 = np.arange(np.round(lo / 2, sigfig) * 2, hi, 10 ** (-sigfig))
        ts0 = np.arange(np.round(lo / 2, sigfig - 1) * 2, hi, 10 ** (-sigfig + 1))
        tsp1 = np.arange(np.round(lo / 2, sigfig - 2) * 2, hi, 10 ** (-sigfig + 2))
        if np.logical_and(tsp1 > lo, tsp1 < hi).sum() >= 3:
            ts = tsp1
        elif np.logical_and(ts0 > lo, ts0 < hi).sum() >= 3:
            ts = ts0
        else:
            ts = tsm1
        ticklabels.append([fmt % t for t in ts])
        strlen = max((len(tl) for tl in ticklabels))
        if np.logical_and(ts > lo, ts < hi).sum() > tick_count:
            ts = ts[::2]
        if strlen > 3 and np.logical_and(ts > lo, ts < hi).sum() > tick_count - 1:
            ts = ts[::2]
        ts = ts[np.logical_and(ts >= lo, ts <= hi)]
        assert len(ts) > 0, ts
        if (ts[-1] - lo) / (hi - lo) > 0.9:
            ts = ts[:-1]
        tickfmts.append(fmt)
        ticks.append(ts)

    for i in range(ndim):
        for j in range(ndim):
            rect = rects[i][j]
            if rect is None:
                continue
            x0, y0, w, h = rect
            if j == i:
                c.setFont(fontname, labelfontsize)
                c.drawCentredString(x0 + w / 2, y0 + h, labels[j])
                c.line(x0, y0, x0 + w, y0)
            else:
                c.rect(x0, y0, w, h, stroke=1, fill=0)
                ts = ticks[i]
                lo, hi = limits[i]
                for t, tl in zip(ts, ticklabels[i]):
                    ty = y0 + (t - lo) * (h / (hi - lo))
                    c.line(x0 - 2, ty, x0 + 2, ty)
                    if j == 0:
                        c.setFont(fontname, tickfontsize)
                        c.drawRightString(x0 - 4, ty - 2, tl)
                if j == 0:
                    c.setFont(fontname, labelfontsize)
                    c.drawRightString(x0 - 10 - labelfontsize, y0 + h / 2, labels[i])
            # ticks bottom
            lo, hi = limits[j]
            ts = ticks[j]
            for t, tl in zip(ts, ticklabels[j]):
                tx = x0 + (t - lo) * (w / (hi - lo))
                c.line(tx, y0, tx, y0 + 2)
                if i == ndim - 1:
                    c.setFont(fontname, tickfontsize)
                    c.drawCentredString(tx, y0 - 1 - tickfontsize, tl)
            if i == ndim - 1:
                c.setFont(fontname, labelfontsize)
                c.drawCentredString(x0 + w / 2, y0 - 10 - labelfontsize, labels[j])
            # ticks left
    c.endForm()


def plot_gmm_corner_pdf(
    outputfile,
    gmm,
    bins=40,
    limits=None,
    levels=[0.393, 0.675, 0.864],
    color=navy,
    scale=1.0,
    width=600,
    height=600,
    margin=40,
    linewidth=0.5,
    labels=None,
    truths=None,
    truthcolor=purple,
    truthlinewidth=1.0,
    tick_count=4,
    fontname="Helvetica"
):
    """Make analytic corner plot from a Gaussian Mixture model.

    Parameters
    ----------
    outputfile: str
        output file name
    gmm: object
        scikit-learn GaussianMixture
    bins: int
        number of bins
    limits: None or list
        axes limits
    levels: list
        Contour levels enclose given confidence intervals (as in corner.py).
    color: reportlab.lib.colors.Color or (r,g,b)
        Stroke color for lines/contours.
    scale: float
        scale factor for labels
    width: int
        plot width
    height: int
        plot height
    margin: int
        margin around plot
    linewidth: float
        Line width in PDF units.
    labels: list
        name for each parameter.
    truths: list
        list of true values for each parameter.
    truthcolor: reportlab.lib.colors.Color or (r,g,b)
        Stroke color for lines indicating true value.
    truthlinewidth: float
        Line width in PDF units.
    tick_count: int
        maximum number of ticks.
    fontname: str
        font name
    """
    n_dim = gmm.means_.shape[1]

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

    weights = gmm.weights_
    means = gmm.means_
    covs = gmm.covariances_

    if limits is None:
        # default limits: mean plus-minus 5 std for each dimension
        limits = []
        for d in range(n_dim):
            lo = min(mean[d] - 3 * cov[d, d] ** 0.5 for mean, cov in zip(means, covs))
            hi = max(mean[d] + 3 * cov[d, d] ** 0.5 for mean, cov in zip(means, covs))
            limits.append((lo, hi))

    grids = [np.linspace(lo, hi, bins) for lo, hi in limits]

    # pdf = CornerPDFTemplate(n_dim, limits, labels, outpath=outputfile, scale=scale, width=width, height=height)
    pdf_c = canvas.Canvas(outputfile, pagesize=(width, height))
    tickfontsize = 6 * scale
    labelfontsize = 8 * scale
    rects = _compute_rects(n_dim, width, height, margin)
    _build_axes_form(
        pdf_c, rects, labels, limits, tick_count, tickfontsize, labelfontsize, fontname=fontname
    )

    color = _to_rl_color(color)
    truthcolor = _to_rl_color(truthcolor)

    # Start a new page and stamp axes
    pdf_c.doForm("axes")

    # Diagonal 1D panels
    for i in range(n_dim):
        rect = rects[i][i]
        x = grids[i]
        means_1d = means[:, i]
        vars_1d = np.array([cov[i, i] for cov in covs], dtype=float)
        pdf1d = _mixture_pdf_1d(x, weights, means_1d, vars_1d)

        # Draw polyline for 1D pdf
        # Map points into panel rect with y scaled to [0, 1.05*max(pdf)]
        x0, y0, w, h = rect
        ymax = float(max(pdf1d.max(), 1e-12))
        # Build and draw path
        p = pdf_c.beginPath()
        px, py = _map_xy(x[0], pdf1d[0], rect, limits[i], (0.0, 1.05 * ymax))
        p.moveTo(px, py)
        for k in range(1, x.size):
            px, py = _map_xy(x[k], pdf1d[k], rect, limits[i], (0.0, 1.05 * ymax))
            p.lineTo(px, py)
        pdf_c.setStrokeColor(color)
        pdf_c.setLineWidth(linewidth)
        pdf_c.drawPath(p, stroke=1, fill=0)

        # Truth vertical line on diagonal
        if np.isfinite(truths[i]):
            _draw_vline_pdf(
                pdf_c, rect, limits[i], (0.0, 1.0), truths[i], color=truthcolor, lw=truthlinewidth
            )

    # Lower-triangular 2D panels
    for i in range(n_dim):
        for j in range(i):
            rect = rects[i][j]
            x = grids[j]
            y = grids[i]
            means_2d = np.stack([means[:, j], means[:, i]], axis=1)
            covs_2d = np.array([cov[[j, i]][:, [j, i]] for cov in covs], dtype=float)
            pdf2d = _mixture_pdf_2d_grid(x, y, weights, means_2d, covs_2d)

            # Compute thresholds for enclosed probability
            thresholds = confidence_contours(pdf2d, levels=levels)

            # Generate contour segments directly (no Matplotlib artists)
            qcg = contour_generator(x=x, y=y, z=pdf2d, name="serial")
            for t in sorted(thresholds):
                level_segs = qcg.lines(t)  # list of (N,2) arrays
                # Draw into panel rect, optionally simplifying
                _draw_contours_pdf(
                    pdf_c,
                    rect,
                    xlim=limits[j],
                    ylim=limits[i],
                    level_segs=level_segs,
                    color=color,
                    lw=linewidth,
                    clip=True,
                )

            # Truth lines on 2D panels
            if np.isfinite(truths[j]):
                _draw_vline_pdf(
                    pdf_c,
                    rect,
                    limits[j],
                    limits[i],
                    truths[j],
                    color=truthcolor,
                    lw=truthlinewidth,
                )
            if np.isfinite(truths[i]):
                _draw_hline_pdf(
                    pdf_c,
                    rect,
                    limits[j],
                    limits[i],
                    truths[i],
                    color=truthcolor,
                    lw=truthlinewidth,
                )

    # Finish the page
    pdf_c.showPage()
    pdf_c.save()
