"""Mixture of Gaussians."""
import numpy as np
from scipy.special import logsumexp

from .gaussian import Gaussian

const2pi = np.log(2.0 * np.pi)


class GaussianMixture:
    """Mixture of Gaussians.

    Parameters
    -----------
    weights: list
        weight for each Gaussian component
    means: list
        mean vector for each Gaussian component.
    covs: list
        covariance matrix for each Gaussian component.
    precisions_cholesky: list
        Cholesky factors of each precision matrix, each computed with:
        `solve_triangular(cholesky(cov, lower=True), eyes[D], lower=True)`

    Attributes
    -----------
    weights: list
        weight for each Gaussian component
    components: list
        list of Gaussian components.
    """

    def __init__(self, weights, means, covs, precisions_cholesky=None):
        assert np.isfinite(weights).all()
        assert len(weights) == len(covs)
        weights = np.asarray(weights)
        assert np.shape(weights) == (len(means),)
        if precisions_cholesky is None:
            precisions_cholesky_maybe = [None] * len(covs)
        else:
            precisions_cholesky_maybe = (precision_cholesky.T for precision_cholesky in precisions_cholesky)
        self.components = [
            Gaussian(mean, cov, precision_cholesky)
            for mean, cov, w, precision_cholesky in zip(means, covs, weights, precisions_cholesky_maybe) if w > 0]
        self.powers = self.components[0].powers
        self.allpowers = self.components[0].allpowers
        self.ndim = self.components[0].ndim
        self.ncomponents = len(self.components)
        self.weights = weights[weights > 0]
        assert len(self.weights) == len(self.components)
        self.log_weights = np.log(self.weights)
        # Precompute spectral bounds once per component
        self.lam_min = []
        self.lam_max = []
        self.log_diag = []
        for cov in covs:
            eigval = np.linalg.eigvalsh(cov)
            lam_min_i = eigval.min()
            if not lam_min_i > 0:
                raise np.linalg.LinAlgError(f'Non-positive eigenvalues in mixture component: {eigval}')
            self.lam_min.append(lam_min_i)
            self.lam_max.append(eigval.max())
            self.log_diag.append(np.log(np.diag(cov)))

    @staticmethod
    def from_pypmc(mix):
        """Initialize from a pypmc Gaussian mixture model (GMM).

        Parameters
        -----------
        mix: `pypmc.density.mixture.GaussianMixture`
            Gaussian mixture.

        Returns
        ----------
        mix: `GaussianMixture`
            Generalized Gaussian mixture.
        """
        return GaussianMixture(
            weights=mix.weights,
            means=[g.mu for g in mix.components],
            covs=[g.sigma for g in mix.components])

    @staticmethod
    def from_sklearn(skgmm):
        """Initialize from a scikit-learn Gaussian mixture model (GMM).

        Parameters
        -----------
        skgmm: `sklearn.mixture.GaussianMixture`
            Gaussian mixture.

        Returns
        ----------
        mix: `GaussianMixture`
            Generalized Gaussian mixture.
        """
        covariance_type = skgmm.covariance_type
        if hasattr(skgmm, '_gmm') and covariance_type == 'full':
            # handle pypmc
            return GaussianMixture(
                weights=skgmm._gmm.weights[0,:,0,0],
                means=skgmm._gmm.means[0,:,:,0],
                covs=skgmm._gmm.covariances.precisions_cholesky_numpy)
        precisions_cholesky = None
        if covariance_type == 'full':
            covs = skgmm.covariances_
            precisions_cholesky = getattr(skgmm, 'precisions_cholesky_', None)
        elif covariance_type == 'tied':
            covs = [skgmm.covariances_] * len(skgmm.weights_)
        elif covariance_type == 'diag':
            covs = [np.diag(cov) for cov in skgmm.covariances_]
        elif covariance_type == 'spherical':
            covs = [np.eye(skgmm.means_.shape[1]) * cov for cov in skgmm.covariances_]
        else:
            raise ValueError(f'unknown covariance_type "{covariance_type}"')
        return GaussianMixture(
            weights=skgmm.weights_,
            means=skgmm.means_,
            covs=covs,
            precisions_cholesky=precisions_cholesky)

    def pdf(self, x, mask):
        """Compute probability density at x.

        Parameters
        -----------
        x: array
            The points (vector) at which to evaluate the probability.
        mask: array
            A boolean mask of the same shape as `x`, indicating whether the entry
            is a value (True) or a upper bound (False).

        Returns
        ----------
        pdf: array
            probability density. One value for each `x`.
        """
        return sum(
            w * g.pdf(x, mask)
            for w, g in zip(self.weights, self.components))

    def logpdf(self, x, mask, cutoff=-100.0, margin=10):
        """Compute logarithm of probability density.

        Parameters
        -----------
        x: array
            The points (vector) at which to evaluate the probability.
        mask: array
            A boolean mask of the same shape as `x`, indicating whether the entry
            is a value (True) or a upper bound (False).
        cutoff: float
            a lowest logpdf value, below which approximations are acceptable
        margin: float
            maximum logpdf difference below which to neglect components.

        Returns
        ----------
        logpdf: array
            logarithm of the probability density. One value for each `x`.
        """
        if mask is Ellipsis:
            return logsumexp([
                w + g.logpdf(x, Ellipsis)
                for w, g in zip(self.log_weights, self.components)],
                axis=0)
        n_rows, d = x.shape
        assert d == self.ndim, (d, self.ndim)
        assert mask.shape == (n_rows, self.ndim), (mask.shape, (len(x), self.ndim))
        assert x.shape == (len(mask), self.ndim), (x.shape, (len(x), self.ndim))
        code = mask.astype(np.int64) + 2 * np.isposinf(x).astype(np.int64)
        powers = code @ self.powers
        unique_powers, unique_indices = np.unique(powers, return_index=True)
        if len(unique_powers) == 1 and unique_powers[0] == self.allpowers:
            return logsumexp([
                w + g.logpdf(x, Ellipsis)
                for w, g in zip(self.log_weights, self.components)],
                axis=0)
        logpdf_values = np.empty(n_rows)
        for power, index in zip(unique_powers, unique_indices):
            members = powers == power
            if power == self.allpowers:
                mask_here = Ellipsis
                k = self.ndim
            else:
                mask_here = mask[index, :]
                k = int(mask_here.sum())

            if k == 0:
                # no observed entries -> PDF contributes zero; only CDF terms exist
                logpdf_values[members] = logsumexp([
                    w + g.conditional_logpdf(x[members, :], mask_here)
                    for w, g in zip(self.log_weights, self.components)], axis=0)
                continue

            nmembers = members.sum()
            WLB_row = np.full(nmembers, cutoff)
            WLB_threshold_row = WLB_row
            X_E = x[members][:, mask_here]  # shape (#rows_group, k)
            contrib = np.full((self.ncomponents, nmembers), -np.inf)
            kept_computed = 0
            # Build per-component upper bounds
            for i, (w, g, lam_min_i, lam_max_i) in enumerate(zip(self.log_weights,
                                                                 self.components,
                                                                 self.lam_min,
                                                                 self.lam_max)):
                v = X_E - g.mean[mask_here][None, :]
                r2 = np.sum(v * v, axis=1)

                # Safe upper bound on subspace log-pdf (works for any subspace E)
                UB = -0.5 * (k * const2pi + k * np.log(lam_min_i) + r2 / lam_max_i)
                WUB_i = w + UB

                # Skip component entirely if it cannot beat the current threshold for any row
                if not np.any(WUB_i >= WLB_threshold_row):
                    continue
                kept_computed += 1
                # Evaluate this component for all members (one call)
                exact_i = self.log_weights[i] + self.components[i].conditional_logpdf(x[members, :], mask_here, key=power)
                contrib[i, :] = exact_i
                # Update per-row lower bound and thresholds
                WLB_row = np.maximum(WLB_row, exact_i)
                WLB_threshold_row = np.maximum(cutoff, WLB_row - margin)
            # print(
            #     power, k, '*' if mask_here is Ellipsis else mask_here * 1,
            #     'pdf' if pure_pdf else 'mix',
            #     f'{nmembers} members, {kept_computed}/{self.ncomponents} kept')
            logpdf_values[members] = logsumexp(contrib, axis=0)
        return logpdf_values
