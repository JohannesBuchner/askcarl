"""Mixture of Gaussians."""
import numpy as np
from scipy.special import logsumexp

from .gaussian import Gaussian


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
        self.weights = weights[weights > 0]
        assert len(self.weights) == len(self.components)
        self.log_weights = np.log(self.weights)

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

    def logpdf(self, x, mask):
        """Compute logarithm of probability density.

        Parameters
        -----------
        x: array
            The points (vector) at which to evaluate the probability.
        mask: array
            A boolean mask of the same shape as `x`, indicating whether the entry
            is a value (True) or a upper bound (False).

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
        assert mask.shape == (len(x), self.ndim), (mask.shape, (len(x), self.ndim))
        assert x.shape == (len(mask), self.ndim), (x.shape, (len(x), self.ndim))
        code = mask.astype(np.int64) + 2 * np.isposinf(x).astype(np.int64)
        powers = code @ self.powers
        unique_powers, unique_indices = np.unique(powers, return_index=True)
        if len(unique_powers) == 1 and unique_powers[0] == self.allpowers:
            return logsumexp([
                w + g.logpdf(x, Ellipsis)
                for w, g in zip(self.log_weights, self.components)],
                axis=0)
        logpdf_values = np.zeros(len(x)) * np.nan
        for power, index in zip(unique_powers, unique_indices):
            members = powers == power
            if power == self.allpowers:
                mask_here = Ellipsis
            else:
                mask_here = mask[index, :]
            logpdf_values[members] = logsumexp([
                w + g.conditional_logpdf(x[members,:], mask_here)
                for w, g in zip(self.log_weights, self.components)],
                axis=0)
        return logpdf_values
