import numpy as np
from scipy.special import logsumexp
from .gaussian import Gaussian


class GaussianMixture:
    """Mixture of Gaussians.

    Attributes
    -----------
    weights: list
        weight for each Gaussian component
    members: list
        list of Gaussian components.
    """

    def __init__(self, weights, means, covs):
        """Initialize.

        Parameters
        -----------
        weights: list
            weight for each Gaussian component
        means: list
            mean vector for each Gaussian component.
        covs: list
            covariance matrix for each Gaussian component.
        """
        assert np.isfinite(weights).all()
        assert len(weights) == len(covs)
        weights = np.asarray(weights)
        assert weights.shape == (len(means),)
        self.weights = weights[weights > 0]
        self.members = [Gaussian(mean, cov) for mean, cov, w in zip(means, covs, weights) if w > 0]
        assert len(self.weights) == len(self.members)
        self.log_weights = np.log(self.weights)

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
            for w, g in zip(self.weights, self.members))

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
        return logsumexp([
            w + g.logpdf(x, mask)
            for w, g in zip(self.log_weights, self.members)],
            axis=0)

