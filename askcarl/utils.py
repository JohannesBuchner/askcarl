"""Utility functions for dealing with Gaussians and their covariances."""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
from scipy.linalg import cholesky, solve_triangular

# jit-compiled multivariate Gaussian logpdf functions
mvn_logpdf_functions = {}


def _mvn_logpdf(X, mean, prec_chol):
    """Compute log-prob of a Gaussian.

    Parameters
    ----------
    X: array
        data, of shape (N, D)
    mean: array
        Mean of Gaussian, of shape (D)
    prec_chol: array
        precision matrix, of shape (D, D)

    Returns
    -------
    logprob: array
        log-probability, one entry for each entry in X, of shape (N)
    """
    if X.ndim == 1:
        X = X[None, :]  # Convert (D,) -> (1, D)
    D = X.shape[1]
    x_centered = X - mean
    y = jnp.dot(x_centered, prec_chol.T)
    log_det = jnp.sum(jnp.log(jnp.diag(prec_chol)))
    quad_form = jnp.sum(y**2, axis=1)
    return log_det - 0.5 * (D * jnp.log(2 * jnp.pi)) - 0.5 * quad_form


def mvn_logpdf(X, mean, prec_chol):
    """Compute log-prob of a Gaussian.

    This keeps jit-compiled functions for each invocation shape.

    Parameters
    ----------
    X: array
        data, of shape (N, D)
    mean: array
        Mean of Gaussian, of shape (D)
    prec_chol: array
        precision matrix, of shape (D, D)

    Returns
    -------
    logprob: array
        log-probability, one entry for each entry in X, of shape (N)
    """
    key = (X.shape[-1], mean is None)
    if key not in mvn_logpdf_functions:
        mvn_logpdf_functions[key] = jax.jit(_mvn_logpdf)
    return mvn_logpdf_functions[key](X, mean, prec_chol)


def mvn_pdf(X, mean, prec_chol):
    """Compute log-prob of a Gaussian.

    Parameters
    ----------
    X: array
        data, of shape (N, D)
    mean: array
        Mean of Gaussian, of shape (D)
    prec_chol: array
        precision matrix, of shape (D, D)

    Returns
    -------
    logprob: array
        log-probability, one entry for each entry in X, of shape (N)
    """
    return jnp.exp(mvn_logpdf(X, mean, prec_chol))


def is_positive_definite(cov, tol=1e-10, condthresh=1e6):
    """Check that the covariance matrix is well behaved.

    Parameters
    ----------
    cov: array
        covariance matrix. shape (D, D)
    tol: float
        smallest eigvalsh value allowed
    condthresh: float
        minimum on matrix condition number

    Returns
    -------
    bool
        True if the matrix is invertable and positive definite
    """
    cond = np.linalg.cond(cov)
    is_invertible = cond < condthresh
    return is_invertible and np.all(np.linalg.eigvalsh(cov) > tol)


# identity matrices
eyes = {}


def cov_to_prec_cholesky(cov):
    """Convert covariance matrix to Cholesky factors of the precision matrix.

    Parameters
    ----------
    cov: array
        covariance matrix. shape (D, D)

    Returns
    -------
    prec_cholesky: array
        Cholesky factors of the precision matrix. shape (D, D)
    """
    D = cov.shape[0]
    if D not in eyes:
        eyes[D] = np.eye(cov.shape[0])
    return solve_triangular(cholesky(cov, lower=True), eyes[D], lower=True)


class univariate_normal:
    """Univariate normal distribution."""

    def __init__(self, mean, cov):
        """Initialise.

        Parameters
        ----------
        mean: array
            Mean of Gaussian, of shape (D)
        cov: array
            covariance matrix. shape (D, D)
        """
        self.mean = mean
        self.std = float(np.sqrt(cov))

    def cdf(self, x):
        """Return cumulative probability.

        Parameters
        ----------
        x: float
            position.

        Returns
        -------
        float
            cdf value.
        """
        return scipy.special.ndtr((x.flatten() - self.mean) / self.std).reshape(
            (len(x),)
        )

    def pdf(self, x):
        """Return probability density.

        Parameters
        ----------
        x: float
            position.

        Returns
        -------
        float
            pdf value.
        """
        return np.exp(-0.5 * ((x - self.mean) / self.std)**2) / np.sqrt(2 * np.pi) / self.std

    def logpdf(self, x):
        """Return log of probability density.

        Parameters
        ----------
        x: float
            position.

        Returns
        -------
        float
            logpdf value.
        """
        return -0.5 * ((x - self.mean) / self.std)**2 - 0.5 * np.log(2 * np.pi * self.std**2)

    def logcdf(self, x):
        """Return log of the cumulative probability.

        Parameters
        ----------
        x: float
            position.

        Returns
        -------
        float
            log(cdf) value.
        """
        return scipy.special.log_ndtr((x.flatten() - self.mean) / self.std).reshape(
            (len(x),)
        )


class multivariate_normal0:
    """Multivariate normal distribution."""

    def __init__(self, cov, precision_cholesky=None, allow_singular=False):
        """Initialise.

        Parameters
        ----------
        mean: array
            Mean of Gaussian, of shape (D)
        cov: array
            covariance matrix. shape (D, D)
        """
        self.cov = cov
        self.allow_singular = allow_singular
        self.precision_cholesky = precision_cholesky
        self.rv = None

    def cdf(self, x):
        """Return cumulative probability.

        Parameters
        ----------
        x: float
            position.

        Returns
        -------
        float
            cdf value.
        """
        if self.rv is None:
            self.rv = scipy.stats.multivariate_normal(
                np.zeros(len(self.cov)), self.cov, allow_singular=self.allow_singular)
        return self.rv.cdf(x)

    def logcdf(self, x):
        """Return log of the cumulative probability.

        Parameters
        ----------
        x: float
            position.

        Returns
        -------
        float
            log(cdf) value.
        """
        if self.rv is None:
            self.rv = scipy.stats.multivariate_normal(
                np.zeros(len(self.cov)), self.cov, allow_singular=self.allow_singular)
        return self.rv.logcdf(x)

    def pdf(self, x):
        """Return probability density.

        Parameters
        ----------
        x: float
            position.

        Returns
        -------
        float
            pdf value.
        """
        if self.rv is None:
            self.rv = scipy.stats.multivariate_normal(
                np.zeros(len(self.cov)), self.cov, allow_singular=self.allow_singular)
        return self.rv.pdf(x)

    def logpdf(self, x):
        """Return log of the probability density.

        Parameters
        ----------
        x: float
            position.

        Returns
        -------
        float
            log(pdf) value.
        """
        if self.precision_cholesky is not None:
            return mvn_logpdf(x, 0, self.precision_cholesky)
        if self.rv is None:
            self.rv = scipy.stats.multivariate_normal(
                np.zeros(len(self.cov)), self.cov, allow_singular=self.allow_singular)
        return self.rv.logpdf(x)
