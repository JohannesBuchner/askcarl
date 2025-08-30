"""Utility functions for dealing with Gaussians and their covariances."""
import jax
import jax.numpy as jnp
import numpy as np
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
    key = X.shape[-1]
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
