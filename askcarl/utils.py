"""Utility functions for dealing with Gaussians and their covariances."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import cholesky, solve_triangular


def mvn_logpdf(X, mean, prec_chol):
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
    return log_det - 0.5 * (D * jnp.log(2 * jnp.pi) + quad_form)


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


def log_prob_gmm_jax(X, centroids, precisions_cholesky, weights):
    """Compute log-prob of GMM.

    Parameters
    ----------
    X: array
        data, of shape (N, D)
    centroids: array
        list of component centers, of shape (K, D)
    precisions_cholesky: array
        list of component precision matrices, of shape (K, D, D)
    weights: array
        list of component weights, of shape (K,)

    Returns
    -------
    logprob: array
        log-probabilities, one entry for each entry in X, of shape (N)
    """
    def log_prob_fn(mu, prec_chol, w):
        return mvn_logpdf(X, mu, prec_chol) + jnp.log(w)

    log_probs = jax.vmap(log_prob_fn)(centroids, precisions_cholesky, weights)  # shape (K, N)
    return jax.scipy.special.logsumexp(log_probs, axis=0)  # shape (N,)


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
    return solve_triangular(cholesky(cov, lower=True), np.eye(cov.shape[0]), lower=True).T
