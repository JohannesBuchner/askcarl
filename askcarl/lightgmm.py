"""A extremely fast-to-train GMM."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

__all__ = ["LightGMM"]

from .utils import mvn_logpdf


def local_covariances(X, indices, centroids, sample_weight=None):
    """Compute covariance of clusters.

    Parameters
    ----------
    X: array
        data. shape (N, D)
    indices: array
        list of selectors on X, one boolean array for each cluster. shape (K, N)
    centroids: array
        list of cluster centers. shape (K, D)
    sample_weight: array
        weights. shape (N,)

    Returns
    -------
    covariances: array
        list of covariance matrices.
    """
    N, D = centroids.shape
    well_defined = np.zeros(N, dtype=bool)
    covariances = np.empty((N, D, D))
    for i, idx in enumerate(indices):
        if not idx.sum() > 2 * D + 1:
            continue
        neighbors = X[idx]
        cov_diag = np.var(neighbors, axis=0)
        if not np.all(cov_diag > 0):
            continue
        cov = np.cov(neighbors, rowvar=False, aweights=sample_weight)

        # assert is_positive_definite(cov)
        well_defined[i] = True
        covariances[i] = cov
    return covariances, well_defined


def log_prob_gmm(X, centroids, covariances, weights):
    """Compute log-prob of GMM.

    Parameters
    ----------
    X: array
        data, of shape (N, D)
    centroids: array
        list of component centers, of shape (K, D)
    covariances: array
        list of component covariance matrices, of shape (K, D, D)
    weights: array
        list of component weights, of shape (K,)

    Returns
    -------
    logprob: array
        log-probabilities, one entry for each entry in X, of shape (N)
    """
    log_probs = np.zeros((len(X), len(centroids)))
    for i, (mu, cov, w) in enumerate(zip(centroids, covariances, weights)):
        try:
            log_probs[:, i] = multivariate_normal.logpdf(X, mean=mu, cov=cov) + np.log(w)
        except np.linalg.LinAlgError:
            continue  # fallback if cov is singular
    return logsumexp(log_probs, axis=1)


@jax.jit
def refine_weights_jax(X, means, precisions_cholesky, sample_weight=None):
    """Derive weights for Gaussian mixture.

    Parameters
    ----------
    X: array
        data, of shape (N, D)
    means: array
        list of component centers, of shape (K, D)
    precisions_cholesky: array
        list of component precision matrices, of shape (K, D, D)
    sample_weight: array
        weights. shape (N,)

    Returns
    -------
    weights: array
        list of component weights, of shape (K,)
    """
    def log_prob_fn(mu, prec_chol):
        return mvn_logpdf(X, mu, prec_chol)
    # Vectorize over components
    log_probs = (jax.vmap(log_prob_fn, in_axes=(0, 0))(
        means, precisions_cholesky)).T  # shape (n_samples, n_components)

    # Log-responsibilities
    log_resp = log_probs - jax.scipy.special.logsumexp(log_probs, axis=1, keepdims=True)

    # Convert to responsibilities
    resp = jnp.exp(log_resp)

    # Compute new weights
    weights = jnp.average(resp, axis=0, weights=sample_weight)
    return weights / weights.sum()


class LightGMM:
    """Wrapper which transforms KMeans results into a GMM."""

    def __init__(
        self, n_components, refine_weights=False,
        init_kwargs=dict(n_init=1, max_iter=1, init='random'),
        warm_start=False, covariance_type='full'
    ):
        """Initialise.

        Parameters
        ----------
        n_components: int
            number of Gaussian components.
        refine_weights: bool
            whether to include a E step at the end.
        init_kwargs: dict
            arguments passed to KMeans
        warm_start: bool
            not supported, has to be False
        covariance_type: str
            only "full" is supported
        """
        assert not warm_start
        assert covariance_type == 'full'
        self.covariance_type = covariance_type
        init_kwargs['n_clusters'] = n_components
        self.refine_weights = refine_weights
        self.init_kwargs = init_kwargs
        self.n_components = n_components
        self.initialised = False

    def _cluster(self, X, sample_weight=None, rng=np.random):
        self.kmeans_ = KMeans(**self.init_kwargs).fit(X, sample_weight=sample_weight)
        self.means_ = np.array(self.kmeans_.cluster_centers_)
        self.labels_ = self.kmeans_.labels_
        self.indices_ = self.labels_[None,:] == jnp.arange(self.n_components)[:,None]
        self.initialised = True

    def _characterize_clusters(self, X, sample_weight=None):
        self.covariances_, well_defined = local_covariances(
            X, self.indices_, self.means_, sample_weight=sample_weight)

        for i in np.where(~well_defined)[0]:
            js = np.where(well_defined)[0]
            j = js[np.argmin(np.abs(js - i))]
            self.covariances_[i] = self.covariances_[j]
            # print(f"setting covariance of component {i} with {j} to numerical issues")

        self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_, 'full')
        if self.refine_weights:
            self.weights_ = refine_weights_jax(X, self.means_, self.precisions_cholesky_, sample_weight=sample_weight)
        else:
            weights_int = jnp.bincount(self.labels_, weights=sample_weight, minlength=self.n_components)
            weights = weights_int / float(weights_int.sum())
            self.weights_ = weights

    def fit(self, X, sample_weight=None, rng=np.random):
        """Fit.

        Parameters
        ----------
        X: array
            data, of shape (N, D)
        sample_weight: array
            weights of observations. shape (N,)
        rng: object
            Random number generator
        """
        self._cluster(X, sample_weight=sample_weight, rng=rng)
        self._characterize_clusters(X, sample_weight=sample_weight)
        self.converged_ = True
        self.n_iter_ = 0

    def to_sklearn(self):
        """Convert to a scikit-learn GaussianMixture object.

        Returns
        -------
        gmm: object
            scikit-learn GaussianMixture
        """
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            warm_start=True,
            weights_init=self.weights_,
            means_init=self.means_,
            precisions_init=self.precisions_cholesky_,
        )
        # This does a warm start at the given parameters
        gmm.converged_ = True
        gmm.lower_bound_ = -np.inf
        gmm.weights_ = self.weights_
        gmm.means_ = self.means_
        gmm.precisions_cholesky_ = self.precisions_cholesky_
        gmm.covariances_ = self.covariances_
        return gmm

    def score_samples(self, X):
        """Compute score of samples.

        Parameters
        ----------
        X: array
            data, of shape (N, D)

        Returns
        -------
        logprob: array
            log-probabilities, one entry for each entry in X, of shape (N)
        """
        return log_prob_gmm(X, self.means_, self.covariances_, self.weights_)

    def score(self, X, sample_weight=None):
        """Compute score of samples.

        Parameters
        ----------
        X: array
            data, of shape (N, D)
        sample_weight: array
            weights of observations. shape (N,)

        Returns
        -------
        logprob: float
            average log-probabilities, one entry for each entry in X, of shape (N)
        """
        return np.average(self.score_samples(X), weights=sample_weight)

    def sample(self, N):
        """Generate samples from model.

        Parameters
        ----------
        N: int
            number of samples

        Returns
        -------
        X: array
            data, of shape (N, D)
        """
        return self.to_sklearn().sample(N)
