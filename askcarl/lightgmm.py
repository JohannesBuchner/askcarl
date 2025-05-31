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


def local_covariances(X, indices, centroids):
    """Compute covariance of clusters.

    Parameters
    ----------
    X: array
        data. shape (N, D)
    indices: array
        list of selectors on X, one boolean array for each cluster. shape (K, N)
    centroids: array
        list of cluster centers. shape (K, D)

    Returns
    -------
    covariances: array
        list of covariance matrices.
    """
    well_defined = np.zeros(len(centroids), dtype=bool)
    covariances = np.empty((len(centroids), X.shape[1], X.shape[1]))
    for i, idx in enumerate(indices):
        neighbors = X[idx]
        cov = np.cov(neighbors, rowvar=False)
        if len(neighbors) > X.shape[1] and is_positive_definite(cov):
            # verify that the cov is reliable:
            # cholesky(cov, lower=True)
            # np.linalg.inv(cov)
            # mark as good
            well_defined[i] = True
        elif len(neighbors) > X.shape[1]:
            # let's try with a diagonal covariance
            cov = np.diag(np.var(neighbors, axis=0))
            if is_positive_definite(cov):
                # cholesky(cov, lower=True)
                # np.linalg.inv(cov)
                # mark as good
                well_defined[i] = True
        covariances[i] = cov
    return covariances, well_defined


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
    D = X.shape[1]
    x_centered = X - mean
    y = jnp.dot(x_centered, prec_chol.T)
    log_det = -jnp.sum(jnp.log(jnp.diag(prec_chol)))
    quad_form = jnp.sum(y**2, axis=1)
    return log_det - 0.5 * (D * jnp.log(2 * jnp.pi) + quad_form)


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
def refine_weights_jax(X, means, precisions_cholesky):
    """Derive weights for Gaussian mixture.

    Parameters
    ----------
    X: array
        data, of shape (N, D)
    means: array
        list of component centers, of shape (K, D)
    precisions_cholesky: array
        list of component precision matrices, of shape (K, D, D)

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
    weights = resp.sum(axis=0) / X.shape[0]
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

    def _cluster(self, X):
        self.kmeans_ = KMeans(**self.init_kwargs).fit(X)
        self.initialised = True
        self.means_ = np.array(self.kmeans_.cluster_centers_)
        self.labels_ = self.kmeans_.labels_
        self.indices_ = self.kmeans_.labels_[None,:] == jnp.arange(self.n_components)[:,None]

    def _characterize_clusters(self, X):
        self.covariances_, well_defined = local_covariances(X, self.indices_, self.means_)

        for i in np.where(~well_defined)[0]:
            js = np.where(well_defined)[0]
            j = js[np.argmin(np.abs(js - i))]
            self.covariances_[i] = self.covariances_[j]
            print(f"setting covariance of component {i} with {j} to numerical issues")

        self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_, 'full')
        if self.refine_weights:
            self.weights_ = refine_weights_jax(X, self.means_, self.precisions_cholesky_)
        else:
            weights_int = jnp.bincount(self.labels_, minlength=self.n_components)
            weights = weights_int / float(weights_int.sum())
            self.weights_ = weights

    def _strip_zero_weight_components(self):
        keep = self.weights_ > 0
        if not keep.all():
            print(f"stripping {(~keep).sum()} of K={len(keep)}, because of zero weight: {self.weights_.min()}")
        self.means_ = self.means_[keep,:]
        self.covariances_ = self.covariances_[keep,:,:]
        self.precisions_cholesky_ = self.precisions_cholesky_[keep,:,:]
        self.weights_ = self.weights_[keep]
        self.n_components_ = keep.sum()

    def fit(self, X):
        """Fit.

        Parameters
        ----------
        X: array
            data, of shape (N, D)
        """
        self._cluster(X)
        self._characterize_clusters(X)
        # self._strip_zero_weight_components()
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

    def score(self, X):
        """Compute score of samples.

        Parameters
        ----------
        X: array
            data, of shape (N, D)

        Returns
        -------
        logprob: float
            average log-probabilities, one entry for each entry in X, of shape (N)
        """
        return self.score_samples(X).mean()

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
