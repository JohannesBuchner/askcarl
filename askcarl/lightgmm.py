"""A extremely fast-to-train GMM."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.ops import segment_sum
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

from .utils import cov_to_prec_cholesky

__all__ = ["LightGMM", "LightGMM2", "LightBaggingGMM"]

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
        """Compute Multivariate Gaussian log-probability density.

        Parameters
        ----------
        mu: array
            mean
        prec_chol: array
            component precision matrix

        Returns
        -------
        float:
            log-probability
        """
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


def kmeans_assign_underpopulated_labels(distances, labels, cardinality, min_cluster_size):
    """Assign more members to underpopulated clusters.

    Parameters
    ----------
    distances: array
        euclidean distance of member to cluster center, of shape (N, K)
    labels: array
        member of label
    cardinality: array
        number of cluster members for each cluster, of shape (K)
    min_cluster_size: int
        minimum number of cluster members, of shape (K)

    Returns
    -------
    labels: array
        modified labels after re-assignment
    """
    underpopulated = cardinality < min_cluster_size
    for label_to_replace in np.where(underpopulated)[0]:
        # find nearest and assign them to the cluster
        nearest = jnp.argsort(distances[:, label_to_replace])[:min_cluster_size]
        labels[nearest] = label_to_replace
        # print(f"boosting underpopulated cluster: {nearest} <- {label_to_replace}")
    return labels


@partial(jax.jit, static_argnames=['K'])
def kmeans_assign_labels(X, centroids, K, sample_weight):
    """Assign members to clusters and compute cluster statistics.

    Parameters
    ----------
    X: array
        data, of shape (N, D)
    centroids: array
        cluster centers, of shape (K, D)
    K: int
        number of clusters.
    sample_weight: array
        weights. shape (N,)

    Returns
    -------
    distances: array
        euclidean distance of member to cluster center, of shape (N, K)
    labels: array
        membership for each data point in X, of shape (N,)
    cardinality: array
        number of cluster members for each cluster, of shape (K)
    counts: array
        cluster weight, of shape (K). If weights are 1 then this is the same as cardinality.
    """
    # compute distances, update labels, update centers, update labels.
    N, D = X.shape

    # First: compute squared distances efficiently
    X_norm = jnp.sum(X ** 2, axis=1, keepdims=True)       # (N, 1)
    C_norm = jnp.sum(centroids ** 2, axis=1, keepdims=True).T  # (1, K)
    distances = X_norm + C_norm - 2 * X @ centroids.T      # (N, K)

    # Assign initial labels
    labels = jnp.argmin(distances, axis=1)

    # Count members per cluster
    # counts = segment_sum(jnp.ones(N, dtype=jnp.int32), labels, K)
    counts = segment_sum(sample_weight, labels, K)
    cardinality = jnp.bincount(labels, minlength=K, length=K)

    return distances, labels, cardinality, counts


@partial(jax.jit, static_argnames=['K'])
def kmeans_compute_cluster_statistics(X, labels, K, sample_weight):
    """Compute cluster statistics from cluster members.

    Parameters
    ----------
    X: array
        data, of shape (N, D)
    labels: array
        membership for each data point in X, of shape (N,)
    K: int
        number of clusters.
    sample_weight: array
        weights. shape (N,)

    Returns
    -------
    centroids: array
        weighted centroids, of shape (K, D)
    counts: array
        cluster weight, of shape (K). If weights are 1 then this is the same as cardinality.
    """
    N, D = X.shape
    weights = sample_weight[:, None]
    weighted_X = X * weights
    counts = segment_sum(sample_weight, labels, K)
    summed = segment_sum(weighted_X, labels, K)
    return summed, counts


def relocate_empty_clusters_dense(X, distances, sample_weight, centers_sum, weight_in_clusters, labels):
    """Change cluster centroids towards distant cluster members.

    Existing cluster centroids statistics (weight and centers) are modified.

    Parameters
    ----------
    X: array
        data, of shape (N, D)
    distances: array
        euclidean distance of member to cluster center, of shape (N, K)
    sample_weight: array
        weights. shape (N,)
    centers_sum: array
        weighted centroids, of shape (K, D). Warning: *modified in place*
    weight_in_clusters: array
        cluster weight, of shape (K).
    labels: array
        membership for each data point in X, of shape (N,)

    Returns
    -------
    centroids: array
        updated centroid positions, of shape (K, D)
    """
    N, D = X.shape
    K, = weight_in_clusters.shape
    assert distances.shape == (N, K)
    assert weight_in_clusters.shape == (K,)
    assert labels.shape == (N,)
    assert sample_weight.shape == (N,)
    empty_clusters = np.where(weight_in_clusters == 0)[0]
    n_empty = empty_clusters.shape[0]
    if n_empty == 0 or np.max(distances) == 0:
        return centers_sum / weight_in_clusters[:, None]

    weight_in_clusters = np.array(weight_in_clusters)

    assigned_distances = distances[np.arange(N), labels]
    far_from_centers = np.argpartition(assigned_distances, -n_empty)[-n_empty:]

    for idx in range(n_empty):
        new_cluster_id = empty_clusters[idx]
        far_idx = far_from_centers[idx]
        weight = sample_weight[far_idx]
        old_cluster_id = labels[far_idx]

        centers_sum[old_cluster_id] -= X[far_idx] * weight
        centers_sum[new_cluster_id] = X[far_idx] * weight

        weight_in_clusters[new_cluster_id] = weight
        weight_in_clusters[old_cluster_id] -= weight

    # to centroid space
    return centers_sum / weight_in_clusters[:, None]


def kmeans_single_iteration(X, centroid_indices, sample_weight=None, min_cluster_size=1):
    """Update of k-means.

    Parameters
    ----------
    X: array
        data, of shape (N, D)
    centroid_indices: array
        indices of X which will serve as initial cluster centroids, of shape (K,)
    sample_weight: array
        weights. shape (N,), or None
    min_cluster_size: int
        minimum number of cluster members, of shape (K)

    Returns
    -------
    labels: array
        membership for each data point in X, of shape (N,)
    centroids: array
        weighted centroids, of shape (K, D).
    cardinality: array
        number of cluster members for each cluster, of shape (K)
    """
    # compute distances, update labels, update centers, update labels.
    N, D = X.shape
    K, = centroid_indices.shape
    centers = X[centroid_indices].copy()

    # === E-step: assign labels to initial centers ===
    distances, labels, _, weight_in_clusters = kmeans_assign_labels(
        X, centers, K=K, sample_weight=sample_weight)

    # === M-step: compute cluster sums and counts ===
    summed, counts = kmeans_compute_cluster_statistics(
        X, labels, K, sample_weight=sample_weight)

    # === Handle empty clusters ===
    centers_relocated = relocate_empty_clusters_dense(
        X, distances, sample_weight, summed, counts, labels)

    distances_final, labels_final, cardinality, _ = kmeans_assign_labels(
        X, centers_relocated, K=K, sample_weight=sample_weight)
    return labels_final, centers_relocated, cardinality


def kmeans_iterate(X, K, sample_weight=None, min_cluster_size=1, rng=np.random, verbose=False, TT=None, invTT=None):
    """Iterate K-means.

    Parameters
    ----------
    X: array
        data, of shape (N, D)
    K: int
        number of clusters.
    sample_weight: array
        weights. shape (N,)
    min_cluster_size: int
        minimum number of cluster members, of shape (K) Returns
    rng: np.random
        Pseudo-random number generator to use.
    verbose: bool
        whether to print to stdout in case of success or failure
    TT: array
        Whitening transform matrix. If none, no whitening is applied.
    invTT: array
        Inverse whitening transform matrix. If none, no whitening is applied.

    Returns
    -------
    labels: array
        membership for each data point in X, of shape (N,)
    centroids: array
        weighted centroids, of shape (K, D).
    covariances: array
        Empirical covariance matrix of the members of each cluster, of shape (K, D, D)
    precisions_chol: array
        list of precision matrices of the members of each cluster, of shape (K, D, D)
    cardinality: array
        fraction of sample N in each cluster, of shape (K,)
    """
    N, D = X.shape
    covariances = np.empty((K, D, D))
    precisions_chol = np.empty((K, D, D))
    if sample_weight is None:
        sample_weight_actual = np.ones(N)
    else:
        sample_weight_actual = sample_weight
    if TT is None:
        XT = X
    else:
        # apply whitening transform
        mean = X.mean(axis=0, keepdims=True)
        XT = (X - mean) @ TT
    while True:
        centroid_indices = rng.choice(N, size=K, replace=False)
        labels, centroids, cardinalities = kmeans_single_iteration(
            XT, centroid_indices, sample_weight=sample_weight_actual, min_cluster_size=min_cluster_size)
        # to fail fast, start with the smallest cluster
        order = np.argsort(cardinalities)
        if cardinalities[order[0]] < min_cluster_size:
            if verbose:
                print('fail, some clusters are too small!', N, D, K, min_cluster_size, cardinalities[order[0]])
            continue
        if invTT is not None:
            # reverse whitening transform
            centroids = centroids @ invTT + mean
        try:
            for k in order:
                mask = labels == k
                covariances[k] = np.cov(
                    X[mask,:], rowvar=False,
                    aweights=None if sample_weight is None else sample_weight[mask])
                precisions_chol[k] = cov_to_prec_cholesky(covariances[k]).T
            if verbose:
                print('success!', cardinalities[order[0]])
            return labels, centroids, covariances, precisions_chol, cardinalities / float(N)
        except np.linalg.LinAlgError:
            # not a successful construction, try again
            continue
        except FloatingPointError as e:
            # not a successful construction, try again
            if verbose:
                print('fail!', e, cardinalities[k])
            continue


class LightGMM:
    """Wrapper which transforms KMeans results into a GMM."""

    def __init__(
        self, n_components, refine_weights=False,
        init_kwargs=dict(n_init=1, max_iter=1, init='random'),
        warm_start=False, covariance_type='full', TT=None, invTT=None
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
        TT: array
            Whitening transform matrix to apply when initialising centroids. If none, no whitening is applied.
        invTT: array
            Inverse transform of TT.
        """
        assert not warm_start
        assert covariance_type == 'full'
        self.covariance_type = covariance_type
        self.n_components = int(n_components)
        init_kwargs['n_clusters'] = self.n_components
        assert refine_weights in (True, False)
        self.refine_weights = refine_weights
        self.init_kwargs = init_kwargs
        self.initialised = False
        self.TT = TT
        self.invTT = invTT

    def _cluster(self, X, sample_weight=None, rng=np.random):
        """Apply clustering.

        Parameters
        ----------
        X: array
            data, of shape (N, D)
        sample_weight: array
            weights. shape (N,)
        rng: np.random
            Pseudo-random number generator to use.
        """
        self.kmeans_ = KMeans(**self.init_kwargs).fit(X, sample_weight=sample_weight)
        self.means_ = np.array(self.kmeans_.cluster_centers_)
        self.labels_ = self.kmeans_.labels_
        self.indices_ = self.labels_[None,:] == jnp.arange(self.n_components)[:,None]
        self.initialised = True

    def _characterize_clusters(self, X, sample_weight=None):
        """Characterize the clusters.

        Parameters
        ----------
        X: array
            data, of shape (N, D)
        sample_weight: NoneType
            weights. shape (N,)
        """
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
        if self.init_kwargs.get('n_init', 0) == 1 and self.init_kwargs.get('max_iter', 0) == 1 and self.init_kwargs.get('init', 'random') == 'random':
            self.labels_, self.means_, self.covariances_, self.precisions_cholesky_, self.weights_ = kmeans_iterate(
                X, self.n_components,
                sample_weight=sample_weight, rng=rng,
                min_cluster_size=2, TT=self.TT, invTT=self.invTT)
            if self.refine_weights:
                self.weights_ = refine_weights_jax(X, self.means_, self.precisions_cholesky_, sample_weight=sample_weight)
        else:
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


class LightGMM2:
    """Wrapper which fits K-folds two LightGMMs results.

    The training data is split into two halfs, and a mixture
    is built from each half. Then, the weights of the mixture
    are optimized with the other half. This should avoid overfitting
    (compared to building a GMM and optimizing on the same data set).
    """

    def __init__(
        self, n_components,
        init_kwargs=dict(n_init=1, max_iter=1, init='random'),
        warm_start=False, covariance_type='full'
    ):
        """Initialise.

        Parameters
        ----------
        n_components: int
            number of Gaussian components.
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
        self.n_components = int(n_components)
        init_kwargs['n_clusters'] = self.n_components
        self.init_kwargs = init_kwargs
        self.initialised = False
        self.gmm1 = LightGMM(n_components, init_kwargs=init_kwargs)
        self.gmm2 = LightGMM(n_components, init_kwargs=init_kwargs)

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
        X1 = X[::2]
        X2 = X[1::2]
        W1 = None if sample_weight is None else sample_weight[::2]
        W2 = None if sample_weight is None else sample_weight[1::2]
        # optimize means and covariances to one half
        self.gmm1.fit(X1, W1, rng=rng)
        self.gmm2.fit(X2, W2, rng=rng)
        # optimize weights onto the left-out sample
        self.gmm1.weights_ = refine_weights_jax(X2, self.gmm1.means_, self.gmm1.precisions_cholesky_, sample_weight=W2)
        self.gmm2.weights_ = refine_weights_jax(X1, self.gmm2.means_, self.gmm2.precisions_cholesky_, sample_weight=W1)
        self.labels_ = self.gmm1.labels_
        self.weights_ = np.concatenate((self.gmm1.weights_, self.gmm2.weights_)) / 2.0
        assert self.weights_.shape == (self.n_components * 2,)
        self.means_ = np.vstack((self.gmm1.means_, self.gmm2.means_))
        assert self.means_.shape == (self.n_components * 2, X.shape[1])
        self.precisions_cholesky_ = np.vstack((self.gmm1.precisions_cholesky_, self.gmm2.precisions_cholesky_))
        assert self.precisions_cholesky_.shape == (self.n_components * 2, X.shape[1], X.shape[1])
        self.covariances_ = np.vstack((self.gmm1.covariances_, self.gmm2.covariances_))
        assert self.covariances_.shape == (self.n_components * 2, X.shape[1], X.shape[1])
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


class LightBaggingGMM:
    """Wrapper which fits B LightGMMs, and averages likelihoods.

    The training data is split into two halfs, and a mixture
    is built from each half. Then, the weights of the mixture
    are optimized with the other half. This should avoid overfitting
    (compared to building a GMM and optimizing on the same data set).
    """

    def __init__(
        self, n_gmms, **kwargs
    ):
        """Initialise.

        Parameters
        ----------
        n_gmms: int
            number of GMMs.
        kwargs: dict
            passed to LightGMM.
        """
        self.n_gmms = n_gmms
        self.gmms = [LightGMM(**kwargs) for i in range(n_gmms)]

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
        for gmm in self.gmms:
            gmm.fit(X=X, sample_weight=sample_weight, rng=rng)

        self.labels_ = gmm.labels_
        self.weights_ = np.concatenate([gmm.weights_ for gmm in self.gmms]) / len(self.gmms)
        assert self.weights_.shape == (gmm.n_components * self.n_gmms,)
        self.means_ = np.vstack([gmm.means_ for gmm in self.gmms])
        assert self.means_.shape == (gmm.n_components * self.n_gmms, X.shape[1])
        self.precisions_cholesky_ = np.vstack([gmm.precisions_cholesky_ for gmm in self.gmms])
        assert self.precisions_cholesky_.shape == (gmm.n_components * self.n_gmms, X.shape[1], X.shape[1])
        self.covariances_ = np.vstack([gmm.covariances_ for gmm in self.gmms])
        assert self.covariances_.shape == (gmm.n_components * self.n_gmms, X.shape[1], X.shape[1])

        gmm = GaussianMixture(
            n_components=gmm.n_components,
            covariance_type='full',
            warm_start=True,
            weights_init=self.weights_,
            means_init=self.means_,
            precisions_init=self.precisions_cholesky_,
        )
        gmm.converged_ = True
        gmm.lower_bound_ = -np.inf
        gmm.weights_ = self.weights_
        gmm.means_ = self.means_
        gmm.precisions_cholesky_ = self.precisions_cholesky_
        gmm.covariances_ = self.covariances_
        self.gmm_ = gmm

    def to_sklearn(self):
        """Convert to a scikit-learn GaussianMixture object.

        Returns
        -------
        gmm: object
            scikit-learn GaussianMixture
        """
        # This does a warm start at the given parameters
        return self.gmm_

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
