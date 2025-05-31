.. _install:
.. highlight:: shell

========
LightGMM
========

LightGMM is a extremely fast construction of a GMM for large data sets.

Method
------

If refine_weights=False, LightGMM relies on scikit-learn's KMeans 
for initialisation, using kmeans++ or random data samples 
('random', very fast) for initally guessing the cluster.

The KMeans steps (n_iter and max_iter parameters) give the cluster centers.
Regularized covariance matrices are then computed from cluster members. 
These set the shapes and locations of the Gaussian mixture model.

The GMM component weights are assigned proportional to number of cluster members.

Alternatively, if refine_weights=True, with a single E step is performed
for maximizing the training set likelihood.

Requirements
------------

You need scikit-learn and jax installed.
These are not installed by default with askcarl.

Usage
-----

The API is scikit-learn-like::

    gmm = LightGMM(**kwargs)
    gmm.fit(X_train)
    logprob = gmm.score_samples(X_test)
    score = gmm.score(X_test)

You can also convert to a scikit-learn GaussianMixture object::

    gmmsk = gmm.to_sklearn()

See the `API documentation of the LightGMM module <askcarl.html#module-askcarl.lightgmm>`_.

Performance
-----------

LightGMM achieves sub-100ms constructions for data sets 
with 50000 samples and ~15 features.

See this benchmark: https://github.com/JohannesBuchner/GMM-benchmark

