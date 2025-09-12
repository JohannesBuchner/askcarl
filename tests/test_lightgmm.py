import numpy as np
from sklearn.mixture import GaussianMixture

from askcarl.lightgmm import LightGMM

import askcarl.mixture


def test_weights():
    N = 1000
    D = 2
    X_orig = np.random.normal(size=(N, D))
    X = X_orig[np.argsort(X_orig[:,0]),:]
    gmm = LightGMM(1, init_kwargs=dict(n_init=1, max_iter=1000, init='random', random_state=42))
    gmm.fit(X)

    gmm2 = LightGMM(1, init_kwargs=dict(n_init=1, max_iter=1000, init='random', random_state=42))
    gmm2.fit(X, np.ones(X.shape[0]))
    
    np.testing.assert_allclose(gmm.means_, gmm2.means_)
    np.testing.assert_allclose(gmm.precisions_cholesky_, gmm2.precisions_cholesky_)
    np.testing.assert_allclose(gmm.covariances_, gmm2.covariances_)

    gmm4 = LightGMM(1, init_kwargs=dict(n_init=1, max_iter=1000, init='random', random_state=42))
    gmm4.fit(X[:500])
    gmm3 = LightGMM(1, init_kwargs=dict(n_init=1, max_iter=1000, init='random', random_state=42))
    gmm3.fit(X, (np.arange(X.shape[0]) < 500)*1.0)
    
    np.testing.assert_allclose(gmm3.means_, gmm4.means_)
    np.testing.assert_allclose(gmm3.precisions_cholesky_, gmm4.precisions_cholesky_)
    np.testing.assert_allclose(gmm3.covariances_, gmm4.covariances_)

    gmm5 = LightGMM(1, init_kwargs=dict(n_init=1, max_iter=1000, init='random', random_state=42), refine_weights=True)
    gmm5.fit(X, (np.arange(X.shape[0]) < 500)*1.0)
    np.testing.assert_allclose(gmm5.means_, gmm4.means_)
    np.testing.assert_allclose(gmm5.precisions_cholesky_, gmm4.precisions_cholesky_)
    np.testing.assert_allclose(gmm5.covariances_, gmm4.covariances_)


def test_nokmeans():
    np.random.seed(234)
    N = 100
    D = 2
    X = np.random.normal(size=(N, D))
    gmm = LightGMM(1, init_kwargs=dict(n_init=1, max_iter=1, init='random'))
    gmm.fit(X)
    assert np.all(gmm.labels_ == 0)
    print(gmm)

def test_single_gauss():
    np.random.seed(234)
    N = 100000
    for D in [2, 5, 20]:
        X = np.random.normal(size=(N, D))
        gmm = LightGMM(1, init_kwargs=dict(n_init=1, max_iter=1000, init='random'))
        gmm.fit(X)
        score = gmm.score(X)
        Y, labels = gmm.sample(10000)
        assert (labels == 0).all()
        np.testing.assert_allclose(Y.mean(axis=0), 0, atol=0.04)
        np.testing.assert_allclose(Y.std(axis=0), 1, atol=0.04)
        
        gmm_ref = GaussianMixture(1)
        gmm_ref.fit(X)
        score_ref = gmm_ref.score(X)
        np.testing.assert_allclose(score, score_ref)

def test_whitening():
    np.random.seed(234)
    N = 100000
    for D in [2, 5, 20]:
        X = np.vstack((np.random.normal(size=(N, D)) + 10, np.random.normal(size=(N, D))))
        invTT = TT = np.eye(D)

        gmm = LightGMM(2, init_kwargs=dict(n_init=1, max_iter=1000, init='random'))
        gmm.fit(X, rng=np.random.RandomState(D))
        gmmT = LightGMM(2, init_kwargs=dict(n_init=1, max_iter=1000, init='random'), TT=TT, invTT=invTT)
        gmmT.fit(X, rng=np.random.RandomState(D))
        
        assert np.all(gmmT.labels_ == gmm.labels_)
        np.testing.assert_allclose(gmmT.means_, gmm.means_)
        np.testing.assert_allclose(gmmT.weights_, gmm.weights_)
        np.testing.assert_allclose(gmmT.score(X), gmm.score(X))

def test_two_gauss():
    np.random.seed(123)
    N = 100000
    for D in [2, 5, 20]:
        X = np.vstack((np.random.normal(size=(N, D)) + 10, np.random.normal(size=(N, D))))
        print(X.shape)
        assert X.shape == (2 * N, D)
        gmm = LightGMM(2, init_kwargs=dict(n_init=2, max_iter=1000, init='random'))
        gmm.fit(X)
        print(gmm.means_[0])
        print(np.diag(gmm.covariances_[0])**0.5)
        print(gmm.means_[1])
        print(np.diag(gmm.covariances_[1])**0.5)
        assert np.logical_or(
            np.logical_and((np.abs(gmm.means_[0] - 10) < 0.04).all(), (np.abs(gmm.means_[1]) < 0.04).all()),
            np.logical_and((np.abs(gmm.means_[1] - 10) < 0.04).all(), (np.abs(gmm.means_[0]) < 0.04).all()))
        score = gmm.score(X)
        Y, labels = gmm.sample(10000)
        A = Y[labels == 0,:]
        B = Y[labels == 1,:]
        print(A.shape, B.shape)
        assert len(A) + len(B) == len(Y)
        if A.mean() > B.mean():
            B, A = A, B
        print(A.mean(axis=0))
        print(B.mean(axis=0))
        np.testing.assert_allclose(A.mean(axis=0), 0, atol=0.02 * D)
        np.testing.assert_allclose(B.mean(axis=0), 10, atol=0.02 * D)
        np.testing.assert_allclose(A.std(axis=0), 1, atol=0.02 * D)
        np.testing.assert_allclose(B.std(axis=0), 1, atol=0.02 * D)
        gmmsk = gmm.to_sklearn()
        for attribute in 'means_', 'precisions_cholesky_', 'weights_', 'covariances_':
            np.testing.assert_allclose(getattr(gmmsk, attribute), getattr(gmm, attribute))
        assert gmmsk.covariance_type == gmm.covariance_type
        np.testing.assert_allclose(A.mean(axis=0), 0, atol=0.02 * D)
        np.testing.assert_allclose(B.mean(axis=0), 10, atol=0.02 * D)
        np.testing.assert_allclose(A.std(axis=0), 1, atol=0.02 * D)
        np.testing.assert_allclose(B.std(axis=0), 1, atol=0.02 * D)
        
        mix = askcarl.mixture.GaussianMixture.from_sklearn(gmm)
        np.testing.assert_allclose(mix.weights, gmm.weights_)
        np.testing.assert_allclose(mix.log_weights, np.log(gmm.weights_))
        for i, comp in enumerate(mix.components):
            np.testing.assert_allclose(comp.mean, gmm.means_[i])
            np.testing.assert_allclose(comp.cov, gmm.covariances_[i])

        gmm_ref = GaussianMixture(2)
        gmm_ref.fit(X)
        score_ref = gmm_ref.score(X)
        np.testing.assert_allclose(score, score_ref)

