import numpy as np
from sklearn.mixture import GaussianMixture

from askcarl.lightgmm import LightGMM

import askcarl.mixture
    
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
        np.logical_or(
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
