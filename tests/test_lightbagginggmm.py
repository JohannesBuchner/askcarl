import numpy as np

from askcarl.lightgmm import LightGMM, LightBaggingGMM

def test_bagging():
    np.random.seed(234)
    N = 100000
    D = 10
    X = np.vstack((np.random.normal(size=(N, D)) + 10, np.random.normal(size=(N, D))))

    bgmm = LightBaggingGMM(10, n_components=2, init_kwargs=dict(n_init=1, max_iter=1000, init='random'))
    bgmm.fit(X)
    gmm = LightGMM(2, init_kwargs=dict(n_init=1, max_iter=1000, init='random'))
    gmm.fit(X)

    assert np.mean(bgmm.score(X)) >= np.mean(gmm.score(X)) - 0.0000001
    print(bgmm)
