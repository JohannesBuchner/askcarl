import numpy as np
from numpy import array
from scipy.stats import norm, multivariate_normal
from scipy.integrate import dblquad
from scipy.special import logsumexp
from numpy.testing import assert_allclose
from hypothesis import given, strategies as st, example, settings, HealthCheck, reproduce_failure
from hypothesis.extra.numpy import arrays
import pypmc.density.mixture
import pytest
import sklearn.mixture

import askcarl


def test_stackoverflow_example():
    rng = np.random.default_rng(238492432)

    n = 6  # dimensionality  
    qc = 4  # number of given coordinates
    q = n - qc  # number of other coordinates (must be 2 if you want check to work)
    x = rng.random(n)  # generate values for all axes
    # the first q are the "other" coordinates for which you want the CDF
    # the rest are "given"

    A = rng.random(size=(n, n))  # generate covariance matrix 
    A = A + A.T + np.eye(n)*n
    mu = rng.random(n)  # generate mean
    dist0 = multivariate_normal(mean=mu, cov=A)

    # Generate MVN conditioned on x[q:] 
    # partition covariance matrix
    s11 = A[:q, :q]  # upper bound covariance
    s12 = A[:q, q:]  # mixed 1
    s21 = A[q:, :q]  # mixed 2
    s22 = A[q:, q:]  # given value covariance
    # partition mean
    mu1 = mu[:q]  # upper bound mean
    mu2 = mu[q:]  # given values mean
    x1 = x[:q]  # "other" values
    x2 = x[q:]  # given values

    print("input: upper", x1, mu1, "given", x2, mu2)
    print("cov_cross:", s12, s21)

    a = x2
    inv_s22 = np.linalg.inv(s22)
    print("inv_s22:", qc, inv_s22, x2)
    assert inv_s22.shape == (qc, qc)
    print((s12 @ inv_s22 @ (a - mu2)).shape)
    mu_c = mu1 + s12 @ inv_s22 @ (a - mu2)
    assert mu_c.shape == (q,)
    print("newcov shape:", (s12 @ inv_s22 @ s21).shape, s12 @ inv_s22 @ s21)
    A_c = s11 - s12 @ inv_s22 @ s21
    assert A_c.shape == (q, q)
    dist = multivariate_normal(mean=mu_c, cov=A_c)
    print("truth:", mu_c, A_c)
    pdf_part = multivariate_normal(mean=mu2, cov=s22).pdf(x2)
    logpdf_part = multivariate_normal(mean=mu2, cov=s22).logpdf(x2)

    # Check (assumes q = 2)
    def pdf(y, x):
        return dist0.pdf(np.concatenate(([x, y], x2)))

    p1 = dblquad(pdf, -np.inf, x[0], -np.inf, x[1])[0]  # joint probability
    p2 = dblquad(pdf, -np.inf, np.inf, -np.inf, np.inf)[0]  # marginal probability

    print("comparison:", p1, p2, dist.cdf(x1), pdf_part)
    # These should match (approximately)
    assert_allclose(dist.cdf(x1) * pdf_part, p1, atol=1e-6)
    #assert_allclose(dist.cdf(x1), 0.25772255281364065)
    #assert_allclose(p1/p2, 0.25772256555864476)

    c1 = askcarl.pdfcdf(x.reshape((1, -1)), np.array([False, False, True, True, True, True]), mean=mu, cov=A)
    #assert_allclose(mu_c, conditional_mean)
    #assert_allclose(A_c, conditional_cov)
    print("truth eval:", x1, dist.mean, dist.cov, dist.cdf(x1), c1)
    assert_allclose(dist.cdf(x1) * pdf_part, c1, atol=1e-6)

    g = askcarl.Gaussian(mean=mu, cov=A)
    c2 = g.conditional_pdf(x.reshape((1, -1)), np.array([False, False, True, True, True, True]))
    assert_allclose(dist.cdf(x1) * pdf_part, c2, atol=1e-6)

    logc2 = g.conditional_logpdf(x.reshape((1, -1)), np.array([False, False, True, True, True, True]))
    assert_allclose(dist.logcdf(x1) + logpdf_part, logc2)

def valid_QR(vectors):
    q, r = np.linalg.qr(vectors)
    return q.shape == vectors.shape and np.all(np.abs(np.diag(r)) > 1e-3) and np.all(np.abs(np.diag(r)) < 1000)

def make_covariance_matrix_via_QR(normalisations, vectors):
    q, r = np.linalg.qr(vectors)
    orthogonal_vectors = q @ np.diag(np.diag(r))
    cov = orthogonal_vectors @ np.diag(normalisations) @ orthogonal_vectors.T
    return cov

def valid_covariance_matrix(A, min_std=1e-6):
    if not np.isfinite(A).all():
        return False
    #if not np.std(A) > min_std:
    #    return False
    if (np.diag(A) <= min_std).any():
        return False

    try:
        np.linalg.inv(np.linalg.inv(A))
    except np.linalg.LinAlgError:
        return False

    try:
        multivariate_normal(mean=np.zeros(len(A)), cov=A)
    except ValueError:
        return False

    return True

@settings(max_examples=100, deadline=None)
@given(
    mu=arrays(np.float64, (6,), elements=st.floats(-10, 10)),
    x=arrays(np.float64, (6,), elements=st.floats(-10, 10)),
    eigval=arrays(np.float64, (6,), elements=st.floats(1e-6, 10)),
    vectors=arrays(np.float64, (6,6), elements=st.floats(-10, 10)).filter(valid_QR),
)
@example(
    mu=array([ 0.5    , -9.     , -2.     ,  0.99999,  0.99999,  0.99999]),
    x=array([0.00000000e+00, 3.86915453e+00, 0.00000000e+00, 0.00000000e+00,
           0.00000000e+00, 1.00000000e-05]),
    eigval=array([1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06]),
    vectors=array([[ 0.00000000e+00, -2.00000000e+00, -2.00000000e+00,
            -2.00000000e+00, -2.00000000e+00, -2.00000000e+00],
           [-2.00000000e+00, -2.00000000e+00, -2.00000000e+00,
            -1.17549435e-38, -2.00000000e+00, -2.00000000e+00],
           [-2.00000000e+00, -2.00000000e+00, -2.00000000e+00,
            -2.00000000e+00, -2.00000000e+00, -2.00000000e+00],
           [-2.00000000e+00, -2.00000000e+00, -2.00000000e+00,
            -2.00000000e+00, -1.40129846e-45, -2.00000000e+00],
           [-2.00000000e+00,  3.33333333e-01, -2.00000000e+00,
            -2.00000000e+00, -2.00000000e+00, -2.00000000e+00],
           [-2.00000000e+00, -2.00000000e+00,  5.00000000e-01,
            -2.00000000e+00, -2.00000000e+00, -2.00000000e+00]]),
).via('discovered failure')
@example(
    mu=array([ 0., -9., 10.,  0.,  0.,  0.]),
    x=array([0., 4., 0., 0., 0., 0.]),
    eigval=array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    vectors=array([[0., 1., 1., 1., 1., 1.],
           [1., 1., 1., 0., 1., 1.],
           [1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 0., 1.],
           [1., 0., 1., 1., 1., 1.],
           [1., 1., 0., 1., 1., 1.]]),
).via('discovered failure')
@example(
    mu=array([0.     , 1.     , 0.5    , 0.03125, 1.     , 0.03125]),
    x=array([ 1.00000000e+01,  6.10351562e-05, -4.24959109e+00,  3.26712313e+00,
           -1.00000000e+01,  0.00000000e+00]),
    eigval=array([0.5, 1. , 0.5, 0.5, 0.5, 0.5]),
    vectors=array([[ 0.  ,  0.  ,  0.  , -0.25, -0.25, -0.25],
           [-0.25, -0.25, -0.25, -0.25,  0.  , -0.25],
           [ 0.  , -0.25, -0.25, -0.25, -0.25, -0.25],
           [-0.25, -0.25, -0.25,  0.  , -0.25, -0.25],
           [-0.25, -0.25,  0.  , -0.25, -0.25, -0.25],
           [-0.25, -0.25, -0.25, -0.25, -0.25, -0.25]]),
).via('discovered failure')
@example(
    mu=array([ 0.        ,  0.        , 10.        ,  0.        ,  6.64641649,
           -1.1       ]),
    x=array([10., 10., 10., 10., 10., 10.]),
    eigval=array([1.00000000e-06, 1.00000000e+00, 4.16143782e-01, 1.00000000e-06,
           6.99209529e-01, 2.42501010e-01]),
    vectors=array([[-1.00000000e-005, -5.67020051e+000, -5.67020051e+000,
            -5.67020051e+000, -5.67020051e+000, -5.67020051e+000],
           [-5.67020051e+000, -5.67020051e+000, -5.67020051e+000,
            -5.67020051e+000, -5.67020051e+000,  1.90000000e+000],
           [-5.67020051e+000, -5.67020051e+000, -5.67020051e+000,
             5.00000000e-001, -5.67020051e+000, -5.67020051e+000],
           [-5.67020051e+000, -1.19209290e-007,  2.22044605e-016,
            -5.67020051e+000, -5.67020051e+000, -5.67020051e+000],
           [-5.67020051e+000, -5.67020051e+000, -5.67020051e+000,
            -5.67020051e+000, -5.67020051e+000, -5.67020051e+000],
           [-5.67020051e+000,  1.11253693e-308, -5.67020051e+000,
            -5.67020051e+000, -5.67020051e+000, -1.17549435e-038]]),
).via('discovered failure')
def test_stackoverflow_like_examples(mu, x, eigval, vectors):
    A = make_covariance_matrix_via_QR(eigval, vectors)
    print("Cov:", A)
    stdevs = np.diag(A)**0.5
    print("stdevs:", stdevs)
    atol = max(stdevs) * 1e-4 * (1 + np.abs(x - mu).max()) + 1e-6
    print("atol:", atol)
    if not valid_covariance_matrix(A):
        return
    n = 6  # dimensionality  
    qc = 4  # number of given coordinates
    q = n - qc  # number of other coordinates (must be 2 if you want check to work)
    # the first q are the "other" coordinates for which you want the CDF
    # the rest are "given"

    A = A + A.T + np.eye(n)*n
    dist0 = multivariate_normal(mean=mu, cov=A)

    # Generate MVN conditioned on x[q:] 
    # partition covariance matrix
    s11 = A[:q, :q]  # upper bound covariance
    s12 = A[:q, q:]  # mixed 1
    s21 = A[q:, :q]  # mixed 2
    s22 = A[q:, q:]  # given value covariance
    # partition mean
    mu1 = mu[:q]  # upper bound mean
    mu2 = mu[q:]  # given values mean
    x1 = x[:q]  # "other" values
    x2 = x[q:]  # given values

    print("input: upper", x1, mu1, "given", x2, mu2)
    print("cov_cross:", s12, s21)

    a = x2
    inv_s22 = np.linalg.inv(s22)
    print("inv_s22:", qc, inv_s22, x2)
    assert inv_s22.shape == (qc, qc)
    print((s12 @ inv_s22 @ (a - mu2)).shape)
    mu_c = mu1 + s12 @ inv_s22 @ (a - mu2)
    assert mu_c.shape == (q,)
    print("newcov shape:", (s12 @ inv_s22 @ s21).shape, s12 @ inv_s22 @ s21)
    A_c = s11 - s12 @ inv_s22 @ s21
    assert A_c.shape == (q, q)
    dist = multivariate_normal(mean=mu_c, cov=A_c)
    print("truth:", mu_c, A_c)
    pdf_part = multivariate_normal(mean=mu2, cov=s22).pdf(x2)

    # Check (assumes q = 2)
    def pdf(y, x):
        return dist0.pdf(np.concatenate(([x, y], x2)))

    p1 = dblquad(pdf, -np.inf, x[0], -np.inf, x[1])[0]  # joint probability

    print("p1:", p1) #, "p2:", p2)
    # These should match (approximately)
    assert_allclose(dist.cdf(x1) * pdf_part, p1, atol=atol, rtol=1e-2)

    c1 = askcarl.pdfcdf(x.reshape((1, -1)), np.array([False, False, True, True, True, True]), mean=mu, cov=A)
    assert_allclose(dist.cdf(x1) * pdf_part, c1, atol=atol)

    g = askcarl.Gaussian(mean=mu, cov=A)
    c2 = g.conditional_pdf(x.reshape((1, -1)), np.array([False, False, True, True, True, True]))
    assert_allclose(dist.cdf(x1) * pdf_part, c2, atol=atol)

def test_trivial_example():
    x = np.zeros((1, 1))
    g = askcarl.Gaussian(mean=np.zeros(1), cov=np.eye(1))
    assert_allclose(norm(0, 1).pdf(x), g.conditional_pdf(x, np.array([True])))

    print("zero")
    x = np.zeros((1, 1))
    g = askcarl.Gaussian(mean=np.zeros(1), cov=np.eye(1))
    assert_allclose(norm(0, 1).cdf(x), g.conditional_pdf(x, np.array([False])))


def test_trivial_mixture():
    x = np.zeros((1, 1))
    mask = np.ones((1, 1), dtype=bool)
    p_truth = norm(0, 1).pdf(x)[0]
    g = askcarl.Gaussian(mean=np.zeros(1), cov=np.eye(1))
    assert_allclose(p_truth, g.pdf(x, mask))
    mix = askcarl.GaussianMixture(means=[np.zeros(1)], covs=[np.eye(1)], weights=[1.0])
    assert_allclose(p_truth, mix.pdf(x, mask))

# Strategy to generate arbitrary dimensionality mean and covariance
@st.composite
def mean_and_cov(draw):
    dim = draw(st.integers(min_value=1, max_value=10))  # Arbitrary dimensionality
    mu = draw(arrays(np.float64, (dim,), elements=st.floats(-10, 10)))  # Mean vector
    eigval = draw(arrays(np.float64, (dim,), elements=st.floats(1e-6, 10)))
    vectors = draw(arrays(np.float64, (dim,dim), elements=st.floats(-10, 10)).filter(valid_QR))
    cov = make_covariance_matrix_via_QR(eigval, vectors)
    return dim, mu, cov


@given(mean_and_cov())
def test_single(mean_cov):
    # a askcarl with one component must behave the same as a single gaussian
    ndim, mu, cov = mean_cov
    if not valid_covariance_matrix(cov):
        return
    assert mu.shape == (ndim,), (mu, mu.shape, ndim)
    assert cov.shape == (ndim,ndim), (cov, cov.shape, ndim)
    
    # a askcarl with one component must behave the same as a single gaussian
    
    rv = askcarl.Gaussian(mu, cov)
    rv_truth = multivariate_normal(mu, cov)

    xi = np.random.randn(1, len(mu))  # A random vector of same dimensionality as `mu`
    assert_allclose(rv.conditional_pdf(xi), rv_truth.pdf(xi[0]))
    assert_allclose(rv.conditional_pdf(xi, np.array([True] * ndim)), rv_truth.pdf(xi[0]))

    assert_allclose(rv.pdf(xi, np.array([[True] * ndim])), rv_truth.pdf(xi[0]))

    assert_allclose(rv.logpdf(xi, np.array([[True] * ndim])), rv_truth.logpdf(xi[0]))

@st.composite
def mean_and_diag_stdevs2(draw):
    # at least 2 dimensions
    dim = draw(st.integers(min_value=2, max_value=10))
    mu = draw(arrays(np.float64, (dim,), elements=st.floats(-1e6, 1e6)))  # Mean vector
    stdevs = draw(arrays(np.float64, (dim,), elements=st.floats(1e-6, 1e6)))
    x = draw(arrays(np.float64, (dim,), elements=st.floats(-1e6, 1e6)))
    i = draw(st.integers(min_value=0, max_value=dim - 1))
    return dim, mu, stdevs, x, i


@given(mean_and_diag_stdevs2())
@example(
    mean_and_cov=(2, array([1., 0.]), array([1., 1.]), array([0., 0.]), 1),
).via('discovered failure')
@example(
    mean_and_cov=(2, array([0., 0.]), array([2., 2.]), array([77., 77.]), 0),
).via('discovered failure')
@example(
    mean_and_cov=(2, array([0., 0.]), array([1., 1.]), array([39., 39.]), 0),
).via('discovered failure')
def test_single_with_UL(mean_and_cov):
    ndim, mu, stdevs, x, i = mean_and_cov
    cov = np.diag(stdevs**2)
    assert mu.shape == (ndim,), (mu, mu.shape, ndim)
    assert cov.shape == (ndim,ndim), (cov, cov.shape, ndim)
    if not valid_covariance_matrix(cov):
        return

    # a askcarl with one component must behave the same as a single gaussian
    print("inputs:", mu, stdevs, cov)
    rv = askcarl.Gaussian(mu, cov)

    mask = np.ones(ndim, dtype=bool)
    mask[i] = False
    rv_truth = multivariate_normal(mu[mask], np.diag(stdevs[mask]**2))

    xi = np.array([x, x])
    assert 0 <= i < ndim
    # set high/low upper limit
    xi[0,i] = 1e200
    xi[1,i] = -1e200
    pa = rv.conditional_pdf(xi, mask)
    pa_expected = np.array([1, 0]) * rv_truth.pdf(xi[:,mask])
    # pa_expected = rv_truth.pdf(xi[:,mask])
    print("for expectation:", xi[0,mask], mu[mask], stdevs[mask], pa, pa_expected)
    #print("Expected:", pa_expected)
    # pa_expected = 1 * rv_truth.pdf(xi)
    assert_allclose(pa, pa_expected)
    pb = rv.pdf(xi, np.array([mask,mask]))
    assert_allclose(pb, pa_expected)
    logpa_expected = np.array([0, -np.inf]) + rv_truth.logpdf(xi[:,mask])
    logpa = rv.logpdf(xi, np.array([mask,mask]))
    assert_allclose(logpa, logpa_expected)

@pytest.mark.parametrize("n_components", [1, 3, 10])
@pytest.mark.parametrize("covariance_type", ['full', 'tied', 'diag', 'spherical'])
def test_import(n_components, covariance_type):
    a = np.vstack((
        np.random.normal(3, 3, size=(10000, 3)),
        np.random.normal(0, 1, size=(3000, 3)),
        np.random.normal(3, 1, size=(10000, 3)),
    ))
    assert a.shape == (23000, 3), a.shape
    skgmm = sklearn.mixture.GaussianMixture(
        n_components=n_components, covariance_type=covariance_type)
    skgmm.fit(a)
    askcarl_fromsklearn = askcarl.GaussianMixture.from_sklearn(skgmm)
    
    means = [g.mean for g in askcarl_fromsklearn.components]
    covs = [g.cov for g in askcarl_fromsklearn.components]
    print(means)
    print([np.diag(cov) for cov in covs])
    if covariance_type in ('full', 'diag') and n_components == 3:
        assert any(np.allclose(mean, 3, atol=0.1) for mean in means)
        assert any(np.allclose(mean, 0, atol=0.1) for mean in means)
    
    target_mixture = pypmc.density.mixture.create_gaussian_mixture(
        means, covs, askcarl_fromsklearn.weights)
    askcarl_frompypmc = askcarl.GaussianMixture.from_pypmc(target_mixture)

    means2 = [g.mean for g in askcarl_frompypmc.components]
    covs2 = [g.cov for g in askcarl_frompypmc.components]

    assert_allclose(means2, means)
    assert_allclose(covs2, covs)

@st.composite
def mixture_strategy(draw):
    dim = draw(st.integers(min_value=1, max_value=10))
    ntest = draw(st.integers(min_value=1, max_value=10))
    ncomponents = draw(st.integers(min_value=1, max_value=10))
    means = [draw(arrays(np.float64, (dim,), elements=st.floats(-10, 10))) for _ in range(ncomponents)]
    covs = [make_covariance_matrix_via_QR(
        draw(arrays(np.float64, (dim,), elements=st.floats(1e-6, 10))),
        draw(arrays(np.float64, (dim,dim), elements=st.floats(-10, 10)).filter(valid_QR))
    ) for _ in range(ncomponents)]
    weights = draw(arrays(np.float64, (ncomponents,), elements=st.floats(0, 1)).filter(lambda weights: (weights>0).any()))
    weights /= weights.sum()
    x = draw(arrays(np.float64, (ntest, dim), elements=st.floats(-10, 10)))
    return dim, ncomponents, means, covs, weights, x

from  sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob

@settings(suppress_health_check=[HealthCheck.filter_too_much], max_examples=1000, deadline=None)
@given(mixture_strategy())
@example(
    mixture=(5,
     2,
     [array([0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0.])],
     [array([[ 1.49244777, -0.07387876, -0.02245019, -0.25755223,  0.35388117],
             [-0.07387876,  1.96311104,  0.58596818,  0.48862124,  0.96229954],
             [-0.02245019,  0.58596818,  1.89882532,  0.54004981,  0.97515668],
             [-0.25755223,  0.48862124,  0.54004981,  2.05494777,  0.91638117],
             [ 0.35388117,  0.96229954,  0.97515668,  0.91638117,  1.1461626 ]]),
      array([[0.90301624, 0.46133735, 0.34701563, 0.13561725, 0.34582217],
             [0.46133735, 0.55336326, 0.4630311 , 0.36990653, 0.46231502],
             [0.34701563, 0.4630311 , 0.89632843, 0.14126309, 0.3479933 ],
             [0.13561725, 0.36990653, 0.14126309, 0.55367394, 0.13887617],
             [0.34582217, 0.46231502, 0.3479933 , 0.13887617, 0.89943142]])],
     array([0., 1.]),
     array([[0., 7., 0., 0., 0.]])),
).via('discovered failure')
@example(
    mixture=(5,
     2,
     [array([0., 0., 0., 0., 0.]), array([0., 8., 0., 0., 0.])],
     [array([[ 1.49244777e-06, -7.38787586e-08, -2.24501872e-08,
              -2.57552228e-07,  3.53881174e-07],
             [-7.38787586e-08,  1.96311104e-06,  5.85968180e-07,
               4.88621241e-07,  9.62299541e-07],
             [-2.24501872e-08,  5.85968180e-07,  1.89882532e-06,
               5.40049813e-07,  9.75156684e-07],
             [-2.57552228e-07,  4.88621241e-07,  5.40049813e-07,
               2.05494777e-06,  9.16381174e-07],
             [ 3.53881174e-07,  9.62299541e-07,  9.75156684e-07,
               9.16381174e-07,  1.14616260e-06]]),
      array([[ 0.2641    ,  0.05401538, -0.03798462,  0.14390769, -0.08798462],
             [ 0.05401538,  0.28357929,  0.23057929,  0.16052426,  0.21807929],
             [-0.03798462,  0.23057929,  0.46757929,  0.06852426,  0.08007929],
             [ 0.14390769,  0.16052426,  0.06852426,  0.31735444,  0.01852426],
             [-0.08798462,  0.21807929,  0.08007929,  0.01852426,  0.50507929]])],
     array([0., 1.]),
     array([[0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]])),
).via('discovered failure')
@example(
    mixture=(7,
     1,
     [array([0., 0., 0., 0., 0., 0., 0.])],
     [array([[156.25637361, -31.26969455,  12.40439959, -31.24564618,
              -31.22989933, -31.24225165, -31.2688804 ],
             [-31.26969455,   6.33006684,  -2.20457805,   6.2198964 ,
                6.16861271,   6.20884123,   6.34227887],
             [ 12.40439959,  -2.20457805,   2.43400622,  -2.56530348,
               -2.80150635,  -2.61622149,  -2.21679021],
             [-31.24564618,   6.2198964 ,  -2.56530348,   7.16206397,
                5.80788728,   5.8481158 ,   6.20768424],
             [-31.22989933,   6.16861271,  -2.80150635,   5.80788728,
                7.34003101,   5.75696927,   6.15640055],
             [-31.24225165,   6.20884123,  -2.61622149,   5.8481158 ,
                5.75696927,   7.23169779,   6.19662906],
             [-31.2688804 ,   6.34227887,  -2.21679021,   6.20768424,
                6.15640055,   6.19662906,   6.36588918]])],
     array([1.]),
     array([[0., 0., 0., 0., 0., 0., 0.]])),
).via('discovered failure')
@example(
    mixture=(1, 1, [array([0.])], [array([[1.]])], array([1.]), array([[0.]])),
).via('discovered failure')
@example(
    mixture=(8,
     1,
     [array([0., 0., 0., 0., 0., 0., 0., 0.])],
     [array([[ 1.53365164e-06, -4.56057424e-07, -2.47157040e-07,
              -1.88188005e-07,  1.47416388e-07,  1.47416391e-07,
               4.49153299e-07,  1.47416391e-07],
             [-4.56057424e-07,  1.76562629e+00,  9.99999625e-01,
               9.99999679e-01, -3.26562499e+00,  1.98437501e+00,
               2.53124938e+00,  1.98437501e+00],
             [-2.47157040e-07,  9.99999625e-01,  1.00000099e+00,
               9.99999580e-01,  9.99999915e-01,  9.99999915e-01,
               1.00000006e+00,  9.99999915e-01],
             [-1.88188005e-07,  9.99999679e-01,  9.99999580e-01,
               1.00000091e+00,  9.99999927e-01,  9.99999927e-01,
               1.00000005e+00,  9.99999927e-01],
             [ 1.47416388e-07, -3.26562499e+00,  9.99999915e-01,
               9.99999927e-01,  2.47656250e+01, -4.48437497e+00,
              -7.53124995e+00, -4.48437497e+00],
             [ 1.47416391e-07,  1.98437501e+00,  9.99999915e-01,
               9.99999927e-01, -4.48437497e+00,  2.26562529e+00,
               2.96875004e+00,  2.26562479e+00],
             [ 4.49153299e-07,  2.53124938e+00,  1.00000006e+00,
               1.00000005e+00, -7.53124995e+00,  2.96875004e+00,
               4.06250038e+00,  2.96875004e+00],
             [ 1.47416391e-07,  1.98437501e+00,  9.99999915e-01,
               9.99999927e-01, -4.48437497e+00,  2.26562479e+00,
               2.96875004e+00,  2.26562529e+00]])],
     array([1.]),
     array([[0., 0., 0., 0., 0., 0., 0., 0.]])),
).via('discovered failure')
@example(
    mixture=(7, 1,
        [array([0., 0., 0., 0., 0., 0., 0.])],
        [array([[ 7.52684752, -4.9545883 , -0.31361835,  0.52269766,  0.52269929,
             6.92515709, -9.70648636],
           [-4.9545883 ,  3.26139136,  0.20644122, -0.3440688 , -0.34406688,
            -4.55852195,  6.38935566],
           [-0.31361835,  0.20644122,  0.01308347, -0.02177316, -0.0217791 ,
            -0.28854817,  0.40443697],
           [ 0.52269766, -0.3440688 , -0.02177316,  0.03630094,  0.0362984 ,
             0.48091352, -0.67406172],
           [ 0.52269929, -0.34406688, -0.0217791 ,  0.0362984 ,  0.03630217,
             0.48091543, -0.67405981],
           [ 6.92515709, -4.55852195, -0.28854817,  0.48091352,  0.48091543,
             6.37156541, -8.9305567 ],
           [-9.70648636,  6.38935566,  0.40443697, -0.67406172, -0.67405981,
            -8.9305567 , 12.51732134]])],
       array([1.]),
       array([[0., 0., 0., 0., 0., 0., 0.]])),
).via('discovered failure')
@example(
    mixture=(3,
     1,
     [array([0., 0., 0.])],
     [array([[32.00000005, 35.99999994,  8.00000004],
             [35.99999994, 40.50000006,  8.99999996],
             [ 8.00000004,  8.99999996,  2.00000005]])],
     array([1.]),
     array([[0., 0., 0.]])),
).via('discovered failure')
@example(
    mixture=(3,
     1,
     [array([0., 0., 0.])],
     [array([[32.00000007, 35.99999993,  8.00000001],
             [35.99999993, 40.50000006,  8.99999998],
             [ 8.00000001,  8.99999998,  2.00000008]])],
     array([1.]),
     array([[0., 0., 0.]])),
).via('discovered failure')
@example(
    mixture=(4,
     1,
     [array([0., 0., 0., 0.])],
     [array([[ 198.43139701,  208.34459689,  178.58081708, -148.82075727],
             [ 208.34459689,  586.13441827,  514.06354813, -278.71597676],
             [ 178.58081708,  514.06354813,  450.99268609, -242.78684396],
             [-148.82075727, -278.71597676, -242.78684396,  152.43362983]])],
     array([1.]),
     array([[0., 0., 0., 0.]])),
).via('discovered failure')
def test_mixture(mixture):
    ndim, ncomponents, means, covs, weights, x = mixture
    mask = np.ones(x.shape, dtype=bool)

    if not all([valid_covariance_matrix(cov) for cov in covs]):
        return

    print("inputs:", ndim, ncomponents, means, covs, weights, x, mask)
    gmm = askcarl.GaussianMixture(weights, means, covs)
    #assert_allclose(gmm.log_weights, np.log(weights))
    askcarl_p = gmm.pdf(x, mask=mask)
    askcarl_logp = gmm.logpdf(x, mask=mask)
    gaussians = [multivariate_normal(mean, cov) for mean, cov in zip(means, covs)]
    if len(gaussians) == 1:
        assert_allclose(gmm.log_weights, 0)
        assert_allclose(gmm.components[0].pdf(x, mask), gaussians[0].pdf(x))
        assert_allclose(gmm.components[0].logpdf(x, mask), gaussians[0].logpdf(x))
        assert_allclose(askcarl_p, gaussians[0].pdf(x))
        assert_allclose(askcarl_logp, gaussians[0].logpdf(x))

    pdf_expected = sum(w * g.pdf(x) for g, w in zip(gaussians, weights))
    logpdf_expected = logsumexp([np.log(w) + g.logpdf(x) for g, w in zip(gaussians, weights)], axis=0)
    assert_allclose(askcarl_p, pdf_expected)
    assert_allclose(askcarl_logp, logpdf_expected)

    target_mixture = pypmc.density.mixture.create_gaussian_mixture(
        means, covs, weights)
    pypmc_logp = np.array([target_mixture.evaluate(xi) for xi in x])
    assert_allclose(askcarl_p, np.exp(pypmc_logp), atol=1e-300, rtol=1e-4)
    assert_allclose(askcarl_logp[pypmc_logp>-100000], pypmc_logp[pypmc_logp>-100000], atol=1)
    assert_allclose(askcarl_logp[askcarl_logp>-100000], askcarl_logp[askcarl_logp>-100000], atol=1)

    precisions = [np.linalg.inv(cov) for cov in covs]
    # compare results of GMM to sklearn
    try:
        skgmm = sklearn.mixture.GaussianMixture(
            n_components=ncomponents, weights_init=weights,
            means_init=means, precisions_init=precisions)
        skgmm._initialize(np.zeros((1, 1)), None)
    except np.linalg.LinAlgError:
        return
    skgmm._set_parameters((weights, np.array(means), covs, skgmm.precisions_cholesky_))
    assert_allclose(skgmm.weights_, weights)
    assert_allclose(skgmm.means_, means)
    assert_allclose(skgmm.covariances_, covs)
    # compare results of GMM to pypmc
    print(x, skgmm.means_, skgmm.precisions_cholesky_)
    sk_logp = logsumexp(
        np.log(weights).reshape((1, -1)) + _estimate_log_gaussian_prob(x, skgmm.means_, skgmm.precisions_cholesky_, 'full'),
            axis=1)
    print(sk_logp)
    assert sk_logp.shape == (len(x),), (sk_logp.shape, len(x))
    print(skgmm.weights_, askcarl_logp, askcarl_p)
    sk_p = skgmm.predict_proba(x)
    # TODO: https://github.com/scikit-learn/scikit-learn/issues/29989
    # commented out for now
    # assert_allclose(askcarl_logp, sk_logp, atol=1e-2, rtol=1e-2)
    # sk_logp1, sk_logp2 = skgmm._estimate_log_prob_resp(x)
    # assert_allclose(sk_logp1, sk_logp)
    # assert_allclose(askcarl_p, sk_p, atol=1e-300, rtol=1e-4)

