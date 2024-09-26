import numpy as np
from numpy import array
from scipy.stats import norm, multivariate_normal, wishart
from scipy.integrate import dblquad
from numpy.testing import assert_allclose
from hypothesis import given, strategies as st, example, settings
from hypothesis.extra.numpy import arrays

import ggmm


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

    # Check (assumes q = 2)
    def pdf(y, x):
        return dist0.pdf(np.concatenate(([x, y], x2)))

    p1 = dblquad(pdf, -np.inf, x[0], -np.inf, x[1])[0]  # joint probability
    p2 = dblquad(pdf, -np.inf, np.inf, -np.inf, np.inf)[0]  # marginal probability

    # These should match (approximately)
    assert_allclose(dist.cdf(x1), p1/p2, atol=1e-6)
    #assert_allclose(dist.cdf(x1), 0.25772255281364065)
    #assert_allclose(p1/p2, 0.25772256555864476)

    c1 = ggmm.pdfcdf(x.reshape((1, -1)), np.array([False, False, True, True, True, True]), mean=mu, cov=A)
    #assert_allclose(mu_c, conditional_mean)
    #assert_allclose(A_c, conditional_cov)
    print("truth eval:", x1, dist.mean, dist.cov, dist.cdf(x1), c1)
    assert_allclose(dist.cdf(x1), c1, atol=1e-6)

    g = ggmm.Gaussian(mean=mu, cov=A)
    c2 = g.pdf(x.reshape((1, -1)), np.array([False, False, True, True, True, True]))
    assert_allclose(dist.cdf(x1), c2, atol=1e-6)

#@st.composite
#def diagonal_cov(draw, stdevs=arrays(np.float64, (6,), elements=st.floats(1e-6, 10))):
#    return np.diag(draw(stdevs)**2)


@settings(max_examples=10, deadline=5000)
@given(
    mu=arrays(np.float64, (6,), elements=st.floats(-10, 10)),
    x=arrays(np.float64, (6,), elements=st.floats(-10, 10)),
    stdevs=arrays(np.float64, (6,), elements=st.floats(1e-2, 10)),
    #A=
    #st.one_of(diagonal_cov,
    #    arrays(np.float64, (6,6), elements=st.floats(-10, 10)).filter(
    #    lambda A: np.std(A)>1e-6 and not (A==0).all() and not (np.diag(A)==0).any()))
)
@example(
    mu=array([0., 0., 0., 0., 0., 0.]),
    x=array([0., 0., 0., 0., 0., 0.]),
    stdevs=array([1., 1., 1., 1., 1., 1.]),
).via('discovered failure')
@example(
    mu=array([0., 0., 0., 0., 0., 0.]),
    x=array([1., 1., 1., 1., 1., 1.]),
    stdevs=array([3., 3., 3., 3., 3., 3.]),
).via('discovered failure')
@example(
    mu=array([0., 0., 0., 0., 0., 0.]),
    x=array([0., 0., 0., 0., 0., 0.]),
    stdevs=array([1., 1., 1., 1., 1., 1.]),
).via('discovered failure')
@example(
    mu=array([0., 0., 0., 0., 0., 0.]),
    x=array([0., 0., 0., 0., 0., 0.]),
    stdevs=array([6., 6., 6., 6., 6., 6.]),
).via('discovered failure')
@example(
    mu=array([0., 0., 0., 0., 0., 0.]),
    x=array([0., 0., 0., 0., 0., 0.]),
    stdevs=array([0.015625, 5.      , 5.      , 5.      , 5.      , 5.      ]),
).via('discovered failure')
@example(
    mu=array([-4., -4., -4., -4., -4., -4.]),
    x=array([0., 0., 0., 0., 0., 0.]),
    stdevs=array([0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125]),
).via('discovered failure')
def test_stackoverflow_like_examples(mu, x, stdevs):
    atol = max(stdevs) * 1e-4 + 1e-6
    A = np.diag(stdevs**2)
    #rng = np.random.default_rng(seed)

    n = 6  # dimensionality  
    qc = 4  # number of given coordinates
    q = n - qc  # number of other coordinates (must be 2 if you want check to work)
    #x = rng.random(n)  # generate values for all axes
    # the first q are the "other" coordinates for which you want the CDF
    # the rest are "given"

    #A = rng.random(size=(n, n))  # generate covariance matrix 
    A = A + A.T + np.eye(n)*n
    # mu = rng.random(n)  # generate mean
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

    a = x2
    inv_s22 = np.linalg.inv(s22)
    mu_c = mu1 + s12 @ inv_s22 @ (a - mu2)
    A_c = s11 - s12 @ inv_s22 @ s21
    dist = multivariate_normal(mean=mu_c, cov=A_c)

    # Check (assumes q = 2)
    def pdf(y, x):
        return dist0.pdf(np.concatenate(([x, y], x2)))

    p1 = dblquad(pdf, -np.inf, x[0], -np.inf, x[1])[0]  # joint probability
    p2 = dblquad(pdf, -np.inf, np.inf, -np.inf, np.inf)[0]  # marginal probability

    # These should match (approximately)
    assert_allclose(dist.cdf(x1), p1/p2, atol=atol)
    #assert_allclose(dist.cdf(x1), 0.25772255281364065)
    #assert_allclose(p1/p2, 0.25772256555864476)

    c1 = ggmm.pdfcdf(x.reshape((1, -1)), np.array([False, False, True, True, True, True]), mean=mu, cov=A)
    #assert_allclose(mu_c, conditional_mean)
    #assert_allclose(A_c, conditional_cov)
    assert_allclose(dist.cdf(x1), c1, atol=atol)

    g = ggmm.Gaussian(mean=mu, cov=A)
    c2 = g.pdf(x.reshape((1, -1)), np.array([False, False, True, True, True, True]))
    assert_allclose(dist.cdf(x1), c2, atol=atol)

def test_trivial_example():
    rng = np.random.RandomState(123)

    x = np.zeros((1, 1))
    g = ggmm.Gaussian(mean=np.zeros(1), cov=np.eye(1))
    assert_allclose(norm(0, 1).pdf(x), g.pdf(x, np.array([True])))

    print("zero")
    x = np.zeros((1, 1))
    g = ggmm.Gaussian(mean=np.zeros(1), cov=np.eye(1))
    assert_allclose(norm(0, 1).cdf(x), g.pdf(x, np.array([False])))

# Helper strategy to generate positive semi-definite covariance matrices
def random_covariance_matrix(dim, seed):
    """Returns a strategy to generate a positive semi-definite covariance matrix of shape (dim, dim)."""
    df = dim  # degrees of freedom (at least dim for valid positive definite matrix)
    scale = np.eye(dim)  # scale matrix, here an identity matrix
    return wishart.rvs(df=df, scale=scale, random_state=np.random.RandomState(seed)).reshape((dim, dim))


# Strategy to generate arbitrary dimensionality mean and covariance
@st.composite
def mean_and_cov(draw):
    dim = draw(st.integers(min_value=1, max_value=10))  # Arbitrary dimensionality
    mu = draw(arrays(np.float64, (dim,), elements=st.floats(-10, 10)))  # Mean vector
    seed = draw(st.integers(min_value=1, max_value=100))
    cov = random_covariance_matrix(dim, seed)  # Covariance matrix
    return dim, mu, cov


@given(mean_and_cov())
def test_single(mean_cov):
    # a ggmm with one component must behave the same as a single gaussian
    ndim, mu, cov = mean_cov
    assert mu.shape == (ndim,), (mu, mu.shape, ndim)
    assert cov.shape == (ndim,ndim), (cov, cov.shape, ndim)
    
    # a ggmm with one component must behave the same as a single gaussian
    
    rv = ggmm.Gaussian(mu, cov)
    rv_truth = multivariate_normal(mu, cov)

    xi = np.random.randn(1, len(mu))  # A random vector of same dimensionality as `mu`
    assert np.allclose(rv.pdf(xi), rv_truth.pdf(xi[0]))

@st.composite
def mean_and_diag_stdevs2(draw):
    # at least 2 dimensions
    dim = draw(st.integers(min_value=2, max_value=10))
    mu = draw(arrays(np.float64, (dim,), elements=st.floats(-10, 10)))  # Mean vector
    seed = draw(st.integers(min_value=1, max_value=100))
    rng = np.random.RandomState(seed)
    stdevs = 1. / rng.uniform(size=dim)
    return dim, mu, stdevs

#@given(mean_and_diag_stdevs2())
#@example(
#    mean_cov=(2, np.array([0., 0.]), np.array([1.  , 1.])),
#).via('discovered failure')
#@example(
#    mean_cov=(3,
#     np.array([0., 0., 0.]),
#     np.array([2.39795500e+00, 1.38826322e+00, 8.74318336e+03])),  # or any other generated value
#).via('discovered failure')
def test_single_with_UL(mean_cov=(2, np.zeros(2), np.ones(2))):
    ndim, mu, stdevs = mean_cov
    cov = np.diag(stdevs)
    assert mu.shape == (ndim,), (mu, mu.shape, ndim)
    assert cov.shape == (ndim,ndim), (cov, cov.shape, ndim)

    # a ggmm with one component must behave the same as a single gaussian
    rv = ggmm.Gaussian(mu, cov)
    rv_truth = multivariate_normal(mu[1:], np.diag(stdevs[1:]))
    #rv_truth = multivariate_normal(mu[1:], np.diag(stdevs[1:]))

    #xi = np.arange(ndim).reshape((1, -1)) * np.ones((2, 1))
    xi = np.zeros((2, ndim))
    # set high upper limit
    xi[0,0] = 1e200
    xi[1,0] = -1e200
    mask = np.ones(ndim, dtype=bool)
    mask[0] = False
    pa = rv.pdf(xi, mask)
    #pa_expected = np.array([[1,0]]) * rv_truth.pdf(xi[:,mask])
    pa_expected = rv_truth.pdf(xi[:,mask])
    print("for expectation:", xi[:,mask], mu[1:], np.diag(stdevs[1:]), pa_expected)
    #print("Expected:", pa_expected)
    # pa_expected = 1 * rv_truth.pdf(xi)
    assert_allclose(pa, pa_expected)
"""
    # set very low upper limit
    xi[0] = -1e200
    pb = rv.pdf(xi, mask)
    pb_expected = 0 * rv_truth.pdf(xi[:,1:])
    assert np.allclose(pb, pa_expected)
"""



