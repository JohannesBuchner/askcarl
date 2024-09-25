import numpy as np
from scipy.stats import norm, multivariate_normal, wishart
from scipy.integrate import dblquad
from numpy.testing import assert_allclose
from hypothesis import given, strategies as st
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
    s11 = A[:q, :q]  # partition covariance matrix
    s12 = A[:q, q:]
    s21 = A[q:, :q]
    s22 = A[q:, q:]
    mu1 = mu[:q]  # partition mean
    mu2 = mu[q:]
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
    assert_allclose(dist.cdf(x1), p1/p2)
    assert_allclose(dist.cdf(x1), 0.25772255281364065)
    assert_allclose(p1/p2, 0.25772256555864476)
    
    c1 = ggmm.pdfcdf(x.reshape((1, -1)), np.array([True, True, False, False, False, False]), mean=mu, cov=A)
    #assert_allclose(mu_c, conditional_mean)
    #assert_allclose(A_c, conditional_cov)
    assert_allclose(dist.cdf(x1), c1)

    g = ggmm.Gaussian(mean=mu, cov=A)
    g.pdf(x.reshape((1, -1)), np.array([True, True, False, False, False, False]))


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
    #assert np.allclose(rv.cdf(xi), rv_truth.cdf(xi[0]))

    #x = np.random.randn(10, len(mu))  # A random vector of same dimensionality as `mu`
    #assert np.allclose(rv.pdf(x), rv_truth.pdf(x))
    #assert np.allclose(rv.cdf(x), rv_truth.cdf(x))




