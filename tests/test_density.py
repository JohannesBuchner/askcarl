import numpy as np
from numpy import array
from scipy.stats import norm, multivariate_normal
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

    c1 = ggmm.pdfcdf(x.reshape((1, -1)), np.array([False, False, True, True, True, True]), mean=mu, cov=A)
    #assert_allclose(mu_c, conditional_mean)
    #assert_allclose(A_c, conditional_cov)
    print("truth eval:", x1, dist.mean, dist.cov, dist.cdf(x1), c1)
    assert_allclose(dist.cdf(x1) * pdf_part, c1, atol=1e-6)

    g = ggmm.Gaussian(mean=mu, cov=A)
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
        np.linalg.inv(A)
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

    c1 = ggmm.pdfcdf(x.reshape((1, -1)), np.array([False, False, True, True, True, True]), mean=mu, cov=A)
    assert_allclose(dist.cdf(x1) * pdf_part, c1, atol=atol)

    g = ggmm.Gaussian(mean=mu, cov=A)
    c2 = g.conditional_pdf(x.reshape((1, -1)), np.array([False, False, True, True, True, True]))
    assert_allclose(dist.cdf(x1) * pdf_part, c2, atol=atol)

def test_trivial_example():
    x = np.zeros((1, 1))
    g = ggmm.Gaussian(mean=np.zeros(1), cov=np.eye(1))
    assert_allclose(norm(0, 1).pdf(x), g.conditional_pdf(x, np.array([True])))

    print("zero")
    x = np.zeros((1, 1))
    g = ggmm.Gaussian(mean=np.zeros(1), cov=np.eye(1))
    assert_allclose(norm(0, 1).cdf(x), g.conditional_pdf(x, np.array([False])))


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
    # a ggmm with one component must behave the same as a single gaussian
    ndim, mu, cov = mean_cov
    if not valid_covariance_matrix(cov):
        return
    assert mu.shape == (ndim,), (mu, mu.shape, ndim)
    assert cov.shape == (ndim,ndim), (cov, cov.shape, ndim)
    
    # a ggmm with one component must behave the same as a single gaussian
    
    rv = ggmm.Gaussian(mu, cov)
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

    # a ggmm with one component must behave the same as a single gaussian
    print("inputs:", mu, stdevs, cov)
    rv = ggmm.Gaussian(mu, cov)

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
