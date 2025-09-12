import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal
from hypothesis import given, strategies as st, example, settings
from hypothesis.extra.numpy import arrays

from askcarl.utils import cov_to_prec_cholesky, mvn_logpdf, mvn_pdf, is_positive_definite

import jax
jax.config.update("jax_enable_x64", True)

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

# Strategy to generate arbitrary dimensionality mean and covariance
@st.composite
def mean_and_cov(draw):
    dim = draw(st.integers(min_value=1, max_value=10))  # Arbitrary dimensionality
    mu = draw(arrays(np.float64, (dim,), elements=st.floats(-10, 10)))  # Mean vector
    eigval = draw(arrays(np.float64, (dim,), elements=st.floats(1e-6, 10)))
    vectors = draw(arrays(np.float64, (dim,dim), elements=st.floats(-10, 10)).filter(valid_QR))
    cov = make_covariance_matrix_via_QR(eigval, vectors)
    return dim, mu, cov

def test_gauss_simple():
    mean = np.ones(2)
    cov = np.eye(2)
    x = np.ones(2)
    prec_chol = cov_to_prec_cholesky(cov)
    logpdf_value = multivariate_normal(mean, cov).logpdf(x)
    logpdf_value2 = mvn_logpdf(x, mean, prec_chol)
    assert_allclose(logpdf_value, logpdf_value2)
    pdf_value = multivariate_normal(mean, cov).pdf(x)
    pdf_value2 = mvn_pdf(x, mean, prec_chol)
    assert_allclose(pdf_value, pdf_value2)

@settings(max_examples=500, deadline=None)
@given(mean_and_cov())
@example(mean_cov=(1, np.array([0.0]), np.array([[2.0]]))).via("discovered failure")
@example(mean_cov=(2, np.zeros(2), np.diag([1.25, 1.25]))).via("discovered failure")
@example(mean_cov=(2, np.zeros(2), np.array([[1.25, 0.75], [0.75, 1.25]]))).via(
    "discovered failure"
)
@example(
    mean_cov=(
        4,
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array(
            [
                [25.00000008, 16.66666663, -16.66666663, 16.66666663],
                [16.66666663, 11.11114238, -11.11109238, 11.11109863],
                [-16.66666663, -11.11109238, 11.11114238, -11.11109863],
                [16.66666663, 11.11109863, -11.11109863, 11.11113613],
            ]
        ),
    )
).via("discovered failure")
@example(
    mean_cov=(
        3,
        np.array([0.0, 0.0, 0.0]),
        np.array(
            [
                [8.0, 0.5, -0.5],
                [0.5, 8.03125002, 7.96874998],
                [-0.5, 7.96874998, 8.03125002],
            ]
        ),
    )
).via("discovered failure")
@example(
    mean_cov=(
        4,
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array(
            [
                [1.64387881, 0.5668371, 1.53617464, -0.72045457],
                [0.5668371, 2.12555721, 0.72270911, 0.94279687],
                [1.53617464, 0.72270911, 1.45482809, -0.55412942],
                [-0.72045457, 0.94279687, -0.55412942, 2.25324107],
            ]
        ),
    )
).via("discovered failure")
@example(
    mean_cov=(
        5,
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array(
            [
                [6.89197531, 2.4537037, 1.22839506, -0.57407407, 1.49382716],
                [2.4537037, 4.69444445, 1.24074074, 3.61111111, -3.07407407],
                [1.22839506, 1.24074074, 6.34567901, 2.18518519, 0.09876543],
                [-0.57407407, 3.61111111, 2.18518519, 6.77777778, -0.51851852],
                [1.49382716, -3.07407407, 0.09876543, -0.51851852, 7.45679012],
            ]
        ),
    )
).via("discovered failure")
def test_mvn_logpdf(mean_cov):
    # a askcarl with one component must behave the same as a single gaussian
    ndim, mu, cov = mean_cov
    if not valid_covariance_matrix(cov):
        return

    rv_truth0 = multivariate_normal(mu * 0, cov)
    logpdf0 = mvn_logpdf(mu * 0, mu * 0, cov_to_prec_cholesky(cov))
    assert_allclose(logpdf0, rv_truth0.logpdf(mu * 0), atol=1e-6, rtol=1e-6)

    rv_truth = multivariate_normal(mu, cov)
    xi = np.random.randn(1, len(mu))  # A random vector of same dimensionality as `mu`
    logpdf1 = mvn_logpdf(xi, mu, cov_to_prec_cholesky(cov))
    assert_allclose(logpdf1, rv_truth.logpdf(xi[0]), atol=3e-6, rtol=1e-5)

def test_gauss_variations2d():
    for shape in 2, (1, 2), (10, 2), 20, (21, 41):
        print()
        print("====", shape)
        print()
        for i in range(50):
            x = np.random.normal(size=shape)
            D = x.shape[-1]
            mean = np.random.normal(size=D)
            cov = np.eye(D) * 3.14
            prec_chol = cov_to_prec_cholesky(cov)
            logpdf_value = multivariate_normal(mean, cov).logpdf(x)
            logpdf_value2 = mvn_logpdf(x, mean, prec_chol)
            assert_allclose(logpdf_value, logpdf_value2, atol=1e-6, rtol=1e-6)
            pdf_value = multivariate_normal(mean, cov).pdf(x)
            pdf_value2 = mvn_pdf(x, mean, prec_chol)
            assert_allclose(pdf_value, pdf_value2, atol=1e-6, rtol=1e-6)

def test_example0():
    mean = np.zeros(4)
    cov = np.array([
        [ 18.23896342, -12.66610018,   8.70397287, -16.98053498],
        [-12.66610018,  34.73512046, -16.51269634,  16.95676303],
        [  8.70397287, -16.51269634,  25.34338581, -16.98053506],
        [-16.98053498,  16.95676303, -16.98053506,  39.98522342]])
    x = np.zeros((1, 4))
    prec_chol = cov_to_prec_cholesky(cov)
    logpdf_value = multivariate_normal(mean, cov).logpdf(x)
    logpdf_value2 = mvn_logpdf(x, mean, prec_chol)
    assert_allclose(logpdf_value, logpdf_value2, atol=1e-6)
    pdf_value = multivariate_normal(mean, cov).pdf(x)
    pdf_value2 = mvn_pdf(x, mean, prec_chol)
    assert_allclose(pdf_value, pdf_value2, atol=1e-6)



def test_example():
    #mean = np.array([10.        ,  0.        ,  6.64641649, -1.1       ])
    mean = np.zeros(4)
    cov = np.array([
        [ 18.23896342, -12.66610018,   8.70397287, -16.98053498],
        [-12.66610018,  34.73512046, -16.51269634,  16.95676303],
        [  8.70397287, -16.51269634,  25.34338581, -16.98053506],
        [-16.98053498,  16.95676303, -16.98053506,  39.98522342]])
    x = np.ones((1, 4))
    prec_chol = cov_to_prec_cholesky(cov)
    logpdf_value = multivariate_normal(mean, cov).logpdf(x)
    logpdf_value2 = mvn_logpdf(x, mean, prec_chol)
    assert_allclose(logpdf_value, logpdf_value2, atol=1e-6, rtol=1e-3)
    pdf_value = multivariate_normal(mean, cov).pdf(x)
    pdf_value2 = mvn_pdf(x, mean, prec_chol)
    assert_allclose(pdf_value, pdf_value2, atol=1e-6, rtol=1e-3)


def test_is_positive_definite():
    assert is_positive_definite(np.eye(2))
    assert not is_positive_definite(np.zeros((2, 2)))
    
