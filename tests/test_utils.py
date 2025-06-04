import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal

from askcarl.utils import mvn_logpdf, mvn_pdf, cov_to_prec_cholesky, is_positive_definite

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

def test_gauss_variations2d():
    for shape in 2, (1, 2), (10, 2), 20, (21, 41):
        print()
        print("====", shape)
        print()
        for i in range(50):
            x = np.random.normal(size=shape)
            D = x.shape[-1]
            mean = np.random.normal(size=D)
            cov = np.eye(D)
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
    
