import os
import numpy as np
from scipy.stats import multivariate_normal
from numpy.testing import assert_allclose

from askcarl.gaussian import Gaussian
from askcarl.utils import cov_to_prec_cholesky, mvn_logpdf, mvn_pdf, is_positive_definite

import jax
jax.config.update("jax_enable_x64", True)

def test_data():
    for field in 'COSMOS', 'eFEDS':
        for z in 'z0.002-0.1 z0.1-0.3 z0.3-0.5 z0.5-0.75 z0.75-1 z1-1.25 z1.25-1.5 z1.5-1.8 z1.8-2.1 z2.1-2.7 z2.7-3.2 z3.2-4 z4-7'.split(' '):
            filename = os.path.join(os.path.dirname(__file__), f'data/gaussian-dump_{field}_{z}.npz')
            if not os.path.exists(filename):
                continue
            data = np.load(filename)
            x = data['x']
            mask = np.isfinite(x)
            assert mask.all()
            for mean, cov in zip(data['means'], data['covs']):
                refrv = multivariate_normal(mean, cov)
                cond = np.linalg.cond(cov)
                tol=1e-10
                condthresh=1e10
                print('cond:', cond)
                assert cond < condthresh, cond
                assert np.all(np.linalg.eigvalsh(cov) > tol), np.linalg.eigvalsh(cov)
                assert is_positive_definite(cov, condthresh=condthresh)

                logpdf1 = mvn_logpdf(x, mean, cov_to_prec_cholesky(cov))
                assert_allclose(logpdf1, refrv.logpdf(x))

                pdf1 = mvn_pdf(x, mean, cov_to_prec_cholesky(cov))
                assert_allclose(pdf1, refrv.pdf(x), atol=1e-6)

                g = Gaussian(mean, cov, cov_to_prec_cholesky(cov))
                pdf = g.pdf(x, mask)
                assert_allclose(pdf, refrv.pdf(x), atol=1e-6)
                logpdf = g.logpdf(x, mask)
                assert_allclose(logpdf, refrv.logpdf(x), atol=1e-6)

                g2 = Gaussian(mean, cov)
                assert_allclose(g.pdf(x, mask), g2.pdf(x, mask), atol=1e-6)
                assert_allclose(g.logpdf(x, mask), g2.logpdf(x, mask), atol=1e-6)
