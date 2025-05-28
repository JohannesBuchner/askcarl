"""Multivariate Gaussians with support for upper limits and missing data."""

import numpy as np
from scipy.stats import multivariate_normal


def pdfcdf(x, mask, mean, cov):
    """
    Compute the mixed PDF and CDF for a multivariate Gaussian distribution.

    Parameters
    -----------
    x: array
        The point (vector) at which to evaluate the probability.
    mask: array
        A boolean mask of the same shape as `x`, indicating whether the entry
        is a value (True) or a upper bound (False).
    mean: array
        mean vector of the multivariate normal distribution.
    cov: array
        covariance matrix of the multivariate normal distribution.

    Returns
    ----------
    pdf: float
        Probability density
    """
    assert x.ndim == 2, x.ndim
    assert mask.shape == (x.shape[1],), (mask.shape, x.shape)
    assert mean.shape == (x.shape[1],), (mean.shape, x.shape)
    assert cov.shape == (x.shape[1],x.shape[1]), (cov.shape, x.shape)

    # Split x into exact values and upper bounds based on the mask
    exact_idx, = np.where(mask)  # Indices of exact values (PDF)
    upper_idx, = np.where(~mask)  # Indices of upper bounds (CDF)
    n_exact = len(exact_idx)
    n_upper = len(upper_idx)

    # Partition mean and covariance matrix accordingly
    mu_exact = mean[exact_idx]  # Mean for exact values
    mu_upper = mean[upper_idx]  # Mean for upper bounds
    cov_exact = cov[np.ix_(exact_idx, exact_idx)]  # Covariance for exact values
    cov_upper = cov[np.ix_(upper_idx, upper_idx)]  # Covariance for upper bounds
    cov_cross = cov[np.ix_(exact_idx, upper_idx)]  # Cross-covariance between exact and upper bounds

    # Extract values from x
    x_exact = x[:,exact_idx]  # Known values for the PDF
    x_upper = x[:,upper_idx]  # Upper bounds for the CDF

    # Compute the conditional mean and covariance for the remaining dimensions (upper bounds)
    if len(upper_idx) > 0:
        inv_cov_exact = np.linalg.inv(cov_exact)
        assert inv_cov_exact.shape == (n_exact, n_exact)
        newcov = np.einsum('ji,jk,mk->mi', cov_cross, inv_cov_exact, x_exact - mu_exact.reshape((1, -1)))
        assert newcov.shape == (len(x), n_upper), (newcov.shape, (len(x), n_upper))
        conditional_mean = mu_upper[None,:] + newcov
        conditional_cov = cov_upper - cov_cross.T @ inv_cov_exact @ cov_cross
        assert conditional_cov.shape == (n_upper, n_upper)
    else:
        # If there are no upper bounds, the conditional mean and cov are just the original ones
        conditional_mean = mu_exact
        conditional_cov = cov_exact

    assert conditional_mean.shape == (len(x), n_upper), (conditional_mean.shape, len(x), n_upper)
    # Create the conditional multivariate normal distributions
    dist_conditional = multivariate_normal(mean=np.zeros(len(conditional_cov)), cov=conditional_cov)  # Conditional MVN

    # Compute the CDF for the upper bounds
    if len(upper_idx) > 0:
        cdf_value = dist_conditional.cdf(x_upper - conditional_mean)
    else:
        # If no upper bounds, use PDF
        cdf_value = 1.0

    pdf_value = multivariate_normal(mu_exact, cov_exact).pdf(x_exact)
    # Return the combined result (PDF * CDF)
    return cdf_value * pdf_value


class Gaussian:
    """Multivariate Gaussians with support for upper limits and missing data.

    Parameters
    -----------
    mean: array
        mean vector of the multivariate normal distribution.
    cov: array
        covariance matrix of the multivariate normal distribution.
    """

    def __init__(self, mean, cov):
        self.ndim = len(mean)
        self.powers = 2**np.arange(self.ndim)
        self.mean = mean
        self.cov = cov
        assert mean.shape == (self.ndim,), (mean.shape,)
        assert cov.shape == (self.ndim, self.ndim), (cov.shape, self.ndim)
        assert np.isfinite(mean).all(), mean
        assert np.isfinite(cov).all(), cov
        self.rvs = {}

    def get_conditional_rv(self, mask):
        """Build conditional distribution.

        Parameters
        -----------
        mask: array
            A boolean mask, indicating whether the entry
            is a value (True) or a upper bound (False).

        Returns
        ----------
        cov_cross: array
            Covariance matrix part of upper bound and value dimensions.
        cov_exact: array
            Covariance matrix part of value dimensions.
        inv_cov_exact: array
            Inverse covariance matrix part of value dimensions.
        rv: scipy.stats.multivariate_normal
            Multivariate Normal Distribution of the upper bound dimensions,
            conditioned with `mask`.
        """
        key = self.powers[mask].sum()
        if key not in self.rvs:
            cov = self.cov
            exact_idx, = np.where(mask)  # Indices of exact values (PDF)
            upper_idx, = np.where(~mask)  # Indices of upper bounds (CDF)
            n_exact = len(exact_idx)
            n_upper = len(upper_idx)

            # Extract values from x
            cov_exact = cov[np.ix_(exact_idx, exact_idx)]  # Covariance for exact values
            cov_upper = cov[np.ix_(upper_idx, upper_idx)]  # Covariance for upper bounds
            cov_cross = cov[np.ix_(exact_idx, upper_idx)]  # Cross-covariance between exact and upper bounds

            # Compute the conditional mean and covariance as a function of the upper bounds dimensions
            # this is conditioned at the position of the exact coordinates.
            if n_upper > 0:
                inv_cov_exact = np.linalg.inv(cov_exact)
                assert inv_cov_exact.shape == (n_exact, n_exact)
                conditional_cov = cov_upper - cov_cross.T @ inv_cov_exact @ cov_cross
                assert conditional_cov.shape == (n_upper, n_upper)
            else:
                # If there are no upper bounds, the conditional covariance is the original one
                inv_cov_exact = None
                conditional_cov = cov_exact

            # Create the conditional multivariate normal distributions
            # print("cov:", conditional_cov)
            # Conditional MVN
            if n_upper > 0:
                rv = multivariate_normal(mean=np.zeros(len(conditional_cov)), cov=conditional_cov)
            else:
                rv = None
            self.rvs[key] = cov_cross, cov_exact, inv_cov_exact, rv

        return self.rvs[key]

    def _prepare_conditional_pdf(self, x, mask=Ellipsis):
        """
        Compute the mixed PDF and CDF for a multivariate Gaussian distribution.

        Parameters:
        - x: The point (vector) at which to evaluate the probability.
             For dimensions where `mask == 0`, this is a value for the PDF.
             For dimensions where `mask == 1`, this is an upper bound for the CDF.
        - mask: A boolean mask of the same shape as `x`.
        - mean: The mean vector of the multivariate normal distribution.
        - cov: The covariance matrix of the multivariate normal distribution.

        Returns:
        - prob: The combined PDF and CDF value.
        """
        mean = self.mean
        if mask is Ellipsis:
            del mask
            mask = np.ones(len(mean), dtype=bool)
        assert mask.shape == (self.ndim,), (self.ndim, mask.shape)

        cov_cross, cov_exact, inv_cov_exact, dist_conditional = self.get_conditional_rv(mask)
        exact_idx, = np.where(mask)  # Indices of exact values (PDF)
        upper_idx, = np.where(~mask)  # Indices of upper bounds (CDF)
        n_exact = len(exact_idx)
        n_upper = len(upper_idx)

        # Partition mean and covariance matrix accordingly
        mu_exact = mean[exact_idx]  # Mean for exact values
        mu_upper = mean[upper_idx]  # Mean for upper bounds

        # Extract values from x
        x_exact = x[:,exact_idx]  # Known values for the PDF
        x_upper = x[:,upper_idx]  # Upper bounds for the CDF

        # Compute the conditional mean for upper bound dimensions
        if n_upper > 0:
            newcov = np.einsum('ji,jk,mk->mi', cov_cross, inv_cov_exact, x_exact - mu_exact.reshape((1, -1)))
            assert newcov.shape == (len(x), n_upper), (newcov.shape, (len(x), n_upper))
            conditional_mean = mu_upper[None,:] + newcov
            assert conditional_mean.shape == ((len(x), n_upper)), (conditional_mean.shape, ((len(x), n_upper)))
        else:
            # If there are no upper bounds, the conditional mean and cov are just the original ones
            conditional_mean = mu_exact.reshape((1, -1))

        assert x_upper.shape == ((len(x), n_upper)), (x_upper.shape, ((len(x), n_upper)))
        return n_upper, n_exact, cov_cross, cov_exact, inv_cov_exact, x_exact, x_upper, mu_exact, mu_upper, conditional_mean, dist_conditional

    def conditional_pdf(self, x, mask=Ellipsis):
        """
        Compute conditional PDF.

        Parameters
        -----------
        x: array
            The points (vector) at which to evaluate the probability.
        mask: array
            A boolean mask of the same shape as `x.shape[1]`, indicating whether the entry
            is a value (True) or a upper bound (False).

        Returns
        ----------
        pdf: array
            Probability density. One value for each `x`.
        """
        n_upper, n_exact, cov_cross, cov_exact, inv_cov_exact, x_exact, x_upper, mu_exact, mu_upper, conditional_mean, dist_conditional = \
            self._prepare_conditional_pdf(x=x, mask=mask)

        # Compute the CDF for the upper bounds
        if n_upper == 0:
            # trivial case: PDF only
            cdf_value = multivariate_normal(np.zeros(self.ndim), self.cov).pdf(x - self.mean.reshape((1, -1)))
        else:
            if n_exact == 0:
                # trivial case: CDF only
                pdf_value = 1
            else:
                pdf_value = multivariate_normal(mu_exact, cov_exact).pdf(x_exact)
            cdf_value = pdf_value * dist_conditional.cdf(x_upper - conditional_mean)

        return cdf_value

    def conditional_logpdf(self, x, mask=Ellipsis):
        """
        Compute conditional log-PDF.

        Parameters
        -----------
        x: array
            The points (vector) at which to evaluate the probability.
        mask: array
            A boolean mask of the same shape as `x.shape[1]`, indicating whether the entry
            is a value (True) or a upper bound (False).

        Returns
        ----------
        logpdf: array
            logarithm of the probability density. One value for each `x`.
        """
        n_upper, n_exact, cov_cross, cov_exact, inv_cov_exact, x_exact, x_upper, mu_exact, mu_upper, conditional_mean, dist_conditional = \
            self._prepare_conditional_pdf(x=x, mask=mask)

        # Compute the CDF for the upper bounds
        if n_upper == 0:
            # trivial case: PDF only
            logcdf_value = multivariate_normal(np.zeros(self.ndim), self.cov).logpdf(x - self.mean.reshape((1, -1)))
        else:
            if n_exact == 0:
                # trivial case: CDF only
                logpdf_value = 0
            else:
                logpdf_value = multivariate_normal(mu_exact, cov_exact).logpdf(x_exact)
            logcdf_value = logpdf_value + dist_conditional.logcdf(x_upper - conditional_mean)

        return logcdf_value

    def pdf(self, x, mask):
        """
        Compute conditional PDF.

        Parameters
        -----------
        x: array
            The points (vector) at which to evaluate the probability.
        mask: array
            A boolean mask of the same shape as `x`, indicating whether the entry
            is a value (True) or a upper bound (False).

        Returns
        ----------
        pdf: array
            probability density. One value for each `x`.
        """
        assert mask.shape == (len(x), self.ndim), (mask.shape, (len(x), self.ndim))
        assert x.shape == (len(mask), self.ndim), (x.shape, (len(x), self.ndim))
        pdf_values = np.zeros(len(x)) * np.nan
        powers = np.einsum('ij,j->i', mask * 1, self.powers)
        unique_powers, unique_indices = np.unique(powers, return_index=True)
        for power, index in zip(unique_powers, unique_indices):
            members = powers == power
            pdf_values[members] = self.conditional_pdf(x[members,:], mask[index, :])
        assert np.isfinite(pdf_values).all(), pdf_values
        return pdf_values

    def logpdf(self, x, mask):
        """
        Compute conditional log-PDF.

        Parameters
        -----------
        x: array
            The points (vector) at which to evaluate the probability.
        mask: array
            A boolean mask of the same shape as `x`, indicating whether the entry
            is a value (True) or a upper bound (False).

        Returns
        ----------
        logpdf: array
            logarithm of the probability density. One value for each `x`.
        """
        assert mask.shape == (len(x), self.ndim), (mask.shape, (len(x), self.ndim))
        assert x.shape == (len(mask), self.ndim), (x.shape, (len(x), self.ndim))
        logpdf_values = np.zeros(len(x)) * np.nan
        powers = (mask * 1) @ self.powers
        unique_powers, unique_indices = np.unique(powers, return_index=True)
        for power, index in zip(unique_powers, unique_indices):
            members = powers == power
            logpdf_values[members] = self.conditional_logpdf(x[members,:], mask[index, :])
        return logpdf_values
