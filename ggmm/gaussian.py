import numpy as np
from scipy.stats import multivariate_normal


def pdfcdf(x, mask, mean, cov):
    """
    Computes the mixed PDF and CDF for a multivariate Gaussian distribution.

    Parameters:
    - x: The point (vector) at which to evaluate the probability.
         For dimensions where `mask == 0`, this is a value for the PDF.
         For dimensions where `mask == 1`, this is an upper bound for the CDF.
    - mask: A boolean mask of the same shape as `x`.
            `mask[i] == 0` means `x[i]` is a value for the PDF.
            `mask[i] == 1` means `x[i]` is an upper bound for the CDF.
    - mean: The mean vector of the multivariate normal distribution.
    - cov: The covariance matrix of the multivariate normal distribution.

    Returns:
    - prob: The combined PDF and CDF value.
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
    def __init__(self, mean, cov):
        self.ndim = len(mean)
        self.powers = 2**np.arange(self.ndim)
        self.mean = mean
        self.cov = cov
        assert mean.shape == (self.ndim,), (mean.shape,)
        assert cov.shape == (self.ndim, self.ndim), (cov.shape, self.ndim)
        self.rvs = {}

    def get_conditional_rv(self, mask):
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

    def pdf(self, x, mask=Ellipsis):
        """
        Computes the mixed PDF and CDF for a multivariate Gaussian distribution.

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
        # Compute the CDF for the upper bounds
        # print("shift:", x_upper, conditional_mean, len(exact_idx), dist_conditional is not None)
        if n_upper == 0:
            # trivial case: PDF only
            # print("trivial case: PDF only", conditional_mean, dist_conditional)
            cdf_value = multivariate_normal(np.zeros(self.ndim), self.cov).pdf(x - self.mean.reshape((1, -1)))
        #elif n_exact == 0:
        #    # trivial case: CDF only
        #    print("trivial case: CDF only", conditional_mean, dist_conditional.mean, dist_conditional.cov)
        #    cdf_value = multivariate_normal(np.zeros(self.ndim), self.cov).cdf(x - self.mean.reshape((1, -1)))
        else:
            # print("conditional CDF case", dist_conditional.mean, dist_conditional.cov, x_upper - conditional_mean)
            if n_exact == 0:
                pdf_value = 1
            else:
                pdf_value = multivariate_normal(mu_exact, cov_exact).pdf(x_exact)
            print(" --", pdf_value, dist_conditional.cdf(x_upper - conditional_mean))
            cdf_value = pdf_value * dist_conditional.cdf(x_upper - conditional_mean)
        # print("result:", cdf_value)
        
        return cdf_value



