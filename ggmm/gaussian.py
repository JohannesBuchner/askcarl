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
        inv_cov_upper = np.linalg.inv(cov_upper)
        newcov = ((cov_cross @ inv_cov_upper) @ (x_upper.T - mu_upper[:,None])).T
        assert newcov.shape == (len(x_upper), len(mu_exact)),  (len(x_upper), len(mu_exact), newcov.shape)
        conditional_mean = mu_exact[None,:] + newcov
        conditional_cov = cov_exact - cov_cross @ inv_cov_upper @ cov_cross.T
    else:
        # If there are no upper bounds, the conditional mean and cov are just the original ones
        conditional_mean = mu_exact
        conditional_cov = cov_exact

    # Create the conditional multivariate normal distributions
    dist_conditional = multivariate_normal(mean=np.zeros(len(conditional_cov)), cov=conditional_cov)  # Conditional MVN

    # Compute the CDF for the upper bounds
    if len(upper_idx) > 0:
        print(x_exact, conditional_mean)
        cdf_value = dist_conditional.cdf(x_exact - conditional_mean)
    else:
        # If no upper bounds, use PDF
        cdf_value = 1.0

    # Return the combined result (PDF * CDF)
    return cdf_value #, conditional_mean, conditional_cov, x_exact


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

            # Extract values from x
            cov_exact = cov[np.ix_(exact_idx, exact_idx)]  # Covariance for exact values
            cov_upper = cov[np.ix_(upper_idx, upper_idx)]  # Covariance for upper bounds
            cov_cross = cov[np.ix_(exact_idx, upper_idx)]  # Cross-covariance between exact and upper bounds

            # Compute the conditional mean and covariance for the remaining dimensions (upper bounds)
            print("conditioning:", len(upper_idx) > 0, upper_idx)
            if len(upper_idx) > 0:
                inv_cov_upper = np.linalg.inv(cov_upper)
                conditional_cov = cov_exact - cov_cross @ inv_cov_upper @ cov_cross.T
            else:
                # If there are no upper bounds, the conditional mean and cov are just the original ones
                inv_cov_upper = None
                conditional_cov = cov_exact

            # Create the conditional multivariate normal distributions
            print("cov:", conditional_cov)
            # Conditional MVN
            if len(exact_idx) > 0:
                rv = multivariate_normal(mean=np.zeros(len(exact_idx)), cov=conditional_cov)
            else:
                rv = None
            self.rvs[key] = cov_cross, inv_cov_upper, rv

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

        cov_cross, inv_cov_upper, dist_conditional = self.get_conditional_rv(mask)
        exact_idx, = np.where(mask)  # Indices of exact values (PDF)
        upper_idx, = np.where(~mask)  # Indices of upper bounds (CDF)

        # Partition mean and covariance matrix accordingly
        mu_exact = mean[exact_idx]  # Mean for exact values
        mu_upper = mean[upper_idx]  # Mean for upper bounds


        #if len(upper_idx) > 0:
        #    return np.ones(len(x))

        # Extract values from x
        x_exact = x[:,exact_idx]  # Known values for the PDF
        x_upper = x[:,upper_idx]  # Upper bounds for the CDF

        # Compute the conditional mean and covariance for the remaining dimensions (upper bounds)
        if len(upper_idx) > 0:
            newcov = ((cov_cross @ inv_cov_upper) @ (x_upper.T - mu_upper[:,None])).T
            conditional_mean = mu_exact[None,:] + newcov
        else:
            # If there are no upper bounds, the conditional mean and cov are just the original ones
            conditional_mean = mu_exact

        # Compute the CDF for the upper bounds
        print("shift:", x_exact - conditional_mean, x_exact, conditional_mean, len(exact_idx), dist_conditional is not None)
        if len(upper_idx) == 0:
            # trivial case: PDF only
            cdf_value = multivariate_normal(np.zeros(self.ndim), self.cov).pdf(x - self.mean.reshape((1, -1)))
        elif len(exact_idx) == 0:
            # trivial case: CDF only
            cdf_value = multivariate_normal(np.zeros(self.ndim), self.cov).cdf(x - self.mean.reshape((1, -1)))
        else:
            cdf_value = dist_conditional.cdf(x_exact - conditional_mean)

        #if dist_conditional is not None:
        #    # cdf_value = dist_conditional.cdf(x_exact - conditional_mean)
        #else:
        #    # If no upper bounds, use PDF
        #    cov_exact = self.cov[np.ix_(exact_idx, exact_idx)]  # Covariance for exact values
        #    print("exact PDF:", x_exact, cov_exact, cov_exact)
        #    cdf_value = multivariate_normal(np.zeros_like(mu_exact), cov_exact).pdf(x_exact - mu_exact.reshape((1, -1)))
        #    #cdf_value = 1.0

        # Return the combined result (PDF * CDF)
        return cdf_value



