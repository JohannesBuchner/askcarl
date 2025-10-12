"""Multivariate Gaussians with support for upper limits and missing data."""

import numpy as np
from scipy.linalg import solve, solve_triangular
from scipy.stats import multivariate_normal

from .utils import (cov_to_prec_cholesky, multivariate_normal0, mvn_logpdf,
                    mvn_pdf, univariate_normal)

const2pi = np.log(2.0 * np.pi)


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
    precision_cholesky: array
        Cholesky factors of the precision matrix, computed with:
        `solve_triangular(cholesky(cov, lower=True), eyes[D], lower=True)`
    """

    def __init__(self, mean, cov, precision_cholesky=None):
        self.ndim = len(mean)
        self.powers = (3 ** np.arange(self.ndim, dtype=np.int64))
        self.allpowers = int(self.powers.sum())
        assert self.allpowers == self.powers.sum()
        self.mean = mean
        self.cov = cov
        self.precision_cholesky = precision_cholesky
        assert mean.shape == (self.ndim,), (mean.shape,)
        assert cov.shape == (self.ndim, self.ndim), (cov.shape, self.ndim)
        if precision_cholesky is not None:
            assert precision_cholesky.shape == (self.ndim, self.ndim), (precision_cholesky.shape, self.ndim)
            assert np.isfinite(precision_cholesky).all(), cov
        assert np.isfinite(mean).all(), mean
        assert np.isfinite(cov).all(), cov
        self.rvs = {}
        self.prec_upper_cache = {}
        if self.precision_cholesky is not None:
            # Global log det(Λ) and Λ
            self.logdet_precision = 2.0 * np.sum(np.log(np.diag(self.precision_cholesky)))
            self.precision = self.precision_cholesky.T @ self.precision_cholesky

    def get_conditional_rv(self, mask, key=None):
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
        if key is None:
            if mask is Ellipsis:
                key = self.allpowers
            else:
                key = self.powers[mask].sum()
                assert mask.shape == (self.ndim,), (self.ndim, mask.shape)
        if key not in self.rvs:
            cov = self.cov
            if mask is Ellipsis:
                n_exact = self.ndim
                n_upper = 0
                exact_idx = None
                upper_idx = None
                mu_exact = self.mean
                mu_upper = upper_idx

                cov_exact = cov
                prec_chol_exact = self.precision_cholesky
                cov_exact_sol = None

                # If there are no upper bounds, the conditional covariance is the original one
                conditional_cov = cov_exact
                cov_cross = None
            else:
                non_mask = ~mask
                exact_idx = None
                upper_idx = None
                n_exact = mask.sum()
                n_upper = self.ndim - n_exact

                # Partition mean and covariance matrix accordingly
                mu_exact = self.mean[mask]  # Mean for exact values
                mu_upper = self.mean[non_mask]  # Mean for upper bounds

                # Compute the conditional mean and covariance as a function of the upper bounds dimensions
                # this is conditioned at the position of the exact coordinates.
                cov_upper = cov[non_mask,:][:,non_mask]  # Covariance for upper bounds
                cov1 = cov[mask,:]
                cov_cross = cov1[:, non_mask]  # Cross-covariance between exact and upper bounds

                # Extract values from x
                cov_exact = cov1[:, mask]  # Covariance for exact values
                cov_exact_sol = solve(cov_exact, cov_cross, assume_a='pos')
                conditional_cov = cov_upper - cov_cross.T @ cov_exact_sol
                prec_chol_exact = None
                # Only factor cov_exact if we do NOT have a global precision factor
                if n_exact > 0 and self.precision_cholesky is None:
                    try:
                        prec_chol_exact = cov_to_prec_cholesky(cov_exact)
                    except ValueError:
                        pass
                assert conditional_cov.shape == (n_upper, n_upper)

            # Create the conditional multivariate normal distributions
            if n_upper == 1:
                rv = univariate_normal(mean=np.zeros(n_upper), cov=conditional_cov)
            elif n_upper > 0:
                rv = multivariate_normal0(
                    cov=conditional_cov,
                    precision_cholesky=prec_chol_exact, allow_singular=True)
            else:
                rv = None
            self.rvs[key] = cov_cross, cov_exact, cov_exact_sol, prec_chol_exact, \
                rv, exact_idx, upper_idx, n_exact, n_upper, mu_exact, mu_upper

        return self.rvs[key]

    def _marginal_logpdf_precision(self, x_exact, mu_exact, exact_idx, upper_idx, mask):
        # Marginal logpdf over observed E using global Λ = R^T R
        # Requires: self.precision_cholesky, self.logdet_precision, self.precision
        R = self.precision_cholesky
        # Select columns for observed and missing sets
        R_E = R[:, mask]         # shape (n, |E|)
        R_U = R[:, ~mask]         # shape (n, |U|)
        # Small precision blocks via Gram matrices
        Lambda_EE = R_E.T @ R_E       # (|E|, |E|)
        Lambda_EU = R_E.T @ R_U       # (|E|, |U|)
        Lambda_UU = R_U.T @ R_U       # (|U|, |U|)
        # Cholesky of Λ_UU (lower)
        L_UU = np.linalg.cholesky(Lambda_UU)
        logdet_UU = 2.0 * np.sum(np.log(np.diag(L_UU)))
        # Residuals on E
        v_E = x_exact - mu_exact[None, :]      # (m, |E|)
        # b_E = v_E @ Λ_EE
        b_E = v_E @ Lambda_EE                  # (m, |E|)
        # b_U^T = v_E @ Λ_EU  => b_U has shape (|U|, m) after transpose
        b_U_T = v_E @ Lambda_EU                # (m, |U|)
        # Solve Λ_UU y_U = b_U for multiple RHS using L_UU
        z = solve_triangular(L_UU, b_U_T.T, lower=True)
        y_U = solve_triangular(L_UU.T, z, lower=False)
        # c_E = (Λ_EU @ y_U)^T = y_U^T @ Λ_EU^T
        c_E = y_U.T @ Lambda_EU.T              # (m, |E|)
        # Quadratic form per row
        q = np.sum(v_E * (b_E - c_E), axis=1)
        # log det Σ_EE = − log det Λ + log det Λ_UU
        logdet_Sigma_EE = -self.logdet_precision + logdet_UU
        return -0.5 * (q + logdet_Sigma_EE + v_E.shape[1] * const2pi)

    def _prepare_conditional_pdf(self, x, mask=Ellipsis, key=None):
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
        cov_cross, cov_exact, cov_exact_sol, prec_chol_exact, dist_conditional, \
            exact_idx, upper_idx, n_exact, n_upper, mu_exact, mu_upper = \
            self.get_conditional_rv(mask, key)
        if n_upper == 0:
            x_exact = x
            # If there are no upper bounds, the conditional mean and cov are just the original ones
            conditional_mean = mu_exact.reshape((1, -1))
            x_upper = None
        else:
            # Compute quantities for upper bound dimensions
            x_exact = x[:,mask]   # Known values for the PDF
            x_upper = x[:,~mask]  # Upper bounds for the CDF
            newcov = (x_exact - mu_exact[None, :]) @ cov_exact_sol
            conditional_mean = mu_upper[None, :] + newcov
            assert newcov.shape == (len(x), n_upper), (newcov.shape, (len(x), n_upper))
            assert conditional_mean.shape == ((len(x), n_upper)), (conditional_mean.shape, ((len(x), n_upper)))
            assert x_upper.shape == ((len(x), n_upper)), (x_upper.shape, ((len(x), n_upper)))

        return n_upper, n_exact, cov_cross, cov_exact, cov_exact_sol, prec_chol_exact, \
            x_exact, x_upper, mu_exact, mu_upper, exact_idx, upper_idx, conditional_mean, dist_conditional

    def conditional_pdf(self, x, mask=Ellipsis, key=None):
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
        # Compute the CDF for the upper bounds
        if mask is Ellipsis or mask.all():
            # trivial case: PDF only
            if self.precision_cholesky is None:
                return multivariate_normal(np.zeros(self.ndim), self.cov).pdf(x - self.mean.reshape((1, -1)))
            else:
                return mvn_pdf(x, self.mean, self.precision_cholesky)

        n_upper, n_exact, cov_cross, cov_exact, cov_exact_sol, prec_chol_exact, \
            x_exact, x_upper, mu_exact, mu_upper, exact_idx, upper_idx, conditional_mean, dist_conditional = \
            self._prepare_conditional_pdf(x=x, mask=mask, key=key)

        if n_exact == 0:
            # trivial case: CDF only
            pdf_value = 1
        else:
            if prec_chol_exact is None:
                if n_exact == 1:
                    pdf_value = univariate_normal(mu_exact, cov_exact).pdf(x_exact)
                else:
                    pdf_value = multivariate_normal(mu_exact, cov_exact).pdf(x_exact)
            else:
                pdf_value = mvn_pdf(x_exact, mu_exact, prec_chol_exact)
        assert dist_conditional is not None, (mask, n_upper, n_exact)
        cdf_value = pdf_value * dist_conditional.cdf(x_upper - conditional_mean)

        return cdf_value

    def conditional_logpdf(self, x, mask=Ellipsis, key=None):
        """
        Compute conditional log-PDF.

        Parameters
        ----------
        x: array
            The points (vector) at which to evaluate the probability.
        mask: array
            A boolean mask of the same shape as `x.shape[1]`, indicating whether the entry
            is a value (True) or a upper bound (False).
        key: int
            key for hashing

        Returns
        -------
        logpdf: array
            logarithm of the probability density. One value for each `x`.
        """
        # Compute the CDF for the upper bounds
        if mask is Ellipsis or mask.all():
            # trivial case: PDF only
            if self.precision_cholesky is not None:
                return mvn_logpdf(x, self.mean, self.precision_cholesky)
            else:
                return multivariate_normal(np.zeros(self.ndim), self.cov).logpdf(x - self.mean.reshape((1, -1)))

        n_upper, n_exact, cov_cross, cov_exact, inv_cov_exact, prec_chol_exact, \
            x_exact, x_upper, mu_exact, mu_upper, exact_idx, upper_idx, conditional_mean, dist_conditional = \
            self._prepare_conditional_pdf(x=x, mask=mask, key=key)
        if n_exact == 0:
            # trivial case: CDF only
            logpdf_value = 0
        else:
            if self.precision_cholesky is not None:
                if key is None:
                    key = self.allpowers if (mask is Ellipsis) else self.powers[mask].sum()
                # Use global precision-based marginal over observed E
                logpdf_value = self._marginal_logpdf_precision(
                    x_exact=x_exact, mu_exact=mu_exact,
                    exact_idx=exact_idx, upper_idx=upper_idx, mask=mask)
            else:
                # Fallback to covariance-based path
                if prec_chol_exact is None:
                    if n_exact == 1:
                        logpdf_value = univariate_normal(mu_exact, cov_exact).logpdf(x_exact)
                    else:
                        logpdf_value = multivariate_normal(mu_exact, cov_exact).logpdf(x_exact)
                else:
                    logpdf_value = mvn_logpdf(x_exact, mu_exact, prec_chol_exact)
        assert dist_conditional is not None, (mask, n_upper, n_exact)

        assert x_upper.ndim == 2
        mask_inf = np.isposinf(x_upper)
        cols_keep = np.any(~mask_inf, axis=0)
        # the trivial case is PDF only, because all are uninformative upper limits
        rows_trivial = mask_inf.all(axis=1)
        rows_nontrivial = ~rows_trivial
        if np.any(rows_nontrivial) and cols_keep.any():
            n_keep = cols_keep.sum()
            if n_keep == len(cols_keep):
                X = x_upper[rows_nontrivial,:]
                dist_reduced = dist_conditional
                M = conditional_mean[rows_nontrivial,:]
            elif n_keep == 1:
                idx = np.where(cols_keep)[0][0]
                dist_reduced = univariate_normal(0, cov=dist_conditional.cov[idx,idx])
                X = x_upper[rows_nontrivial, idx]
                M = conditional_mean[rows_nontrivial, idx]
            else:
                cov_remaining = dist_conditional.cov[cols_keep,:][:, cols_keep]
                prec_chol_exact = cov_to_prec_cholesky(cov_remaining)
                dist_reduced = multivariate_normal0(
                    cov=cov_remaining, precision_cholesky=prec_chol_exact,
                    allow_singular=True)
                X = x_upper[:, cols_keep][rows_nontrivial,:]
                M = conditional_mean[:, cols_keep][rows_nontrivial,:]
            # Only rows with any finite bounds need the CDF
            logcdf_value = np.zeros(len(x))
            try:
                logcdf_nontrivial = dist_reduced.logcdf(X - M)
            except ValueError:
                # invalid construction.
                # this data point is not possible with this Gaussian.
                return -np.inf + logcdf_value
            if np.iscomplexobj(logcdf_nontrivial):
                logcdf_nontrivial = np.real(logcdf_nontrivial)
            logcdf_value[rows_nontrivial] += logcdf_nontrivial
            return logcdf_value + logpdf_value
        else:
            return logpdf_value

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
        code = mask.astype(np.int64) + 2 * np.isposinf(x).astype(np.int64)
        powers = code @ self.powers
        unique_powers, unique_indices = np.unique(powers, return_index=True)
        for power, index in zip(unique_powers, unique_indices):
            members = powers == power
            if power == self.allpowers:
                mask_here = Ellipsis
            else:
                mask_here = mask[index, :]
            pdf_values[members] = self.conditional_pdf(x[members,:], mask_here, key=power)
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
        if mask is Ellipsis:
            return self.conditional_logpdf(x, Ellipsis).reshape((len(x),))
        assert mask.shape == (len(x), self.ndim), (mask.shape, (len(x), self.ndim))
        assert x.shape == (len(mask), self.ndim), (x.shape, (len(x), self.ndim))
        code = mask.astype(np.int64) + 2 * np.isposinf(x).astype(np.int64)
        powers = code @ self.powers
        unique_powers, unique_indices = np.unique(powers, return_index=True)
        if len(unique_powers) == 1 and unique_powers[0] == self.allpowers:
            return self.conditional_logpdf(x, Ellipsis).reshape((len(x),))
        logpdf_values = np.zeros(len(x)) * np.nan
        for power, index in zip(unique_powers, unique_indices):
            members = powers == power
            if power == self.allpowers:
                mask_here = Ellipsis
            else:
                mask_here = mask[index, :]
            logpdf_values[members] = self.conditional_logpdf(x[members,:], mask_here, key=power)
        return logpdf_values
