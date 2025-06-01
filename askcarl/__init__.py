"""Multivariate Gaussians with support for upper limits and missing data."""

__version__ = '1.2.1'
from .gaussian import Gaussian, pdfcdf
from .mixture import GaussianMixture
